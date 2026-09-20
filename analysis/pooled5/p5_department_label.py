#!/usr/bin/env python
"""把 医疗8 与 医疗随机 的每一个不同 query 判一个**科室**，码本见 `work/科室/codebook.md`。

为什么要这一层：两份语料都没有能用的科室标签——医疗8 根本没有科室列（它的 `l2` 只覆盖 4.4% 的行，
而且是内容类型不是科室），医疗随机 的 `legacy_dept` 44.6% 是空档、词表里没有肿瘤科、「内科」把
消化/心血管/内分泌/肿瘤/肾/呼吸/神经七个二级科目混在一起。详见码本开头的实测数字。

这是**纯后处理**：不碰任何运行产物，不改任何已交付文件，只往 `work/科室/` 写新东西。

    # 主标注（DeepSeek，全量）
    P5_DEPT_PROVIDER=deepseek .venv/bin/python analysis/pooled5/p5_department_label.py
    # 跨模型复核（Kimi，抽样）——同一个模型的多个视角只能测自洽，换供应商才是信度
    P5_DEPT_PROVIDER=kimi P5_DEPT_SAMPLE=1500 .venv/bin/python analysis/pooled5/p5_department_label.py

断点续跑：每批写一行 JSONL，重跑时已完成的 sid 直接跳过。
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

QM = Path(__file__).resolve().parents[2]
BASE = QM / "analysis/pooled5/work/科室"
CORPORA = {"医疗8": "data/raw/pooled5/医疗8_pooled5.parquet",
           "医疗随机": "data/raw/pooled5/医疗随机_pooled5.parquet"}

CLINICAL = ["NK_HX", "NK_XH", "NK_SJ", "NK_XX", "NK_XY", "NK_SN", "NK_NF", "NK_FS",
            "WK_PW", "WK_SJ", "WK_GK", "WK_MN", "WK_XX", "WK_SZ",
            "FCK_FK", "FCK_CK", "FCK_SZ", "EK", "YK", "EB", "KQ", "PF", "PF_XB", "MR",
            "JS", "GR", "ZL", "JZ", "KF", "TT", "ZY", "NANKE", "JC", "QK"]
NONCLINICAL = ["YP_GENERIC", "YS", "JY", "FYL", "WFPD"]
CODES = set(CLINICAL) | set(NONCLINICAL)

PROVIDERS = {
    # flash is the primary: this is a bounded classification against a written codebook, which is
    # exactly what the pipeline's own annotator role uses flash for. Measured on the first 1,000
    # strings, pro ran at 227 strings/min — 4.2 hours for the corpus — so pro's output is kept as a
    # CROSS-CAPACITY probe (`deepseek_pro_probe`) rather than as the primary pass.
    "deepseek": {"url": "https://api.deepseek.com/chat/completions", "model": "deepseek-v4-flash",
                 "key": "DEEPSEEK_API_KEY"},
    "deepseek_pro": {"url": "https://api.deepseek.com/chat/completions", "model": "deepseek-v4-pro",
                     "key": "DEEPSEEK_API_KEY"},
    "kimi": {"url": "https://openrouter.ai/api/v1/chat/completions", "model": "moonshotai/kimi-k3",
             "key": "OPENROUTER_API_KEY"},
}
PROVIDER = os.environ.get("P5_DEPT_PROVIDER", "deepseek")
SAMPLE = int(os.environ.get("P5_DEPT_SAMPLE", "0"))
BATCH = int(os.environ.get("P5_DEPT_BATCH", "40"))
WORKERS = int(os.environ.get("P5_DEPT_WORKERS", "16"))
ATTEMPTS = 6
OUT = BASE / PROVIDER if not SAMPLE else BASE / f"{PROVIDER}_s{SAMPLE}"


def _key(name: str) -> str:
    for line in (QM / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} missing from QMine/.env")


def build_inputs() -> pd.DataFrame:
    """One row per DISTINCT query across both corpora, with a stable sid.

    Distinct rather than per-row because the label is a property of the string; the
    per-row join back onto each corpus happens in the aggregation step.
    """
    seen: dict[str, None] = {}
    for p in CORPORA.values():
        for q in pd.read_parquet(QM / p)["query"].astype(str):
            seen.setdefault(q, None)
    rows = [{"sid": f"d{i:06d}", "query": q} for i, q in enumerate(sorted(seen))]
    return pd.DataFrame(rows)


PROMPT = """你是中国三甲医院的分诊台护士长，按《医疗机构诊疗科目名录》给每一条搜索/问答 query 判一个科室。

{codebook}

下面是 {n} 条待判的串（sid<TAB>query）：
{listing}

只输出 JSON：{{"labels": [{{"sid": "...", "code": "..."}}], "n_read": 读过的串数}}
每个 sid 恰好出现一次，code 必须是码表里的码。"""


def _call(codebook: str, rows: list[tuple[str, str]], key: str) -> str:
    cfg = PROVIDERS[PROVIDER]
    listing = "\n".join(f"{sid}\t{q}" for sid, q in rows)
    body = {"model": cfg["model"], "temperature": 0, "max_tokens": 8000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "你只输出 JSON，不解释。"},
                         {"role": "user", "content": PROMPT.format(codebook=codebook, n=len(rows), listing=listing)}]}
    with httpx.Client(timeout=600) as c:
        r = c.post(cfg["url"], json=body, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    ch = r.json()["choices"][0]
    if ch.get("finish_reason") == "content_filter":
        raise RuntimeError("content_filter")
    return ch["message"]["content"]


def label(codebook: str, rows: list[tuple[str, str]], key: str) -> list[dict]:
    want = {sid for sid, _ in rows}
    err = ""
    for attempt in range(ATTEMPTS):
        try:
            text = _call(codebook, rows, key)
        except Exception as e:  # noqa: BLE001 — network, HTTP 429/5xx, provider filter
            err = f"{type(e).__name__}: {str(e)[:160]}"
            if len(rows) > 1 and "content_filter" in err:
                mid = len(rows) // 2
                return label(codebook, rows[:mid], key) + label(codebook, rows[mid:], key)
            time.sleep(8 * (attempt + 1))
            continue
        try:
            t = text.strip()
            m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
            j = json.loads(m.group(1) if m else t)
            labs = j.get("labels") or []
            got = {x.get("sid"): x.get("code") for x in labs}
            bad = sorted({c for c in got.values() if c not in CODES})
            if bad:
                raise ValueError(f"codes outside the codebook: {bad[:3]}")
            missing = want - set(got)
            if missing:
                raise ValueError(f"{len(missing)} sids unlabelled")
            return [{"sid": s, "code": got[s]} for s in sorted(want)]
        except Exception as e:  # noqa: BLE001
            err = f"parse: {str(e)[:160]}"
    # ONE UNANSWERABLE BATCH MUST NOT LOSE THE OTHER 948. The first full run died here:
    # a batch of 60 came back with an empty body six times and the RuntimeError propagated
    # out of ThreadPoolExecutor.map, aborting every worker. Split instead, so the failure is
    # isolated to the single string that causes it; a lone string that still fails is recorded
    # as UNLABELLED and counted, never silently dropped and never fatal.
    if len(rows) > 1:
        mid = len(rows) // 2
        return label(codebook, rows[:mid], key) + label(codebook, rows[mid:], key)
    print(f"  ⚠ giving up on 1 string after {ATTEMPTS} attempts ({err})", flush=True)
    return [{"sid": rows[0][0], "code": "__FAILED__"}]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    codebook = (BASE / "codebook.md").read_text(encoding="utf-8")
    assert codebook.strip(), "codebook.md is empty — write it before labelling"
    src = BASE / "inputs.csv"
    if not src.exists():
        build_inputs().to_csv(src, index=False, encoding="utf-8-sig")
    g = pd.read_csv(src, encoding="utf-8-sig", keep_default_na=False)
    if SAMPLE:
        g = g.sample(n=min(SAMPLE, len(g)), random_state=20260917).sort_values("sid")

    raw, done_path = OUT / "raw.jsonl", OUT / "labels.csv"
    done: set[str] = set()
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            try:
                done |= {x["sid"] for x in json.loads(line)["labels"]}
            except Exception:  # noqa: BLE001 — a truncated final line on a killed run
                pass
    todo = [(r.sid, r.query) for r in g.itertuples() if r.sid not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{PROVIDER}/{PROVIDERS[PROVIDER]['model']}: {len(g):,} strings, {len(done):,} already done, "
          f"{len(batches):,} batches of {BATCH} on {WORKERS} workers", flush=True)
    key = _key(PROVIDERS[PROVIDER]["key"])
    lock_out: list[str] = []
    n_done = [0]

    def run(b: list[tuple[str, str]]) -> None:
        labs = label(codebook, b, key)
        lock_out.append(json.dumps({"labels": labs}, ensure_ascii=False))
        n_done[0] += 1
        if n_done[0] % 25 == 0:
            with raw.open("a", encoding="utf-8") as f:
                while lock_out:
                    f.write(lock_out.pop() + "\n")
            print(f"  {n_done[0]}/{len(batches)} batches", flush=True)

    if batches:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(run, batches))
        with raw.open("a", encoding="utf-8") as f:
            while lock_out:
                f.write(lock_out.pop() + "\n")

    rows = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        try:
            rows += json.loads(line)["labels"]
        except Exception:  # noqa: BLE001
            pass
    out = pd.DataFrame(rows).drop_duplicates("sid")
    out.loc[out["code"] == "__FAILED__", "code"] = pd.NA
    out = g.merge(out, on="sid", how="left")
    unl = int(out["code"].isna().sum())
    out.to_csv(done_path, index=False, encoding="utf-8-sig")
    print(f"→ {done_path}  ({len(out):,} strings, {unl} unlabelled)")
    print(out["code"].value_counts().head(12).to_string())
    return 1 if unl else 0


if __name__ == "__main__":
    random.seed(20260917)
    raise SystemExit(main())
