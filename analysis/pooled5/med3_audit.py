#!/usr/bin/env python
"""健康管家 Top1w 的盲标：三个独立视角判「产品层 vs 用户内容」，口径与 health-pool3 那一轮相同。

码本 `work/医疗3/audit_codebook.md` 是**重建**的（原来写在工作流提示里，没落盘），所以这一轮除了
989 个新的歧义串，还**重标了原来 1,192 串里的一个随机样本**：拿新票和当时存下来的票逐条比，
才能说这份重建的码本复不复刻得了原来的仪器。没有这个校准，新快照的清洗口径就只是"看起来一样"。

    # 三个视角 + 校准样本（默认 300）
    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/med3_audit.py
    # 只跑聚合（三个视角都已完成时）
    MED3_AGG_ONLY=1 HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/med3_audit.py

断点续跑：每个视角一份 JSONL，重跑跳过已完成的 sid。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
BASE = QM / "analysis/pooled5/work/医疗3"
PREV = QM / "analysis/pooled5/work/医疗随机"

LENS_ORDER = ["origin", "standalone", "form"]
CODES = ["A", "F", "P", "D", "C", "U"]
LENS_FOCUS = {
    "origin": "你现在只用**来源视角**：先不管这串长什么样，只问「这句话最可能是谁打出来的」——"
              "一个正在用健康助手的人会在输入框里打出它吗，还是它更像界面上被点了一下的东西？",
    "standalone": "你现在只用**独立性视角**：把这串单独拿出来看，它还成立吗？"
                  "用户键入的内容通常自带诉求、脱离上下文也读得懂；作答选项与分面按钮往往必须挂在上一个问题下才有意义。",
    "form": "你现在只用**形态视角**：只看表面形态——长度、是否完整句、有无人称、有无问句标记、"
            "是否平行短语、是否像一个标签而不是一句话。不要揣测意图，只描述这串写成了什么样子。",
}
URL, MODEL, KEYNAME = "https://api.deepseek.com/chat/completions", "deepseek-v4-pro", "DEEPSEEK_API_KEY"
BATCH = int(os.environ.get("MED3_BATCH", "40"))
WORKERS = int(os.environ.get("MED3_WORKERS", "12"))
CALIB = int(os.environ.get("MED3_CALIB", "300"))
ATTEMPTS = 6


def _key() -> str:
    for line in (QM / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(KEYNAME + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{KEYNAME} missing from QMine/.env")


def _call(codebook: str, lens: str, rows: list[tuple[str, str]], key: str) -> str:
    listing = "\n".join(f"{sid}\t{q}" for sid, q in rows)
    user = (f"{codebook}\n\n---\n\n{LENS_FOCUS[lens]}\n\n"
            f"下面是 {len(rows)} 条待判的串（sid<TAB>串）：\n{listing}\n\n"
            '只输出 JSON：{"labels": [{"sid": "...", "code": "A|F|P|D|C|U"}]}\n'
            "每个 sid 恰好一次。拿不准判 U。")
    body = {"model": MODEL, "temperature": 0, "max_tokens": 8000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "你只输出 JSON，不解释。"},
                         {"role": "user", "content": user}]}
    with httpx.Client(timeout=600) as c:
        r = c.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def label(codebook: str, lens: str, rows: list[tuple[str, str]], key: str) -> list[dict]:
    want = {sid for sid, _ in rows}
    err = ""
    for attempt in range(ATTEMPTS):
        try:
            text = _call(codebook, lens, rows, key)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(8 * (attempt + 1))
            continue
        try:
            t = text.strip()
            m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
            j = json.loads(m.group(1) if m else t)
            got = {x.get("sid"): x.get("code") for x in (j.get("labels") or [])}
            bad = sorted({c for c in got.values() if c not in CODES})
            if bad:
                raise ValueError(f"codes outside the codebook: {bad[:3]}")
            missing = want - set(got)
            if missing:
                raise ValueError(f"{len(missing)} sids unlabelled")
            return [{"sid": s, "code": got[s]} for s in sorted(want)]
        except Exception as e:  # noqa: BLE001
            err = f"parse: {str(e)[:150]}"
    if len(rows) > 1:                      # isolate the offender, never lose the batch
        mid = len(rows) // 2
        return label(codebook, lens, rows[:mid], key) + label(codebook, lens, rows[mid:], key)
    print(f"  ⚠ {lens}: giving up on 1 string ({err})", flush=True)
    return [{"sid": rows[0][0], "code": "U"}]     # the codebook's own tie-break


def fleiss_kappa(M: np.ndarray) -> float:
    n = M.sum(1)[0]
    N = M.shape[0]
    p = M.sum(0) / (N * n)
    P = ((M * M).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P.mean(), (p * p).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def run_lens(codebook: str, lens: str, sub: pd.DataFrame, key: str) -> pd.DataFrame:
    out = BASE / "audit" / lens
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done: set[str] = set()
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            try:
                done |= {x["sid"] for x in json.loads(line)["labels"]}
            except Exception:  # noqa: BLE001
                pass
    todo = [(r.sid, r.query) for r in sub.itertuples() if r.sid not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"  {lens}: {len(sub):,} strings, {len(done):,} done, {len(batches)} batches", flush=True)
    if batches:
        buf: list[str] = []
        def go(b):
            buf.append(json.dumps({"labels": label(codebook, lens, b, key)}, ensure_ascii=False))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(go, batches))
        with raw.open("a", encoding="utf-8") as f:
            for line in buf:
                f.write(line + "\n")
    rows = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        try:
            rows += json.loads(line)["labels"]
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(rows).drop_duplicates("sid").set_index("sid")["code"]


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    codebook = (BASE / "audit_codebook.md").read_text(encoding="utf-8")
    sub = pd.read_csv(BASE / "audit_subset.csv", encoding="utf-8-sig", keep_default_na=False)
    key = _key()
    votes = {}
    for lens in LENS_ORDER:
        votes[lens] = run_lens(codebook, lens, sub, key) if not os.environ.get("MED3_AGG_ONLY") \
            else pd.read_csv(BASE / "audit" / lens / "labels.csv", encoding="utf-8-sig").set_index("sid")["code"]
        (BASE / "audit" / lens).mkdir(parents=True, exist_ok=True)
        votes[lens].reset_index().to_csv(BASE / "audit" / lens / "labels.csv", index=False, encoding="utf-8-sig")

    rows = []
    for r in sub.itertuples():
        v = "".join(votes[l].get(r.sid, "U") for l in LENS_ORDER)
        cnt = pd.Series(list(v)).value_counts()
        maj = cnt.index[0] if cnt.iloc[0] >= 2 else "NOMAJ"
        rows.append({"sid": r.sid, "query": r.query, "kind": r.kind, "votes": v,
                     "majority": maj, "unanimous": len(set(v)) == 1})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "audit_labels.csv", index=False, encoding="utf-8-sig")

    new, cal = out[out["kind"] == "new"], out[out["kind"] == "calibration"]
    M = np.stack([[sum(1 for l in LENS_ORDER if votes[l].get(s, "U") == c) for c in CODES] for s in sub["sid"]])
    summ = {"n": len(out), "n_new": len(new), "n_calibration": len(cal),
            "fleiss_kappa_全部": round(fleiss_kappa(M), 3),
            "三票全同占比%": round(100 * float(out["unanimous"].mean()), 1),
            "新串_多数分布": new["majority"].value_counts().to_dict(),
            "新串_三票一致判为产品层": int(((new["unanimous"]) & (new["majority"] != "U")).sum())}
    # 校准：重建的码本复刻得了原来的仪器吗？
    if len(cal):
        prev = pd.read_csv(PREV / "ai_audit_labels.csv", encoding="utf-8-sig").set_index("query")
        j = cal.set_index("query").join(prev[["votes", "majority"]], rsuffix="_prev", how="inner")
        summ["校准"] = {
            "n": int(len(j)),
            "多数码一致率%": round(100 * float((j["majority"] == j["majority_prev"]).mean()), 1),
            "删/留结论一致率%": round(100 * float(
                ((j["majority"] != "U") & (j["votes"].map(lambda v: len(set(v)) == 1))
                 == ((j["majority_prev"] != "U") & (j["votes_prev"].map(lambda v: len(set(v)) == 1)))).mean()), 1),
            "逐票一致率%": round(100 * float(np.mean([a == b for va, vb in zip(j["votes"], j["votes_prev"])
                                                     for a, b in zip(va, vb)])), 1),
        }
    (BASE / "audit_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
