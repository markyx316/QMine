#!/usr/bin/env python
"""语用子功能盲标的跨模型信度：另一家模型（DeepSeek）用同一份码本、同样不给快照，独立标一个分层子样本，
与三名 Claude 视角的多数标签比 Cohen κ。

为什么需要：三个 Claude 视角只差一行「读法」提示，码本又是按字面规则写的，三者 κ=0.99 量到的是**同一模型的自洽**，
不是这套子功能划分的信度（实测 NEWS_EVENT：三视角各自手工逐行给码，没有互相读文件，仍然 99.3% 全同）。
换一家模型独立标，κ 才回答「换一个认真读码本的标注者，会不会给出同样的划分」。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_facet_crossmodel.py 金融8 NEWS_EVENT [更多意图…]

输入：facets/<CODE>_codebook.json、facets/<CODE>_labels.csv（p5_facet_aggregate 的输出）。
输出：facets/crossmodel/<CODE>_{raw.jsonl,labels.csv,confusion.csv}、facets/crossmodel/summary.json。
抽样：每个快照最多 PER_SNAPSHOT 行（固定种子），小快照全取；按块发送，每块覆盖检查（缺码、重复、码本外的码都拒收并重试）。
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK  # noqa: E402

QM = Path(__file__).resolve().parents[2]
MODEL = "deepseek-v4-pro"
URL = "https://api.deepseek.com/chat/completions"
PER_SNAPSHOT = 25
CHUNK = 40
SEED = 20260915


def _key() -> str:
    for line in (QM / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DEEPSEEK_API_KEY missing from QMine/.env")


def _call(messages: list[dict], key: str) -> str:
    # httpx, not urllib: behind this machine's proxy settings urllib's connection is closed at once (measured), httpx gets HTTP 200
    body = {"model": MODEL, "messages": messages, "temperature": 0, "response_format": {"type": "json_object"}, "max_tokens": 8000}
    with httpx.Client(timeout=600) as c:
        r = c.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _parse(text: str) -> dict:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    return json.loads(t)


def code_chunk(cb: dict, rows: pd.DataFrame, key: str, raw_log: list) -> dict:
    valid = [c["code"] for c in cb["codes"]] + ([cb["unsure_code"]] if cb.get("unsure_code") else [])
    listing = "\n".join(f"{r.sid}\t{r.query}" for r in rows.itertuples())
    sys_msg = ("You are a careful annotator of Chinese finance search/assistant queries. Apply the codebook exactly as written, "
               "one code per row, reading each row on its own. Do not invent codes. Output JSON only.")
    user = (f"CODEBOOK for intent {cb.get('intent', '')}:\n{json.dumps(cb, ensure_ascii=False)}\n\n"
            f"ROWS (sid<TAB>query), {len(rows)} rows, order is arbitrary and the source is hidden:\n{listing}\n\n"
            'Return {"labels": [{"sid": "...", "code": "..."}]} with every sid exactly once. '
            f"Allowed codes: {', '.join(valid)}.")
    want = set(rows["sid"])
    last_err = ""
    for attempt in range(3):
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user + (f"\n\nYour previous answer was rejected: {last_err}" if last_err else "")}]
        try:
            text = _call(msgs, key)
        except Exception as e:  # network or HTTP error: back off and retry
            last_err = f"call failed: {e}"
            time.sleep(5 * (attempt + 1))
            continue
        raw_log.append({"attempt": attempt, "sids": sorted(want), "response": text})
        try:
            labs = _parse(text)["labels"]
            got = {}
            for x in labs:
                s, c = str(x["sid"]), str(x["code"])
                if s in got:
                    raise ValueError(f"duplicate sid {s}")
                got[s] = c
            missing, extra = want - set(got), set(got) - want
            bad = sorted({c for c in got.values() if c not in valid})
            if missing or extra or bad:
                raise ValueError(f"missing {len(missing)} sids, {len(extra)} unknown sids, codes outside codebook {bad[:5]}")
            return got
        except Exception as e:
            last_err = str(e)[:300]
    raise RuntimeError(f"chunk failed after 3 attempts: {last_err}")


def run(domain: str, code: str, key: str) -> dict:
    fdir = WORK / domain / "intent_structure" / "facets"
    out = fdir / "crossmodel"
    out.mkdir(parents=True, exist_ok=True)
    cb = json.loads((fdir / f"{code}_codebook.json").read_text(encoding="utf-8"))
    cb = {"intent": code, **cb}
    lab = pd.read_csv(fdir / f"{code}_labels.csv")
    rng = np.random.default_rng(SEED)
    parts = []
    for s, g in lab.groupby("source", sort=True):
        take = g if len(g) <= PER_SNAPSHOT else g.iloc[rng.choice(len(g), PER_SNAPSHOT, replace=False)]
        parts.append(take)
    sub = pd.concat(parts)
    sub = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
    chunks = [sub.iloc[i:i + CHUNK] for i in range(0, len(sub), CHUNK)]
    raw_log: list = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda ch: code_chunk(cb, ch[["sid", "query"]], key, raw_log), chunks))
    ds = {}
    for r in results:
        ds.update(r)
    assert set(ds) == set(sub["sid"]), "cross-model coverage mismatch"
    with (out / f"{code}_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in raw_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sub["deepseek"] = sub["sid"].map(ds)
    lenses = [c for c in ("answer", "form", "goal") if c in sub.columns]
    sub[["sid", "source", "多数", *lenses, "deepseek"]].to_csv(out / f"{code}_labels.csv", index=False, encoding="utf-8-sig")
    maj = sub[sub["多数"] != "无多数"]
    k_major = cohen_kappa_score(maj["多数"], maj["deepseek"])
    k_lens = {l: round(float(cohen_kappa_score(sub[l], sub["deepseek"])), 3) for l in lenses}
    k_within = {f"{a}~{b}": round(float(cohen_kappa_score(sub[a], sub[b])), 3) for i, a in enumerate(lenses) for b in lenses[i + 1:]}
    conf = pd.crosstab(maj["多数"], maj["deepseek"])
    conf.to_csv(out / f"{code}_confusion.csv", encoding="utf-8-sig")
    per_code = {}
    for c in sorted(set(maj["多数"]) | set(maj["deepseek"])):
        a, b = maj["多数"] == c, maj["deepseek"] == c
        per_code[c] = {"claude_n": int(a.sum()), "deepseek_n": int(b.sum()), "both": int((a & b).sum())}
    dis = maj[maj["多数"] != maj["deepseek"]]
    pairs = (dis["多数"] + " -> " + dis["deepseek"]).value_counts().head(8).to_dict()
    res = {"n_sub": len(sub), "n_with_majority": len(maj), "model": MODEL, "raw_agreement_%": round(100 * float((maj["多数"] == maj["deepseek"]).mean()), 1),
           "kappa_claude_majority_vs_deepseek": round(float(k_major), 3), "kappa_each_lens_vs_deepseek": k_lens,
           "kappa_between_claude_lenses_same_rows": k_within, "per_code": per_code, "top_disagreements": pairs}
    print(code, json.dumps(res, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    domain, codes = sys.argv[1], sys.argv[2:]
    key = _key()
    sp = WORK / domain / "intent_structure" / "facets" / "crossmodel" / "summary.json"
    summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for c in codes:
        summ[c] = run(domain, c, key)
        sp.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
