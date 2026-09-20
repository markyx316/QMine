#!/usr/bin/env python
"""医疗随机：把三个盲标视角的结果聚合成 `work/医疗随机/ai_audit_labels.csv`，并报一致度。

输入：盲标工作流的结果（每个视角一份 labels：sid + code），以及 `work/医疗随机/ai_audit_subset.csv`（sid → query）。
输出：每个串一行——三票 `votes`（按 origin/standalone/form 顺序）、`majority`（>=2 票；三票各异记 U，
因为码本的兜底规则就是「拿不准就当用户内容」）、是否全同。另报 Fleiss κ 与两两一致率。

    python analysis/pooled5/medrand_audit_aggregate.py <workflow_result.json>

覆盖是硬条件：每个视角必须给出**全部** sid 的码，多一个少一个都拒收（health-pool2 的教训：
一个视角少标 12 条时，majority 会静默变成两票制）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent / "work" / "医疗随机"
LENS_ORDER = ["origin", "standalone", "form"]
CODES = ["A", "F", "P", "D", "C", "U"]


def fleiss_kappa(M: np.ndarray) -> float:
    n = M.sum(1)[0]
    N = M.shape[0]
    p = M.sum(0) / (N * n)
    P = ((M * M).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P.mean(), (p * p).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def main() -> int:
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("result", raw)
    sub = pd.read_csv(WORK / "ai_audit_subset.csv", encoding="utf-8-sig")
    sids = list(sub["sid"])
    votes = {}
    for item in items:
        lens = item["lens"]
        assert lens in LENS_ORDER, f"unknown lens {lens}"
        labs = {r["sid"]: r["code"] for r in item["labels"]}
        assert len(labs) == len(item["labels"]), f"{lens}: duplicate sids"
        missing, extra = set(sids) - set(labs), set(labs) - set(sids)
        assert not missing and not extra, f"{lens}: {len(missing)} sids unlabelled, {len(extra)} unknown sids"
        bad = sorted({c for c in labs.values() if c not in CODES})
        assert not bad, f"{lens}: codes outside the codebook {bad}"
        votes[lens] = labs
    assert set(votes) == set(LENS_ORDER), f"expected lenses {LENS_ORDER}, got {sorted(votes)}"
    rows = []
    for sid, q in zip(sub["sid"], sub["query"]):
        v = "".join(votes[l][sid] for l in LENS_ORDER)
        cnt = pd.Series(list(v)).value_counts()
        maj = cnt.index[0] if cnt.iloc[0] >= 2 else "NOMAJ"   # three different readings: no majority.
        # The BUILDER keeps those rows as user rows (codebook tie-break: if unsure, keep); writing NOMAJ
        # here rather than U keeps the report's "no majority" count honest instead of silently 0.
        rows.append({"sid": sid, "query": q, "votes": v, "majority": maj, "unanimous": len(set(v)) == 1})
    out = pd.DataFrame(rows)
    # the product-layer table aggregates PV by majority code, so carry each string's PV alongside
    raw = pd.read_excel(Path(__file__).resolve().parents[2] / "data/raw/健康管家-医疗-随机10000-20260915..xlsx", dtype=str)
    pv = dict(zip(raw["query"].astype(str), pd.to_numeric(raw["total_pv"], errors="coerce")))
    out["pv_raw"] = out["query"].map(pv).astype(float)
    assert out["pv_raw"].notna().all(), "a labelled string is not in the raw export"
    out.to_csv(WORK / "ai_audit_labels.csv", index=False, encoding="utf-8-sig")
    M = np.stack([[sum(1 for l in LENS_ORDER if votes[l][sid] == c) for c in CODES] for sid in sids])
    pair = {f"{a}~{b}": round(float(np.mean([votes[a][s] == votes[b][s] for s in sids])), 3)
            for i, a in enumerate(LENS_ORDER) for b in LENS_ORDER[i + 1:]}
    summ = {"n": len(out), "fleiss_kappa": round(fleiss_kappa(M), 3),
            "unanimous_%": round(100 * float(out["unanimous"].mean()), 1),
            "majority_counts": out["majority"].value_counts().to_dict(),
            "unanimous_product_rows": int(((out["unanimous"]) & (out["majority"] != "U")).sum()),
            "pairwise_agreement": pair,
            "three_way_split_rows": int(out["votes"].map(lambda v: len(set(v)) == 3).sum())}
    (WORK / "ai_audit_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
