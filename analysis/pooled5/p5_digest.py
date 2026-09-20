# -*- coding: utf-8 -*-
"""A compact, readable digest of one domain's canonical tables — what an analyst needs in front
of them before interpreting anything. Everything here is a lookup from work/<domain>/*.csv, so
the digest and the tables cannot disagree.

    python analysis/pooled5/p5_digest.py 金融 > work/金融/digest.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import CONTRASTS, SOURCES, SRC_ZH, WORK, available, load, work_file

pd.set_option("display.width", 200, "display.max_colwidth", 60)


def digest(domain: str) -> None:
    w = WORK / domain
    load(domain)  # not used below — called for its positional-join assertion
    print(f"================ {domain} · POOLED-5 数字底稿 ================")
    print("\n[1] 语料构成（清洗后进入挖掘的行数 / 原始行数 / 剔除 PV%）")
    print(pd.read_csv(w / "composition.csv").to_string(index=False))
    print("\n[1b] 清洗剔除的行（按来源×层）")
    print(pd.read_csv(w / "cleaning_removed.csv").to_string(index=False))
    print("\n[2] 本次挖出的意图体系（L1）")
    defs = pd.read_csv(w / "class_definitions.csv")
    if len(defs):
        for _, r in defs.iterrows():
            print(f"  · {r['code']} {r['name']}: {str(r['definition'])[:150]}")
    print("\n[3] 各来源的意图占比（行占比 %；括号内为该来源内 Wilson 95% 区间）")
    s = pd.read_csv(w / "shares_td_l1_name.csv")
    piv = s.pivot_table(index="label", columns="source", values="row_share").reindex(
        columns=[c for c in SOURCES if c in set(s.source)])
    piv = (piv * 100).round(1)
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    piv.columns = [SRC_ZH[c] for c in piv.columns]
    print(piv.to_string())
    print("\n[3b] 同一张表，按流量（pv_norm，来源内归一）")
    pv = pd.read_csv(w / "pv_shares_td_l1_name.csv", index_col=0)
    pv = (pv / pv.sum() * 100).round(1)
    pv.columns = [SRC_ZH.get(c, c) for c in pv.columns]
    print(pv.loc[piv.index.intersection(pv.index)].to_string())
    print("\n[4] 四条对比线的整体距离（TVD=需要换类的行占比；V=Cramér's V）")
    print(pd.read_csv(w / "contrast_summary_td_l1_name.csv").to_string(index=False))
    print("\n[5] 每条对比线上显著变化的类目（Newcombe 95% 区间不含 0，按幅度排序，取前 8）")
    t = pd.read_csv(w / "diffs_td_l1_name.csv")
    for a, b, why in CONTRASTS:
        sub = t[(t.a == a) & (t.b == b) & t.sig]
        if not len(sub):
            continue
        sub = sub.reindex(sub.diff_pp.abs().sort_values(ascending=False).index).head(8)
        print(f"\n  — {SRC_ZH[a]} → {SRC_ZH[b]}（{why}）")
        for _, r in sub.iterrows():
            print(f"     {r.label[:26]:<28} {r.share_a*100:5.1f}% → {r.share_b*100:5.1f}%  "
                  f"{r.diff_pp:+6.1f}pp [{r.lo_pp:+.1f},{r.hi_pp:+.1f}] {r.kind}")
    print("\n[6] 写法标记（行占比）")
    f = pd.read_csv(w / "form.csv").set_index("source")
    f = f.reindex([x for x in SOURCES if x in f.index])
    f.index = [SRC_ZH[i] for i in f.index]
    # The length columns are characters and the marker columns are proportions — scaling them
    # together printed 疑问句 as 0.4 when it is 40.5%.
    num = f.select_dtypes("number")
    lens = [c for c in num.columns if c.startswith("len_")]
    props = [c for c in num.columns if c not in lens + ["n"]]
    print(pd.concat([num[["n"] + lens].round(1), (num[props] * 100).round(1)], axis=1).to_string())
    print("\n[7] 逐字重合（A 的不同字符串里有多少也出现在 B）")
    ov = pd.read_csv(w / "overlap.csv", index_col=0)
    ov.index = [SRC_ZH.get(i, i) for i in ov.index]
    ov.columns = [SRC_ZH.get(c, c) for c in ov.columns]
    print((ov * 100).round(1).to_string())
    print("\n[8] 体系适配度（这套体系是在 87% 搜索行上拟合的）")
    fit = pd.read_csv(w / "fit.csv")
    fit["source"] = fit["source"].map(SRC_ZH)
    print(fit.round(3).to_string(index=False))
    nn = work_file("semantic_nn.parquet")
    if nn.exists():
        x = pd.read_parquet(nn)
        x = x[x.domain == domain]
        if len(x):
            print("\n[9] 在 2026 搜索里能否找到近邻（bge-base-zh 余弦；2025搜索=时间对照）")
            bands = [(0.999, 1.01, "完全相同"), (0.90, 0.999, "≥0.90"), (0.80, 0.90, "0.80–0.90"),
                     (0.70, 0.80, "0.70–0.80"), (-1, 0.70, "<0.70")]
            rows = []
            for src in SOURCES:
                g = x[x.source == src]
                if not len(g):
                    continue
                rows.append({"来源": SRC_ZH[src], "n": len(g), "中位相似度": round(float(g.sim.median()), 3),
                             **{lab: round(float(((g.sim >= lo) & (g.sim < hi)).mean() * 100), 1) for lo, hi, lab in bands}})
            print(pd.DataFrame(rows).to_string(index=False))
    wr = work_file("wrap_summary.csv")
    if wr.exists():
        x = pd.read_csv(wr)
        x = x[x.domain == domain]
        if len(x):
            print("\n[10] 包装对（搜索词被原样写进更长的助手查询）")
            print(x.to_string(index=False))
    td = work_file("taxonomy_delta_summary.csv")
    if td.exists():
        x = pd.read_csv(td)
        x = x[x.domain == domain]
        if len(x):
            print("\n[11] 加入助手行之后，同样 2 万条搜索行的划分变了多少（对照：*-pool 只有搜索）")
            print(x.to_string(index=False))
            mix = pd.read_csv(WORK / domain / "taxonomy_delta_newclass_mix.csv")
            print("  新体系里助手占比最高的类：")
            print(mix.head(8).to_string(index=False))
    print("\n[12] 每个类目在每个来源里的例子 —— 见 examples_td_l1_name.csv（top_pv 与随机各 6 条）")
    print(f"[13] 逐行数据：analysis/pooled5/deliverables/{domain}_pooled5_labeled.parquet")


if __name__ == "__main__":
    for x in (sys.argv[1:] or available()):
        digest(x)
