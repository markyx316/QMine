# -*- coding: utf-8 -*-
"""Intent tables T12–T20 for SEARCH_VS_ASSISTANT_2026.zh.md, computed from sva_final_rows2.parquet and the
sva_intent2 outputs (shares/diffs CSV, assistant_nn parquet with ladder, wrap pairs). Appends nothing: writes
report_tables_intent_v3.md."""
import os, sys, math, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
F = pd.read_parquet(os.environ.get("FINAL_OUT", f"{SP}/sva_final_rows2.parquet")); F["query"] = F["query"].astype(str)
PFX = os.environ.get("INTENT_PREFIX", f"{SP}/sva_intent2")
SH = pd.read_csv(PFX + "_shares.csv"); DF = pd.read_csv(PFX + "_diffs.csv")
A = pd.read_parquet(PFX + "_assistant_nn.parquet"); PW = pd.read_csv(PFX + "_wrap_pairs.csv")
fm = frame(); CODES = list(fm.FRAME); NAME = {k: v[0] for k, v in fm.FRAME.items()}
SURFS = [x for x in ["search_top1000", "search_top10k", "assistant_top1k", "assistant_random1k"] if x in set(SH.surface)]
SL = {"search_top1000": "搜索前1000", "search_top10k": "搜索前1万", "assistant_top1k": "助手头部1k", "assistant_random1k": "助手随机1k"}
DOMS = list(CAT) + ["合计"]
U = F[F.tier == "user"]
OUT = open(f"{SP}/report_tables_intent_v3.md", "w", encoding="utf-8")
def W(x=""): print(x, file=OUT)
def pc(x, d=1): return "—" if x != x else f"{x*100:.{d}f}%"
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p, max(0, c-h), min(1, c+h)
def kappa(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    if n == 0: return float("nan"), float("nan"), 0
    po = (a == b).mean(); pe = sum((a == c).mean()*(b == c).mean() for c in set(a) | set(b))
    return ((po-pe)/(1-pe) if pe < 1 else float("nan")), po, n
def dn(d): return "5域合计" if d == "合计" else d

W("### T12 统一意图占比（可判定行，去掉 U13；行占比 [Wilson 95% CI]）")
for d in DOMS:
    x = SH[(SH.view == "interpretable") & (SH.domain == d)]
    W(f"\n**{dn(d)}**（n：" + "，".join(f"{SL[sf]} {int(x[x.surface == sf].n.iloc[0]):,}" for sf in SURFS) + "）\n")
    W("| 意图 | " + " | ".join(SL[sf] for sf in SURFS) + " |"); W("|---|" + "---|"*len(SURFS))
    for c in CODES:
        if c == "U13": continue
        W(f"| {c} {NAME[c]} | " + " | ".join((lambda r: f"{r.share*100:.1f} [{r.lo*100:.1f}–{r.hi*100:.1f}]")(x[(x.surface == sf) & (x.code == c)].iloc[0]) for sf in SURFS) + " |")
W("\n### T12b U13（无法判定）在全部 user 行中的占比")
W("| 领域 | " + " | ".join(f"{SL[sf]} U13" for sf in SURFS) + " |"); W("|---|" + "---|"*len(SURFS))
for d in DOMS:
    x = SH[(SH.view == "all") & (SH.domain == d) & (SH.code == "U13")]
    W(f"| {dn(d)} | " + " | ".join((lambda r: f"{r.share*100:.1f} [{r.lo*100:.1f}–{r.hi*100:.1f}]")(x[x.surface == sf].iloc[0]) for sf in SURFS) + " |")

W("\n### T12c 头部切片按流量（PV）加权的意图占比（可判定行；%）")
HS=[x for x in ["search_top1000","search_top10k","assistant_top1k"] if x in SURFS]
W("| 领域 | 意图 | " + " | ".join(SL[sf] for sf in HS) + " |"); W("|---|---|" + "---|"*len(HS))
for d in DOMS:
    x = SH[(SH.view == "interpretable") & (SH.domain == d)]
    for c in CODES:
        if c == "U13": continue
        vals = [x[(x.surface == sf) & (x.code == c)].pv_share.iloc[0] for sf in HS]
        if max(vals) >= 0.03: W(f"| {dn(d)} | {c} {NAME[c]} | " + " | ".join(f"{v*100:.1f}" for v in vals) + " |")
W("\n### T13 最大的意图差异（可判定行；百分点 [Newcombe 95% CI]；只列 CI 不含 0 且 |差异|≥4pp 的）")
PAIRS = [("search_top1000", "assistant_top1k"), ("search_top10k", "assistant_top1k"), ("assistant_top1k", "assistant_random1k")]
for d in DOMS:
    W(f"\n**{dn(d)}**\n"); W("| 对比 | 变化最大的意图（从 → 到，差异） |"); W("|---|---|")
    for s1, s2 in PAIRS:
        if s1 not in SURFS: continue
        x = DF[(DF.view == "interpretable") & (DF.domain == d) & (DF.frm == s1) & (DF.to == s2) & (DF.code != "U13") & (DF.sig) & (DF["diff"].abs() >= 0.04)].copy()
        x = x.reindex(x["diff"].abs().sort_values(ascending=False).index).head(6)
        W(f"| {SL[s1]} → {SL[s2]} | " + "；".join(f"{r.code}{r.name} {r.p_from*100:.1f}→{r.p_to*100:.1f}（{r.diff*100:+.1f} [{r.lo*100:+.1f}, {r.hi*100:+.1f}]）" for r in x.itertuples()) + " |")

W("\n### T14 标注可靠性：deepseek-v4-flash vs qwen3.8-flash（去重 query；两者都有标签的子集）")
R = U.dropna(subset=["u_ds", "u_qw"])
W("| 范围 | n | 一致率 | Cohen κ |"); W("|---|---|---|---|")
for nm, x in [("全部", R.drop_duplicates("query"))] + [(SL[sf], R[R.surface == sf].drop_duplicates("query")) for sf in SURFS] + [(d, R[R.domain == d].drop_duplicates("query")) for d in CAT]:
    k, po, n = kappa(x.u_ds, x.u_qw); W(f"| {nm} | {n:,} | {pc(po)} | {k:.3f} |")
Rp = U.dropna(subset=["p_ds", "p_qw"]).drop_duplicates("query"); k, po, n = kappa(Rp.p_ds.astype(int), Rp.p_qw.astype(int))
W(f"| p（个人处境）字段 | {n:,} | {pc(po)} | {k:.3f} |")
Rq = R.drop_duplicates("query")
W("\n| 意图 | deepseek 判为该类的条数 | 其中 qwen 也判为该类 | 同判率 |"); W("|---|---|---|---|")
for c in CODES:
    nd = int((Rq.u_ds == c).sum()); ag = int(((Rq.u_ds == c) & (Rq.u_qw == c)).sum()); W(f"| {c} {NAME[c]} | {nd:,} | {ag:,} | {pc(ag/max(nd,1))} |")

W("\n### T15 陈述个人/家人具体处境（p=1）的占比（全部 user 行；[Wilson 95% CI]）")
W("| 领域 | " + " | ".join(SL[sf] for sf in SURFS) + " |"); W("|---|" + "---|"*len(SURFS))
for d in DOMS:
    t = []
    for sf in SURFS:
        x = U[(U.surface == sf) & ((U.domain == d) if d != "合计" else True)].dropna(subset=["p_ds"]); k = int(x.p_ds.sum()); p, lo, hi = wilson(k, len(x))
        t.append(f"{p*100:.1f} [{lo*100:.1f}–{hi*100:.1f}]")
    W(f"| {dn(d)} | " + " | ".join(t) + " |")

LV = ["A 原样搜索词", "B 近似改写(≥0.80)", "C 同题延伸(0.70–0.80)", "D 搜索无近邻(<0.70)", "E 个人处境/个案判断", "F 对话/委托"]
W("\n### T16 从搜索到助手的六级阶梯（助手 user 行；优先级 F>E>A>B>C>D；行占比，头部附 PV 占比）")
W("| 领域·切片 | n | " + " | ".join(LV) + " |"); W("|---|---|" + "---|"*len(LV))
for d in DOMS:
    for sf in ("assistant_top1k", "assistant_random1k"):
        x = A[(A.surface == sf) & ((A.domain == d) if d != "合计" else True)].dropna(subset=["level"]); w = x.pv.astype(float)
        W(f"| {dn(d)}·{SL[sf]} | {len(x):,} | " + " | ".join(pc((x.level == lv).mean()) + (f"（PV {pc((w*(x.level == lv)).sum()/w.sum())}）" if sf == "assistant_top1k" else "") for lv in LV) + " |")

W("\n### T17 意图 × 离搜索多远（5域合计助手 user 行；每类中“搜索前1万里无近邻(<0.70)”的比例，按头部排序）")
W("| 意图 | 助手头部1k n | 头部：原样重合 | 头部：无近邻 | 助手随机1k n | 随机：无近邻 |"); W("|---|---|---|---|---|---|")
rows = []
for c in CODES:
    h = A[(A.surface == "assistant_top1k") & (A.u_ds == c)].dropna(subset=["band"]); t = A[(A.surface == "assistant_random1k") & (A.u_ds == c)].dropna(subset=["band"])
    if len(h) >= 15 or len(t) >= 15:
        rows.append(((h.band == "远").mean() if len(h) else 1, c, len(h), (h.band == "完全相同").mean() if len(h) else float("nan"), len(t), (t.band == "远").mean() if len(t) else float("nan")))
for far, c, nh, ex, nt, ft in sorted(rows):
    W(f"| {c} {NAME[c]} | {nh:,} | {pc(ex)} | {pc(far) if nh else '—'} | {nt:,} | {pc(ft)} |")

W("\n### T18 同一需求的意图转移（包装对 = 搜索头部查询被原样包含进更长的助手查询；近邻对 = 0.70≤余弦<0.999）")
near = A[(A.sim >= 0.70) & (A.sim < 0.999)].dropna(subset=["u_ds", "u_nn"])
for nm, df, src, dst in (("包装对", PW, "u_core", "u_wrap"), ("近邻对", near, "u_nn", "u_ds")):
    for sf in ("assistant_top1k", "assistant_random1k"):
        x = df[df.surface == sf]
        if not len(x): continue
        W(f"\n**{nm}·{SL[sf]}**：n={len(x):,}，意图不变 {pc((x[src] == x[dst]).mean())}\n")
        W("| 搜索侧意图 → 助手侧意图 | 对数 | 占全部对 |"); W("|---|---|---|")
        for (c1, c2), v in x[x[src] != x[dst]].groupby([src, dst]).size().sort_values(ascending=False).head(8).items():
            W(f"| {c1}{NAME[c1]} → {c2}{NAME[c2]} | {v} | {pc(v/len(x))} |")

W("\n### T19 同一意图，不同写法（5域合计 user 行；长度中位 / 疑问句% / 第一人称% / p=1%）")
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_metrics.py"), encoding="utf-8").read(); ns = {}; exec(src[src.index("M={"):src.index("C={d:")], {}, ns); M = ns["M"]
import re
def has(series, pat): p = re.compile(pat); return series.map(lambda x: bool(p.search(x)))
W("| 意图 | " + " | ".join(SL[sf] for sf in SURFS) + " |"); W("|---|" + "---|"*len(SURFS))
for c in CODES:
    t = []; ok = 0
    for sf in SURFS:
        x = U[(U.surface == sf) & (U.u_ds == c)]; ok += len(x) >= 20
        t.append("—" if len(x) < 20 else f"n={len(x):,}；{x['query'].str.len().median():.0f}字 / {pc(has(x['query'], M['疑问句']).mean(),0)} / {pc(has(x['query'], M['第一人称']).mean(),0)} / {pc(x.p_ds.mean(),0)}")
    if ok >= 2: W(f"| {c} {NAME[c]} | " + " | ".join(t) + " |")

W("\n### T20 各自 taxonomy 经 crosswalk 映射后，与统一标注一致的比例（同一行）")
W("| 切片 | n | 一致 |"); W("|---|---|---|")
CW = fm.CROSSWALK
for sf in SURFS:
    x = U[U.surface == sf].dropna(subset=["u_ds"]); cw = [fm.frame_code(r, c) if (r, c) in CW else None for r, c in zip(x.own_run, x.own_code)]
    x = x.assign(cw=cw).dropna(subset=["cw"]); W(f"| {SL[sf]} | {len(x):,} | {pc((x.cw == x.u_ds).mean())} |")
OUT.close(); print("ok")
