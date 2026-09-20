# -*- coding: utf-8 -*-
"""Every number the cross-domain synthesis derives that is not already in a cross table.

`work/cross/*` answers questions inside one domain. This script answers the four that only
exist across domains, and writes them to `work/cross/synthesis_derived.csv` so the synthesis
can cite an artifact instead of a paragraph:

  A. 轴的可排序性 — TVD intervals overlap or not, within a domain and across domains.
     An overlap means the two axes (or the two domains) MUST NOT be ordered.
  B. 统一13类的行级对齐 — the five taxonomies use different class names, so the only
     harmonisation that does not lose rows is the row-level U01–U13 map already measured in
     `work/<domain>/vs_unified_crosstab_by_source.csv`. Per-class dominant-U is NOT used:
     31 of 94 classes have dominant-U purity below 0.5.
  C. 时间线的分解 — a string gets one label per run, so the shared strings contribute exactly
     zero to the year-on-year difference. Therefore
         TVD(time) = churn_rate x TVD(what fell off, what came in)
     exactly. The script checks the identity rather than assuming it.
  D. 语音是不是头尾的混合 — the minimum-TVD mixture a*head + (1-a)*tail, its residual, and
     whether the residual survives dropping the repeated rows and the ~40-char truncations.
  F. 小格子复核 — B says a class rose in 4 or 5 domains. Several of those cells hold 1-4 rows.
     A direction carried by a 1-row cell is not a cross-domain regularity, so F re-reports the
     raw counts, the >=20-row floor of section 2.1, AND an exact test, because the Newcombe
     interval on a 0->1 cell is marginal in a way the exact test is not. (The synthesis shipped
     `4/5 rules` whose 医疗/人物/金融 members were 2, 4 and 1 rows; verifier 2026-09-13.)
  G. 共有桶的 U 层 — sections 3.2/8-4 called the shared half "bare entity" in all five domains.
     That was a merge by CLASS NAME, which 2.1 forbids. G maps the shared rows themselves.
  H. 语音残差的逐类分解 — section 4.3 quoted `voice minus head` as if it were the residual of the
     4.2 mixture. It is not: the three classes it named have residuals of about zero. H computes
     the real residual voice - (a*head + (1-a)*tail) per class, and asserts that.
  I. 包装对的 same_intent 头尾差 — section 3.3 read 人物 as "the one domain that reverses".
     Its interval contains 0, so the direction is not measured there.

    python analysis/pooled5/p5_synthesis_recompute.py
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK, cross_dir, load, newcombe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SVA = ROOT / "analysis/sva2026/work"   # the U01-U13 labels, same source as p5_vs_unified.py

UN = {"U01": "导航直达", "U02": "裸实体", "U03": "事实与数值", "U04": "解释与介绍", "U05": "核实与动态",
      "U06": "办事与操作", "U07": "个案判断与建议", "U08": "清单与推荐", "U09": "获取现成内容",
      "U10": "生成与编辑", "U11": "会话与系统指令", "U12": "违规或灰色内容", "U13": "无法判定"}
DOMS = ["金融", "医疗", "教育", "影视", "人物"]
AXIS = {"时间（搜索内部，2025→2026）": "时间", "界面（同为头部流量）": "界面",
        "深度（助手头部→长尾）": "深度", "输入方式（打字头部→语音）": "输入·对头部",
        "输入方式（打字长尾→语音）": "输入·对长尾", "界面+深度（不可单独归因）": "界面+深度"}


def tvd(a: pd.Series, b: pd.Series) -> float:
    i = sorted(set(a.index) | set(b.index))
    return float(0.5 * (a.reindex(i).fillna(0) - b.reindex(i).fillna(0)).abs().sum())


def a_orderability(rows: list) -> None:
    ci = pd.read_csv(cross_dir() / "tvd_ci.csv")
    ci["ax"] = ci["why"].map(AXIS)
    for dom, g in ci.groupby("domain"):
        g = g[g.ax != "界面+深度"]
        for x, y in itertools.combinations(list(g.itertuples()), 2):
            ov = not (x.lo > y.hi or y.lo > x.hi)
            rows.append({"block": "A 轴可排序性·域内", "domain": dom, "key": f"{x.ax} vs {y.ax}",
                         "value": "区间重叠·不可排序" if ov else "可排序",
                         "detail": f"{x.tvd:.4f}[{x.lo:.4f},{x.hi:.4f}] vs {y.tvd:.4f}[{y.lo:.4f},{y.hi:.4f}]"})
    for ax, g in ci.groupby("ax"):
        for x, y in itertools.combinations(list(g.sort_values("tvd", ascending=False).itertuples()), 2):
            ov = not (x.lo > y.hi or y.lo > x.hi)
            rows.append({"block": "A 轴可排序性·跨域", "domain": ax, "key": f"{x.domain} vs {y.domain}",
                         "value": "区间重叠·不可排序" if ov else "可排序",
                         "detail": f"{x.tvd:.4f}[{x.lo:.4f},{x.hi:.4f}] vs {y.tvd:.4f}[{y.lo:.4f},{y.hi:.4f}]"})


def b_unified(rows: list) -> None:
    lines = [("2026search", "assistant_top", "界面"), ("assistant_top", "assistant_random", "深度"),
             ("2026search", "assistant_random", "界面+深度")]
    per = {}
    for d in DOMS:
        x = pd.read_csv(WORK / d / "vs_unified_crosstab_by_source.csv")
        per[d] = x.groupby("source")[list(UN)].sum()
    tally: dict = {}
    for d in DOMS:
        g = per[d]
        for a, b, nm in lines:
            na, nb = int(g.loc[a].sum()), int(g.loc[b].sum())
            for u, uname in UN.items():
                dd, lo, hi = newcombe(int(g.loc[a, u]), na, int(g.loc[b, u]), nb)
                sig = lo > 0 or hi < 0
                rows.append({"block": f"B 统一13类Δ·{nm}", "domain": d, "key": uname,
                             "value": round(dd * 100, 2),
                             "detail": f"{g.loc[a,u]/na*100:.2f}%→{g.loc[b,u]/nb*100:.2f}% "
                                       f"[{lo*100:+.2f},{hi*100:+.2f}] sig={sig}"})
                if sig:
                    tally.setdefault((nm, uname, "升" if dd > 0 else "降"), []).append(d)
    for (nm, uname, sgn), ds in sorted(tally.items()):
        if len(ds) >= 4:
            rows.append({"block": f"B 跨领域一致·{nm}", "domain": "—", "key": f"{uname} {sgn}",
                         "value": f"{len(ds)}/5", "detail": "、".join(ds)})


def c_time(rows: list) -> None:
    ci = pd.read_csv(cross_dir() / "tvd_ci.csv")
    t = ci[ci.why.str.startswith("时间")].set_index("domain")
    for d in DOMS:
        T = pd.read_csv(WORK / d / "turnover_classes.csv")
        piv = T.pivot_table(index="label", columns="bucket", values="share").fillna(0.0)
        n = T.groupby("bucket")["n"].first()
        stay, gone, new = n["留存(两年都在)"], n["掉榜(只在2025)"], n["新进(只在2026)"]
        s25, s26 = stay / (stay + gone), stay / (stay + new)
        gn = tvd(piv["掉榜(只在2025)"], piv["新进(只在2026)"])
        rec = tvd(s25 * piv["留存(两年都在)"] + (1 - s25) * piv["掉榜(只在2025)"],
                  s26 * piv["留存(两年都在)"] + (1 - s26) * piv["新进(只在2026)"])
        assert abs(rec - t.loc[d, "tvd"]) < 0.002, (d, rec, t.loc[d, "tvd"])
        rows.append({"block": "C 时间线分解", "domain": d, "key": "换血率(2026侧)",
                     "value": round(1 - s26, 4), "detail": f"留存{int(stay)} 掉榜{int(gone)} 新进{int(new)}"})
        rows.append({"block": "C 时间线分解", "domain": d, "key": "TVD(掉榜,新进)",
                     "value": round(gn, 4),
                     "detail": f"TVD时间={t.loc[d,'tvd']:.4f}=换血率×该值，验算{rec:.4f}"})


def d_voice(rows: list) -> None:
    for d in ["金融", "医疗"]:
        D = load(d)
        g = {s: x for s, x in D.groupby("source")}
        top = g["assistant_top"]["td_l1_name"].value_counts(normalize=True)
        ran = g["assistant_random"]["td_l1_name"].value_counts(normalize=True)
        v = g["assistant_voice"]
        for nm, vv in (("原样", v), ("去重复串", v.drop_duplicates(subset=["query"])),
                       ("去≥38字疑似截断", v[v["query"].astype(str).str.len() < 38])):
            voi = vv["td_l1_name"].value_counts(normalize=True)
            best = min(((a, tvd(voi, a * top.add(0 * ran, fill_value=0) + (1 - a) * ran.add(0 * top, fill_value=0)))
                        for a in np.arange(0, 1.001, 0.005)), key=lambda z: z[1])
            rows.append({"block": "D 语音混合拟合", "domain": d, "key": f"{nm}(n={len(vv)})",
                         "value": f"α头部={best[0]:.2f} 残差TVD={best[1]:.4f}",
                         "detail": f"TVD(语音,头部)={tvd(voi,top):.4f} TVD(语音,长尾)={tvd(voi,ran):.4f} "
                                   f"TVD(头部,长尾)={tvd(top,ran):.4f}"})


def e_anchor(rows: list) -> None:
    """The interface line anchors on ONE search snapshot. If that snapshot is itself the product
    of a within-search writing shift (人物, §3.1 of dom_ppl.md), the interface reading inherits it.
    Re-anchor on 2025 and report both the TVD move and any >=4pp mover that changes sign."""
    for d in DOMS:
        D = load(d)
        g = {s: x for s, x in D.groupby("source")}
        n26, n25, nt = len(g["2026search"]), len(g["2025search"]), len(g["assistant_top"])
        c26 = g["2026search"]["td_l1_name"].value_counts()
        c25 = g["2025search"]["td_l1_name"].value_counts()
        ct = g["assistant_top"]["td_l1_name"].value_counts()
        labs = sorted(set(c26.index) | set(c25.index) | set(ct.index))
        t26 = 0.5 * sum(abs(c26.get(l, 0) / n26 - ct.get(l, 0) / nt) for l in labs)
        t25 = 0.5 * sum(abs(c25.get(l, 0) / n25 - ct.get(l, 0) / nt) for l in labs)
        flips, big = [], 0
        for l in labs:
            d26, lo26, hi26 = newcombe(int(c26.get(l, 0)), n26, int(ct.get(l, 0)), nt)
            d25, _, _ = newcombe(int(c25.get(l, 0)), n25, int(ct.get(l, 0)), nt)
            if (lo26 > 0 or hi26 < 0) and abs(d26) >= 0.04:
                big += 1
                if d26 * d25 < 0:
                    flips.append(f"{l} {d26*100:+.1f}pp→{d25*100:+.1f}pp")
        rows.append({"block": "E 界面线换锚", "domain": d, "key": "TVD锚2026→锚2025",
                     "value": f"{t26:.4f}→{t25:.4f} ({t25-t26:+.4f})",
                     "detail": f"≥4pp显著类 {big} 个，换锚后变号 {len(flips)} 个"
                               + ("：" + "；".join(flips) if flips else "")})


def umap() -> pd.Series:
    """The U01-U13 label per string, rebuilt exactly as `p5_vs_unified.py: umap()` does."""
    a = pd.read_parquet(SVA / "label_full/unique_labels.parquet")[["query", "u_ds"]]
    b = pd.read_parquet(SVA / "label_search10k/labels.parquet")[["query", "u_ds"]]
    both = pd.concat([a, b], ignore_index=True).dropna(subset=["u_ds"])
    both = both.drop_duplicates(subset=["query"], keep="first")
    return both.set_index(both["query"].astype(str))["u_ds"]


def f_smallcells(rows: list) -> None:
    """A 4/5 that three domains join with 1, 2 and 4 rows is not a 4/5."""
    lines = [("2026search", "assistant_top", "界面"), ("2026search", "assistant_random", "界面+深度")]
    for u in ["U10", "U11"]:
        for a, b, nm in lines:
            for d in DOMS:
                g = pd.read_csv(WORK / d / "vs_unified_crosstab_by_source.csv").groupby("source")[list(UN)].sum()
                ka, na = int(g.loc[a, u]), int(g.loc[a].sum())
                kb, nb = int(g.loc[b, u]), int(g.loc[b].sum())
                dd, lo, hi = newcombe(ka, na, kb, nb)
                pf = float(fisher_exact([[ka, na - ka], [kb, nb - kb]])[1])
                rows.append({"block": f"F 小格子复核·{nm}", "domain": d, "key": UN[u],
                             "value": f"{ka}→{kb} 行",
                             "detail": f"Δ{dd*100:+.2f}pp [{lo*100:+.2f},{hi*100:+.2f}] "
                                       f"newcombe_sig={lo > 0 or hi < 0} fisher_p={pf:.3g} "
                                       f"过20行最小格子={max(ka, kb) >= 20}"})
    # The three cells the synthesis counted as "成立" and must not: they are 1, 2 and 4 rows.
    for d, u, a, b, k in [("医疗", "U11", "2026search", "assistant_top", 2),
                          ("人物", "U11", "2026search", "assistant_top", 4),
                          ("金融", "U10", "2026search", "assistant_top", 1)]:
        g = pd.read_csv(WORK / d / "vs_unified_crosstab_by_source.csv").groupby("source")[list(UN)].sum()
        assert int(g.loc[b, u]) == k and int(g.loc[a, u]) == 0, (d, u, int(g.loc[b, u]))
        assert k < 20, (d, u, k)


def g_shared(rows: list, U: pd.Series) -> None:
    """What the two surfaces literally share, mapped ROW-WISE to U — not by class name."""
    for d in DOMS:
        D = load(d)
        search = set(D[D.source.isin(["2025search", "2026search"])]["query"].astype(str))
        typed = D[D.source.isin(["assistant_top", "assistant_random"])].copy()
        typed["u"] = typed["query"].astype(str).map(U)
        sh = typed[typed["query"].astype(str).isin(search)]
        vc = sh["td_l1_name"].value_counts(normalize=True)
        sub = sh[sh.td_l1_name == vc.index[0]]
        su = sub["u"].value_counts(normalize=True)
        rows.append({"block": "G 共有桶的U层", "domain": d, "key": vc.index[0],
                     "value": f"该类占共有桶 {vc.iloc[0]*100:.0f}%，其 U 层第一大类 {UN[su.index[0]]} {su.iloc[0]*100:.0f}%",
                     "detail": f"共有 {len(sh)}/{len(typed)} 行（{len(sh)/len(typed)*100:.1f}%）；"
                               f"整桶 裸实体={sh['u'].value_counts(normalize=True).get('U02', 0)*100:.1f}%"})
    # 教育 and 金融 are the two the "5/5 都是裸实体型" sentence could not have: their first
    # class maps to 无法判定 and to 事实与数值.
    assert all(r["block"] != "G 共有桶的U层" or r["domain"] != "教育" or "无法判定" in r["value"] for r in rows)


def h_voice_residual(rows: list) -> None:
    """voice - (a*head + (1-a)*tail), per class. NOT `voice - head`, which 4.3 used by mistake."""
    named = {"金融": ["查询金融资产实时行情"],
             "医疗": ["仅给出名称的裸实体查询", "查询中药材/食材/茶饮的功效与作用"]}
    for d in ["金融", "医疗"]:
        D = load(d)
        g = {s: x for s, x in D.groupby("source")}
        labs = sorted(set(g["assistant_top"].td_l1_name) | set(g["assistant_random"].td_l1_name)
                      | set(g["assistant_voice"].td_l1_name))
        T = g["assistant_top"]["td_l1_name"].value_counts(normalize=True).reindex(labs).fillna(0)
        R = g["assistant_random"]["td_l1_name"].value_counts(normalize=True).reindex(labs).fillna(0)
        V = g["assistant_voice"]["td_l1_name"].value_counts(normalize=True).reindex(labs).fillna(0)
        al = min(np.arange(0, 1.001, 0.005), key=lambda a: float(0.5 * (V - (a * T + (1 - a) * R)).abs().sum()))
        fit = al * T + (1 - al) * R
        res = (V - fit) * 100
        for cl in named[d]:
            rows.append({"block": "H 语音逐类残差·正文点名的类", "domain": d, "key": cl,
                         "value": round(float(res[cl]), 2),
                         "detail": f"语音{V[cl]*100:.1f}% 头部{T[cl]*100:.1f}% 长尾{R[cl]*100:.1f}% "
                                   f"拟合{fit[cl]*100:.1f}%（α={al:.2f}）；语音−头部={100*(V[cl]-T[cl]):+.1f}pp，"
                                   f"这才是正文误当成残差的那个差"})
            # The point of the whole block: the mixture explains these classes almost exactly.
            assert abs(float(res[cl])) < 1.5, (d, cl, float(res[cl]))
        for cl, v in res.sort_values().head(2).items():
            rows.append({"block": "H 语音逐类残差·最负", "domain": d, "key": cl, "value": round(float(v), 2),
                         "detail": f"语音{V[cl]*100:.1f}% 拟合{fit[cl]*100:.1f}%（α={al:.2f}）"})
        for cl, v in res.sort_values(ascending=False).head(3).items():
            rows.append({"block": "H 语音逐类残差·最正", "domain": d, "key": cl, "value": round(float(v), 2),
                         "detail": f"语音{V[cl]*100:.1f}% 拟合{fit[cl]*100:.1f}%（α={al:.2f}）"})
    # The two domains' most-negative classes must not be the same thing, or 4.4 could claim a
    # shared voice direction. They are 个股决策 and 残余 — assert they differ.
    neg = [r["key"] for r in rows if r["block"] == "H 语音逐类残差·最负"]
    assert len(set(neg[:1]) & set(neg[2:3])) == 0, neg


def i_wrap(rows: list) -> None:
    """same_intent head vs tail, with an interval. 人物 reverses in the point estimate only."""
    W = pd.concat([pd.read_csv(p) for p in sorted(WORK.glob("wrap_summary_*.csv"))])
    for d in DOMS:
        w = W[W.domain == d].set_index("source")
        kt, nt = round(w.loc["assistant_top", "same_intent"] * w.loc["assistant_top", "pairs"]), int(w.loc["assistant_top", "pairs"])
        kr, nr = round(w.loc["assistant_random", "same_intent"] * w.loc["assistant_random", "pairs"]), int(w.loc["assistant_random", "pairs"])
        dd, lo, hi = newcombe(int(kt), nt, int(kr), nr)
        rows.append({"block": "I 包装对·意图不变率 头部→长尾", "domain": d,
                     "key": f"{int(kt)}/{nt} → {int(kr)}/{nr}", "value": round(dd * 100, 1),
                     "detail": f"{kt/nt*100:.1f}% → {kr/nr*100:.1f}%，Δ[{lo*100:+.1f},{hi*100:+.1f}] "
                               f"sig={lo > 0 or hi < 0}"})
    ppl = [r for r in rows if r["block"].startswith("I 包装对") and r["domain"] == "人物"][0]
    assert "sig=False" in ppl["detail"], ppl   # the reversal is not measurable


def main() -> None:
    rows: list = []
    a_orderability(rows)
    b_unified(rows)
    c_time(rows)
    d_voice(rows)
    e_anchor(rows)
    f_smallcells(rows)
    g_shared(rows, umap())
    h_voice_residual(rows)
    i_wrap(rows)
    out = pd.DataFrame(rows)
    out.to_csv(cross_dir() / "synthesis_derived.csv", index=False)
    print(f"wrote {WORK/'cross'/'synthesis_derived.csv'}  ({len(out)} rows)")
    print(out[out.block.str.startswith("B 跨领域一致")].to_string(index=False))
    print(out[out.block == "C 时间线分解"].to_string(index=False))
    print(out[out.block == "D 语音混合拟合"].to_string(index=False))
    print(out[out.block == "E 界面线换锚"].to_string(index=False))
    print(out[out.block.str.startswith("F 小格子")].to_string(index=False))
    print(out[out.block == "G 共有桶的U层"].to_string(index=False))
    print(out[out.block.str.startswith("H 语音逐类残差")].to_string(index=False))
    print(out[out.block.str.startswith("I 包装对")].to_string(index=False))


if __name__ == "__main__":
    main()
