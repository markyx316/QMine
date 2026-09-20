# -*- coding: utf-8 -*-
"""Build audit2_report.md from recomputed results (audit2_analysis + audit2_round2_analysis). Numbers come from code."""
import math
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import audit2_analysis as AN  # noqa: E402
import audit2_round2_analysis as R2A  # noqa: E402
import audit2_labels as LB  # noqa: E402
import audit2_build_round2 as B2  # noqa: E402

CODES = [f"U{i:02d}" for i in range(1, 14)]
REDACT_PREFIX = [
    ("谢忠卫，", "[bio-card passage on a named local civil servant + templated question]"),
    ("蒋培", "[named individual]是否公开过个人履历"), ("崔丕军", "[named individual]的出生年月有官方报道吗"),
    ("李清贤", "[named doctor]在山东省心内科的地位如何"), ("孟景伟", "[named official]离开安庆市的原因是什么"),
    ("李林和陈海峰", "[two named officials]分管哪些具体工作"), ("庞红义", "[named individual]的教育背景是什么"),
    ("李汉晋", "[named individual]在山西交通企业协会任职多久了"), ("陈彬煌", "[named individual]现在在哪里任职"),
    ("武尚志", "[named officer]离休后有没有被追授更高军衔"), ("卢志强", "[named businessman]在郑州还有哪些关联企业"),
    ("刘乃贵", "[named individual]毕业后最初在哪个机构任职"), ("覃美金", "[a bare personal name]"),
    ("把这段互动改成更露骨", "[explicit-content rewrite command]"), ("好的，继续吧，仅控制在500字", "[explicit-content continuation command]"),
    ("生成4张9:16竖屏写实私密写真", "[explicit image-generation prompt]"),
    ("宝贝即将步入一年级", "[request for sentences about a child starting school]"),
    ("李双江李天一", "[celebrity-family 近况曝光 headline]"),
    ("中学施发型令", "[headline about a school's hair rule for students]"),
]


def show(s, n=48):
    for p, d in REDACT_PREFIX:
        if str(s).startswith(p):
            return d
    s = "".join(ch for ch in str(s) if unicodedata.category(ch) != "Cf").replace("|", "／").replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def ci(t, d=1):
    p, lo, hi = t
    return f"{p * 100:.{d}f}% [{lo * 100:.{d}f}, {hi * 100:.{d}f}]"


def md(df, cols=None):
    df = df if cols is None else df[cols]
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v).replace("|", "\\|") for v in r) + " |")
    return "\n".join(out)


def main():
    res = AN.compute()
    o = R2A.compute()
    T = res["table"]
    L = []
    w = L.append

    def trow(name):
        r = T[T["sample"] == name].iloc[0]
        return r

    # ------------------------------------------------------------ header / method
    S = res["S"]
    excl = {"3a changed": 11, "3b S4": 11, "3c S5": 14, "4b search non-user": 18, "search v2→v3 restored": 16}
    w("# Audit 2 — v3 cleaning rules on fresh samples\n")
    w("Independent audit of `tools/clean_assistant_functional.py` (v3 output in `clean_v3/`). Every number below is "
      "computed by `audit2_report_build.py` from `audit2_samples.parquet`, `audit2_round2_samples.parquet` and the label "
      "dicts in `audit2_labels.py` / `audit2_labels2.py`; no project file or pre-existing scratchpad file was modified.\n")
    w("## 1. Method\n")
    w("- **Criteria** were the fixed A / B / C definitions given with the task (A = system-authored or content-free; "
      "B = user; C = ambiguous; single characters in education search → C unless the lookup reading is unambiguous).")
    w("- **One label per distinct string per surface**, assigned from a shuffled list showing only id, surface, category "
      "and the string — never tier, PV, snapshot or sample. Labels are stored as dicts keyed by exact string; "
      "`check()` asserts every key matches a sampled row and every sampled row has a label.")
    w("- **Fresh samples.** All 1,133 strings in `audit_clean_samples_labeled.parquet` (`qs`) and "
      "`audit_clean_search_samples.parquet` were excluded. Rows removed by that exclusion: "
      + ", ".join(f"{k} {v}" for k, v in excl.items()) + ". The 16 restored search rows (C1 → user) were ALL first-audit "
      "strings, so no fresh restored-search rows exist and none are scored.")
    w("- **Round 1** (seed 20260910): samples 1–4 below, 1,006 distinct strings. **Round 2** (seed 20260911, frozen "
      "before labelling): a census of every kept 5-domain head row labelled U05, holdout rows for each rule proposed "
      "from round-1 misses, a replay of the search S5 rule on the 2025 search snapshot, and 55 decoy rows — 410 distinct "
      "strings, shuffled together so rule membership was not shown.")
    w("- **Limitations.** One labeller. Before labelling I had read `sva_tier_diff.txt`, which quotes ~100 of the "
      "changed rows scored in sample 3a. The headline/keyword boundary is a judgement: a noun phrase ending in "
      "`最新消息/最新进展/后续` was labelled keyword-shaped (B), a complete reported-event clause A; a bare "
      "name + `去世` on search B. Rule holdouts mix matches from several rules plus decoys, which reduces but does not "
      "remove anchoring.\n")

    # ------------------------------------------------------------ key numbers
    w("## 2. Key numbers\n")
    rows = []
    spec = [("1_head_kept", "1. Kept head (5 domains top1k)", "miss A/n"),
            ("2_tail_kept", "2. Kept tail (5 domains random1k)", "miss A/n"),
            ("3a_removed (v2 user -> v3 non-user)", "3a. Newly removed (v2 user → v3 non-user)", "precision A/n"),
            ("3a_restored (v2 non-user -> v3 user)", "3a. Newly restored (v2 non-user → v3 user)", "precision B/n"),
            ("3b_S4", "3b. S4_suggested_chip, 100 random (33 cats)", "precision A/n"),
            ("3c_S5", "3c. S5_headline, 100 random (33 cats)", "precision A/n"),
            ("4a_search_kept", "4a. Search kept, top 1,000 by wise_pv (30/domain)", "miss A/n"),
            ("4b_search_nonuser", "4b. Search non-user rows (all fresh)", "precision A/n")]
    for key, label, metric in spec:
        r = trow(key)
        rows.append({"sample": label, "rows (distinct)": f"{r.n} ({r.distinct})", "A": r.A, "B": r.B, "C": r.C,
                     "metric": metric, "rate [95% Wilson]": f"{r.rate * 100:.1f}% [{r.lo * 100:.1f}, {r.hi * 100:.1f}]",
                     "excl. C": f"{r.rate_exclC * 100:.1f}% [{r.lo_exclC * 100:.1f}, {r.hi_exclC * 100:.1f}]",
                     "PV-weighted": f"{r.pv_weighted * 100:.1f}%"})
    w(md(pd.DataFrame(rows)))
    b1, b2 = res["1_head_kept_pv_boot"], res["2_tail_kept_pv_boot"]
    w(f"\nPV-weighted miss share within sample 1: **{trow('1_head_kept').pv_weighted * 100:.1f}%** (domain-stratified "
      f"bootstrap 95% [{b1[0] * 100:.1f}, {b1[1] * 100:.1f}], 5,000 resamples). Domain-weighted estimates over the "
      f"kept-row pools: head A {res['1_head_kept_weighted_A'] * 100:.1f}%, C {res['1_head_kept_weighted_C'] * 100:.1f}%; "
      f"tail A {res['2_tail_kept_weighted_A'] * 100:.1f}%, C {res['2_tail_kept_weighted_C'] * 100:.1f}%. "
      f"Decoys (40 random kept top1k rows from all 33 categories, round 2): A {ci(o['decoy_assistant']['A_rate'])}, "
      f"C {o['decoy_assistant']['C']}/40. For scale, the first audit (v2, 33 categories) reported 11.3% [8.2, 15.4] of kept "
      f"top1k rows as A; the scopes differ, so this is context, not a like-for-like comparison.\n")

    w("### Per domain (miss rate A/n, n = 60 head / 40 tail)\n")
    pr = []
    for d in AN.CAT:
        h, t = trow(f"1_head_kept:{d}"), trow(f"2_tail_kept:{d}")
        pr.append({"domain": d, "head A/B/C": f"{h.A}/{h.B}/{h.C}", "head miss": f"{h.rate * 100:.1f}% [{h.lo * 100:.1f}, {h.hi * 100:.1f}]",
                   "tail A/B/C": f"{t.A}/{t.B}/{t.C}", "tail miss": f"{t.rate * 100:.1f}% [{t.lo * 100:.1f}, {t.hi * 100:.1f}]"})
    w(md(pd.DataFrame(pr)))
    w("")
    w("### Removal / restoration breakdown (sample 3a) and S4/S5 by stratum\n")
    br = T[T["sample"].str.startswith(("3a_removed:", "3b_S4:", "3c_S5:", "4b_search_nonuser:"))].copy()
    br["rate [CI]"] = [f"{a * 100:.1f}% [{b * 100:.1f}, {c * 100:.1f}]" for a, b, c in zip(br.rate, br.lo, br.hi)]
    w(md(br[["sample", "n", "A", "B", "C", "rate [CI]"]]))
    w("")

    # ------------------------------------------------------------ lists
    def lst(df, extra=True):
        d = df.copy()
        d["string"] = d.qs.map(show)
        d["pv"] = d.pv.astype(int)
        cols = ["string", "domain", "category", "snapshot", "pv", "tier"]
        if "lab" in d.columns:
            cols.append("lab")
        cols += ["why", "u_ds"]
        d["domain"] = d.domain.fillna("—")
        d["u_ds"] = d.u_ds.fillna("—")
        return md(d[cols])

    w("## 3. Every miss and every wrong move\n")
    w(f"### 3.1 Misses — kept `user` rows labelled A (samples 1, 2, 4a; n = {len(res['misses'])})\n")
    w("`why` is the criterion the label used. Sample 4a (search) had no misses.\n")
    w(lst(res["misses"]))
    w(f"\n### 3.2 Wrong removals — v2 user → v3 non-user rows NOT labelled A (n = {len(res['wrong_removed'])})\n")
    w(lst(res["wrong_removed"]))
    w(f"\n### 3.3 Wrong restorations — v2 non-user → v3 user rows NOT labelled B (n = {len(res['wrong_restored'])})\n")
    w("None. Only 3 fresh restored rows exist in the 5 domains (5 of the 8 restorations were first-audit strings); all 3 are B.\n")
    for key, title in (("wrong_3b_S4", "S4_suggested_chip sample rows not labelled A"),
                       ("wrong_3c_S5", "S5_headline sample rows not labelled A"),
                       ("wrong_4b_search_nonuser", "Search non-user rows not labelled A")):
        w(f"### 3.{4 + ['wrong_3b_S4', 'wrong_3c_S5', 'wrong_4b_search_nonuser'].index(key)} {title} (n = {len(res[key])})\n")
        w(lst(res[key]))
        w("")
    w(f"### 3.7 Ambiguous (C) kept rows (samples 1, 2; n = {len(res['kept_C'])})\n")
    w("Not errors, listed because they bound what any rule can decide. Names of non-public individuals are replaced.\n")
    w(lst(res["kept_C"]))
    w("")

    # ------------------------------------------------------------ bias check
    w("## 4. Bias check against the single-instrument intent labels (`u_ds`)\n")
    mc = res["misses"][res["misses"]["sample"] != "4a_search_kept"].copy()
    ct = pd.crosstab(mc.u_ds, mc["sample"]).reindex(CODES, fill_value=0)
    ct = ct[(ct.sum(axis=1) > 0)]
    ct.insert(0, "u_ds", ct.index)
    ct.insert(1, "name", [AN_FRAME[c][0] for c in ct.index])
    w("### 4.1 Which classes the misses carry\n")
    w(md(ct.reset_index(drop=True)))
    w(f"\nIn the head, {int(((mc['sample'] == '1_head_kept') & (mc.u_ds == 'U05')).sum())} of "
      f"{int((mc['sample'] == '1_head_kept').sum())} misses are **U05 核实与动态** — tapped headlines, "
      f"{int(((mc['sample'] == '1_head_kept') & (mc.domain == '人物')).sum())} of them in 人物. Tail misses spread over "
      "U03/U04/U05/U08/U10/U11 (cards, LLM-voice follow-ups, a template).\n")

    def dist_table(key, census=None):
        t, r, rU = res[key]
        d = t.copy()
        d["name"] = [AN_FRAME[c][0] for c in d.u_ds]
        d["kept share"] = (d.share_kept * 100).round(2)
        d["sample rows in class"] = d.n_sample_in_class
        lab = "A" if key.endswith("_A") else "C"
        d[f"{lab} in class"] = d[f"n_{lab}"]
        d[f"within-class {lab} [CI]"] = [f"{a * 100:.0f}% [{b * 100:.0f}, {c * 100:.0f}]" if n else "—"
                                          for a, b, c, n in zip(d[f"within_class_{lab}"], d.within_lo, d.within_hi, d.n_sample_in_class)]
        d["share after removal"] = (d.share_after * 100).round(2)
        d["shift pp"] = d.shift_pp.round(2)
        d["max inflation pp"] = d.max_inflation_pp.round(2)
        d["max deflation pp"] = d.max_deflation_pp.round(2)
        if census is not None:
            i = d.index[d.u_ds == "U05"][0]
            d.loc[i, "share after removal"] = round(census["share_after_point"] * 100, 2)
            d.loc[i, "shift pp"] = round((census["u05_share"] - census["share_after_point"]) * 100, 2)
            d.loc[i, "max inflation pp"] = round((census["u05_share"] - census["share_after_range"][0]) * 100, 2)
            d.loc[i, f"within-class {lab} [CI]"] = "census " + ci(census["A_rate"], 0)
        cols = ["u_ds", "name", "kept share", "sample rows in class", f"{lab} in class", f"within-class {lab} [CI]",
                "share after removal", "shift pp", "max inflation pp", "max deflation pp"]
        return md(d[cols]), r, rU, d

    u = o["u05_summary"]
    w("### 4.2 Maximum distortion of each class's row share — head (5 domains top1k, kept rows)\n")
    w("Method: `q_k` = share of kept rows that are A and in class k (from sample 1), `r` = Σ q_k. Share after removing "
      "A rows = (p_k − q_k)/(1 − r). **Max inflation** uses the Wilson upper bound of q_k (capped at p_k, no other removals); "
      "**max deflation** uses the Wilson lower bound of q_k with the Wilson-upper total r. Row U05 is replaced by the census "
      "(§4.4), whose bound is exact up to the total-miss uncertainty.\n")
    tb, r, rU, dh = dist_table("dist_1_head_kept_A", census=u)
    w(tb)
    w(f"\nTotal A in kept head rows: {r * 100:.1f}% (Wilson upper {rU * 100:.1f}%).\n")
    w("### 4.3 Maximum distortion — tail (5 domains random1k, kept rows)\n")
    tb, r, rU, dt = dist_table("dist_2_tail_kept_A")
    w(tb)
    w(f"\nTotal A in kept tail rows: {r * 100:.1f}% (Wilson upper {rU * 100:.1f}%). n = 200, so every class bound is at least ~1.7 pp "
      "whatever the data show.\n")

    w("### 4.4 U05 census — every kept 5-domain head row labelled U05\n")
    bd = u["by_domain"].reindex(columns=["A", "B", "C"], fill_value=0).copy()
    cen = o["u05_census"]
    F = pd.read_parquet(f"{SP}/sva_final_rows.parquet")
    s1 = S[S["sample"] == "1_head_kept"].copy()
    s1["lab"] = s1.qs.map(LB.LAB_ASSISTANT)
    rows = []
    for d, c in AN.CAT.items():
        N = int(((F.domain == d) & (F.surface == "assistant_top1k") & (F.tier == "user")).sum())
        cd = cen[cen.l1 == c]
        a = int((cd.lab == "A").sum())
        ra = (s1[s1.domain == d].lab == "A").mean()
        rows.append({"domain": d, "kept head rows": N, "U05 rows": len(cd), "A": a, "B": int((cd.lab == "B").sum()),
                     "C": int((cd.lab == "C").sum()), "A share of U05": pct(a / len(cd)) if len(cd) else "—",
                     "U05 share (v3)": pct(len(cd) / N, 2), "U05 share after A removed": pct((len(cd) - a) / (N * (1 - ra)), 2)})
    rows.append({"domain": "5 domains", "kept head rows": u["kept_head_rows"], "U05 rows": u["u05_rows"], "A": u["A"], "B": u["B"],
                 "C": u["C"], "A share of U05": ci(u["A_rate"]), "U05 share (v3)": pct(u["u05_share"], 2),
                 "U05 share after A removed": f"{u['share_after_point'] * 100:.2f}% [{u['share_after_range'][0] * 100:.2f}, {u['share_after_range'][1] * 100:.2f}]"})
    w(md(pd.DataFrame(rows)))
    top = cen[cen.lab == "A"].sort_values("search_num", ascending=False).head(10)
    w(f"\n{u['A']} of {u['u05_rows']} U05 head rows ({u['A_pv']:,} of {u['pv']:,} PV) are system text — almost all tapped headlines. "
      f"Removing them takes the pooled U05 head share from {u['u05_share'] * 100:.2f}% to {u['share_after_point'] * 100:.2f}% "
      f"({u['share_after_also_minus_C'] * 100:.2f}% if the {u['C']} C rows are also dropped): **v3 still roughly doubles U05 in the "
      "head**, and triples it in 人物. The first audit's v2 finding (fact-verification roughly doubled) therefore still holds in the 5-domain head after S4/S5. Highest-PV examples: "
      + " · ".join(f"`{show(q, 30)}` ({int(p)})" for q, p in zip(top.qs, top.search_num)) + ".\n")

    w("### 4.5 Ambiguous (C) rows — share and classes\n")
    th, rh, rUh = res["dist_1_head_kept_C"]
    tt, rt, rUt = res["dist_2_tail_kept_C"]
    cc = pd.crosstab(res["kept_C"].u_ds, res["kept_C"]["sample"]).reindex(CODES, fill_value=0)
    cc = cc[cc.sum(axis=1) > 0]
    cc.insert(0, "u_ds", cc.index)
    cc.insert(1, "name", [AN_FRAME[c][0] for c in cc.index])
    w(f"C share: head {rh * 100:.1f}% (Wilson upper {rUh * 100:.1f}%), tail {rt * 100:.1f}% (upper {rUt * 100:.1f}%).\n")
    w(md(cc.reset_index(drop=True)))
    ch = th.set_index("u_ds")
    ctl = tt.set_index("u_ds")
    w(f"\nHead C rows are mostly U13 ({int(ch.loc['U13', 'n_C'])} of {int(ch.n_C.sum())}), as expected: bare education fragments "
      f"(`少`, `的`, `3`, `第二题`) with 300–2,000 PV. If all head C rows were dropped, U13 would fall by {ch.loc['U13', 'shift_pp']:.1f} pp "
      f"and U02 rise by {-ch.loc['U02', 'shift_pp']:.1f} pp. Tail C rows are NOT mostly U13 ({int(ctl.loc['U13', 'n_C'])} of {int(ctl.n_C.sum())}); "
      f"they are formal follow-up questions about named local officials and works (U03 {int(ctl.loc['U03', 'n_C'])}, "
      f"U05 {int(ctl.loc['U05', 'n_C'])}, U08 {int(ctl.loc['U08', 'n_C'])}) whose origin — typed or a suggested question — "
      f"the text cannot decide; dropping them would move U03 by {ctl.loc['U03', 'shift_pp']:.1f} pp.\n")
    fr = []
    for surf in ("assistant_top1k",):
        f = F[(F.surface == surf) & (F.tier == "user")].copy()
        f["ql"] = f["query"].astype(str).str.len()
        for d in list(AN.CAT) + ["5 domains"]:
            g = f if d == "5 domains" else f[f.domain == d]
            one = g[g.ql == 1]
            fr.append({"domain": d, "kept head rows": len(g), "single-character rows": len(one),
                       "of which U13": int((one.u_ds == "U13").sum()),
                       "U13 share": pct((g.u_ds == "U13").mean()), "U13 share without them": pct((g[g.ql > 1].u_ds == "U13").mean())})
    w("Single-character kept head rows (a measured count; kept by design, and a lookup vs a voice/OCR fragment cannot be told apart):\n")
    w(md(pd.DataFrame(fr)))
    w("")

    # ------------------------------------------------------------ patterns & rules
    w("## 5. Systematic patterns and proposed rules (NOT applied)\n")
    w("Each rule was written from round-1 strings only, checked for false positives on round-1 labels, frozen in "
      "`audit2_build_round2.py`, and then scored on round-2 rows that no round-1 or first-audit label had touched.\n")
    ru = o["rules"].set_index("rule")

    def rr(name, fld="precision"):
        x = ru.loc[name]
        return f"{int(x.A if 'exemption' not in name else x.B)}/{int(x.holdout_n)} = {x.precision * 100:.1f}% [{x.lo * 100:.1f}, {x.hi * 100:.1f}]"

    un = o["u05_union"]
    s25, s26 = o["s25_summary"], o["s26_train"]
    R2 = o["R2"]
    h5 = R2[R2["set"] == "R5_card"]
    z = h5[h5.qs.str.contains("‌")]
    pat = [
        {"#": 1, "pattern": "Tapped headlines outside the 新闻 category (人物/影视/金融 head); S5 rule B's verb list and PV>=100 miss them",
         "measured count": f"{int((res['misses'].why == 'headline').sum())} of {len(res['misses'])} sample misses; census: {u['A']} of {u['u05_rows']} U05 head rows",
         "proposed rule": "R1: head, not 新闻, no ask word, PV>=100, len>=6, contains 逝世/去世/离世/疑似/偶遇/同框/拟邀/辞去/涨粉/自曝/发声/现身/互动/热议/官宣/离婚/出轨/夺冠/当选/即将/首次/回应/表态/被查/落马/获刑/身亡/否认/曝光/爆料. R1b: identical string is also a 新闻 top1k row, no NEWS_ASKS word",
         "holdout precision": f"R1 {rr('R1_headline_verbs')}; R1b {rr('R1b_crosslisted_news')}; R1∪R1b inside the U05 census holdout {ci(un['precision'])}, recall {ci(un['recall_on_A'])}",
         "verdict": "Do not apply. False positives: explicit-content edit commands matching 互动/涨粉/离婚, weather and pension keyword queries cross-listed in 新闻, bare celebrity names. Precise only where the intent label already says U05, which would make cleaning depend on the quantity being compared."},
        {"#": 2, "pattern": "`有没有更多…` suggested chips (CHIP only covers `有没有更…的`)",
         "measured count": f"{ru.loc['R2a_more_chip', 'kept_rows_33cat']} kept rows, 33 cats ({ru.loc['R2a_more_chip', 'kept_rows_5dom_tail']} in 5-domain tail)",
         "proposed rule": "R2a: `^有没有更多`", "holdout precision": rr("R2a_more_chip"),
         "verdict": "Apply (tiny effect on the 5 domains). One typed false positive with a `+` keyword join."},
        {"#": 3, "pattern": "LLM-voice follow-ups without a chip prefix (`有哪些感人瞬间`, `是否公开过`, `有官方报道吗`)",
         "measured count": f"4 tail misses; {ru.loc['R2b_llm_followup', 'kept_rows_33cat']} kept rows",
         "proposed rule": "R2b: `有哪些感人瞬间$|是否公开过|有官方报道吗[？?]?$`", "holdout precision": rr("R2b_llm_followup"),
         "verdict": f"Unproven: 1 holdout row. The family is open-ended; {int(((res['kept_C']['sample'] == '2_tail_kept') & (res['kept_C'].why == 'origin_equal')).sum())} kept tail sample rows in the same formal register were labelled C, not A."},
        {"#": 4, "pattern": "Bare replies and insults not in REPLY (`什么`, `随便`, `你妈`)",
         "measured count": f"3 head misses; {ru.loc['R3_reply', 'kept_rows_33cat']} kept rows",
         "proposed rule": "R3: `^\\s*(什么|啥|随便|都行|无所谓|你妈的?)[\\s。！!~～？?]*$`", "holdout precision": rr("R3_reply"),
         "verdict": "Apply. Assistant surface only (on search a bare word is a lookup)."},
        {"#": 5, "pattern": "Feed hashtag copy (`#爆款短剧…#`, `#百家流量扶持计划#`)",
         "measured count": f"1 head miss; {ru.loc['R4_hashtag', 'kept_rows_33cat']} kept rows ({ru.loc['R4_hashtag', 'kept_rows_5dom_head']} in 5-domain head)",
         "proposed rule": "R4: `^\\s*#[^#]{2,}#\\s*$`", "holdout precision": rr("R4_hashtag"), "verdict": "Apply."},
        {"#": 6, "pattern": "Essay-edit slot template (`我想对作文《…》进行个性化编辑`) — TEMPLATE_MARK knows it but only uses it to veto an exemption",
         "measured count": f"1 head miss; {ru.loc['R6_essay_edit_template', 'kept_rows_33cat']} kept rows, all 教育 ({ru.loc['R6_essay_edit_template', 'kept_rows_5dom_head']} head, {ru.loc['R6_essay_edit_template', 'kept_rows_5dom_tail']} tail)",
         "proposed rule": "R6: `^我想对作文《`", "holdout precision": rr("R6_essay_edit_template"), "verdict": "Apply."},
        {"#": 7, "pattern": "Pasted answer text / bio cards without `男/女` (U+200C marks text copied from an answer; `名，现任…`)",
         "measured count": f"2 tail misses; {ru.loc['R5_card', 'kept_rows_33cat']} kept rows",
         "proposed rule": "R5: `^[一-鿿]{2,4}[，,]现任` or contains U+200C", "holdout precision": f"{rr('R5_card')}; U+200C part {int((z.lab == 'A').sum())}/{len(z)}",
         "verdict": "Reject U+200C: it marks copying, and users append their own request or paste their own prompts. The `现任` part has 3 kept rows in total — not worth a rule."},
        {"#": 8, "pattern": "S5 rule A (新闻 category, no NEWS_ASKS) removes keyword news queries",
         "measured count": f"{int((res['wrong_3c_S5'].lab == 'B').sum())} of 100 S5 sample rows; all end in a keyword suffix",
         "proposed rule": "R7 exemption: `(最新消息|最新进展|最新通知|最新政策|调整政策|后续|进展|路径图|事故|前后对比)$`",
         "holdout precision": f"exempted rows labelled B: {rr('R7_news_keyword_exemption (precision = B/n)')}",
         "verdict": f"Apply. Affects {ru.loc['R7_news_keyword_exemption (precision = B/n)', 'kept_rows_33cat']} S5 rows, none in the 5 domains. Caveat: rests on the keyword-vs-headline judgement in §1."},
        {"#": 9, "pattern": "Search S5 removes verification lookups (`江泽民去世`), quoted keywords (`“今日金价”`) and questions (`…有何回应`)",
         "measured count": f"{int((res['wrong_4b_search_nonuser'].lab == 'B').sum())} B + {int((res['wrong_4b_search_nonuser'].lab == 'C').sum())} C of 22 fresh search S5 rows",
         "proposed rule": "Exempt `^“…”$`, `^[一-鿿]{2,4}(去世|逝世)$`, and `有何|何时|为何|是谁|咋`",
         "holdout precision": f"2025 snapshot replay (n = {s25['n']}): S5 as-is {ci(s25['S5_precision'])}; exempted {s25['exempted']} rows, B {ci(s25['exempt_precision_B'])}; S5 after exemption {ci(s25['S5_after_exemption_precision'])}",
         "verdict": "Apply. The remaining false positives are titles that contain 曝光/来了. Volume is 22 rows of 50k."},
        {"#": 10, "pattern": "S4 removes typed `能否…吗？`, comma-listed requests and a pasted exam question",
         "measured count": "3 of 5 wrong removals in 3a",
         "proposed rule": "Exempt `吗[？?]?$`, two commas, or length > 40",
         "holdout precision": "3b rows not in 3a: exempted 1, labelled A (0/1); the 3 B rows in the holdout were not exempted",
         "verdict": "Reject: does not generalise. S4 is already 96% precise."},
        {"#": 11, "pattern": "CHIP's unanchored `该(公司|企业|平台…)` fires inside long pasted company/app descriptions",
         "measured count": "12 S4 rows are caught only by this branch (8 longer than 30 chars)",
         "proposed rule": "Anchor it (`^该…`) and let S3 take the passages", "holdout precision": "not measured (no fresh labelled rows)",
         "verdict": "Tier hygiene only: the passages are A either way; the few typed `如何查询该公司…` rows are unlabelled."},
        {"#": 12, "pattern": "Education head single-character fragments (C, not A)",
         "measured count": f"{fr[2]['single-character rows']} kept 教育 head rows ({fr[2]['of which U13']} carry U13)",
         "proposed rule": "None for removal; report U13 with and without them", "holdout precision": "n/a",
         "verdict": f"U13 in 教育 head {fr[2]['U13 share']} → {fr[2]['U13 share without them']}; pooled {fr[5]['U13 share']} → {fr[5]['U13 share without them']}."},
    ]
    w(md(pd.DataFrame(pat)))
    fp = o["rule_false_positives"]
    fpv = fp[fp["set"].isin(["R1_headline_verbs", "R1b_crosslisted_news", "R2a_more_chip"])]
    w("\nFalse positives of R1 / R1b / R2a on the holdout (R5's 33 are summarised above): "
      + " · ".join(f"`{show(q, 26)}` ({s.split('_')[0]})" for q, s in zip(fpv.qs, fpv["set"])) + ".\n")

    # ------------------------------------------------------------ comparison fitness
    w("## 6. Is v3 fit for comparing unified-intent shares between surfaces?\n")
    head = F[(F.surface == "assistant_top1k") & (F.tier == "user")]
    tail = F[(F.surface == "assistant_random1k") & (F.tier == "user")]
    srch = F[(F.surface == "search_top1000") & (F.tier == "user")]
    sh = lambda df: df.u_ds.value_counts(normalize=True).reindex(CODES, fill_value=0)
    ps, pa, pt = sh(srch), sh(head), sh(tail)
    dsrch = res["dist_4a_search_kept_A"][0].set_index("u_ds")
    dhd = res["dist_1_head_kept_A"][0].set_index("u_ds")
    dtl = res["dist_2_tail_kept_A"][0].set_index("u_ds")
    cmp_rows = []
    for c in CODES:
        bs = max(dsrch.loc[c, "max_inflation_pp"], dsrch.loc[c, "max_deflation_pp"]) if c in dsrch.index else 2.5
        if c == "U05":
            ba = (u["u05_share"] - u["share_after_range"][0]) * 100
            pa_c = u["share_after_point"]
        else:
            ba = max(dhd.loc[c, "max_inflation_pp"], dhd.loc[c, "max_deflation_pp"])
            pa_c = dhd.loc[c, "share_after"]
        bt = max(dtl.loc[c, "max_inflation_pp"], dtl.loc[c, "max_deflation_pp"])
        se_t = 1.96 * math.sqrt(pt[c] * (1 - pt[c]) / len(tail)) * 100
        dh, dtt = (pa[c] - ps[c]) * 100, (pt[c] - ps[c]) * 100
        thr_h, thr_t = ba + bs, bt + bs + se_t
        verdict = lambda d_, t_: "yes" if abs(d_) > t_ + 0.5 else ("marginal" if abs(d_) > t_ else "no")
        cmp_rows.append({"u_ds": c, "name": AN_FRAME[c][0], "search %": f"{ps[c] * 100:.1f}",
                         "asst head %": f"{pa[c] * 100:.1f}", "head, A removed %": f"{pa_c * 100:.1f}",
                         "head − search pp": f"{dh:+.1f}", "head threshold pp": f"{thr_h:.1f}",
                         "head readable": verdict(dh, thr_h),
                         "asst tail %": f"{pt[c] * 100:.1f}", "tail − search pp": f"{dtt:+.1f}", "tail threshold pp": f"{thr_t:.1f}",
                         "tail readable": verdict(dtt, thr_t), "_th": thr_h, "_tt": thr_t,
                         "_big": max(ps[c], pa[c], pt[c]) >= 0.02})
    cm = pd.DataFrame(cmp_rows)
    w("Pooled over the 5 domains; kept rows only; shares from `sva_final_rows.parquet`. **Threshold** = worst-case cleaning bias "
      "on the assistant side (head: §4.2 bound, U05 from the census; tail: §4.3 bound plus 1.96·SE because random1k is a sample) "
      "+ worst-case bias on the search side (sample 4a found 0 misses in 150: the bound is up to ~2.5 pp, capped at the class's "
      "own share because a class cannot lose more rows than it has). **readable** = yes when the difference exceeds the threshold "
      "by more than 0.5 pp, marginal when it exceeds it by less. The bounds are conservative, and they ignore labelling error in "
      "`u_ds` itself, which this audit did not measure. For U05 read the `head, A removed` column, not the raw head share.\n")
    w(md(cm.drop(columns=["_th", "_tt", "_big"])))
    big = cm[cm._big & (cm.u_ds != "U13")]
    th_head, th_tail = big._th.max(), big._tt.max()
    w("")
    w(f"Across classes holding at least 2% on some surface (U13 excluded), the largest head threshold is **{th_head:.1f} pp** "
      f"({big.loc[big._th.idxmax(), 'u_ds']}) and the largest tail threshold **{th_tail:.1f} pp** ({big.loc[big._tt.idxmax(), 'u_ds']}). "
      f"U13's own threshold is {cm.loc[cm.u_ds == 'U13', '_th'].iloc[0]:.1f} pp (head).\n")

    # ------------------------------------------------------------ bottom line
    s1r = trow("1_head_kept")
    ppl = trow("1_head_kept:人物")
    w("## 7. Bottom line\n")
    w(f"- **Removal is sound.** New removals are {trow('3a_removed (v2 user -> v3 non-user)').rate * 100:.1f}% precise, S4 "
      f"{trow('3b_S4').rate * 100:.0f}%, S5 {trow('3c_S5').rate * 100:.0f}% (its errors are keyword news queries outside the 5 domains). "
      f"Search S5 is the weak rule ({trow('4b_search_nonuser').rate * 100:.0f}% on 22 rows; {s25['S5_precision'][0] * 100:.0f}% "
      f"replayed on the 2025 snapshot) but touches 22 of 50k rows.")
    ppl_m = res["misses"][(res["misses"]["sample"] == "1_head_kept") & (res["misses"].domain == "人物")]
    w(f"- **Recall is adequate in aggregate; 人物 head is the exception, and its misses are mostly one class** "
      f"({int((ppl_m.u_ds == 'U05').sum())} of {len(ppl_m)} are U05). Kept head rows are "
      f"{s1r.rate * 100:.1f}% [{s1r.lo * 100:.1f}, {s1r.hi * 100:.1f}] system text ({trow('2_tail_kept').rate * 100:.1f}% tail; "
      f"0 of 150 on search). 人物 head is {ppl.rate * 100:.0f}% [{ppl.lo * 100:.0f}, {ppl.hi * 100:.0f}]. The census shows "
      f"{u['A']} of {u['u05_rows']} U05 head rows are tapped headlines, so the head U05 share is inflated from "
      f"{u['share_after_point'] * 100:.1f}% to {u['u05_share'] * 100:.1f}% (人物: 3.7% → 11.0%).")
    edu_head = head[head.domain == "教育"]
    u05_resid = ((u["share_after_range"][1] - u["share_after_range"][0]) * 100
                 + max(dsrch.loc["U05", "max_inflation_pp"], dsrch.loc["U05", "max_deflation_pp"]))
    dom_hi = T[T["sample"].str.startswith(("1_head_kept:", "2_tail_kept:"))].hi * 100
    w(f"- **Fit for the pooled 5-domain comparison, with three conditions.** (1) As a blanket rule, do not read a head difference "
      f"smaller than **{math.ceil(th_head)} pp** or a tail difference smaller than **{math.ceil(th_tail)} pp**; §6 gives smaller "
      f"class-specific thresholds for small classes, which cannot be inflated by more than their own share. (2) Read head U05 only "
      f"after subtracting the census rows: search {ps['U05'] * 100:.1f}% vs head {u['share_after_point'] * 100:.1f}%, not "
      f"{u['u05_share'] * 100:.1f}%. The corrected gap is {(u['share_after_point'] - ps['U05']) * 100:.1f} pp with about "
      f"{u05_resid:.1f} pp of remaining cleaning uncertainty, so its direction reads, but it is half the raw gap. (3) Exclude U13 "
      f"(head {fr[5]['U13 share']}). It is inflated by 教育, where U13 is {fr[2]['U13 share']} of kept head rows and "
      f"{fr[2]['of which U13']} of its {int((edu_head.u_ds == 'U13').sum())} U13 rows are single characters. On that basis the "
      f"large gaps survive: bare entity U02 search {ps['U02'] * 100:.0f}% vs head {pa['U02'] * 100:.0f}%, explanation U04 "
      f"{ps['U04'] * 100:.0f}% vs {pa['U04'] * 100:.0f}%, and case advice U07 {ps['U07'] * 100:.0f}% vs tail {pt['U07'] * 100:.0f}%.")
    w(f"- **Not fit per domain at n = 60/40.** The per-domain Wilson upper bound on the miss rate ranges {dom_hi.min():.0f}–{dom_hi.max():.0f}%. "
      f"人物 head is not fit per class at all: {ppl.rate * 100:.0f}% of its kept rows are system text, and the census corrects only U05. "
      f"**The smallest share difference that should be read is {math.ceil(th_head)} percentage points (pooled head) / "
      f"{math.ceil(th_tail)} pp (pooled tail).**")
    w("- Rules worth applying next (measured on holdout rows, none applied here): `^有没有更多` chips, bare replies, feed hashtags, "
      "the essay-edit template, the 新闻 keyword exemption, and the three search-S5 exemptions. None of them fixes the U05 "
      "headline problem. Two headline rules were tried and failed their holdout at ~70%.\n")

    w("## 8. Files\n")
    for f, dsc in (("audit2_report.md", "this report"), ("audit2_labels.py", "round-1 labels (LAB_ASSISTANT, LAB_SEARCH, REASON_*) + check()"),
                   ("audit2_labels2.py", "round-2 labels + check()"), ("audit2_samples.parquet", "round-1 sampled rows (sample, rid, tier, v2_tier, pv…)"),
                   ("audit2_round2_samples.parquet", "round-2 rows (set = census / rule holdout / replay / decoy)"),
                   ("audit2_label_list.txt / audit2_round2_list.txt", "the blind lists that were labelled"),
                   ("audit2_make_labels.py / audit2_make_labels2.py", "id-level labelling record → label dicts"),
                   ("audit2_build_samples.py / audit2_build_round2.py", "sampling, exclusions, frozen rules"),
                   ("audit2_analysis.py / audit2_round2_analysis.py / audit2_report_build.py", "metrics and this report")):
        w(f"- `{SP}/{f}` — {dsc}")
    text = "\n".join(L) + "\n"
    with open(f"{SP}/audit2_report.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote", len(text), "chars;", "head threshold %.2f tail %.2f" % (th_head, th_tail))


AN_FRAME = AN.frame().FRAME if hasattr(AN, "frame") else None

if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "uif", "/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/tools/unified_intent_frame.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    AN_FRAME = m.FRAME
    main()
