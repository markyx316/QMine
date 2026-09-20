"""Every class, in every snapshot: the tables the comparison is actually made of.

`p10b_drift` answers "how far apart are these snapshots" with one summary
distance per level. That is the right first question and the wrong only question:
a reader who is told two snapshots differ by TVD 0.147 still cannot say *which
classes moved*, *whether a class missing from one is really absent*, or *which
class is characteristic of which snapshot*. This module answers those, and it
answers them per level, because a difference that is invisible at L1 is often
obvious at the leaf.

Three rules decide what these numbers can be read to mean, and each is enforced
here rather than left to the prose:

1. **Every share is within-snapshot.** Snapshots differ in size by an order of
   magnitude; a raw count across them mostly reports which export was bigger.
2. **An absence needs a detectability figure.** Zero of 950 rows is consistent
   with a share up to 0.39%; zero of 10,000 is consistent with 0.037%. So every
   zero is shipped with an expected count, a P(0) and a verdict, and the prose
   quotes the verdict.
3. **Raw-count exclusivity is a sample-size artefact.** Alongside 独占快照 the
   tables carry 均衡归属% — the class's within-snapshot share divided by the sum
   of its within-snapshot shares — which asks where a class sits when every
   snapshot is treated as the same size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .manifest import SnapshotManifest
from .stats import (absence_verdict, codes, cramers_v, newcombe, one_sided_upper,
                    rng_for, tvd, tvd_with_bounds, wilson)

SNAP = "snapshot"

#: The four label levels, and what each is called to a reader. Both routes are
#: reported at both depths because they disagree in informative ways — the
#: taxonomy is written by an architect, the leaves fall out of the embedding, and
#: "which leaf does this intent mostly land in" is therefore a measurement.
LEVELS: list[tuple[str, str]] = [
    ("td_l1", "自上而下 L1 意图"),
    ("td_l2", "自上而下 L2 子意图"),
    ("bu_family_final", "自下而上 家族"),
    ("bu_leaf", "自下而上 叶"),
]

TOPN_CUTS = (1, 3, 5, 10)


# ----------------------------------------------------------------- class frame

def class_frame(d: pd.DataFrame, level: str, names: dict[str, dict]) -> pd.DataFrame:
    """Collapse one level to key / display name / definition.

    The key is always the identifier, never the name: a run can produce two
    leaves with the same name, and adding them together reads two different
    clusters as one. Where names collide the display name gets the key appended,
    so the tables stay readable without the ambiguity.
    """
    out = d.copy()
    out["_key"] = out[level].astype(str)
    keys = list(pd.unique(out["_key"]))
    tbl = names.get(level, {})
    disp = {k: (tbl.get(k, {}).get("name") or _fallback_name(level, k)) for k in keys}
    defi = {k: (tbl.get(k, {}).get("definition") or "") for k in keys}
    seen: dict[str, list[str]] = {}
    for k, v in disp.items():
        seen.setdefault(v, []).append(k)
    disp = {k: (f"{v}#{k}" if len(seen[v]) > 1 else v) for k, v in disp.items()}
    out["_disp"] = out["_key"].map(disp)
    out["_def"] = out["_key"].map(defi)
    return out


def _fallback_name(level: str, key: str) -> str:
    if level == "bu_leaf":
        return f"叶{key}"
    if level == "bu_family_final":
        return f"家族{key}"
    return str(key)


# --------------------------------------------------------- class × snapshot

def matrix(d: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    """The main table: one row per class, one block of columns per snapshot."""
    snaps = man.snapshots
    n_s = {s: int((d[SNAP] == s).sum()) for s in snaps}
    pv_s = {s: float(d.loc[d[SNAP] == s, "pv_norm"].sum()) for s in snaps}
    base = d["_key"].value_counts(normalize=True)
    rows = []
    for key, g in d.groupby("_key", sort=False):
        r: dict[str, Any] = {"key": key, "类目": g["_disp"].iloc[0], "定义": g["_def"].iloc[0],
                             "全语料条数": len(g),
                             "全语料占比%": round(100 * len(g) / len(d), 3)}
        sh = {}
        for s in snaps:
            z = man.display(s)
            k = int((g[SNAP] == s).sum())
            lo, hi = wilson(k, n_s[s])
            sh[s] = k / n_s[s] if n_s[s] else 0.0
            r[f"{z}_条数"] = k
            r[f"{z}_占比%"] = round(100 * sh[s], 3)
            r[f"{z}_95CI低%"] = round(100 * lo, 3)
            r[f"{z}_95CI高%"] = round(100 * hi, 3)
            r[f"{z}_流量占比%"] = (round(100 * g.loc[g[SNAP] == s, "pv_norm"].sum() / pv_s[s], 3)
                                   if pv_s[s] else float("nan"))
        tot = sum(sh.values())
        for s in snaps:
            z = man.display(s)
            # Where the class sits when every snapshot is weighted equally.
            r[f"{z}_均衡归属%"] = round(100 * sh[s] / tot, 2) if tot else float("nan")
            # Concentration against the WHOLE corpus — reads high for a class
            # that is common in a snapshot the corpus is mostly made of.
            r[f"{z}_指数"] = round(sh[s] / base[key], 4) if base[key] else float("nan")
            # Concentration against the unweighted mean of the snapshots —
            # immune to the corpus being 90% one export.
            r[f"{z}_均衡指数"] = round(sh[s] / (tot / len(snaps)), 4) if tot else float("nan")
        present = [s for s in snaps if r[f"{man.display(s)}_条数"] > 0]
        absent = [s for s in snaps if r[f"{man.display(s)}_条数"] == 0]
        hi_s = max(snaps, key=lambda s: sh[s])
        lo_s = min(snaps, key=lambda s: sh[s])
        r["出现快照数"] = len(present)
        r["缺席快照"] = " / ".join(man.display(s) for s in absent)
        r["独占快照"] = man.display(present[0]) if len(present) == 1 else ""
        r["最高快照"] = man.display(hi_s)
        r["最低快照"] = man.display(lo_s)
        r["极差pp"] = round(100 * (sh[hi_s] - sh[lo_s]), 3)
        r["最高比最低倍数"] = round(sh[hi_s] / sh[lo_s], 2) if sh[lo_s] > 0 else float("inf")
        rows.append(r)
    t = pd.DataFrame(rows)
    for s in snaps:
        z = man.display(s)
        t[f"{z}_排名"] = t[f"{z}_占比%"].rank(ascending=False, method="min").astype(int)
    return t.sort_values("全语料条数", ascending=False).reset_index(drop=True)


def enrich(t: pd.DataFrame, d: pd.DataFrame, level: str,
           names: dict[str, dict], quotable: pd.Series) -> pd.DataFrame:
    """Attach the other route's correspondence, and each level's own extras.

    The two routes are produced independently — the intents are written by an
    architect, the leaves fall out of the embedding — so "which intent does this
    leaf mostly land in" is a measurement rather than a definition.
    """
    other = "td_l1" if level.startswith("bu_") else "bu_leaf"
    lab = "主导意图" if level.startswith("bu_") else "主导叶"
    if other not in d.columns:
        # The cross-route correspondence is a measurement, not a definition —
        # a run that delivered only one route still has everything else, and
        # raising here would cost it the entire comparison.
        return t.copy()
    other_names = names.get(other, {})
    dom, share = {}, {}
    for key, g in d.groupby("_key", sort=False):
        vc = g[other].astype(str).value_counts(normalize=True)
        top = str(vc.index[0])
        dom[key] = other_names.get(top, {}).get("name") or _fallback_name(other, top)
        share[key] = round(100 * float(vc.iloc[0]), 1)
    t = t.copy()
    t[lab] = t["key"].map(dom)
    t[f"{lab}占比%"] = t["key"].map(share)
    meta = names.get(level, {})
    for col, field in _EXTRA_FIELDS.get(level, []):
        if any(field in (meta.get(k) or {}) for k in meta):
            t[col] = t["key"].map(lambda k: (meta.get(k) or {}).get(field, ""))
    if level == "td_l2":
        # Sub-intents are a geometric subdivision and carry no name, so the
        # table hands the reader representative strings instead — the highest
        # traffic rows that the quote guard allows.
        rep = {}
        for key, g in d.groupby("_key", sort=False):
            pool = g[quotable.reindex(g.index, fill_value=False)]
            top = pool.sort_values("pv_norm", ascending=False)["query"].head(3).tolist()
            rep[key] = " / ".join(str(x)[:24] for x in top) if top else "（该子意图无可引行）"
        t["代表串(流量最高，仅供辨识)"] = t["key"].map(rep)
    return t


_EXTRA_FIELDS: dict[str, list[tuple[str, str]]] = {
    "bu_leaf": [("盲评一致性", "coherence"), ("风险标注", "risk_flag"), ("user_need", "user_need")],
    "bu_family_final": [("家族内部一致", "coherent"), ("家族审计意见", "audit_notes")],
    "td_l1": [("体系内风险标注", "risk"), ("架构师预估占比", "expected_share"),
              ("user_need", "user_need")],
}


# ------------------------------------------------------------------- absence

def absence(d: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    """Every "this class has zero rows in this snapshot", with its detectability.

    A zero never proves absence on its own. The verdict column is what the prose
    is allowed to quote; the raw zero is not.
    """
    snaps = man.snapshots
    n_s = {s: int((d[SNAP] == s).sum()) for s in snaps}
    rows = []
    for key, g in d.groupby("_key", sort=False):
        cnt = {s: int((g[SNAP] == s).sum()) for s in snaps}
        for s in snaps:
            if cnt[s]:
                continue
            k_o = len(g) - cnt[s]
            n_o = sum(n_s[t] for t in snaps if t != s)
            p_ref = k_o / n_o if n_o else 0.0
            exp = p_ref * n_s[s]
            p0 = (1 - p_ref) ** n_s[s]
            best = (max((t for t in snaps if t != s and n_s[t]),
                        key=lambda t: cnt[t] / n_s[t], default=None) if n_o else None)
            rows.append({
                "key": key, "类目": g["_disp"].iloc[0], "缺席快照": man.display(s),
                "该快照n": n_s[s], "其余快照条数": k_o, "其余快照n": n_o,
                "参照占比%": round(100 * p_ref, 3), "期望条数": round(exp, 2),
                "P(0)": round(p0, 4), "该快照上界%": round(100 * one_sided_upper(n_s[s]), 3),
                "最高占比快照": man.display(best) if best else "",
                "最高占比%": round(100 * cnt[best] / n_s[best], 3) if best else float("nan"),
                "判定": absence_verdict(exp, p0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ pairwise change

def pairwise_change(d: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    """Every class's change across every snapshot pair, with a Newcombe interval."""
    snaps = man.snapshots
    disp = d.drop_duplicates("_key").set_index("_key")["_disp"]
    rows = []
    for i, a in enumerate(snaps):
        for b in snaps[i + 1:]:
            ga, gb = d[d[SNAP] == a], d[d[SNAP] == b]
            na, nb = len(ga), len(gb)
            if not na or not nb:
                continue
            ca, cb = ga["_key"].value_counts(), gb["_key"].value_counts()
            for key in sorted(set(ca.index) | set(cb.index)):
                ka, kb = int(ca.get(key, 0)), int(cb.get(key, 0))
                diff, lo, hi = newcombe(ka, na, kb, nb)
                rows.append({"key": key, "类目": disp.get(key, key),
                             "a": man.display(a), "b": man.display(b),
                             "k_a": ka, "k_b": kb, "n_a": na, "n_b": nb,
                             "占比_a%": round(100 * ka / na, 3),
                             "占比_b%": round(100 * kb / nb, 3),
                             "差_pp": round(100 * diff, 3), "低_pp": round(100 * lo, 3),
                             "高_pp": round(100 * hi, 3), "显著": bool(lo > 0 or hi < 0),
                             "类型": ("b独有" if ka == 0 and kb else
                                      "a独有" if kb == 0 and ka else "共有")})
    return pd.DataFrame(rows)


def pairwise_distance(d: pd.DataFrame, man: SnapshotManifest, tag: str = "",
                      n_boot: int | None = None, n_null: int | None = None) -> pd.DataFrame:
    """Snapshot-to-snapshot distance, with a bootstrap interval AND a noise ceiling.

    Rank correlation sits beside TVD because they answer different questions: two
    snapshots can differ a lot in magnitude while keeping the same ordering (an
    overall rescale), or barely differ in magnitude while reordering completely.
    """
    order = sorted(d["_key"].unique())
    k = len(order)
    arr = {s: codes(d.loc[d[SNAP] == s, "_key"], order) for s in man.snapshots}
    kw: dict[str, int] = {}
    if n_boot is not None:
        kw["n_boot"] = n_boot
    if n_null is not None:
        kw["n_null"] = n_null
    rows = []
    for i, a in enumerate(man.snapshots):
        for b in man.snapshots[i + 1:]:
            ga, gb = arr[a], arr[b]
            if not len(ga) or not len(gb):
                continue
            bounds = tvd_with_bounds(ga, gb, k, tag=(tag, a, b), **kw)
            ct = pd.crosstab(
                pd.concat([d.loc[d[SNAP] == a, "_key"], d.loc[d[SNAP] == b, "_key"]]),
                pd.concat([d.loc[d[SNAP] == a, SNAP], d.loc[d[SNAP] == b, SNAP]]))
            ca = d.loc[d[SNAP] == a, "_key"].value_counts().reindex(order).fillna(0)
            cb = d.loc[d[SNAP] == b, "_key"].value_counts().reindex(order).fillna(0)
            rho = float(ca.rank().corr(cb.rank(), method="spearman"))
            top_a = list(ca.sort_values(ascending=False).head(5).index)
            top_b = list(cb.sort_values(ascending=False).head(5).index)
            rows.append({"a": man.display(a), "b": man.display(b),
                         "n_a": len(ga), "n_b": len(gb), "秩相关rho": round(rho, 4),
                         "前5重合个数": len(set(top_a) & set(top_b)),
                         "首位是否相同": bool(top_a and top_b and top_a[0] == top_b[0]),
                         "TVD": round(bounds["tvd"], 4),
                         "TVD低": round(bounds["lo"], 4), "TVD高": round(bounds["hi"], 4),
                         "同源噪声上界": round(bounds["noise_ceiling"], 4),
                         # NAMED FOR THE BAR IT USES. The conditional-mix table
                         # has a column of the same meaning computed against the
                         # POINT estimate; one name for two different tests is
                         # how a reader concludes the wrong thing from the right
                         # number.
                         "超出噪声(区间下端)": bounds["exceeds_noise"],
                         "CramersV": round(cramers_v(ct.values), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- reliability

def confidence(d: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    """How securely this class was labelled, in this snapshot.

    A share difference says nothing about whether the boundary holds. A class can
    be perfectly stable in share while every row in one snapshot was a coin-flip
    for the classifier — the run leaves per-row readings behind, and aggregating
    them per (class, snapshot) says which cells to discount.
    """
    have = [c for c in ("td_confidence", "td_ambiguous", "bu_ambiguous") if c in d.columns]
    if not have:
        return pd.DataFrame()
    rows = []
    for key, g in d.groupby("_key", sort=False):
        r: dict[str, Any] = {"key": key, "类目": g["_disp"].iloc[0], "全语料条数": len(g)}
        for s in man.snapshots:
            z, gg = man.display(s), g[g[SNAP] == s]
            r[f"{z}_n"] = len(gg)
            for col, suffix, scale in (("td_confidence", "置信度均值", 1),
                                       ("td_ambiguous", "意图歧义%", 100),
                                       ("bu_ambiguous", "聚类歧义%", 100)):
                if col in have:
                    r[f"{z}_{suffix}"] = (round(scale * float(gg[col].astype(float).mean()), 4)
                                          if len(gg) else float("nan"))
        if "td_confidence" in have:
            solid = [s for s in man.snapshots if r[f"{man.display(s)}_n"] >= 20]
            vals = [r[f"{man.display(s)}_置信度均值"] for s in solid]
            worst = min(solid, key=lambda s: r[f"{man.display(s)}_置信度均值"], default=None)
            r["置信度最低快照"] = man.display(worst) if worst else ""
            r["置信度极差"] = round(max(vals) - min(vals), 4) if len(vals) >= 2 else float("nan")
            r["全语料置信度均值"] = round(float(g["td_confidence"].astype(float).mean()), 4)
        rows.append(r)
    t = pd.DataFrame(rows)
    return t.sort_values("全语料置信度均值") if "全语料置信度均值" in t.columns else t


# ------------------------------------------------------------- characteristic

def signature(d: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    """A snapshot's characteristic classes: higher here than in EVERY other one.

    Raw exclusivity is almost always empty — one row elsewhere disqualifies a
    class — so it carries nearly no information about what a snapshot is like.
    This standard carries weight instead: every pairwise Newcombe interval must
    agree in direction and exclude zero, which gets *harder* as snapshots are
    added rather than easier.
    """
    snaps = man.snapshots
    if len(snaps) < 2:
        return pd.DataFrame()
    n_s = {s: int((d[SNAP] == s).sum()) for s in snaps}
    rows = []
    for key, g in d.groupby("_key", sort=False):
        cnt = {s: int((g[SNAP] == s).sum()) for s in snaps}
        for s in snaps:
            others = [t for t in snaps if t != s]
            above = below = True
            worst_hi, worst_lo = 1.0, -1.0
            for t in others:
                _diff, lo, hi = newcombe(cnt[t], n_s[t], cnt[s], n_s[s])
                if not lo > 0:
                    above = False
                if not hi < 0:
                    below = False
                worst_hi = min(worst_hi, lo)
                worst_lo = max(worst_lo, hi)
            if above or below:
                rows.append({
                    "key": key, "类目": g["_disp"].iloc[0], "快照": man.display(s),
                    "方向": "显著高于其余全部" if above else "显著低于其余全部",
                    "该快照条数": cnt[s], "该快照n": n_s[s],
                    "该快照占比%": round(100 * cnt[s] / n_s[s], 3) if n_s[s] else float("nan"),
                    "其余快照合并占比%": round(100 * (len(g) - cnt[s]) / (len(d) - n_s[s]), 3)
                    if len(d) > n_s[s] else float("nan"),
                    "最弱一对的区间端点pp": round(100 * (worst_hi if above else worst_lo), 3),
                    "对比了几个快照": len(others)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- surface split

def surface_split(d: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    """Collapse the snapshots into their declared groups and compare the groups.

    Only produced when every snapshot is assigned to a group — a partial
    assignment silently drops rows from an n that is then reported as if it were
    the corpus, which is a defect this study has actually shipped.
    """
    if not man.has_surfaces:
        return pd.DataFrame()
    groups = man.surface_groups()
    if len(groups) != 2:
        return _surface_multi(d, man, groups)
    (ga_name, ga_snaps), (gb_name, gb_snaps) = list(groups.items())
    da, db = d[d[SNAP].isin(ga_snaps)], d[d[SNAP].isin(gb_snaps)]
    na, nb = len(da), len(db)
    ca, cb = da["_key"].value_counts(), db["_key"].value_counts()
    disp = d.drop_duplicates("_key").set_index("_key")["_disp"]
    rows = []
    for key in sorted(set(ca.index) | set(cb.index)):
        ka, kb = int(ca.get(key, 0)), int(cb.get(key, 0))
        diff, lo, hi = newcombe(ka, na, kb, nb)
        exp = p0 = float("nan")
        verdict = ""
        if kb == 0 and na:
            exp, p0 = (ka / na) * nb, (1 - ka / na) ** nb
        elif ka == 0 and nb:
            exp, p0 = (kb / nb) * na, (1 - kb / nb) ** na
        if ka == 0 or kb == 0:
            verdict = absence_verdict(exp, p0)
        rows.append({"key": key, "类目": disp.get(key, key),
                     f"{ga_name}_条数": ka, f"{ga_name}_n": na,
                     f"{ga_name}_占比%": round(100 * ka / na, 3) if na else float("nan"),
                     f"{gb_name}_条数": kb, f"{gb_name}_n": nb,
                     f"{gb_name}_占比%": round(100 * kb / nb, 3) if nb else float("nan"),
                     "差_pp": round(100 * diff, 3), "低_pp": round(100 * lo, 3),
                     "高_pp": round(100 * hi, 3), "显著": bool(lo > 0 or hi < 0),
                     "归属": (f"仅{ga_name}" if kb == 0 else
                              f"仅{gb_name}" if ka == 0 else "两侧都有"),
                     "另一边期望条数": round(exp, 2) if exp == exp else float("nan"),
                     "另一边P(0)": round(p0, 4) if p0 == p0 else float("nan"),
                     "缺席判定": verdict,
                     f"{gb_name}行占该类比例%": round(100 * kb / (ka + kb), 2) if (ka + kb) else float("nan")})
    return pd.DataFrame(rows).sort_values("差_pp")


def _surface_multi(d: pd.DataFrame, man: SnapshotManifest,
                   groups: dict[str, list[str]]) -> pd.DataFrame:
    """Three or more groups: shares per group, without the two-sided test."""
    disp = d.drop_duplicates("_key").set_index("_key")["_disp"]
    sizes = {g: int(d[SNAP].isin(s).sum()) for g, s in groups.items()}
    rows = []
    for key, g in d.groupby("_key", sort=False):
        r: dict[str, Any] = {"key": key, "类目": disp.get(key, key)}
        for name, snaps in groups.items():
            k = int(g[SNAP].isin(snaps).sum())
            r[f"{name}_条数"] = k
            r[f"{name}_n"] = sizes[name]
            r[f"{name}_占比%"] = round(100 * k / sizes[name], 3) if sizes[name] else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def exclusive_power(d: pd.DataFrame, man: SnapshotManifest, level_name: str) -> dict:
    """If a class really were unique to the smaller group, would this data see it?

    "Zero classes unique to side B" is a conclusion only with a power figure
    attached. This takes the rarest class on the smaller side and asks what the
    same rate would produce on the larger one. The asymmetry it exposes is
    structural, not incidental — the side with fewer rows can almost never
    support the claim in the other direction.
    """
    if not man.has_surfaces:
        return {}
    groups = man.surface_groups()
    if len(groups) != 2:
        return {}
    sized = sorted(groups.items(), key=lambda kv: int(d[SNAP].isin(kv[1]).sum()))
    (small_name, small_snaps), (big_name, big_snaps) = sized
    small, big = d[d[SNAP].isin(small_snaps)], d[d[SNAP].isin(big_snaps)]
    if not len(small) or not len(big):
        return {}
    ks = small["_key"].value_counts()
    kmin = int(ks.min())
    p = kmin / len(small)
    return {"层级": level_name, "较小侧": small_name, "较大侧": big_name,
            "较小侧n": len(small), "较大侧n": len(big),
            "较小侧最小类的条数": kmin, "该发生率%": round(100 * p, 4),
            "同发生率下较大侧期望条数": round(p * len(big), 1),
            "较大侧一条都不出现的概率": round((1 - p) ** len(big), 4),
            f"实际仅{small_name}类数": int(
                (big["_key"].value_counts().reindex(ks.index).fillna(0) == 0).sum())}


# -------------------------------------------------------------- profile / topn

def coverage(d: pd.DataFrame, man: SnapshotManifest, level_zh: str) -> pd.DataFrame:
    rows = []
    for s in man.snapshots:
        g = d[d[SNAP] == s]
        if not len(g):
            continue
        p = g["_key"].value_counts(normalize=True).to_numpy()
        ent = float(-(p * np.log(p)).sum())
        rows.append({"快照": man.display(s), "层级": level_zh, "n": len(g),
                     "类目数": int(g["_key"].nunique()),
                     "有效类目数": round(float(np.exp(ent)), 2), "熵": round(ent, 4),
                     "HHI": round(float((p ** 2).sum()), 4),
                     "首位类目": g["_disp"].value_counts().index[0],
                     "首位占比%": round(100 * p.max(), 2),
                     "前3占比%": round(100 * np.sort(p)[::-1][:3].sum(), 2),
                     "前5占比%": round(100 * np.sort(p)[::-1][:5].sum(), 2)})
    return pd.DataFrame(rows)


def topn(d: pd.DataFrame, man: SnapshotManifest,
         level_zh: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Head concentration per snapshot, plus the top-10 membership it is computed from.

    READ THIS TABLE DOWN A COLUMN, NOT ACROSS. Each snapshot ranks its own
    classes, and rank 10 is almost never the same class twice — so it answers
    "how concentrated is this snapshot's head", not "what do those ten classes
    do elsewhere".

    The ordering is defined HERE and the report only displays it. There used to
    be a second ordering path in the renderer, and on a tie the two paths picked
    different classes: the traffic total for "these same ten classes" was then
    computed over a different ten, measured at 5.25pp apart. Ties are also
    reported, because the row total is tie-immune while the traffic total is not.
    """
    rows, members = [], []
    for s in man.snapshots:
        g = d[d[SNAP] == s]
        n = len(g)
        if not n:
            continue
        cnt = g["_key"].value_counts()
        pv = g.groupby("_key")["pv_norm"].sum()
        pv_tot = float(pv.sum())
        order = sorted(cnt.index, key=lambda k: (-int(cnt[k]), str(k)))
        r: dict[str, Any] = {"层级": level_zh, "快照": man.display(s), "n": n,
                             "类目数": len(order)}
        for k in TOPN_CUTS:
            sel = order[:k]
            c = int(cnt.loc[sel].sum())
            r[f"前{k}条数"] = c
            # Not rounded here: two tables format from this same unrounded value,
            # and double rounding printed one quantity as 98.24 and 98.23.
            r[f"前{k}行占比%"] = 100 * c / n
            r[f"前{k}流量占比%"] = (100 * float(pv.loc[sel].sum()) / pv_tot
                                   if pv_tot else float("nan"))
        for i, k in enumerate(order[:10], 1):
            members.append({"层级": level_zh, "快照": man.display(s), "名次": i, "key": k,
                            "类目": g.loc[g["_key"] == k, "_disp"].iloc[0],
                            "条数": int(cnt[k]), "行占比%": 100 * int(cnt[k]) / n,
                            "流量占比%": (100 * float(pv[k]) / pv_tot
                                         if pv_tot else float("nan"))})
        # The effective n for the TRAFFIC figures is not the row count: one row
        # can hold several percent of a snapshot's weight. Without this a reader
        # applies the row-share precision to the traffic share.
        w = g["pv_norm"].to_numpy(dtype=float)
        tot = w.sum()
        w = w / tot if tot else w
        r["流量有效n"] = float(1.0 / np.square(w).sum()) if tot else float("nan")
        r["单行最大流量%"] = float(100 * w.max()) if len(w) else float("nan")
        r["前10之外类数"] = max(len(order) - 10, 0)
        r["前10之外条数"] = n - r["前10条数"]
        r["前10之外行占比%"] = 100 - r["前10行占比%"]
        r["前10之外流量占比%"] = 100 - r["前10流量占比%"]
        r.update(_tie_band(order, cnt, pv, pv_tot, r["前10流量占比%"]))
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(members)


def _tie_band(order, cnt, pv, pv_tot, point) -> dict[str, Any]:
    """How much the top-10 traffic total could move if the tie broke the other way."""
    if len(order) <= 10:
        return {"第10名条数": int(cnt[order[-1]]) if order else 0, "并列类数": 0,
                "并列需选": 0, "边界并列": False, "前10流量占比%低": point,
                "前10流量占比%高": point, "前10流量摆动pp": 0.0}
    thr = int(cnt[order[9]])
    strict = [k for k in order if int(cnt[k]) > thr]
    tied = [k for k in order if int(cnt[k]) == thr]
    need = 10 - len(strict)
    base = float(pv.loc[strict].sum()) if strict else 0.0
    cand = sorted(float(pv[k]) for k in tied)
    lo, hi = base + sum(cand[:need]), base + sum(cand[len(cand) - need:])
    lo_pct = 100 * lo / pv_tot if pv_tot else float("nan")
    hi_pct = 100 * hi / pv_tot if pv_tot else float("nan")
    return {"第10名条数": thr, "并列类数": len(tied), "并列需选": need,
            "边界并列": bool(len(tied) > need), "前10流量占比%低": lo_pct,
            "前10流量占比%高": hi_pct, "前10流量摆动pp": hi_pct - lo_pct}


# --------------------------------------------------------------- conditional

def conditional_mix(d: pd.DataFrame, outer: str, inner: str, man: SnapshotManifest,
                    min_n: int = 30, tag: str = "",
                    n_null: int = 300) -> pd.DataFrame:
    """Same class, different snapshots — what is it made of inside?

    This is the test for "the share is unchanged but the thing itself is not".
    The inner TVD is meaningless without its own-source noise ceiling: at the
    tens-to-hundreds of rows these cells hold, two samples from ONE distribution
    routinely give 0.2–0.3. Two independent reviewers had to compute this
    column by hand before it existed, and it overturned several claims of the
    form "unchanged in a year / changed the moment the interface changed".
    """
    rows = []
    snaps = man.snapshots
    for key, g in d.groupby(outer, sort=False):
        cnt = {s: int((g[SNAP] == s).sum()) for s in snaps}
        inner_order = sorted(g[inner].astype(str).unique())
        k = len(inner_order)
        for i, a in enumerate(snaps):
            for b in snaps[i + 1:]:
                if cnt[a] < min_n or cnt[b] < min_n:
                    continue
                ga = g.loc[g[SNAP] == a, inner].astype(str)
                gb = g.loc[g[SNAP] == b, inner].astype(str)
                pa = ga.value_counts(normalize=True).reindex(inner_order).fillna(0)
                pb = gb.value_counts(normalize=True).reindex(inner_order).fillna(0)
                obs = float(0.5 * (pa - pb).abs().sum())
                pool = np.concatenate([ga.to_numpy(), gb.to_numpy()])
                cd = pd.Index(inner_order).get_indexer(pool)
                rng = rng_for("conditional_mix", tag, str(key), a, b)
                null = np.empty(n_null)
                for t in range(n_null):
                    idx = rng.permutation(len(cd))
                    x, y = cd[idx[:cnt[a]]], cd[idx[cnt[a]:]]
                    null[t] = tvd(x, y, k)
                hi = float(np.percentile(null, 95))
                rows.append({
                    "外层": g["_disp"].iloc[0] if outer == "_key" else str(key),
                    "a": man.display(a), "b": man.display(b), "n_a": cnt[a], "n_b": cnt[b],
                    "内层TVD": round(obs, 4), "同源噪声上界": round(hi, 4),
                    "超出噪声(点估计)": bool(obs > hi),
                    "a主导内层": str(ga.value_counts().index[0]),
                    "a主导占比%": round(100 * float(pa.max()), 2),
                    "b主导内层": str(gb.value_counts().index[0]),
                    "b主导占比%": round(100 * float(pb.max()), 2),
                    "主导是否相同": bool(ga.value_counts().index[0] == gb.value_counts().index[0])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ examples

def examples(d: pd.DataFrame, man: SnapshotManifest, quotable: pd.Series,
             per: int = 3) -> pd.DataFrame:
    """Real rows per (class, snapshot) — highest traffic, then a random draw.

    A cell whose every row is blocked by the quote guard says so explicitly
    rather than going blank: a blank reads as "no data", which is a different
    and wrong statement.
    """
    rows = []
    for (key, s), g in d.groupby(["_key", SNAP], sort=False):
        if s not in man.sizes:
            continue
        pool = g[quotable.reindex(g.index, fill_value=False)]
        base = {"key": key, "类目": g["_disp"].iloc[0], "快照": man.display(s),
                "该类该快照条数": len(g), "可引条数": len(pool)}
        if not len(pool):
            rows.append({**base, "取法": "不引原文（该类该快照的行全部命中风控图层或引用护栏）",
                         "query": "", "pv_raw": float("nan")})
            continue
        pv_raw = pool["pv_raw"] if "pv_raw" in pool.columns else pool["pv_norm"]
        for _, r in pool.assign(_pv=pv_raw).sort_values("pv_norm", ascending=False).head(per).iterrows():
            rows.append({**base, "取法": "流量最高", "query": r["query"], "pv_raw": r["_pv"]})
        rest = pool.iloc[per:] if len(pool) > per else pool.iloc[0:0]
        if len(rest):
            pvr = rest["pv_raw"] if "pv_raw" in rest.columns else rest["pv_norm"]
            for _, r in rest.assign(_pv=pvr).sample(
                    min(per, len(rest)), random_state=7).iterrows():
                rows.append({**base, "取法": "随机", "query": r["query"], "pv_raw": r["_pv"]})
    return pd.DataFrame(rows)
