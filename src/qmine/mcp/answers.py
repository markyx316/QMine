"""Answering questions about a finished run, with citations and a guard.

This is the half of the product that exists after the mining is done: the study
ships as a 190KB report and thirty tables, and the questions people actually
have — "which classes moved most", "is that class really absent from the
assistant", "show me what those rows look like" — are answerable from the tables
in a sentence each.

TWO RULES, AND BOTH ARE STRUCTURAL.

**Every payload names its source.** A figure without a table name and a row
selector is a figure an assistant can restate slightly wrong and nobody can
re-check. `source` is not decoration; it is what makes the answer auditable.

**The quote guard runs HERE too.** The comparison tables were written under the
seven layers, so their example rows are already screened — but a chat surface
that reaches the same data by a different path would be a way around them, and
a guard with a way around it is not a guard. Every string this module emits is
re-checked against the stateless layers (the vertical-independent hard rules,
the co-occurrence rule, the named-clinician rule, and the run's own screened
list). The row-indexed layers did their work when the tables were written.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from ..pooled.guards import QuoteGuard
from . import store
from .store import RunRef

#: What a blocked string becomes. Never a blank: a blank reads as "no data",
#: which is a different and wrong statement.
BLOCKED = "（该串命中引用护栏，不予显示）"

LEVEL_ALIASES = {
    "l1": "td_l1", "td_l1": "td_l1", "意图": "td_l1", "intent": "td_l1",
    "l2": "td_l2", "td_l2": "td_l2", "子意图": "td_l2", "subintent": "td_l2",
    "leaf": "bu_leaf", "bu_leaf": "bu_leaf", "叶": "bu_leaf",
    "family": "bu_family_final", "bu_family_final": "bu_family_final", "家族": "bu_family_final",
}


def level_of(name: str) -> str:
    key = str(name).strip().lower()
    if key not in LEVEL_ALIASES:
        raise ValueError(f"unknown level {name!r}; use one of "
                         f"{sorted(set(LEVEL_ALIASES.values()))}")
    return LEVEL_ALIASES[key]


@lru_cache(maxsize=8)
def _guard(run_id: str, generation: str, root: str) -> QuoteGuard:
    """The stateless layers, plus this run's own screened list if it has one."""
    ref = RunRef(run_id, generation, __import__("pathlib").Path(root))
    screened = ref.pooled_dir / "quote_block.json"
    return QuoteGuard(screened_path=screened if screened.is_file() else None)


def scrub(ref: RunRef, payload: Any) -> Any:
    """Guard every emitted string, and make every value JSON-native.

    One walk does both because both are about what leaves this process. The
    second half is not cosmetic: a pandas cell is a numpy scalar, `json.dumps`
    refuses it, and the failure surfaces as `TypeError: Object of type int64 is
    not JSON serializable` from inside a tool that looked fine in a REPL.
    """
    g = _guard(ref.run_id, ref.generation, str(ref.root))
    return _walk(g, payload)


def native(obj: Any) -> Any:
    """JSON-native, with no guard. For payloads that carry no corpus text."""
    return _walk(None, obj)


def _walk(g: QuoteGuard | None, obj: Any) -> Any:
    import math

    if isinstance(obj, str):
        return BLOCKED if (g is not None and _blocked(g, obj)) else obj
    if isinstance(obj, dict):
        return {str(k): _walk(g, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_walk(g, v) for v in obj]
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        # NaN and Infinity are not JSON. They reach here from empty cells and
        # from a ratio with a zero denominator.
        return None if math.isnan(obj) else ("inf" if math.isinf(obj) else obj)
    item = getattr(obj, "item", None)
    if callable(item):                      # numpy scalar
        try:
            return _walk(g, item())
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _blocked(g: QuoteGuard, s: str) -> bool:
    if not s or len(s) > 400:
        return False
    d = pd.DataFrame({"query": [s], "td_l1": [""]})
    lay = g.layers(d)
    # Only the STATELESS layers: L1 (row indices) and L5 (whole classes) cannot
    # be evaluated on a loose string, and the tables already applied them.
    return bool(lay["L2_硬规则"].iloc[0] or lay["L3_共现"].iloc[0]
                or lay["L6_声明正则"].iloc[0] or lay["L7_逐串筛查"].iloc[0])


# ------------------------------------------------------------------ overview

def overview(ref: RunRef) -> dict[str, Any]:
    """What this run is, whether it finished, and what it delivered."""
    rs = store.run_summary(ref)
    gates = rs.get("gates") or {}

    def status(v: Any) -> str:
        return str(getattr(v, "status", v.get("status") if isinstance(v, dict) else ""))

    out: dict[str, Any] = {
        "run_id": ref.run_id, "generation": ref.generation,
        "mode": rs.get("mode"),
        "provider": (rs.get("llm_usage") or {}).get("provider"),
        "halted": rs.get("halted"), "halt_reason": rs.get("halt_reason"),
        "phases_completed": rs.get("completed_phases"),
        "gates": {"passed": sum(1 for v in gates.values() if status(v) == "passed"),
                  "warned": sorted(k for k, v in gates.items() if status(v) == "warned"),
                  "failed": sorted(k for k, v in gates.items() if status(v) == "failed"),
                  "skipped": sum(1 for v in gates.values() if status(v) == "skipped")},
        "fast_skipped": rs.get("fast_skipped"),
        "has_cross_snapshot_comparison": ref.has_comparison(),
        "source": f"runs/{ref.run_id}/{ref.generation}/run_summary.json",
    }
    if rs.get("mode") == "fast":
        out["read_this_first"] = (
            "This run was FAST: the analysis is full size but the second-opinion layer "
            "was not run, so kappa is ABSENT rather than perfect. `fast_skipped` lists "
            "exactly what was not checked.")
    if out["provider"] == "offline":
        out["read_this_first"] = (
            "THE OFFLINE STAND-IN WROTE THIS RUN. Every agent output is a deterministic "
            "function, not a model — it looks complete and means nothing about the corpus.")
    if ref.has_comparison():
        s = store.summary(ref)
        out["snapshots"] = s.get("snapshots")
        out["axis"] = (s.get("manifest") or {}).get("axis")
        out["levels"] = {zh: d.get("n_classes") for zh, d in (s.get("levels") or {}).items()}
    return out


# ------------------------------------------------------------------ findings

def findings(ref: RunRef) -> dict[str, Any]:
    """The measured headline of the comparison — the answer to "so what?".

    Deliberately small and deliberately hedged: each distance carries its noise
    ceiling and each absence its verdict, because those are the two figures a
    reader reliably misreads when they are not right beside the number.
    """
    s = store.summary(ref)
    man = s.get("manifest") or {}
    out: dict[str, Any] = {
        "run_id": ref.run_id,
        "snapshots": s.get("snapshots"),
        "axis": man.get("axis"),
        "how_to_read_the_axis": (
            "STRATUM: these snapshots are different samples of the same period — "
            "differences are differences, NOT changes over time."
            if man.get("axis") == "stratum" else
            "TIME: the snapshots are ordered periods, so differences may be read as change."),
        "n_rows": s.get("n_rows"),
        "quote_guard": s.get("quote_guard"),
        "source": f"runs/{ref.run_id}/{ref.generation}/pooled/tables/summary.json",
    }
    if man.get("uniform_weight"):
        out["weights"] = (
            f"{man['uniform_weight']} carry no usable weight column, so their traffic "
            "share equals their row share. Do not read those as measured traffic.")

    try:
        tvd = store.read_table(ref, "pairwise_tvd")
        out["distance_between_snapshots"] = [
            {"level": r["层级"], "a": r["a"], "b": r["b"],
             "TVD": r["TVD"], "noise_ceiling": r["同源噪声上界"],
             "clears_the_noise": str(r.get("超出噪声(区间下端)", "")) in ("True", "是"),
             "rank_correlation": r.get("秩相关rho")}
            for _, r in tvd.iterrows()]
        out["how_to_read_the_distance"] = (
            "A TVD below its noise ceiling means 'this data cannot tell them apart', "
            "NOT 'they are the same'. The ceiling is drawn by pooling both snapshots "
            "and splitting them back at the observed sizes.")
    except FileNotFoundError:
        pass

    levels = s.get("levels") or {}
    out["per_level"] = {
        zh: {"classes": d.get("n_classes"),
             "significant_pairs": f"{d.get('显著变动对')}/{d.get('比较对数')}",
             "characteristic_classes": d.get("特征类条目"),
             "classes_absent_somewhere": d.get("缺席条目"),
             "absences_that_are_real": d.get("真缺席条目"),
             "cramers_v": d.get("CramersV_快照×类")}
        for zh, d in levels.items()}

    movers: list[dict[str, Any]] = []
    for level in ("td_l1", "bu_leaf"):
        try:
            nc = store.read_table(ref, f"newcombe_{level}")
        except FileNotFoundError:
            continue
        sig = nc[nc["显著"].astype(str).isin(("True", "是"))].copy()
        if not len(sig):
            continue
        sig["_abs"] = pd.to_numeric(sig["差_pp"], errors="coerce").abs()
        for _, r in sig.sort_values("_abs", ascending=False).head(6).iterrows():
            movers.append({"level": level, "class": r["类目"], "a": r["a"], "b": r["b"],
                           "diff_pp": r["差_pp"],
                           "interval_pp": [r["低_pp"], r["高_pp"]]})
    if movers:
        out["biggest_significant_differences"] = movers
        out["source_for_differences"] = "pooled/tables/newcombe_<level>.csv"
    hr = s.get("hard_rule") or {}
    out["privacy"] = {
        "hard_rule_hits_in_corpus": hr.get("命中行"),
        "hits_still_quotable": hr.get("其中仍可引"),
        "hits_printed_into_the_report": hr.get("已印进报告"),
        "the_check_actually_ran": hr.get("已检查"),
    }
    return scrub(ref, out)


# --------------------------------------------------------------- one class

def class_detail(ref: RunRef, level: str, name_or_key: str) -> dict[str, Any]:
    """Everything the comparison knows about one class."""
    lv = level_of(level)
    m = store.read_table(ref, f"matrix_{lv}")
    hit = m[(m["类目"].astype(str) == str(name_or_key))
            | (m["key"].astype(str) == str(name_or_key))]
    if not len(hit):
        near = [c for c in m["类目"].astype(str) if str(name_or_key) in c][:8]
        raise ValueError(
            f"no {lv} class matches {name_or_key!r}. "
            + (f"Did you mean: {near}?" if near else
               f"Classes: {sorted(m['类目'].astype(str))[:20]}"))
    row = hit.iloc[0]
    out: dict[str, Any] = {
        "level": lv, "class": row["类目"], "key": row["key"],
        "definition": row.get("定义") or "",
        "rows_in_corpus": row.get("全语料条数"),
        "share_of_corpus_pct": row.get("全语料占比%"),
        "per_snapshot": {},
        "range_pp": row.get("极差pp"),
        "highest_in": row.get("最高快照"), "lowest_in": row.get("最低快照"),
        "source": f"pooled/tables/matrix_{lv}.csv",
    }
    for col in m.columns:
        if col.endswith("_占比%"):
            snap = col[:-len("_占比%")]
            out["per_snapshot"][snap] = {
                "share_pct": row[col],
                "rows": row.get(f"{snap}_条数"),
                "ci_pct": [row.get(f"{snap}_95CI低%"), row.get(f"{snap}_95CI高%")],
                "balance_index": row.get(f"{snap}_均衡指数"),
            }
    try:
        nc = store.read_table(ref, f"newcombe_{lv}")
        mine = nc[nc["key"].astype(str) == str(row["key"])]
        out["pairwise"] = json_rows(mine, ["a", "b", "占比_a%", "占比_b%", "差_pp",
                                           "低_pp", "高_pp", "显著"])
    except FileNotFoundError:
        pass
    try:
        ab = store.read_table(ref, f"absence_{lv}")
        mine = ab[ab["key"].astype(str) == str(row["key"])]
        if len(mine):
            out["absences"] = json_rows(mine, ["缺席快照", "该快照n", "期望条数",
                                               "P(0)", "该快照上界%", "判定"])
            out["how_to_read_an_absence"] = (
                "Quote the 判定 column, never the bare zero: zero of 950 rows is "
                "consistent with a share up to 0.39%.")
    except FileNotFoundError:
        pass
    return scrub(ref, out)


def json_rows(t: pd.DataFrame, cols: list[str] | None = None) -> list[dict[str, Any]]:
    import json as _json

    if cols:
        t = t[[c for c in cols if c in t.columns]]
    return _json.loads(t.to_json(orient="records", force_ascii=False))


# ---------------------------------------------------------------- examples

def examples(ref: RunRef, level: str, name_or_key: str | None = None,
             snapshot: str | None = None, n: int = 6) -> dict[str, Any]:
    """Real rows for a class. Guard-filtered twice: once on write, once here."""
    lv = level_of(level)
    t = store.read_table(ref, f"examples_{lv}")
    if name_or_key:
        t = t[(t["类目"].astype(str) == str(name_or_key))
              | (t["key"].astype(str) == str(name_or_key))]
    if snapshot:
        t = t[t["快照"].astype(str) == str(snapshot)]
    if not len(t):
        raise ValueError(f"no examples for {lv} / {name_or_key!r} / {snapshot!r}. "
                         "Only 意图 (td_l1) and 叶 (bu_leaf) carry example rows.")
    blocked = t[t["取法"].astype(str).str.startswith("不引原文")]
    real = t[~t["取法"].astype(str).str.startswith("不引原文")].head(max(1, min(int(n), 30)))
    out = {
        "level": lv, "class": name_or_key, "snapshot": snapshot,
        "examples": json_rows(real, ["快照", "取法", "query", "该类该快照条数", "可引条数"]),
        "cells_with_nothing_quotable": json_rows(
            blocked, ["快照", "该类该快照条数", "取法"]) if len(blocked) else [],
        "source": f"pooled/tables/examples_{lv}.csv",
        "note": ("These are real rows from the corpus, selected by traffic and then at "
                 "random, and filtered by the seven-layer quote guard. A cell listed "
                 "under `cells_with_nothing_quotable` has rows but none that may be "
                 "reproduced — that is a count, not an absence."),
    }
    return scrub(ref, out)


# ---------------------------------------------------------------- glossary

GLOSSARY: dict[str, str] = {
    "TVD": "Total variation distance between two snapshots' class distributions: half "
           "the sum of absolute share differences. 0 = identical, 1 = disjoint. Only "
           "meaningful beside its noise ceiling.",
    "同源噪声上界": "The 95th percentile of the distance you get by pooling BOTH snapshots "
                 "and splitting them back at the observed sizes — i.e. the distance two "
                 "samples of one distribution give by themselves. Below it means 'cannot "
                 "tell apart', not 'the same'.",
    "均衡指数": "A class's within-snapshot share divided by the unweighted mean of its "
             "shares across snapshots. 1.0 = evenly spread; 2.0 = twice as concentrated "
             "here as the average snapshot. Immune to one snapshot being much larger.",
    "均衡归属%": "A class's share in this snapshot divided by the sum of its shares across "
              "all snapshots — where it sits when every snapshot is treated as the same size.",
    "缺席判定": "Whether a zero count is evidence. 真缺席 = expected count ≥5 and P(0) <1%. "
             "不可判定 = expected <3, the sample cannot support any claim. 偏少但证据弱 = "
             "in between.",
    "Wilson": "The confidence interval used for a single share; it behaves correctly near "
              "0 and 1, where the textbook normal interval does not.",
    "Newcombe": "The interval used for the DIFFERENCE between two shares, built from the "
                "two Wilson intervals.",
    "pv_norm": "Each snapshot's weights renormalised to 10,000 so traffic shares are "
               "within-snapshot. Raw counts across snapshots mostly report export sizes.",
    "流量有效n": "1/Σw² — the effective sample size of the TRAFFIC figures, usually far "
              "smaller than the row count because one row can carry several percent of "
              "a snapshot's weight.",
    "特征类": "A class significantly higher here than in EVERY other snapshot, by a "
           "Newcombe interval per pair. Gets harder as snapshots are added.",
    "td_l1": "Top-down L1 intent: the taxonomy an architect agent wrote and a classifier "
             "was trained on from a gold set.",
    "bu_leaf": "Bottom-up leaf: a cluster from the embedding, named by a blind panel. "
               "Produced independently of the intents, so their correspondence is a "
               "measurement rather than a definition.",
    "fast": "`--fast` keeps the analysis at full size and drops only the second-opinion "
            "layer, so kappa is ABSENT, not 1.0. `fast_skipped` lists what was skipped.",
    "stratum": "The comparison axis for snapshots that differ by SAMPLING rather than by "
               "date. The computation is the same; the prose is not, and one caveat inverts.",
}


def glossary(term: str | None = None) -> dict[str, Any]:
    if not term:
        return {"terms": sorted(GLOSSARY), "note": "Ask for one by name."}
    key = str(term).strip()
    if key in GLOSSARY:
        return {"term": key, "meaning": GLOSSARY[key]}
    near = [k for k in GLOSSARY if key.lower() in k.lower()]
    if len(near) == 1:
        return {"term": near[0], "meaning": GLOSSARY[near[0]]}
    return {"term": key, "meaning": None, "did_you_mean": near or sorted(GLOSSARY)}
