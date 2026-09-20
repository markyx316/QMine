"""Apply a plan, and record everything it did — including what it was refused.

The output is two tables and a report:

* `prepared.parquet` — the rows that enter the mining, with `query`, `snapshot`,
  `pv_raw`, `pv_norm` and whatever the plan kept.
* `prepared_all_rows.parquet` — EVERY row, including the removed ones, with the
  tier that removed it. This is what makes the cleaning checkable.
* `prep_report.json` — per input, per operation: how many rows it touched, and
  every refusal.

ORDER IS LOAD-BEARING IN THREE PLACES, and each is here rather than in a comment
on the plan, because a plan cannot enforce its own order:

* a head cut runs BEFORE tiering, or rows outside the intended sample get tiered
  and then counted as product noise;
* an aggregation runs BEFORE tiering, because per-day rows are not per-query
  rows and a rule that reads a weight reads the wrong one;
* `pv_norm` is computed AFTER the tier filter, because the denominator is the
  traffic that was KEPT, not the traffic that was exported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inspect import has_content, read_any
from .plan import PrepOp, PrepPlan

USER_TIER = "user"
NORM_TOTAL = 10_000.0

#: Operations whose whole purpose is to remove a large share, so the
#: `max_drop_share` guard does not apply to them.
_UNBOUNDED = {"drop_empty", "head_cut", "drop_duplicates", "aggregate_by_query",
              "merge_collisions"}


class PrepError(RuntimeError):
    pass


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise PrepError(f"{where}: pattern {pattern!r} does not compile: {exc}") from exc


def _mask(rx: re.Pattern[str], s: pd.Series) -> np.ndarray:
    """Python `re`, never `Series.str.contains`. See the module docstring."""
    return s.astype(str).map(lambda x: bool(rx.search(x))).to_numpy()


def apply_ops(df: pd.DataFrame, spec: Any, *, max_drop_share: float
              ) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one input's operations. Returns (frame, op records, refusals)."""
    d = df.copy()
    d["query"] = d[spec.text_column].astype(str)
    d["query_raw"] = d["query"]
    d["wrapper_stripped"] = False
    d["tier"] = USER_TIER
    d["pv_raw"] = (pd.to_numeric(d[spec.weight_column], errors="coerce")
                   if spec.weight_column and spec.weight_column in d.columns
                   else np.nan)
    records: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    n0 = len(d)

    for i, op in enumerate(spec.ops):
        before_rows, before_user = len(d), int((d["tier"] == USER_TIER).sum())
        d, note = _apply_one(d, op, spec, refusals, max_drop_share=max_drop_share, index=i)
        records.append({
            "序": i, "op": op.op, "tier": op.name or "", "why": op.why,
            "作用前行数": before_rows, "作用后行数": len(d),
            "本步移出挖掘": before_user - int((d["tier"] == USER_TIER).sum()),
            "说明": note})
    records.append({"序": len(spec.ops), "op": "(总计)", "tier": "", "why": "",
                    "作用前行数": n0, "作用后行数": len(d),
                    "本步移出挖掘": int((d["tier"] != USER_TIER).sum()), "说明": ""})
    return d, records, refusals


def _apply_one(d: pd.DataFrame, op: PrepOp, spec: Any, refusals: list[dict[str, Any]],
               *, max_drop_share: float, index: int) -> tuple[pd.DataFrame, str]:
    where = f"{Path(spec.path).name} op#{index} {op.op}"
    tier = op.name or op.op

    if op.op == "trim":
        d["query"] = d["query"].astype(str).str.strip()
        return d, ""

    if op.op == "drop_empty":
        hit = ~d["query"].map(has_content).to_numpy()
        d.loc[hit & (d["tier"] == USER_TIER), "tier"] = tier
        return d, f"{int(hit.sum())} 行无实质内容"

    if op.op == "drop_duplicates":
        hit = d.duplicated(subset=["query"], keep="first").to_numpy()
        d.loc[hit & (d["tier"] == USER_TIER), "tier"] = tier
        return d, f"{int(hit.sum())} 行是本文件内的重复串"

    if op.op == "length_filter":
        n = d["query"].astype(str).str.len()
        hit = np.zeros(len(d), dtype=bool)
        if op.min_len is not None:
            hit |= (n < op.min_len).to_numpy()
        if op.max_len is not None:
            hit |= (n > op.max_len).to_numpy()
        hit = _bounded(hit, d, op, where, refusals, max_drop_share)
        d.loc[hit & (d["tier"] == USER_TIER), "tier"] = tier
        return d, f"{int(hit.sum())} 行超出 [{op.min_len}, {op.max_len}]"

    if op.op == "strip_prefix":
        rx = _compile(op.pattern, where)
        q = d["query"].astype(str)
        stripped = q.map(lambda s: rx.sub("", s, count=1).strip())
        changed = (stripped != q) & (stripped.str.len() > 0)
        # A row that is NOTHING but the wrapper keeps its original text and falls
        # to whatever rule handles content-free strings. Emptying it here would
        # delete a row without saying so.
        d.loc[changed, "query"] = stripped[changed]
        d.loc[changed, "wrapper_stripped"] = True
        return d, f"{int(changed.sum())} 行剥掉了包装前缀（没有删任何行）"

    if op.op == "flag_regex":
        rx = _compile(op.pattern, where)
        hit = _mask(rx, d["query"])
        col = f"flag_{tier}"
        d[col] = hit
        return d, f"{int(hit.sum())} 行命中（只标记，仍进入挖掘）"

    if op.op == "drop_regex":
        rx = _compile(op.pattern, where)
        hit = _mask(rx, d["query"])
        hit = _bounded(hit, d, op, where, refusals, max_drop_share)
        d.loc[hit & (d["tier"] == USER_TIER), "tier"] = tier
        return d, f"{int(hit.sum())} 行命中"

    if op.op == "head_cut":
        n = int(op.n or 0)
        if n <= 0 or n >= len(d):
            return d, "不需要截断"
        w = pd.to_numeric(d["pv_raw"], errors="coerce")
        # NOT just "any weight at all". A column populated for 4 of 60 rows
        # ranks those 4 and tiers away the other 93% — a catastrophic removal
        # that `head_cut`'s exemption from the over-large-removal guard lets
        # straight through, because taking a head IS meant to remove a lot.
        covered = float(w.notna().mean()) if len(w) else 0.0
        if covered < 0.9:
            refusals.append({
                "where": where, "op": op.op, "coverage": round(covered, 4),
                "reason": (f"the weight column is populated for only "
                           f"{100 * covered:.1f}% of this input, so a head cut would "
                           "rank a handful of rows and tier away the rest — refused, "
                           "nothing removed")})
            return d, f"拒绝：权重列只有 {100 * covered:.1f}% 有值，头部边界无法验证"
        order = w.rank(ascending=False, method="min")
        cut = float(w.sort_values(ascending=False).iloc[n - 1])
        ties = int((w == cut).sum())
        keep = (order <= n) | (w > cut)
        d.loc[(~keep).to_numpy() & (d["tier"] == USER_TIER), "tier"] = tier or "below_head_cut"
        return d, (f"保留权重最高的 {n} 行；边界权重 {cut:g}，并列 {ties} 行"
                   + ("（并列跨越边界，实际保留数可能多于 n）" if ties > 1 else ""))

    if op.op == "aggregate_by_query":
        keep_first = [c for c in d.columns if c not in ("pv_raw",)]
        agg = d.groupby("query", sort=False, as_index=False).agg(
            {**{c: "first" for c in keep_first if c != "query"}, "pv_raw": "sum"})
        # pandas sums an all-NaN group to 0.0. A weight that does not exist must
        # stay absent: a 0.0 reads as "no traffic" and, worse, it disarmed
        # `head_cut`'s "no usable weight" refusal, which then silently tiered
        # away everything below rank n on a column of zeros.
        had = d.groupby("query", sort=False)["pv_raw"].apply(lambda x: x.notna().any())
        agg.loc[~had.to_numpy(), "pv_raw"] = np.nan
        agg["n_rows_raw"] = d.groupby("query", sort=False).size().to_numpy()
        return agg.reset_index(drop=True), f"{len(d):,} 行合并成 {len(agg):,} 个串"

    if op.op == "merge_collisions":
        dup = int(d.duplicated(["query"]).sum())
        if not dup:
            return d, "剥包装后没有产生重复串"
        before = len(d)
        d["pv_raw"] = d.groupby("query")["pv_raw"].transform("sum")
        n_raw = d.groupby("query")["query"].transform("size")
        d = d.assign(n_rows_raw=n_raw).drop_duplicates(["query"]).reset_index(drop=True)
        # MERGING IS NOT REMOVING, and the difference has to be visible. These
        # rows' text and weight are both preserved — but the all-rows table now
        # holds fewer rows than the file, so `n_rows_raw` records how many input
        # rows each surviving row stands for, and the total is asserted.
        assert int(d["n_rows_raw"].sum()) == before, (
            f"merge lost rows: {int(d['n_rows_raw'].sum())} accounted for of {before}")
        return d, (f"{dup} 个剥包装后重合的串已合并，权重相加"
                   f"（{before:,} 行 → {len(d):,} 行，n_rows_raw 记录每行代表几条原始行）")

    raise PrepError(f"{where}: unknown operation {op.op!r}")


def _bounded(hit: np.ndarray, d: pd.DataFrame, op: PrepOp, where: str,
             refusals: list[dict[str, Any]], max_drop_share: float) -> np.ndarray:
    """Refuse an over-large removal by turning it into a flag.

    Returning an all-False mask is what makes this a REFUSAL rather than a
    warning: the rows stay in the mining, the flag column records the match, and
    the report names the operation. A guard that only warns is a guard that gets
    read after the corpus has already changed.
    """
    if op.op in _UNBOUNDED:
        return hit
    share = float(hit.mean()) if len(hit) else 0.0
    if share <= max_drop_share:
        return hit
    refusals.append({
        "where": where, "op": op.op, "tier": op.name, "pattern": op.pattern,
        "share": round(share, 4), "limit": max_drop_share,
        "reason": (f"would remove {100 * share:.1f}% of this input, over the "
                   f"{100 * max_drop_share:.0f}% limit — applied as a FLAG instead, "
                   "so the rows stay in the mining and the match is still recorded"),
        "flag_column": f"flag_refused_{op.name or op.op}"})
    d[f"flag_refused_{op.name or op.op}"] = hit
    return np.zeros(len(hit), dtype=bool)


def execute(plan: PrepPlan, out_dir: Path | str, *, max_drop_share: float = 0.25,
            previous: Path | str | None = None) -> dict[str, Any]:
    """Run the whole plan and write the corpus, the all-rows table and the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tags = plan.tags()
    if len(set(tags)) != len(tags):
        raise PrepError(
            f"two inputs resolve to the same snapshot tag ({tags}). The comparison "
            "could not tell them apart; give them distinguishable names.")
    groups = [i.group for i in plan.inputs]
    if any(groups) and not all(groups):
        raise PrepError(
            "some inputs have a group and some do not. A partial grouping drops the "
            "unassigned snapshots out of the group tables while their rows still count "
            "toward the reported n — either group every input or none.")

    parts, report = [], {"axis": plan.comparison_axis, "rationale": plan.rationale,
                         "confidence": plan.confidence, "concerns": list(plan.concerns),
                         "max_drop_share": max_drop_share, "inputs": []}
    for spec in plan.inputs:
        raw = read_any(spec.path)
        if spec.text_column not in raw.columns:
            raise PrepError(
                f"{Path(spec.path).name} has no column {spec.text_column!r}. "
                f"Columns: {sorted(map(str, raw.columns))}")
        d, records, refusals = apply_ops(raw, spec, max_drop_share=max_drop_share)
        d["snapshot"] = spec.snapshot
        d["snapshot_display"] = spec.display or spec.snapshot
        d["group"] = spec.group
        d["source_file"] = Path(spec.path).name
        parts.append(d)
        kept = int((d["tier"] == USER_TIER).sum())
        report["inputs"].append({
            "path": spec.path, "snapshot": spec.snapshot,
            "display": spec.display or spec.snapshot, "group": spec.group,
            "text_column": spec.text_column, "weight_column": spec.weight_column,
            "原始行数": int(len(raw)), "处理后行数": int(len(d)),
            # MERGED is not REMOVED. Reporting only a removal percentage against
            # the post-merge count said "0.0% removed" for a file that went from
            # 3 rows to 2, which is true and unreadable.
            "合并掉的行": int(len(raw) - len(d)),
            "进入挖掘": kept, "被清洗剔除": int(len(d) - kept),
            "剔除占比%": round(100 * (len(d) - kept) / max(len(d), 1), 2),
            "占原始行数%": round(100 * kept / max(len(raw), 1), 2),
            "各步": records, "拒绝执行": refusals, "notes": spec.notes})

    allrows = pd.concat(parts, ignore_index=True)
    allrows["kept"] = allrows["tier"] == USER_TIER
    allrows["row_id_all"] = range(len(allrows))
    mine = allrows[allrows["kept"]].reset_index(drop=True).copy()
    if not len(mine):
        raise PrepError("every row was removed by the plan — refusing to write an "
                        "empty corpus. Read `prep_report.json` for which step did it.")
    # PV NORMALISATION IS AFTER THE FILTER: the denominator is the KEPT traffic.
    mine["pv_norm"] = _normalise(mine)
    mine["row_id"] = range(len(mine))

    if previous:
        report["superset_check"] = _superset_check(Path(previous), mine)

    keep = ["query", "query_raw", "wrapper_stripped", "snapshot", "snapshot_display",
            "group", "source_file", "pv_raw", "pv_norm", "tier", "row_id", "row_id_all"]
    keep += [c for c in mine.columns if c.startswith("flag_")]
    for spec in plan.inputs:
        keep += [c for c in spec.keep_columns if c in mine.columns]
    keep = list(dict.fromkeys(c for c in keep if c in mine.columns))

    corpus_path = out / "prepared_corpus.parquet"
    mine[keep].to_parquet(corpus_path, index=False)
    allrows.to_parquet(out / "prepared_all_rows.parquet", index=False)
    report["corpus"] = str(corpus_path)
    report["n_mined"] = int(len(mine))
    report["n_all"] = int(len(allrows))
    report["snapshots"] = {s: int((mine["snapshot"] == s).sum()) for s in tags}
    report["columns"] = keep
    report["removed_by_tier"] = (allrows.loc[~allrows["kept"]]
                                 .groupby(["snapshot", "tier"]).size()
                                 .reset_index(name="n").to_dict("records"))
    (out / "prep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return report


def _normalise(mine: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=mine.index, dtype=float)
    for snap, g in mine.groupby("snapshot", sort=False):
        w = pd.to_numeric(g["pv_raw"], errors="coerce").where(lambda x: x > 0)
        if w.notna().sum() == 0:
            out.loc[g.index] = NORM_TOTAL / max(len(g), 1)
            continue
        # THE SMALLEST OBSERVED WEIGHT, NOT THE MEDIAN. A row whose traffic we
        # do not know should not be handed the traffic of a typical row: the
        # median promotes every blank into the middle of the distribution and
        # inflates whatever class those blanks land in.
        w = w.fillna(float(w.min()))
        out.loc[g.index] = w * (NORM_TOTAL / float(w.sum()))
    return out


def _superset_check(previous: Path, mine: pd.DataFrame) -> dict[str, Any]:
    """A rebuilt corpus must reproduce the snapshots it already delivered.

    Adding a snapshot to a corpus that has already been mined and reported on is
    the commonest way to invalidate a delivered study without noticing: the
    shared snapshots must come out byte-for-byte the same, or every table under
    the old report now describes different rows.
    """
    try:
        old = pd.read_parquet(previous)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "why": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"ran": True, "snapshots": {}}
    for snap in pd.unique(old["snapshot"].astype(str)) if "snapshot" in old.columns else []:
        a = old[old["snapshot"].astype(str) == snap].reset_index(drop=True)
        b = mine[mine["snapshot"].astype(str) == snap].reset_index(drop=True)
        same_len = len(a) == len(b)
        diffs = 0
        if same_len:
            for col in ("query", "pv_raw", "pv_norm"):
                if col in a.columns and col in b.columns:
                    va, vb = a[col].astype(str).to_numpy(), b[col].astype(str).to_numpy()
                    diffs += int((va != vb).sum())
        out["snapshots"][str(snap)] = {
            "rows_before": int(len(a)), "rows_now": int(len(b)),
            "same_length": same_len, "cell_differences": diffs if same_len else None,
            "reproduced": bool(same_len and diffs == 0)}
    out["all_reproduced"] = all(v["reproduced"] for v in out["snapshots"].values()) \
        if out["snapshots"] else None
    return out
