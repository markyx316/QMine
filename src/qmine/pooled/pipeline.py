"""One call that turns a finished pooled run into the comparison deliverables.

Used from two places — the `p10c_pooled` phase node and `qmine compare` — so the
two cannot drift apart.

NOTHING HERE MAKES A MODEL CALL. Every number is computed from the run's own
artifacts, which is what lets a fast run ship the same comparison as a full one:
fast mode removes the second-opinion layer, and this is analysis, not checking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .figures import build_figures
from .frame import class_names, load_frame
from .guards import QuoteGuard, printed_strings
from .manifest import SnapshotManifest
from .report import build_report
from .source import RunSource
from .tables import build_tables
from .verify import check, render_verification
from .workbook import write_rows_workbook, write_tables_workbook


@dataclass
class PooledOptions:
    """Everything a user may say about a comparison. All of it optional."""

    axis: str = "time"
    labels: dict[str, str] = field(default_factory=dict)
    surfaces: dict[str, str] = field(default_factory=dict)
    contrasts: list[list[Any]] = field(default_factory=list)
    weight_column: str | None = None
    extra_quote_patterns: list[str] = field(default_factory=list)
    never_quote_classes: list[str] = field(default_factory=list)
    extra_hard_rules: dict[str, str] = field(default_factory=dict)
    screened_quote_block: str | None = None
    title: str = ""
    n_boot: int | None = None
    n_null: int | None = None
    min_conditional_n: int = 30
    narrative: dict[str, str] = field(default_factory=dict)


@dataclass
class PooledResult:
    out_dir: Path
    report_path: Path
    verification_path: Path
    tables_dir: Path
    tables_workbook: Path
    rows_workbook: Path
    figures: list[Path]
    summary: dict[str, Any]
    manifest: SnapshotManifest
    verification: dict[str, Any]
    hard_rule_hits: pd.DataFrame


def run_comparison(src: RunSource, opts: PooledOptions, *,
                   out_dir: Path | None = None) -> PooledResult:
    """Build every cross-snapshot deliverable for one finished pooled generation."""
    gen = src.gen_dir()
    out = Path(out_dir) if out_dir else gen / "pooled"
    tables_dir = out / "tables"
    img_dir = out / "img"

    d, man = load_frame(src, weight_column=opts.weight_column, axis=opts.axis,
                        labels=opts.labels, surfaces=opts.surfaces,
                        contrasts=opts.contrasts or None)
    if len(man.snapshots) < 2:
        raise ValueError(
            f"this run has one snapshot ({man.snapshots}); there is nothing to compare. "
            "Pool several inputs into ONE run — two separate runs' labels share no codes.")

    guard = QuoteGuard(
        risk_screen=src.json("risk_screen"),
        tree_naming=src.json("tree_naming"),
        taxonomy=src.json("taxonomy_v2") or src.json("taxonomy"),
        extra_patterns=opts.extra_quote_patterns,
        declared_never_quote=opts.never_quote_classes,
        screened_path=opts.screened_quote_block,
        extra_hard_rules=opts.extra_hard_rules)

    names = class_names(src)
    run_id = src.run_id()
    built = build_tables(d, man, names, guard, tables_dir, n_boot=opts.n_boot,
                         n_null=opts.n_null, min_conditional_n=opts.min_conditional_n,
                         tag=run_id)
    frames, summary = built["frames"], built["summary"]

    figures = build_figures(frames, man, img_dir)

    # FIXED FILENAMES. These land inside `runs/<id>/<gen>/`, which already names
    # the corpus, so a per-corpus prefix buys nothing and costs the index its
    # literal filenames — the delivered-document table matches on the name.
    title = opts.title or run_id
    rows_wb = out / "跨快照对比_逐行标注.xlsx"
    tables_wb = out / "跨快照对比_对照表.xlsx"
    report_path = out / "跨快照对比_逐类对照.zh.md"

    text = build_report(frames, summary, man, title=title, run_id=run_id,
                        generation=gen.name, img_dir="img",
                        narrative=opts.narrative, workbook_name=rows_wb.name)
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")

    write_tables_workbook(frames, summary, man, tables_wb, run_id=run_id,
                          generation=gen.name)
    write_rows_workbook(d, man, summary, rows_wb, run_id=run_id, generation=gen.name)

    quotable_mask = guard.quotable(d)
    corpus = set(d["query"].astype(str))
    ver = check(text, tables_dir, corpus=corpus,
                quotable=set(d.loc[quotable_mask, "query"].astype(str)),
                year_like=_year_like(man),
                extra_numbers=_rendered_numbers(man))
    verification_path = out / "跨快照对比_叙述复核.md"
    verification_path.write_text(render_verification(ver, report_path), encoding="utf-8")

    # THE HARD-RULE SCAN READS WHAT WAS ACTUALLY PRINTED. Its `printed` set must
    # be non-empty or "0 printed" is silence being read as a pass.
    # AN EMPTY PRINTED SET IS A SKIP, NEVER A PASS. "0 hard-rule strings
    # printed" is only a measurement when the report actually quoted corpus
    # rows; with nothing quoted the check has not run, and a gate reading PASS
    # would be indistinguishable from one that looked.
    printed = printed_strings([report_path], corpus=corpus,
                              tables=sorted(tables_dir.glob("examples_*.csv")))
    hits = guard.hard_rule_scan(d, printed)
    if len(hits):
        hits.to_csv(out / "硬规则命中.csv", index=False, encoding="utf-8-sig")

    summary["report"] = report_path.name
    summary["figures"] = [p.name for p in figures]
    summary["verification"] = {k: v for k, v in ver.items() if k != "blocks"}
    summary["hard_rule"] = {
        "已检查": bool(printed),
        "命中行": int(len(hits)),
        "其中仍可引": int((~hits["已被护栏拦下"]).sum()) if len(hits) else 0,
        "已印进报告": int(hits["出现在交付文档里"].sum()) if len(hits) else 0,
        "报告里引用的语料串": len(printed)}
    (tables_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    return PooledResult(out_dir=out, report_path=report_path,
                        verification_path=verification_path, tables_dir=tables_dir,
                        tables_workbook=tables_wb, rows_workbook=rows_wb,
                        figures=figures, summary=summary, manifest=man,
                        verification=ver, hard_rule_hits=hits)


def _rendered_numbers(man: SnapshotManifest) -> list[float]:
    """Figures the REPORT computes and no table holds.

    The preamble prints a snapshot-size ratio and a detectability bound at the
    smallest and largest snapshot. They are rendered, so they are true; a pool
    that lacks them reports them as unmatched the moment an author quotes one.
    """
    from .stats import one_sided_upper

    sizes = list(man.sizes.values())
    if not sizes:
        return []
    lo, hi = min(sizes), max(sizes)
    return [hi / max(lo, 1), 100 * one_sided_upper(lo), 100 * one_sided_upper(hi),
            float(lo), float(hi), float(sum(sizes))]


def _year_like(man: SnapshotManifest) -> Sequence[str]:
    """Snapshot names are NAMES, not measurements — `2026` in a tag is not a figure."""
    out: set[str] = set()
    for s in man.snapshots:
        for token in (str(s), man.display(s)):
            out.update(t for t in _digits(token))
    return sorted(out)


def _digits(s: str) -> list[str]:
    cur, out = "", []
    for ch in s:
        if ch.isdigit():
            cur += ch
        elif cur:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out
