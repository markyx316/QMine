"""The planner that reads the measurements — and the checks that bound it.

Follows the domain scout's contract exactly: the agent produces HYPOTHESES in a
schema, a program validates them, and nothing it returns changes a parameter
without passing that validation. Specifically:

* every operation must be one of the closed vocabulary (pydantic refuses others);
* every regex must compile, or the plan is rejected by name;
* every input the agent names must be one that was actually offered;
* a removal that is too large is downgraded to a flag by the executor, whatever
  the plan says;
* if anything fails, the deterministic plan is used and the failure is recorded.

A model here is worth having for one thing the measurements cannot give: reading
a column of repeated prefixes and a length distribution and saying *what kind of
export this is*. It is not worth handing it the power to delete rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from .inspect import InputProfile
from .plan import PrepPlan
from .planner import heuristic_plan


class CorpusPrepAgent(Agent):
    """Proposes a `PrepPlan` over measured input profiles."""

    role = "corpus_prep"
    prompt_name = "prepare_corpus"
    schema = PrepPlan

    def build_user(self, *, profiles: list[dict[str, Any]] | None = None,
                   baseline: dict[str, Any] | None = None, **_: Any) -> str:
        parts = [
            "## 待合并的导出（全部是测量值，不是原始行）\n",
            json.dumps(profiles or [], ensure_ascii=False, indent=1),
            "\n\n## 一份纯机械推导出来的方案，供你对照\n",
            "它只用上面的数字，没有任何判断。你可以照搬、修改或推翻它，"
            "但每一处改动都要在 `why` 里说明依据。\n",
            json.dumps(baseline or {}, ensure_ascii=False, indent=1),
            "\n\n给出完整方案：每个输入的列、快照名、操作序列，以及 "
            "`comparison_axis`、`rationale`、`confidence`、`concerns`。",
        ]
        return "".join(parts)


def plan_corpus(profiles: list[InputProfile], ctx: Any = None, *,
                text_column: str | None = None, axis: str | None = None,
                ) -> tuple[PrepPlan, dict[str, Any]]:
    """The plan, and a record of how it was arrived at.

    Fails soft by design: a planner that cannot run must not stop a preparation
    that would otherwise be fine, and the deterministic plan is a real plan
    rather than a placeholder.
    """
    base = heuristic_plan(profiles, text_column=text_column, axis=axis)
    meta: dict[str, Any] = {"source": "heuristic", "agent": None}
    if ctx is None:
        return base, meta

    offered = {str(p.path) for p in profiles}
    try:
        out = CorpusPrepAgent(ctx).run(
            profiles=[p.brief() for p in profiles],
            baseline=json.loads(base.model_dump_json()))
    except Exception as exc:  # noqa: BLE001
        meta["agent"] = f"unavailable: {type(exc).__name__}: {exc}"
        return base, meta

    problems = validate_plan(out, offered)
    if problems:
        meta["agent"] = "rejected"
        meta["problems"] = problems
        return base, meta
    meta["source"] = "agent"
    meta["agent"] = "accepted"
    meta["differences"] = diff_plans(base, out)
    return out, meta


#: Operations whose result later ops depend on, and where they have to sit.
#: The executor's docstring calls the order load-bearing and then trusted the
#: plan to get it right; a plan that tiers before cutting a head, or reads a
#: weight before aggregating a per-day export, produces a corpus that looks fine.
_MUST_PRECEDE = [
    ("aggregate_by_query", ("head_cut", "drop_regex", "length_filter", "flag_regex"),
     "a per-day export's weights are per-day until it is aggregated, so anything "
     "that reads a weight or tiers a row must come after"),
    ("head_cut", ("drop_regex", "length_filter", "flag_regex"),
     "tiering a row that the head cut would have removed anyway files it as "
     "product noise, which it is not"),
    ("strip_prefix", ("merge_collisions",),
     "collisions only exist once the wrapper is off"),
]


def validate_plan(plan: PrepPlan, offered: set[str]) -> list[str]:
    """Everything a plan must satisfy before it is allowed to run."""
    import re as _re

    problems: list[str] = []
    if not plan.inputs:
        problems.append("the plan names no inputs")
    missing = offered - {str(s.path) for s in plan.inputs} - {
        p for p in offered if any(Path(p).name == Path(s.path).name for s in plan.inputs)}
    if missing:
        # Quietly dropping an input leaves the run comparing fewer snapshots
        # than the person asked for, and nothing downstream can tell.
        problems.append(
            f"{sorted(Path(m).name for m in missing)} were offered and the plan does "
            "not name them — a dropped input silently shrinks the comparison")
    seen: set[str] = set()
    for spec in plan.inputs:
        name = Path(spec.path).name
        if str(spec.path) not in offered and not any(
                Path(o).name == name for o in offered):
            problems.append(f"{spec.path!r} is not one of the inputs that were offered")
        if not spec.text_column:
            problems.append(f"{name}: no text column named")
        if spec.snapshot in seen:
            problems.append(f"{name}: snapshot tag {spec.snapshot!r} is used twice")
        seen.add(spec.snapshot)
        for i, op in enumerate(spec.ops):
            if op.op in ("strip_prefix", "flag_regex", "drop_regex"):
                if not op.pattern:
                    problems.append(f"{name} op#{i} {op.op}: no pattern")
                    continue
                try:
                    _re.compile(op.pattern)
                except _re.error as exc:
                    problems.append(f"{name} op#{i} {op.op}: pattern does not compile ({exc})")
            if op.op in ("drop_regex", "length_filter", "head_cut") and not op.why.strip():
                # A removal whose reason is not written down cannot be reviewed,
                # and an unreviewable removal is the one that ships.
                problems.append(f"{name} op#{i} {op.op}: removes rows with no stated reason")
            if op.op == "head_cut" and not op.n:
                problems.append(f"{name} op#{i} head_cut: no n")
        ops = [o.op for o in spec.ops]
        for first, laters, why in _MUST_PRECEDE:
            if first not in ops:
                continue
            i = ops.index(first)
            late = [o for o in laters if o in ops and ops.index(o) < i]
            if late:
                problems.append(f"{name}: {late} come before {first} — {why}")
    groups = [s.group for s in plan.inputs]
    if any(groups) and not all(groups):
        problems.append("some inputs are grouped and some are not — group all or none")
    return problems


def diff_plans(base: PrepPlan, other: PrepPlan) -> list[str]:
    """What the agent changed. Printed so a reviewer reads the delta, not the plan."""
    out: list[str] = []
    if base.comparison_axis != other.comparison_axis:
        out.append(f"axis {base.comparison_axis} → {other.comparison_axis}")
    b = {Path(s.path).name: s for s in base.inputs}
    for s in other.inputs:
        k = Path(s.path).name
        if k not in b:
            out.append(f"{k}: added")
            continue
        o = b[k]
        if s.text_column != o.text_column:
            out.append(f"{k}: text column {o.text_column} → {s.text_column}")
        if s.weight_column != o.weight_column:
            out.append(f"{k}: weight column {o.weight_column} → {s.weight_column}")
        if s.snapshot != o.snapshot:
            out.append(f"{k}: snapshot {o.snapshot} → {s.snapshot}")
        ops_b = [x.op for x in o.ops]
        ops_s = [x.op for x in s.ops]
        if ops_b != ops_s:
            out.append(f"{k}: ops {ops_b} → {ops_s}")
    for k in b:
        if not any(Path(s.path).name == k for s in other.inputs):
            out.append(f"{k}: DROPPED from the plan")
    return out
