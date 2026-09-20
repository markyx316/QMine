"""What may be done to an input, expressed as data an executor can check.

THE AGENT FILLS THIS IN; IT DOES NOT ACT. Every preparation step is one of a
closed list of named operations with bounded parameters, so what a model
proposes can be read, argued with, diffed between runs and refused — none of
which is true of a model that writes and runs cleaning code.

TWO RULES THE EXECUTOR ENFORCES AND THE PLAN CANNOT OVERRIDE:

1. **Nothing is deleted.** Every operation assigns a row a TIER. Rows whose tier
   is not `user` do not enter the mining, and they ship in a companion table
   with the tier that removed them. Cleaning nobody can see is cleaning nobody
   can check — and the product layer these exports carry is precisely what a
   reader wants to look at.
2. **A removal that is too large is downgraded, not obeyed.** An op that would
   tier away more than `max_drop_share` of a source is applied as a FLAG instead
   and recorded as refused. A regex that is one character too greedy otherwise
   removes a third of a snapshot silently, and the analysis that follows is
   about a corpus nobody chose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Every operation the executor knows how to perform. An `op` outside this list
#: fails validation, which is the point: a plan cannot introduce a step nobody
#: has read.
OpName = Literal[
    "trim",               # strip surrounding whitespace
    "drop_empty",         # no letter, digit or CJK character — never a query
    "drop_duplicates",    # keep the first occurrence within this input
    "length_filter",      # outside [min_len, max_len]
    "strip_prefix",       # remove a product's own wrapper; NEVER drops the row
    "flag_regex",         # tag matching rows; they still enter the mining
    "drop_regex",         # tier matching rows out of the mining
    "head_cut",           # keep the top-n rows by weight (a declared head sample)
    "aggregate_by_query",  # one row per string, weights summed (per-day exports)
    "merge_collisions",   # after stripping, sum the weights of now-identical rows
]


class PrepOp(BaseModel):
    """One preparation step."""

    op: OpName
    #: The tier a removed or flagged row is assigned. Reported verbatim, so it
    #: should read as a reason: `product_answer_option`, `system_template`.
    name: str = ""
    #: Python regex, for strip_prefix / flag_regex / drop_regex. Evaluated with
    #: `re`, never `Series.str.contains` — pandas 3 routes that to RE2, which
    #: rejects a `一-鿿` class and silently makes `\\w` ASCII-only.
    pattern: str = ""
    min_len: int | None = None
    max_len: int | None = None
    n: int | None = Field(default=None, description="head_cut: how many rows to keep.")
    #: Why this step exists, in one sentence. Required for anything that removes
    #: rows: a removal whose reason is not written down cannot be reviewed.
    why: str = ""


class InputSpec(BaseModel):
    """One file, and what it becomes."""

    path: str
    #: The snapshot tag. Must be unique across the plan — two inputs resolving to
    #: the same tag cannot be told apart afterwards.
    snapshot: str
    display: str = ""
    #: An optional grouping across snapshots: an interface, a product, a
    #: sampling method. Either every input has one or none does.
    group: str = ""
    text_column: str
    weight_column: str | None = None
    ops: list[PrepOp] = Field(default_factory=list)
    #: Columns to carry through to the corpus (legacy labels, product, source).
    keep_columns: list[str] = Field(default_factory=list)
    notes: str = ""


class PrepPlan(BaseModel):
    """The whole preparation, as one reviewable object."""

    inputs: list[InputSpec]
    comparison_axis: Literal["time", "stratum"] = "time"
    #: Why these snapshots are comparable and what differs between them. This is
    #: the sentence a reader needs and the one nothing else in the pipeline
    #: supplies.
    rationale: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    #: Anything the planner could not settle from the measurements. A concern
    #: stated is a concern a person can act on; a concern suppressed to look
    #: confident is the failure this field exists to prevent.
    concerns: list[str] = Field(default_factory=list)

    def tags(self) -> list[str]:
        return [i.snapshot for i in self.inputs]
