"""Two planners: one that reasons, one that cannot.

`heuristic_plan` is built from the measurements alone. It is what runs offline,
what the tests exercise, and what the agent's proposal is compared against — a
model that agrees with it has told us nothing new, and a model that disagrees
has to have a reason that survives the executor's checks.

It is also the floor. If the agent is unavailable, refuses, or returns something
that does not validate, preparation still happens and still says what it did.
"""

from __future__ import annotations

import re
from pathlib import Path

from .inspect import InputProfile
from .plan import InputSpec, PrepOp, PrepPlan

_DATEY = re.compile(r"^(?:19|20)\d{2}[-/]?\d{2}[-/]?\d{2}$|^\d{8}$")

#: A prefix carried by at least this share of an input's rows is the product's
#: own framing, not something users typed. Set where it is because a genuinely
#: popular user phrasing does not reach a fifth of a corpus, while a wrapper
#: reaches 86% of one measured export.
WRAPPER_SHARE = 0.20
#: Above this repeat rate, the export is almost certainly one row per
#: (query, day) rather than one row per query, and every weight is a day's
#: weight rather than the string's.
AGGREGATE_ABOVE = 0.05


def heuristic_plan(profiles: list[InputProfile], *,
                   text_column: str | None = None,
                   axis: str | None = None) -> PrepPlan:
    """A plan derived from the measurements, with no model involved."""
    inputs: list[InputSpec] = []
    concerns: list[str] = []
    for p in profiles:
        text = text_column or (p.text_candidates[0] if p.text_candidates else None)
        if text is None:
            # Not `continue`: a dropped input leaves the run comparing fewer
            # snapshots than the person asked for, and the concern scrolls past.
            raise ValueError(
                f"{Path(p.path).name}: no column looks like free-text queries. "
                f"Columns present: {[c.name for c in p.columns]}. Name it with "
                "--text-column, or leave this file out on purpose.")
        weight = p.weight_candidates[0] if p.weight_candidates else None
        ops = [PrepOp(op="trim", why="前后空白不是查询的一部分")]
        if p.duplicate_rate > AGGREGATE_ABOVE:
            ops.append(PrepOp(
                op="aggregate_by_query",
                why=f"{100 * p.duplicate_rate:.1f}% 的行重复了前面出现过的串——"
                    "这份导出更像是按天一行而不是按串一行，先合并再做任何按权重的判断"))
        wrapper = next((pre for pre, frac in p.repeated_prefixes if frac >= WRAPPER_SHARE),
                       None)
        if wrapper:
            ops.append(PrepOp(
                op="strip_prefix", pattern="^" + re.escape(wrapper),
                why=f"{Path(p.path).name} 里有 "
                    f"{100 * dict(p.repeated_prefixes)[wrapper]:.1f}% 的行以「{wrapper}」开头，"
                    "这是产品自己的包装而不是用户打的字；剥掉，不删行"))
            ops.append(PrepOp(op="merge_collisions", why="剥包装后重合的串，权重相加"))
        ops.append(PrepOp(op="drop_empty", name="content_free",
                          why="没有任何文字、数字或汉字的串不是查询"))
        inputs.append(InputSpec(
            path=p.path, snapshot=p.suggested_tag,
            display=p.suggested_tag, text_column=text, weight_column=weight,
            ops=ops,
            keep_columns=[c.name for c in p.columns
                          if c.name not in (text, weight) and 1 < c.n_unique <= 64],
            notes="；".join(p.notes)))
        if weight is None:
            concerns.append(f"{Path(p.path).name}: 没有可用的权重列，这个快照按均匀权重处理，"
                            "它的流量占比会等于行占比")

    tags = [i.snapshot for i in inputs]
    if len(set(tags)) != len(tags):
        # Distinct tags are a hard requirement downstream, so fix it here rather
        # than letting the executor refuse a plan nobody can repair by hand.
        seen: dict[str, int] = {}
        for i in inputs:
            if tags.count(i.snapshot) > 1:
                seen[i.snapshot] = seen.get(i.snapshot, 0) + 1
                i.snapshot = f"{i.snapshot}#{seen[i.snapshot]}"
                i.display = f"{Path(i.path).stem}"
        concerns.append("两个输入解析出了同一个快照名，已按文件名区分——"
                        "请给它们更可读的名字（--label）")

    derived = axis or _axis(inputs)
    if not axis:
        # THE AXIS IS AN ASSUMPTION, AND IT IS STATED AS ONE. It is tempting to
        # settle it from cross-file overlap, and that was measured: two random
        # 1w samples of the SAME surface a year apart share 0.02% of their
        # strings, two head exports of that same surface share 63.3%, and a
        # different surface shares 0.00%. Overlap separates head from tail, not
        # time from interface — so a rule built on it would be confidently wrong
        # half the time. It goes in front of a person instead.
        concerns.append(
            "对比轴是**推出来的**，不是你声明的：快照名"
            + ("看起来是不同日期，所以按**时间**处理。" if derived == "time"
               else "看不出时间先后，所以按**分层**处理。")
            + "这决定了正文把差异写成「变化」还是「差别」，弄反了整段告诫都会反过来。"
              "如果不对，用 --axis 说明。（跨文件重合度**不能**定这件事：实测同一界面的两份"
              "随机抽样重合 0.02%，同一界面的两份头部导出重合 63.3%，不同界面重合 0.00%。）")
    return PrepPlan(
        inputs=inputs, comparison_axis=derived,  # type: ignore[arg-type]
        rationale=_rationale(profiles, derived),
        # An inferred axis is never "high": the one thing this plan cannot check
        # is the one that inverts the prose.
        confidence="high" if (axis and len(concerns) == 0) else "medium",
        concerns=concerns)


def _axis(inputs: list[InputSpec]) -> str:
    """Time only when every tag is a distinct date. Otherwise stratum.

    The default matters: the computation is axis-agnostic but the prose is not,
    and one caveat inverts. Calling a sampling difference a change over time is
    the more damaging of the two errors, so the ambiguous case goes to stratum.
    """
    tags = [i.snapshot for i in inputs]
    return "time" if len(tags) > 1 and all(_DATEY.match(t) for t in tags) \
        and len(set(tags)) == len(tags) else "stratum"


def _rationale(profiles: list[InputProfile], axis: str) -> str:
    heads = [Path(p.path).name for p in profiles if p.weight_sorted_desc]
    rand = [Path(p.path).name for p in profiles if not p.weight_sorted_desc]
    bits = [f"{len(profiles)} 份导出合并成一份语料，一次运行导出一套体系，所以它们之间可以直接比较。"]
    if heads and rand:
        bits.append(f"其中 {len(heads)} 份权重降序且高度集中（像头部导出），"
                    f"{len(rand)} 份不是（像随机或去重抽样）——"
                    "这两者的差别既是界面也是流量层，单独一对无法归因到其中一个。")
    bits.append("按**时间**读：快照有先后，差异可以读成变化。" if axis == "time"
                else "按**分层**读：快照之间没有先后，差异不要读成变化。")
    return "".join(bits)
