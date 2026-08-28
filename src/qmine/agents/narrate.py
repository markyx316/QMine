"""The final report, written by an agent — and the four checks that let it ship.

Everything else this pipeline delivers is assembled by Python. That is why those
documents are correct and why they read like a changelog: each section is
generated independently, so the reading order is the order the code ran, and
every defect ever fixed added a caveat paragraph where it happened rather than
where a reader needs it. A template cannot write a through-line, because a
template does not know what the run turned out to be about.

So this one is not templated. The prose is the agent's, start to finish. What
code retains is the evidence and the verdict on whether the prose may ship.

**Two passes, because one is the documented failure mode.** A single call over a
whole run's evidence is the configuration long-form grounded generation is worst
at: the context is large, the content is dense, and the writer drifts off its
sources as it goes. Pass 1 sees a MAP of the run — bundle titles and what may not
be omitted, no numbers — and writes its own outline. Pass 2 writes one section at
a time against evidence scoped to that section, carrying the outline it wrote and
the tail of the section before it. Structure is global and agent-authored;
grounding is local and mechanical.

**Four checks, and the second is the one that is new here.**

1. *Precision* — `check_numbers`: every number in the section is in that
   section's sheet. Existing machinery, unchanged.
2. *Coverage* — every must-cover item is addressed in the assembled document.
   `check_numbers` is silent about omission: a section that reports a clean
   result and drops the gate that passed with slack, or reports the chosen K
   without the three others that stood up equally, passes it perfectly and
   misleads completely. Precision-only grounding is what lets a writer report
   selectively, and this is the counterweight.
3. *Figures* — every image reference resolves to a file that exists.
4. *Language* — no English prose in a Chinese deliverable.

**It fails closed, visibly.** A section that cannot pass ships as a marked block
naming what was rejected, not as silence and not as unchecked prose. A hole a
reader can see is recoverable; a confident fabricated number is not. The marker
is also what the pre-delivery auditor reads downstream.

**It cannot decide anything.** No parameter, no threshold, no selection. It
explains what the measurements already settled. Its only authority is over
sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..report.i18n import looks_like_english_prose
from ..report.narrative_brief import (
    Bundle,
    citable_numbers,
    MustCover,
    build_catalogue,
    digest,
    must_cover,
    numbers_in,
    sheet,
)
from .verify import check_numbers

MAX_SECTIONS = 14
MAX_ATTEMPTS = 3
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass
class SectionResult:
    id: str
    heading: str
    markdown: str = ""
    covered: list[str] = field(default_factory=list)
    ok: bool = False
    attempts: int = 0
    rejections: list[str] = field(default_factory=list)


def _reject(section_md: str, bundles: list[Bundle], allowed_figs: set[str],
            language: str, shown: str = "") -> list[str]:
    """Every reason this draft may not ship. Empty means it may."""
    problems: list[str] = []

    facts = {b.id: b.facts for b in bundles}
    # The pool must be what the sheet SHOWS, not only what `_flatten` reaches.
    # `sheet` renders dotted paths, so a dict keyed by id displays numbers the
    # value-only pool rejects — live42 refused `32, 40, 42, 43, 44, 45`, the leaf
    # ids the section had just been shown, three times, and the section shipped
    # as a hole. See `citable_numbers`.
    # EVERYTHING THE WRITER WAS SHOWN IS CITABLE; anything else is entrapment.
    #
    # `shown` carries the channels that are not the fact sheet: the must-cover
    # items, the figure captions, and the previous section. A must-cover arrives
    # under "必须原样包含这句话", so a number inside one ORDERS the writer to
    # state something this check would then refuse — a hole no retry can close,
    # the same shape as the negative-number defect in `verify._NUMBER`.
    #
    # The OUTLINE is deliberately excluded. `_plan` verifies evidence ids and
    # section count and nothing numeric, so pooling the thesis would let one
    # unchecked planning call launder a number into every section below it.
    # The rejection notice is excluded for the same reason: it PRINTS the
    # offending numbers, and pooling it would make any rejected number citable
    # on the retry — the check would pass on its second attempt, always.
    pool = {**facts, "_sheet": citable_numbers(facts)}
    if shown:
        pool["_shown"] = numbers_in(shown)
    res = check_numbers(section_md, pool)
    if not res.ok:
        bad = ", ".join(c.raw for c in res.unsupported[:8])
        problems.append(
            f"这些数字不在事实表里, 不能出现在正文中: {bad}。"
            "请改用事实表中的数字, 或者不写这个数字。"
            "注意: 自己算出来的比值/百分比也算 — 事实表里没有就不能写。")

    for ref in IMAGE_RE.findall(section_md):
        name = ref.split("/")[-1].strip()
        if name not in allowed_figs:
            problems.append(
                f"图 `{name}` 不在本节可用的图里。可用的是: "
                + (", ".join(sorted(allowed_figs)) or "本节没有可用的图"))

    if language == "zh":
        english = [ln for ln in section_md.splitlines()
                   if looks_like_english_prose(ln)]
        if english:
            problems.append(
                "以下整句是英文, 交付语言是中文, 请改写: "
                + " / ".join(ln.strip()[:80] for ln in english[:3]))

    if len(section_md.strip()) < 120:
        problems.append("这一节太短, 没有把论点讲完。")

    return problems


def _write_section(deps: Any, sec: Any, bundles: list[Bundle],
                   musts: list[MustCover], outline_text: str, previous: str,
                   language: str) -> SectionResult:
    from .roles import StoryWriterAgent

    out = SectionResult(id=str(getattr(sec, "id", "")),
                        heading=str(getattr(sec, "heading", "")))
    allowed_figs = {f for b in bundles for f, _ in b.figures}
    facts_text = "\n\n".join(
        f"### {b.id} — {b.title}\n{sheet(b.facts)}" for b in bundles)
    figs_text = "\n".join(
        f"- `{f}` — {cap}" for b in bundles for f, cap in b.figures)
    musts_text = "\n".join(
        (f"- `{m.id}` — {m.what}\n  **必须原样包含这句话** (不要改写, 不要改标点):\n"
         f"  {m.verbatim}") if m.verbatim else f"- `{m.id}` — {m.what}"
        for m in musts)
    sec_text = (f"{out.heading}\n\n本节要论证: "
                f"{getattr(sec, 'intent', '') or out.heading}")

    agent = StoryWriterAgent(deps.agent_ctx())
    rejected = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        out.attempts = attempt
        try:
            draft = agent.run(outline=outline_text, section=sec_text,
                              facts=facts_text, figures=figs_text,
                              musts=musts_text, previous=previous[-1200:],
                              rejected=rejected, language=language)
        except Exception as exc:                              # noqa: BLE001
            out.rejections.append(f"{type(exc).__name__}: {exc}")
            rejected = (f"上一次调用直接失败了 ({type(exc).__name__})。"
                        "请只返回一个 JSON 对象, 字段为 markdown 与 covered。")
            continue
        md = (getattr(draft, "markdown", "") or "").strip()
        if not md:
            # AN EMPTY RETURN MUST CARRY FEEDBACK LIKE ANY OTHER REJECTION.
            #
            # This appended a rejection and `continue`d WITHOUT setting
            # `rejected`, so the retry re-sent a byte-identical prompt and got a
            # byte-identical answer. live42's three empty sections each burned all
            # three attempts that way — one call's outcome, paid for three times.
            # `interpret.py` states the rule this broke: a retry must carry
            # EXTERNAL feedback, because re-asking unchanged is intrinsic
            # self-correction, which is the configuration shown not to help.
            #
            # `reporter` runs on a model in plain-JSON mode, so a whole section of
            # Chinese prose has to survive as an escaped JSON string; the field
            # arriving empty or missing is a parse-shaped failure, and saying so
            # is the only thing that gives the retry a chance.
            out.rejections.append("空白正文 (markdown 字段为空或缺失)")
            rejected = (
                "你上一稿返回的 JSON 里 `markdown` 字段是空的或者根本不存在, "
                "因此这一节没有任何正文。请重新返回一个 JSON 对象, 其中 "
                "`markdown` 必须是这一节完整的正文字符串 —— 不要把正文写在 JSON "
                "之外, 不要只返回 `covered`。正文里的换行请正常转义, 若担心过长, "
                "宁可写得紧凑一些, 也不要返回空字段。")
            deps.emit(f"    ⚠ §{out.id} 第 {attempt} 次返回空正文 — 已告知模型重试")
            continue
        problems = _reject(md, bundles, allowed_figs, language,
                           shown="\n".join((musts_text, figs_text, previous)))
        if not problems:
            out.markdown = md
            out.covered = [str(c) for c in (getattr(draft, "covered", []) or [])]
            out.ok = True
            deps.emit(f"    §{out.id} 通过 (第 {attempt} 次尝试)")
            return out
        out.rejections.extend(problems)
        rejected = "\n".join(f"- {p}" for p in problems)
        deps.emit(f"    ⚠ §{out.id} 第 {attempt} 次退回 — {problems[0][:100]}")

    deps.emit(f"    §{out.id} 三次都没通过, 该节以「未通过校验」标记交付")
    return out


@dataclass
class _Section:
    id: str
    heading: str
    intent: str = ""
    evidence: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)


@dataclass
class _Outline:
    title: str = ""
    thesis: str = ""
    sections: list[_Section] = field(default_factory=list)


#: The order an investigation is explained in, which is not the order the phases
#: ran in. Only bundles this run actually produced are kept.
_SPINE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("corpus", "这批查询是什么", ("corpus",)),
    ("topdown", "自上而下: 先想清楚用户想干什么", ("topdown_taxonomy", "topdown_gold")),
    ("topdown_quality", "这套体系标得准不准", ("topdown_adversarial", "topdown_gold")),
    ("representation", "换个方向: 让数据自己说话, 先要一把尺子", ("representation",)),
    ("granularity", "切多少个家族, 以及还有哪些切法同样站得住", ("granularity",)),
    ("hierarchy", "两层树是怎么长出来的", ("hierarchy", "naming")),
    ("governance", "审计提出的处方, 最后真的执行了", ("governance", "risk")),
    ("panel", "两条路线放在同一把尺子上", ("panel", "decisions")),
    ("deployment", "交付了什么, 新查询怎么被打上标签", ("deployment", "samples")),
    ("quality", "哪些地方还不够好", ("gates", "findings")),
    ("meta", "这次运行本身", ("run_meta",)),
)


def _fallback_outline(catalogue: dict[str, Bundle]) -> _Outline:
    """A structure chosen by code, for when the planner cannot produce one.

    This is the ONE piece of the final report that a template may decide, and it
    decides only the running order — every sentence is still the agent's. It
    exists because delivering nothing when the planner misbehaves is worse than
    delivering an agent-written report in a sensible fixed order, and because the
    offline stand-in cannot produce a valid outline, which would otherwise leave
    the entire section-writing path unexercised by any test that runs the graph.

    A document built this way says so, in `_assemble`.
    """
    secs: list[_Section] = []
    for sid, heading, ev in _SPINE:
        have = [e for e in ev if e in catalogue]
        if have:
            secs.append(_Section(id=sid, heading=heading, intent=heading,
                                 evidence=have,
                                 figures=[f for e in have
                                          for f, _ in catalogue[e].figures]))
    return _Outline(title="最终报告", thesis="", sections=secs)


def _plan(deps: Any, catalogue: dict[str, Bundle], musts: list[MustCover],
          language: str) -> Any:
    """Pass 1, retried against structural requirements rather than taste."""
    from .roles import StoryPlannerAgent

    agent = StoryPlannerAgent(deps.agent_ctx())
    dig = digest(catalogue, musts)
    rejected = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            outline = agent.run(digest=dig, language=language, rejected=rejected)
        except Exception as exc:                              # noqa: BLE001
            rejected = f"上一版调用失败: {type(exc).__name__}"
            continue
        secs = list(getattr(outline, "sections", []) or [])[:MAX_SECTIONS]
        problems: list[str] = []
        if not secs:
            problems.append("大纲里一节都没有。")
        unknown = sorted({e for s in secs for e in (getattr(s, "evidence", []) or [])
                          if e not in catalogue})
        if unknown:
            problems.append(
                f"这些 evidence id 不存在: {', '.join(unknown)}。只能用: "
                + ", ".join(sorted(catalogue)))
        # Placement of the must-cover items is NOT verified here. The planner is
        # asked to place them, but `_assign_musts` routes every one of them onto
        # a section mechanically afterwards, so an item the planner forgot still
        # reaches a writer. A check here would only be able to reject the outline
        # for something the pipeline can simply fix.
        if problems:
            rejected = "\n".join(f"- {p}" for p in problems)
            deps.emit(f"  ⚠ 大纲第 {attempt} 次退回 — {problems[0][:110]}")
            continue
        outline.sections = secs
        deps.emit(f"  大纲通过: {len(secs)} 节 (第 {attempt} 次尝试)")
        return outline, "agent"
    fb = _fallback_outline(catalogue)
    deps.emit(f"  ⚠ 大纲三次都没通过, 改用固定顺序的 {len(fb.sections)} 节骨架 "
              "(正文仍由 agent 撰写)")
    return (fb, "fallback") if fb.sections else (None, "none")


def _assign_musts(outline: Any, musts: list[MustCover],
                  catalogue: dict[str, Bundle]) -> dict[str, list[MustCover]]:
    """Route each must-cover to the section whose evidence can actually carry it.

    The planner is asked to place these, but placement is not left to it alone:
    an item assigned nowhere would simply be absent, and the only check that
    would catch it fires after the whole document is written. Routing on the
    evidence a section was given is mechanical and cannot be forgotten.
    """
    secs = list(getattr(outline, "sections", []) or [])
    by_sec: dict[str, list[MustCover]] = {str(s.id): [] for s in secs}
    if not secs:
        return by_sec

    def owner(m: MustCover) -> str:
        want = ("gates" if m.id.startswith(("gate:", "slack:"))
                else "findings" if m.id.startswith("finding:")
                else "granularity" if m.id == "k_tie_set"
                else "panel" if m.id in ("both_routes", "structurally_invisible")
                else "")
        for s in secs:
            if want and want in (getattr(s, "evidence", []) or []):
                return str(s.id)
        return str(secs[-1].id)

    for m in musts:
        by_sec[owner(m)].append(m)
    return by_sec


def _coverage(authored: str, musts: list[MustCover]) -> list[MustCover]:
    """Must-cover items whose anchors never appear in the AGENT'S OWN prose.

    Scope is the whole point. Checking the assembled document instead lets the
    machine-written parts satisfy the requirement: the provenance banner names
    the scripted reports, so `自下而上聚类最终报告.md` in the boilerplate marked
    the "both routes must be explained" item covered while no section had
    explained either route. A check a fixed string can satisfy is not a check, so
    only the sections the agent actually wrote and that actually passed are
    searched here.
    """
    missing: list[MustCover] = []
    for m in musts:
        # A SUPPLIED SENTENCE IS CHECKED EXACTLY, and that is the whole point of
        # it: `check_numbers` guarantees numbers, so an attribution — a noun —
        # slips through with every figure correctly sourced. live42 wrote the
        # wrong reference into a sentence whose numbers were all valid. Requiring
        # the pipeline's own wording verbatim is the only mechanical guarantee
        # available for a claim that is not a number.
        if m.verbatim and m.verbatim not in authored:
            missing.append(m)
            continue
        if not all(str(a) in authored for a in (m.anchors or [])):
            missing.append(m)
    return missing


def narrate(state: Any, deps: Any) -> dict[str, Any]:
    """Write the final report. Returns the markdown and a provenance record."""
    language = getattr(getattr(deps, "cfg", None), "report_language", "zh") or "zh"
    gen = Path(deps.store.gen_dir)

    catalogue = build_catalogue(state, deps)
    if not catalogue:
        return {"ran": False, "skipped": "本次运行没有可用的证据 bundle"}
    musts = must_cover(state, deps, catalogue)
    deps.emit(f"  证据 bundle {len(catalogue)} 个, 必写项 {len(musts)} 条")

    outline, outline_source = _plan(deps, catalogue, musts, language)
    if outline is None:
        return {"ran": False, "skipped": "没有可用的大纲, 也没有可用的骨架"}

    by_sec = _assign_musts(outline, musts, catalogue)
    outline_text = "\n".join(
        [f"标题: {getattr(outline, 'title', '')}",
         f"主线: {getattr(outline, 'thesis', '')}", ""]
        + [f"{i}. {s.heading} — {getattr(s, 'intent', '')}"
           for i, s in enumerate(getattr(outline, "sections", []), 1)])

    results: list[SectionResult] = []
    previous = ""
    for sec in getattr(outline, "sections", []):
        bundles = [catalogue[e] for e in (getattr(sec, "evidence", []) or [])
                   if e in catalogue]
        if not bundles:
            bundles = list(catalogue.values())[:3]
        r = _write_section(deps, sec, bundles, by_sec.get(str(sec.id), []),
                           outline_text, previous, language)
        results.append(r)
        if r.ok:
            previous = r.markdown

    document = _assemble(outline, results, state, deps, gen, outline_source)
    # Only what the agent wrote AND that passed — a section that failed closed
    # ships as a marked hole, and a hole covers nothing.
    authored = "\n\n".join(r.markdown for r in results if r.ok)
    missing = _coverage(authored, musts)
    if missing:
        document += _disclose_missing(missing)
        deps.emit(f"  ⚠ {len(missing)} 条必写项在全文中找不到, 已在文末披露")

    n_ok = sum(1 for r in results if r.ok)
    deps.emit(f"  最终报告: {n_ok}/{len(results)} 节通过校验, "
              f"必写项 {len(musts) - len(missing)}/{len(musts)} 覆盖")
    return {
        "ran": True,
        "markdown": document,
        "title": getattr(outline, "title", "") or "最终报告",
        "thesis": getattr(outline, "thesis", ""),
        "outline_source": outline_source,
        "n_sections": len(results),
        "n_sections_ok": n_ok,
        "n_musts": len(musts),
        "n_musts_missing": len(missing),
        "missing": [{"id": m.id, "what": m.what} for m in missing],
        "sections": [{"id": r.id, "heading": r.heading, "ok": r.ok,
                      "attempts": r.attempts, "covered": r.covered,
                      "rejections": r.rejections[:6]} for r in results],
        "bundles_used": sorted({e for s in getattr(outline, "sections", [])
                                for e in (getattr(s, "evidence", []) or [])}),
    }


def _assemble(outline: Any, results: list[SectionResult], state: Any, deps: Any,
              gen: Path, outline_source: str = "agent") -> str:
    """Stitch the agent's sections into one document, disclosing what it is."""
    title = getattr(outline, "title", "") or "最终报告"
    L = [f"# {title}", ""]
    thesis = getattr(outline, "thesis", "")
    if thesis:
        L += [f"> {thesis}", ""]
    L += [
        f"**运行**: `{state.get('run_id')}` / {gen.name}",
        "",
        "---",
        "",
        "> **这份文档是怎么来的。** 正文由一个 agent 撰写 —— 大纲、组织、每一句话都是它写的, "
        "没有任何模板。它能写什么则由机器决定: 每一节只拿到该节需要的那部分事实表, "
        "**正文里的每一个数字都必须出现在事实表中**, 逐值核对, 不通过就打回重写; "
        "本次运行提出过的告警、并列结果与未关闭问题构成一份**不得省略清单**, "
        "在全文范围内逐条核对是否真的写到了。它**不能**决定任何参数, 也不能改动任何 artifact —— "
        "所有取舍都已由脚本按实测指标定下, 它负责把这些讲清楚。",
        "",
        "> 更细的证据、完整的表格与失败史在 `自下而上聚类最终报告.md`、"
        "`自上而下类目体系最终报告.md` 与 `统一度量面板.md` 里, 那三份由脚本生成, "
        "本文与它们的数字同源。",
        "",
        "---",
        "",
    ]
    if outline_source == "fallback":
        L += ["> ⚠️ **本文的章节顺序是固定骨架, 不是 agent 排的** —— 规划这一步没有通过"
              "结构校验, 因此改用了预设的叙述顺序。每一节的正文仍然由 agent 撰写并逐值核对, "
              "但「先讲什么、后讲什么」这一层不是它决定的。", ""]
    for i, r in enumerate(results, 1):
        L += [f"## {i}. {r.heading}", ""]
        if r.ok:
            L += [r.markdown, ""]
        else:
            # SAY WHICH OF THE TWO THINGS WENT WRONG, AND SAY EACH ONE ONCE.
            #
            # One notice covered both outcomes and read "未通过校验" — did not
            # pass validation. For a blank return nothing was ever validated:
            # the writer produced no text at all, which is a different fact
            # about the run and points at a different remedy. live42 shipped
            # three sections whose whole explanation was "空白正文" repeated
            # three times, which tells a reader neither what happened nor that
            # it was the SAME failure three times rather than three findings.
            blank_only = all("空白正文" in p for p in r.rejections) and r.rejections
            L += [
                ("> ⚠️ **本节没有正文: 撰写这一节的 agent 三次都没有返回任何文字。** "
                 "这不是校验退回 —— 没有产出可供校验。这里刻意留空, 而不是补一段没有"
                 "来源的文字。")
                if blank_only else
                ("> ⚠️ **本节未通过校验, 因此没有正文。** 这是刻意保留的空缺: "
                 "与其交付一段没有核对过的文字, 不如让读者看见这里缺了什么。"),
                "",
            ]
            # De-duplicated in order: three identical rejections are one fact
            # about the run, not three.
            reasons: list[str] = []
            for p in r.rejections:
                if p not in reasons:
                    reasons.append(p)
            if not blank_only:
                L += ["> 被退回的原因:", *[f"> - {p}" for p in reasons[:4]], ""]
            L += [
                f"> 尝试次数: {r.attempts}。",
                "",
                "> 本节要讲的内容在脚本生成的报告里是完整的, 请查阅上面列出的那三份文档。",
                "",
            ]
    return "\n".join(L)


def _disclose_missing(missing: list[MustCover]) -> str:
    """Say what the narrator failed to cover, in the document itself.

    A coverage failure that is only logged is a coverage failure the reader never
    learns about — which is precisely the omission the check exists to catch,
    displaced by one step.
    """
    L = ["", "---", "", "## 附: 本文没有覆盖到的必写项", "",
         "下列内容是本次运行**规定不得省略**的, 但在全文中没有找到对应的表述。"
         "这一节由机器生成, 用来保证这些内容即使没被写进正文, 也不会从交付中消失。", ""]
    for m in missing:
        L.append(f"- `{m.id}` — {m.what}")
    return "\n".join(L) + "\n"
