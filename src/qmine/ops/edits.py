"""Letting an agent change a delivered report, without letting it ship a mistake.

Every other agent in this pipeline describes. This one is allowed to **act**: it
reads the finished deliverables together with every warning the run accumulated,
and edits the reports where it finds a real defect. That is a genuine grant of
authority, and it is safe only because of the shape of the operation it is given.

**It cannot rewrite. It can only replace an anchor it can prove is there.**

    Edit(file, anchor, replacement, reason, artifact_key, check)

The anchor must occur **exactly once** in the file. Zero occurrences means the
agent is editing something it misread; two means it cannot know which one it
meant. Both are refusals, not repairs. This is the same discipline the project
already applies to its own scripted edits — `assert old in s` before
`str.replace` — for the same reason: a silent no-op cost a full debugging cycle.

**The citation defines the fact sheet.** An edit names the artifact key its
correction comes from, that key must resolve, and every number in the replacement
must be present **in that key's own subtree**. Checking against the whole run
would let a real number belonging to a different quantity through, which is the
documented blind spot of `agents.verify` — its docstring says to keep fact sheets
small, and this is how: the citation is not decoration, it is the pool.

**A number may not vanish without a replacement.** If the anchor carried 29 and
the replacement carries no number at all, the edit deletes evidence rather than
correcting it. Corrections are welcome; quiet removals are not.

**Language.** A run configured `zh` had English shipped into half its
deliverables, and the fix for it introduced fresh English of its own. So a
replacement is checked against the report language before it lands.

**Nothing is destroyed.** The pre-edit file is written beside the new one. Every
applied edit and every REFUSED edit is recorded with its reason — a refusal is a
finding about the auditor, and hiding it would make the audit report a list of
successes rather than a record of what happened.

**What it may not touch, at all.** Artifacts (`.json`, `.npy`, `.csv`,
`.parquet`), code, configuration, and the executable notebook. Reports are
*presentation*: a wrong number in prose is a document that disagrees with the
data, and fixing it changes no measurement. Editing an artifact would change the
measurement itself, and no agent gets that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Only rendered prose. An artifact is a measurement; a report is a description
#: of one, and only the description is in scope.
EDITABLE_SUFFIXES = (".md",)

#: A ceiling on how much one audit may change. Not a budget — a blast radius. An
#: auditor proposing forty edits to a report has stopped correcting defects and
#: started rewriting, and that is a different operation with different risks.
MAX_EDITS = 12

_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")
_HAN = re.compile(r"[一-鿿]")


@dataclass
class ProposedEdit:
    file: str
    anchor: str
    replacement: str
    reason: str = ""
    artifact_key: str = ""
    severity: str = "warn"
    check: str = ""

    def as_record(self) -> dict[str, Any]:
        return {"file": self.file, "anchor": self.anchor[:200],
                "replacement": self.replacement[:200], "reason": self.reason[:400],
                "artifact_key": self.artifact_key, "severity": self.severity,
                "check": self.check}


@dataclass
class EditOutcome:
    applied: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "n_applied": len(self.applied), "n_refused": len(self.refused),
            "files_changed": self.files_changed,
            "applied": self.applied, "refused": self.refused,
            "note": ("Every edit is an anchored replacement whose numbers were checked "
                     "against the artifact it cites. A refusal is recorded, not hidden — "
                     "it is a finding about the auditor."),
        }


def numbers_in(text: str) -> set[str]:
    return {m.group(1).replace(",", "") for m in _NUM.finditer(text or "")}


def _pool_from(value: Any, depth: int = 0) -> set[str]:
    """Every number reachable under the cited key, as normalised strings."""
    out: set[str] = set()
    if depth > 8:
        return out
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        v = float(value)
        for cand in (v, v * 100.0, round(v, 4), round(v * 100.0, 1), round(v * 100.0)):
            s = f"{cand:.10g}"
            out.add(s)
            if s.endswith(".0"):
                out.add(s[:-2])
        return out
    if isinstance(value, str):
        out |= numbers_in(value)
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            out |= numbers_in(str(k)) | _pool_from(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        for v in value:
            out |= _pool_from(v, depth + 1)
    return out


def _sourced(num: str, pool: set[str]) -> bool:
    if num in pool:
        return True
    try:
        v = float(num)
    except ValueError:
        return False
    for p in pool:
        try:
            if abs(float(p) - v) < 1e-9 or (v and abs(float(p) - v) / abs(v) < 1e-4):
                return True
        except ValueError:
            continue
    return False


def validate_edit(
    edit: ProposedEdit,
    texts: dict[str, str],
    artifacts: dict[str, Any],
    *,
    language: str = "zh",
) -> tuple[bool, str]:
    """Decide whether one edit may land. Returns (ok, reason_if_not).

    Ordered cheapest-and-most-decisive first, so the recorded reason names the
    rule that actually bit rather than whichever check happened to run.
    """
    from .checks import evaluate

    name = Path(edit.file).name
    if name not in texts:
        return False, f"{name} is not one of this run's editable deliverables"
    if Path(name).suffix not in EDITABLE_SUFFIXES:
        return False, f"{Path(name).suffix} files are artifacts, not descriptions of them"
    if not edit.anchor.strip():
        return False, "no anchor — an edit must name the exact text it replaces"
    if edit.anchor == edit.replacement:
        return False, "replacement is identical to the anchor"

    n = texts[name].count(edit.anchor)
    if n == 0:
        return False, ("the anchor does not appear in the file — the auditor is "
                       "correcting text that is not there")
    if n > 1:
        return False, f"the anchor appears {n} times; an edit must be unambiguous"

    found, cited = _resolve(edit.artifact_key, artifacts)
    if not found:
        return False, (f"cites {edit.artifact_key!r}, which is not an artifact of this "
                       "run — an unsourced correction is just a rewrite")

    pool = _pool_from(cited)
    unsourced = sorted(x for x in numbers_in(edit.replacement) if not _sourced(x, pool))
    if unsourced:
        return False, (f"{', '.join(unsourced[:5])} in the replacement "
                       f"{'is' if len(unsourced) == 1 else 'are'} not in "
                       f"`{edit.artifact_key}` — every number an edit introduces must "
                       "come from the artifact it cites")

    lost = numbers_in(edit.anchor) - numbers_in(edit.replacement)
    if lost and not numbers_in(edit.replacement):
        return False, (f"the replacement drops {', '.join(sorted(lost)[:4])} and adds no "
                       "number — that removes evidence rather than correcting it")

    if language == "zh" and edit.replacement.strip() and not _HAN.search(edit.replacement):
        if _NUM.sub("", edit.replacement).strip(" .,;:%()[]{}|`*-—·\n\t"):
            return False, ("the replacement carries no Chinese on a run configured "
                           "`zh` — this is how English got into half the deliverables")

    if edit.check:
        # AN EDIT'S CHECK IS THE OPPOSITE OF AN OBSERVATION'S, AND THIS WAS BACKWARDS.
        #
        # An OBSERVATION asserts what *should* hold; the assertion FAILING is what
        # confirms a defect. An EDIT asserts what the artifacts *do* say — the
        # ground truth the document is being aligned to — so the assertion HOLDING
        # is what makes the correction well-founded.
        #
        # This refused an edit whose check evaluated true, i.e. it rejected every
        # correctly-sourced correction. Measured on live40: the auditor wrote
        # `adversarial_validation.estimated_accuracy == 0.82` — true — to fix a
        # report claiming adversarial accuracy was HIGHER than cross-validation
        # when 0.82 < 0.8625. The fix was refused for being right, and the wrong
        # claim shipped.
        res = evaluate(edit.check, artifacts)
        if res.verdict == "confirmed":
            return False, (f"its own check `{edit.check}` is FALSE against the artifacts — "
                           "an edit's check must state what the artifacts DO say, so a "
                           "failing one means the correction is not sourced")
    if not edit.reason.strip():
        return False, "no reason given; an undocumented edit cannot be reviewed"
    return True, ""


def _resolve(key: str, artifacts: dict[str, Any]) -> tuple[bool, Any]:
    from ..agents.observe import resolve_key

    return resolve_key(key, artifacts)


def apply_edits(
    gen_dir: Path | str,
    edits: list[ProposedEdit],
    artifacts: dict[str, Any],
    *,
    language: str = "zh",
    max_edits: int = MAX_EDITS,
    dry_run: bool = False,
) -> EditOutcome:
    """Validate every edit, apply the ones that pass, and record all of them.

    Applied against an in-memory copy first and written once per file, so a file
    is never left half-edited by a later refusal — and so two edits to one file
    are validated against the SAME original text, not against each other's
    output.
    """
    gen = Path(gen_dir)
    texts = {p.name: p.read_text(encoding="utf-8")
             for p in gen.glob("*.md") if p.is_file()}
    working = dict(texts)
    out = EditOutcome()

    for edit in edits:
        if len(out.applied) >= max_edits:
            out.refused.append({**edit.as_record(),
                                "why": f"the cap of {max_edits} edits per audit was reached — "
                                       "beyond this it is a rewrite, not a correction"})
            continue
        # Validated against the ORIGINAL text: an anchor made unique or made
        # ambiguous by an earlier edit in the same batch would make the outcome
        # depend on ordering, and the auditor never saw the intermediate file.
        ok, why = validate_edit(edit, texts, artifacts, language=language)
        if not ok:
            out.refused.append({**edit.as_record(), "why": why})
            continue
        name = Path(edit.file).name
        if working[name].count(edit.anchor) != 1:
            out.refused.append({**edit.as_record(),
                                "why": "an earlier edit in this batch changed the anchor"})
            continue
        working[name] = working[name].replace(edit.anchor, edit.replacement, 1)
        out.applied.append({**edit.as_record(), "applied": True})

    if not dry_run:
        for name, text in working.items():
            if text == texts[name]:
                continue
            # Keep the pre-audit file. Every change stays reversible and diffable,
            # which is the difference between an audited edit and a silent one.
            (gen / f"{Path(name).stem}.pre_audit.md").write_text(texts[name], encoding="utf-8")
            (gen / name).write_text(text, encoding="utf-8")
            out.files_changed.append(name)
    return out
