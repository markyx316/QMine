"""The guardrail that makes agent-written prose safe to ship.

`ReporterAgent`'s prompt has always said "every number must come from here". Nothing
enforced it, and nothing ever called the agent — so the reports are pure templating
and the guarantee was never tested. This module is that enforcement.

**The contract.** An authoring agent is handed a *fact sheet*: a flat mapping of
name → value, every entry read from a run artifact. It may write any prose it
likes, but **every number it writes must be one of those values**. A number that is
not is a hallucination, and the call is rejected and retried rather than shipped.

**Why this shape and not "have an agent review the draft".** Two results decide it:

* LLMs do not reliably self-correct without *external* feedback, and performance
  can degrade when they try (Huang et al., "Large Language Models Cannot
  Self-Correct Reasoning Yet", ICLR 2024). So the check must compare against
  artifacts, never against the model's own judgement.
* LLM judges are dominated by **style** bias (reported around 0.76–0.92, far
  exceeding position bias), with self-preference worth roughly 10–25%. Asking a
  model "is this report good?" measures how well it is written. Asking "does
  49,999 appear in the fact sheet?" is arithmetic, and arithmetic does not have a
  style preference.

So: agents supply *reasoning and language*; this module supplies *truth*, and it
does so mechanically.

**What this does NOT catch, stated plainly.** It is a value-level check, not a
semantic one. It rejects a number that is not in the sheet, and — since a percent
sign asserts a share — it rejects a share matched against a count. It does NOT
catch a number that is in the sheet but attached to the wrong quantity: prose
saying "the families fragment at 2.479" passes, because 2.479 is a real value,
even though it belongs to the leaves. Catching that needs to know which fact the
sentence refers to, which is a judgement, and routing it back through a model
would reintroduce exactly the bias this module exists to avoid. The mitigation is
structural instead: keep fact sheets SMALL and scoped to the one question being
asked, so there are few wrong facts available to grab.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Numbers that carry no factual claim: section numbers, list ordinals, years,
#: and the small counts that appear in ordinary phrasing ("两条路线", "第 3 阶段").
#: Kept deliberately tiny — the temptation is to widen it until nothing fails,
#: which is how a guardrail becomes decoration.
_STRUCTURAL = re.compile(
    r"(?:^|[^\d])(?:"
    r"第\s*\d+\s*(?:阶段|节|章|轮|条|列|步)"      # 第 3 阶段
    r"|[§#]\s*\d+(?:\.\d+)*"                       # §2.2
    r"|\d+(?:\.\d+)*\s*[.、)]\s"                   # "1. " / "2.3) " list markers
    r"|20\d\d\s*年"                                # a year
    r")"
)

#: A number as it appears in prose: 1,234 / 0.8221 / 36 / 9.9% / 97.6％
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|％)?")


@dataclass
class Claim:
    """One number an agent wrote, with enough context to explain a rejection."""

    raw: str
    value: float
    is_pct: bool
    context: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.raw}{'%' if self.is_pct else ''} in “…{self.context}…”"


@dataclass
class CheckResult:
    supported: list[Claim] = field(default_factory=list)
    unsupported: list[Claim] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def message(self) -> str:
        if self.ok:
            return f"all {len(self.supported)} numeric claims trace to the fact sheet"
        bad = "; ".join(str(c) for c in self.unsupported[:6])
        return (f"{len(self.unsupported)} of "
                f"{len(self.supported) + len(self.unsupported)} numeric claims are not "
                f"in the fact sheet: {bad}")


def _flatten(facts: Any, prefix: str = "") -> dict[str, float]:
    """Every number reachable in a fact sheet, keyed by dotted path."""
    out: dict[str, float] = {}
    if isinstance(facts, dict):
        for k, v in facts.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(facts, (list, tuple)):
        for i, v in enumerate(facts):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    elif isinstance(facts, bool):
        pass                                     # True is not the number 1 here
    elif isinstance(facts, (int, float)):
        out[prefix] = float(facts)
    return out


def extract_claims(text: str) -> list[Claim]:
    """Pull every numeric claim out of prose, skipping structural numbers."""
    claims: list[Claim] = []
    blocked: list[tuple[int, int]] = [m.span() for m in _STRUCTURAL.finditer(text)]
    for m in _NUMBER.finditer(text):
        if any(s <= m.start() < e for s, e in blocked):
            continue
        raw = m.group(1)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:                        # pragma: no cover - regex guards
            continue
        lo, hi = max(0, m.start() - 34), min(len(text), m.end() + 34)
        claims.append(Claim(raw=raw, value=value, is_pct=bool(m.group(2)),
                            context=text[lo:hi].replace("\n", " ")))
    return claims


def _matches(raw: str, value: float, is_pct: bool, pool: Iterable[float]) -> bool:
    """Is this written number one of the known values, allowing only for rounding?

    A report says 9.9% for 0.09862, and 49,999 for 49999.0. Both must pass. But
    "50,000" must NOT pass for 49,999 and "0.80" must NOT pass for 0.8221 — and
    the first version of this function accepted both, because it took the
    precision from the parsed float (`0.80` → `0.8` → one decimal → ±0.05) and
    gave every number a flat 5e-4 relative slack (±25 at a scale of 50,000).

    Precision therefore comes from the DIGITS THE AUTHOR WROTE. An author who
    writes a rounder number than the artifact supports is making a claim the
    artifact does not license, which is exactly what this exists to catch.
    """
    dp = _written_dp(raw)
    # A bare integer COUNT must be exact — 36 leaves is not 35. A bare integer
    # PERCENTAGE is a rounding of a measured share, so "10%" legitimately stands
    # for 9.862%. Treating both the same way either rejects honest rounding or
    # lets a miscount through, and only one of those is recoverable by a reader.
    tol = 0.5 * 10 ** -dp if (dp or is_pct) else 0.0
    if is_pct:
        # A percent sign ASSERTS that this is a share, so it is checked only
        # against share-shaped facts. Without this, "12%" was accepted because
        # the sheet happened to contain `families = 12` — the check matched a
        # share against a count. That is a misattribution, not an invention, and
        # it is the one class of error this module can still be tightened against
        # cheaply.
        cands = [value / 100.0]
    else:
        cands = [value, value * 100.0]            # a share may be written either way
    for fact in pool:
        for c in cands:
            if fact == c:
                return True
            # The tolerance travels with the candidate's own scale: a claim
            # written as a percent is checked at percent precision against the
            # fraction, and vice versa.
            scale = c / value if value else 1.0
            if abs(fact - c) <= max(tol * abs(scale), 1e-9):
                return True
    return False


def _written_dp(raw: str) -> int:
    """Decimal places the AUTHOR wrote — `"0.80"` is two, not one."""
    return len(raw.split(".")[1]) if "." in raw else 0


def check_numbers(text: str, facts: Any) -> CheckResult:
    """Every number in `text` must appear in `facts`.

    This is the whole guarantee. It is deliberately mechanical: it cannot be
    talked out of a rejection, it has no opinion about how well the prose reads,
    and it fails closed — an author that cites a number the artifacts do not
    contain does not ship.
    """
    pool = set(_flatten(facts).values())
    # Percent-encoded twins, so 0.09862 in the sheet supports "9.9%" in the prose.
    pool |= {v * 100.0 for v in list(pool) if abs(v) <= 1.0}
    res = CheckResult()
    for c in extract_claims(text):
        ok = _matches(c.raw.replace(",", ""), c.value, c.is_pct, pool)
        (res.supported if ok else res.unsupported).append(c)
    return res


def fact_sheet(pairs: dict[str, Any]) -> str:
    """Render a fact sheet for a prompt: one `name = value` per line.

    Given to the author as the ONLY admissible source of numbers, and to
    `check_numbers` as the pool. Both sides reading the same object is the point
    — a fact sheet the checker does not share is a guarantee about nothing.
    """
    lines: list[str] = []
    for k, v in _flatten(pairs).items():
        lines.append(f"{k} = {v:g}" if isinstance(v, float) else f"{k} = {v}")
    return "\n".join(sorted(lines))
