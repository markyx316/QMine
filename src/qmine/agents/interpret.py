"""Agent-written interpretation that cannot ship an unverified number.

This is the loop that makes `InterpreterAgent` safe enough to put in a
deliverable. Three properties, each chosen against a specific failure:

**It fails closed.** If the agent cannot produce prose whose every number is in
the fact sheet, the section ships WITHOUT agent prose rather than with unchecked
prose. A deliverable missing an explanation is recoverable; one carrying a
confident fabricated number is not.

**Its retry carries external feedback.** A rejected attempt is re-asked with the
exact offending numbers quoted back. Re-asking without them would be intrinsic
self-correction, which is the configuration shown not to improve reasoning and
sometimes to degrade it (Huang et al., ICLR 2024). The fact sheet and the
rejection list are both external ground truth.

**It records provenance.** Every accepted interpretation carries the role, the
model, and the number of attempts, so a reader can tell templated text from
authored text, and so a run can be audited for how much of its prose was written
by an agent.

What this does NOT do, deliberately: it never lets the agent change a number,
choose a parameter, or veto a result. Selection stays with the measured metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .verify import check_numbers, fact_sheet


@dataclass
class Interpreted:
    """One verified interpretation, or a recorded failure to produce one."""

    ok: bool
    reading: str = ""
    caveats: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    attempts: int = 0
    author: str = ""
    reason: str = ""

    def as_markdown(self, label: str = "解读") -> str:
        """Render for a report, always disclosing that an agent wrote it."""
        if not self.ok or not self.reading.strip():
            return ""
        out = [f"> **{label}** (由 `{self.author}` 撰写, 所有数字已对照产物核验): "
               f"{self.reading.strip()}"]
        for c in self.caveats:
            if str(c).strip():
                out.append(f"> - ⚠️ {str(c).strip()}")
        if self.unavailable:
            out.append("> - 以下数字本次未产出, 因此未在解读中给出: "
                       + ", ".join(f"`{u}`" for u in self.unavailable))
        return "\n".join(out) + "\n"


def interpret(
    deps: Any,
    question: str,
    facts: dict[str, Any],
    *,
    context: str = "",
    language: str = "zh",
    max_attempts: int = 3,
    suffix: str = "",
) -> Interpreted:
    """Ask for an interpretation and refuse to return an unverified one."""
    from .roles import InterpreterAgent

    sheet = fact_sheet(facts)
    agent = InterpreterAgent(deps.agent_ctx(), suffix=suffix)
    rejected = ""
    last = ""
    for attempt in range(1, max_attempts + 1):
        try:
            out = agent.run(question=question, facts=sheet, context=context,
                            language=language, rejected=rejected)
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            continue
        reading = (out.reading or "").strip()
        if not reading:
            last = "empty reading"
            continue
        # Check the caveats too — a fabricated number is no safer for being in a
        # bullet than in the paragraph above it.
        blob = reading + "\n" + "\n".join(str(c) for c in (out.caveats or []))
        res = check_numbers(blob, facts)
        if res.ok:
            deps.emit(f"  interpretation accepted on attempt {attempt} "
                      f"({len(res.supported)} numeric claims verified)")
            return Interpreted(
                ok=True, reading=reading,
                caveats=[str(c) for c in (out.caveats or [])],
                unavailable=[str(u) for u in (out.unavailable or [])],
                attempts=attempt, author=f"{agent.role}@{getattr(agent, 'model', 'routed')}",
            )
        last = res.message()
        rejected = "\n".join(f"- {c}" for c in res.unsupported[:8])
        deps.emit(f"  ⚠ interpretation attempt {attempt} rejected — {last[:160]}")

    deps.emit(f"  interpretation abandoned after {max_attempts} attempts; "
              "the section ships without agent prose")
    return Interpreted(ok=False, attempts=max_attempts, reason=last)
