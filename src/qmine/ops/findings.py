"""A run-level ledger of open findings, so a real one cannot be forgotten.

This codebase has already run the experiment where it can be. A critic agent
identified the kappa defect *before* the run that shipped it; the finding was
written to an artifact; nothing read it; the defect shipped. `agents/observe.py`
fixed half of that by turning an observation into a gate on its own phase. This
module fixes the other half.

**The gap it closes.** `p6_observer` reported a real, checkable defect on live39.
It warned, the run finished, `run_summary.json` recorded four words about it, and
the next run would have started with no memory of it at all. A finding that
survives exactly one run is a finding the next run rediscovers or loses.

So findings live beside `cache/` at the **run root**, not in a generation. A new
generation inherits them, exactly as it inherits paid LLM calls, and for the same
reason: the work of establishing them was real and re-doing it is waste.

**What makes an entry close.** Not a human ticking a box, and not the absence of
a re-report — either would let a defect fall out of the ledger while still being
present. An entry carrying a `check` closes when **its own expression evaluates
true against the current artifacts**: the assertion that failed now holds, which
is the definition of fixed. Entries without a check can only be closed by an
explicit, recorded human judgement, because nothing else can settle them.

**One defect is one entry.** Findings are keyed by a fingerprint over phase and
claim, so the same defect seen in gen01 and gen03 is a single row with a history
rather than two rows that each look new.

The ledger decides nothing and changes nothing. It is a list that refuses to
shorten on its own.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FINDINGS_FILE = "findings.json"

OPEN, FIXED, REFUTED, WAIVED = "open", "fixed", "refuted", "waived"


def fingerprint(phase: str, claim: str, artifact_key: str = "") -> str:
    """A stable id for one defect.

    Normalised hard on purpose: the same observer re-run on the same defect
    rewords its sentence, and a fingerprint that moves with the wording produces
    a ledger that grows by one row per run and is therefore never read.
    """
    norm = re.sub(r"[\s\d.,;:!?()\[\]{}'\"`—–-]+", "", f"{phase}|{claim}|{artifact_key}".lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


@dataclass
class Finding:
    id: str
    phase: str
    severity: str
    claim: str
    artifact_key: str = ""
    evidence: str = ""
    check: str = ""
    #: "confirmed" once a check has failed. A confirmed finding is a measurement,
    #: not an agent's opinion, and is the only kind allowed to block a run.
    verdict: str = "unverifiable"
    status: str = OPEN
    first_seen: str = ""
    last_seen: str = ""
    times_seen: int = 1
    resolution: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.status == OPEN and self.severity == "blocking" and self.verdict == "confirmed"

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


class FindingLedger:
    """Load, merge, re-check, and persist. Never deletes an entry."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.entries: dict[str, Finding] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return                      # a corrupt ledger must not stop a run
        for row in raw.get("findings", []) or []:
            try:
                known = {f for f in Finding.__dataclass_fields__}
                self.entries[row["id"]] = Finding(**{k: v for k, v in row.items() if k in known})
            except (KeyError, TypeError):
                continue

    # -- writing ----------------------------------------------------------
    def record(self, *, phase: str, severity: str, claim: str, artifact_key: str = "",
               evidence: str = "", check: str = "", verdict: str = "unverifiable",
               seen_at: str = "") -> Finding:
        """Add a finding, or note that an existing one is still here."""
        fid = fingerprint(phase, claim, artifact_key)
        cur = self.entries.get(fid)
        if cur is None:
            cur = Finding(id=fid, phase=phase, severity=severity, claim=claim,
                          artifact_key=artifact_key, evidence=evidence, check=check,
                          verdict=verdict, first_seen=seen_at, last_seen=seen_at)
            self.entries[fid] = cur
            return cur
        cur.times_seen += 1
        cur.last_seen = seen_at or cur.last_seen
        cur.evidence = evidence or cur.evidence
        # A re-sighting REOPENS. Otherwise a finding waived once is invisible
        # forever, including on the run where it finally breaks something.
        if cur.status in (FIXED, REFUTED):
            cur.history.append({"at": seen_at, "was": cur.status, "why": "seen again"})
            cur.status = OPEN
        # Never downgrade: a check that could not be evaluated this time does not
        # unmake a confirmation that was measured last time.
        if verdict == "confirmed" or not cur.check:
            cur.verdict, cur.check = verdict or cur.verdict, check or cur.check
        return cur

    def recheck(self, artifacts: dict[str, Any], *, at: str = "") -> list[Finding]:
        """Re-evaluate every open, checkable finding against current artifacts.

        This is the only automatic path out of the ledger, and it is deliberately
        the narrowest one: the assertion that failed now holds. A finding closed
        this way was closed by a measurement, not by anyone's opinion that it had
        been dealt with.
        """
        from .checks import evaluate

        closed: list[Finding] = []
        for f in self.entries.values():
            if f.status != OPEN or not f.check:
                continue
            res = evaluate(f.check, artifacts)
            if res.verdict == "refuted":            # the assertion holds again
                f.status = FIXED
                f.resolution = f"check now passes: {f.check}"
                f.history.append({"at": at, "was": OPEN, "why": "check passes"})
                closed.append(f)
            elif res.verdict == "confirmed":
                f.verdict = "confirmed"
        return closed

    def waive(self, fid: str, why: str, *, at: str = "") -> bool:
        """Record a human judgement that a finding should not be acted on.

        Requires a reason, and the reason is kept. A waiver with no argument is
        indistinguishable from forgetting, which is what this file exists to stop.
        """
        f = self.entries.get(fid)
        if f is None or not why.strip():
            return False
        f.history.append({"at": at, "was": f.status, "why": f"waived: {why}"})
        f.status, f.resolution = WAIVED, why
        return True

    # -- reading ----------------------------------------------------------
    @property
    def open_findings(self) -> list[Finding]:
        return sorted(
            (f for f in self.entries.values() if f.status == OPEN),
            key=lambda f: ({"blocking": 0, "warn": 1, "note": 2}.get(f.severity, 3),
                           0 if f.verdict == "confirmed" else 1, f.phase),
        )

    @property
    def confirmed_open(self) -> list[Finding]:
        return [f for f in self.open_findings if f.verdict == "confirmed"]

    def save(self, *, run_id: str = "", generation: str = "") -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id, "generation": generation,
            "n_open": len(self.open_findings),
            "n_confirmed_open": len(self.confirmed_open),
            "note": ("Findings survive a new generation, like the LLM cache. An entry "
                     "with a `check` closes ONLY when that expression evaluates true "
                     "again; one without a check needs a recorded human waiver. "
                     "Nothing here is ever deleted."),
            "findings": [f.as_record() for f in sorted(
                self.entries.values(), key=lambda f: (f.status != OPEN, f.phase))],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path


def all_json_artifacts(gen_dir: Path | str, *, max_bytes: int = 4_000_000) -> dict[str, Any]:
    """Every JSON artifact in a generation, keyed by name — the recheck's universe.

    Read from disk rather than from `deps` so the recheck sees the run as it was
    actually delivered, including artifacts written by a phase that has already
    returned. `max_bytes` skips anything too large to be a check's subject.
    """
    out: dict[str, Any] = {}
    for p in sorted(Path(gen_dir).glob("*.json")):
        try:
            if p.stat().st_size > max_bytes:
                continue
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def recheck_run(deps: Any) -> "FindingLedger":
    """Re-evaluate the whole ledger against this generation's artifacts.

    Runs at report time, when every artifact exists. This is the step that lets a
    finding close itself: a defect fixed in code stops reproducing, its assertion
    holds again, and the entry moves to `fixed` without anyone remembering to
    tick it off. It is also what makes an inherited finding honest — a generation
    that did not fix it will see it still open rather than starting clean.
    """
    led = FindingLedger(Path(deps.store.root) / FINDINGS_FILE)
    at = f"{getattr(deps, 'run_id', '')}/{Path(deps.store.gen_dir).name}"
    closed = led.recheck(all_json_artifacts(deps.store.gen_dir), at=at)
    for f in closed:
        deps.emit(f"  ✔ finding closed — its check passes again: {f.claim[:100]}")
    still = led.confirmed_open
    if still:
        deps.emit(f"  ⚠ {len(still)} CONFIRMED finding(s) still open — carried to the "
                  "next generation and printed in the panel report:")
        for f in still[:5]:
            deps.emit(f"      [{f.phase}] {f.claim[:110]}")
    led.save(run_id=getattr(deps, "run_id", ""), generation=at)
    return led
