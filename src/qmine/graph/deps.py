"""The runtime container every node reaches through.

State carries pointers; this carries the machinery.  Keeping them separate is
what lets the graph checkpoint a kilobyte after a node that just produced a
150 MB matrix — the matrix went to the artifact store, its ``ArtifactRef`` went
to state, and :class:`Deps` holds the loader that can bring it back.

The in-process cache matters more than it looks.  Phases 3 through 9 all read
the same embedding matrix; without a cache, "state holds only pointers" would
mean re-reading 150 MB from disk a dozen times per run.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..agents.base import AgentContext
from ..artifacts import ArtifactRef, ArtifactStore
from ..config import QMineConfig
from ..llm.registry import ModelRegistry
from ..memory.context import BlindnessFirewall
from ..memory.store import QMineMemory
from ..records import DecisionRecord, GateResult, LessonRecord

log = logging.getLogger("qmine.graph")


@dataclass
class Deps:
    """Everything a node needs that is too large or too stateful to sit in state."""

    cfg: QMineConfig
    store: ArtifactStore
    registry: ModelRegistry
    memory: QMineMemory
    firewall: BlindnessFirewall = field(default_factory=BlindnessFirewall)
    run_id: str = ""
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _decision_seq: int = 0
    _prescription_seq: int = 0
    #: `Deps` is shared by every node, and the graph runs the top-down and
    #: bottom-up branches CONCURRENTLY. Three things here are read-modify-write
    #: and were safe only while the graph was a strict chain:
    #:
    #: * the id counters — `+= 1` is not atomic, and p2a and p7_audit both raise
    #:   prescriptions, so two branches could be handed the same id;
    #: * `_cache` — `if key in cache: ... else: rebuild()` lets two branches both
    #:   miss and both rebuild, which for `recover()` means paying twice for the
    #:   same artifact;
    #: * `firewall` — armed in p7 from the taxonomy.
    #:
    #: Re-entrant because `recover()` calls `emit()` and `has()` while holding it.
    _lock: Any = field(default_factory=threading.RLock, repr=False)
    #: Set when `template_masks(trusted=True)` had to fall back to unvalidated
    #: mined groups. Read by p1 so the run carries a gate rather than a log line.
    _trusted_fallback: dict[str, Any] = field(default_factory=dict, repr=False)
    #: Progress lines surfaced to the CLI as the run proceeds.
    on_event: Any = None

    # -- artifact access ----------------------------------------------------
    def load(self, name: str) -> Any:
        """Load an artifact by logical name, memoised for the process lifetime."""
        with self._lock:
            if name in self._cache:
                return self._cache[name]
        value = self.store.load(name)          # I/O outside the lock
        with self._lock:
            return self._cache.setdefault(name, value)

    def cache_put(self, name: str, value: Any) -> None:
        with self._lock:
            self._cache[name] = value

    def recover(self, key: str, artifact: str, rebuild: Any = None, default: Any = None) -> Any:
        """Get a value from process memory, or rebuild it from the artifact store.

        This exists because of a real and quiet failure. Several phases used to
        read intermediate values straight out of ``_cache`` — which is process
        memory. That works within a single run and returns ``None`` on a
        *resumed* run, because the new process has an empty cache. The result
        was not a crash: Phase 10 simply omitted the top-down label column from
        the delivered table, and Phase 7 named zero clusters. Both look like
        success in the logs.

        Checkpointing makes a run resumable only if every phase can rebuild what
        it needs from disk. Anything read through this helper can; anything read
        through a bare ``_cache.get`` cannot.
        """
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self.store.has(artifact):
                raw = self.store.load(artifact)
                value = rebuild(raw) if rebuild else raw
                self._cache[key] = value
                self.emit(f"  recovered {key!r} from artifact {artifact!r} (resumed run)")
                return value
        return default

    def has(self, name: str) -> bool:
        return name in self._cache or self.store.has(name)

    @property
    def df(self) -> pd.DataFrame:
        return self.load("corpus")

    @property
    def queries(self) -> list[str]:
        return self.df[self.cfg.data.text_column].astype(str).tolist()

    def embedding(self, name: str = "emb_hybrid") -> np.ndarray:
        return self.load(name)

    # -- derived state ------------------------------------------------------
    #
    # These were originally plain in-process caches, which was a silent bug: a
    # resumed run starts with an empty cache, so Phase 8 recomputed its metric
    # deltas over zero template families and reported nothing. Anything a later
    # phase depends on must be *rebuildable from artifacts*, not merely
    # remembered — otherwise "resume" restores the graph but not the reasoning.

    def template_masks(self, *, trusted: bool = True) -> dict[str, np.ndarray]:
        """Phrasing-family membership, rebuilt from the stored groups if needed.

        ``trusted=True`` returns only the families allowed to judge a
        representation; ``False`` returns every family, for coverage and display.
        """
        key = "template_masks" if trusted else "template_masks_all"
        if key in self._cache:
            return self._cache[key]
        if not self.has("template_groups"):
            return {}
        from ..ops.templates import group_masks
        from ..records import TemplateGroup

        payload = self.load("template_groups")
        groups = [TemplateGroup.model_validate(g) for g in payload.get("groups", [])]
        masks = group_masks(groups, self.df, text_col=self.cfg.data.text_column,
                            trusted_only=trusted)
        if trusted and not masks:
            # "FALL BACK LOUDLY" — IT WAS SILENT. No log, no gate, no artifact.
            #
            # This is the portability case, and it is the one that matters: with no
            # domain profile there are no seeds, so nothing is trusted, and K is
            # then located by AMI against UNVALIDATED mined groups. `generic.yaml`
            # ships 0 seeds, so that is the default path for any new corpus. Mined
            # groups are measurably worse references — on live40, `suffix:是什么`
            # spans 7 top-down intents at 42% purity, against 87.8% median for the
            # seeded ones — and nothing said so.
            masks = group_masks(groups, self.df, text_col=self.cfg.data.text_column)
            self._trusted_fallback = {
                "fell_back": True, "n_groups_used": len(masks),
                "why": ("no seeded phrasing group survived, so the K locator's "
                        "reference is mined groups that passed no cohesion test"),
            }
            self.emit("  ⚠ 没有任何**种子**措辞群存活 — K 的定位参照改用"
                      f"{len(masks)} 个**未经验证的**挖掘群。这会直接影响定下来的 K。")
        self._cache[key] = masks
        return masks

    def taxonomy(self) -> Any:
        """The taxonomy, rebuilt from the artifact if this process did not build it."""
        if "taxonomy_obj" in self._cache:
            return self._cache["taxonomy_obj"]
        from ..records import Taxonomy

        for name in ("taxonomy_v2", "taxonomy"):
            if self.has(name):
                payload = self.load(name)
                tax = Taxonomy.model_validate(payload["taxonomy"])
                self._cache["taxonomy_obj"] = tax
                return tax
        return None

    def leaf_family_final(self) -> np.ndarray:
        """Post-governance families, falling back to pre-governance."""
        if "leaf_family_final" in self._cache:
            return self._cache["leaf_family_final"]
        return self.load("leaf_family_final" if self.has("leaf_family_final") else "leaf_family")

    def leaf_labels_final(self) -> np.ndarray:
        """Post-governance leaf assignments; identical to pre unless a split ran."""
        if "leaf_labels_final" in self._cache:
            return self._cache["leaf_labels_final"]
        return self.load("leaf_labels_final" if self.has("leaf_labels_final") else "leaf_labels")

    def leaf_centroids_final(self) -> np.ndarray:
        if "leaf_centroids_final" in self._cache:
            return self._cache["leaf_centroids_final"]
        return self.load("leaf_centroids_final" if self.has("leaf_centroids_final") else "leaf_centroids")

    # -- agent context ------------------------------------------------------
    def agent_ctx(self) -> AgentContext:
        return AgentContext(
            cfg=self.cfg,
            registry=self.registry,
            store=self.store,
            memory=self.memory,
            firewall=self.firewall,
            run_id=self.run_id,
        )

    # -- record helpers ------------------------------------------------------
    def decision(
        self,
        phase: str,
        question: str,
        choice: str,
        rationale: str,
        *,
        decided_by: str = "metric",
        evidence: dict[str, Any] | None = None,
        rejected: list[dict[str, Any]] | None = None,
        decisive_metrics: list[str] | None = None,
    ) -> DecisionRecord:
        """Record a choice — and, just as importantly, what it beat.

        The rejected list is not decoration.  Phase 11's failure-history section
        is a projection of these records, and a report that cannot show what
        lost is asking to be trusted rather than showing its work.
        """
        # The id must be READ inside the lock too: incrementing atomically and
        # then reading the field is still a race — a sibling branch can bump it
        # between the two, and both decisions ship with the same id.
        with self._lock:
            self._decision_seq += 1
            _seq = self._decision_seq
        rec = DecisionRecord(
            id=f"D{_seq:03d}",
            phase=phase,
            question=question,
            choice=choice,
            rationale=rationale,
            decided_by=decided_by,
            evidence=evidence or {},
            rejected=rejected or [],
            decisive_metrics=decisive_metrics or [],
        )
        try:
            self.memory.remember_decision(rec)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not persist decision: %s", exc)
        return rec

    def lesson(
        self, situation: str, action: str, outcome: str, lesson: str, *, phase: str = "", severity: str = "warning"
    ) -> LessonRecord:
        rec = LessonRecord(
            id=f"L{int(time.time() * 1000) % 10_000_000}",
            situation=situation,
            action=action,
            outcome=outcome,
            lesson=lesson,
            phase=phase,
            domain=self.cfg.domain.key,
            severity=severity,  # type: ignore[arg-type]
        )
        try:
            self.memory.remember_lesson(rec)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not persist lesson: %s", exc)
        return rec

    def next_prescription_id(self) -> str:
        # p2a and p7_audit both raise prescriptions and now sit in CONCURRENT
        # branches. `+= 1` is a read-modify-write, so without this two of them
        # can carry the same id — and a prescription is matched to its executed
        # change by id, so a collision silently merges two different remedies.
        with self._lock:
            self._prescription_seq += 1
            return f"P{self._prescription_seq:03d}"

    # -- gates ---------------------------------------------------------------
    def gate(
        self,
        name: str,
        phase: str,
        *,
        passed: bool,
        observed: dict[str, Any],
        threshold: dict[str, Any],
        message: str = "",
        remediation: str = "",
        warn_only: bool = False,
    ) -> GateResult:
        blocking = name in self.cfg.gates.blocking and not warn_only
        status = "passed" if passed else ("warned" if warn_only or not blocking else "failed")
        g = GateResult(
            name=name,
            phase=phase,
            status=status,  # type: ignore[arg-type]
            blocking=blocking,
            observed=observed,
            threshold=threshold,
            message=message,
            remediation=remediation,
        )
        if not passed:
            self.lesson(
                situation=f"{phase} gate {name}: observed {observed}",
                action="ran the phase with the current configuration",
                outcome=f"gate {status}",
                lesson=remediation or message or f"{name} did not meet {threshold}",
                phase=phase,
                severity="critical" if blocking else "warning",
            )
        self.emit(f"gate {name}: {status.upper()} — {message or observed}")
        return g

    def recall_block(self, situation: str) -> str:
        """Remembered lessons and settled decisions, rendered for a prompt."""
        try:
            return self.memory.context_block(situation)
        except Exception as exc:  # noqa: BLE001
            log.debug("recall failed: %s", exc)
            return ""

    # -- progress -----------------------------------------------------------
    def emit(self, msg: str) -> None:
        log.info(msg)
        if self.on_event:
            try:
                self.on_event(msg)
            except Exception:  # noqa: BLE001
                pass


def artifact_summary(refs: dict[str, ArtifactRef], limit: int = 40) -> str:
    """A compact listing of what exists, for prompts and CLI output."""
    lines = []
    for name in sorted(refs)[:limit]:
        r = refs[name]
        shape = f" {tuple(r.shape)}" if r.shape else ""
        lines.append(f"- {name} ({r.kind}{shape}, {r.bytes / 1024:.0f} KB) — {r.summary or Path(r.path).name}")
    return "\n".join(lines)
