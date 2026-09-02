"""Phase node implementations, one module per stage of the playbook."""
from __future__ import annotations

from typing import Any


def observe(deps: Any, phase: str, artifacts: dict[str, Any],
            *, decisions: Any = None, gates: Any = None) -> dict[str, Any]:
    """Run the phase observer if it is enabled, and hand back gates to register.

    One helper rather than the guard repeated at every call site, because the
    guard has three parts — the flag, smoke_mode, and `as_state_gates()` — and the
    third is the one that silently does nothing when it is forgotten. `deps.gate`
    returns a GateResult and registers nothing, so an observer whose gate is not
    merged into the node's return reaches the log and no operator: that failure
    already happened once, with 10 gates recorded where 15 were created.

    Returns `{}` when observation is off, so a call site can always spread it.
    """
    if not getattr(deps.cfg, "observe_phases", False) or deps.cfg.smoke_mode:
        return {}
    from ...agents.observe import observe_phase

    return observe_phase(deps, phase, artifacts,
                         decisions=decisions, gates=gates).as_state_gates()
