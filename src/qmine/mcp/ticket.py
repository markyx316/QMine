"""Proof that the approval gate was consulted for THIS run, before it starts.

WHY THIS EXISTS. The `PreToolUse` hook is the only channel that can carry real
consent, and on 2026-09-21 it silently did not: the command referenced an
unquoted `${CLAUDE_PROJECT_DIR}`, this checkout's path contains spaces, the
shell split it mid-path, `qmine` never ran — and a hook that emits no decision
does not block anything. dsh falls through to `next()`, which is **allow**. The
paid run started with no dialog and nobody was told the gate was missing.

That is the failure mode to design against: not a gate that says no, a gate that
is not there. So in `ask` mode the server requires a TICKET that only the hook
writes, and refuses when there is none. A missing or broken hook now fails
CLOSED and says so, instead of quietly becoming no gate at all.

WHAT A TICKET IS AND IS NOT. It proves the gate ran for this run id, just now.
It does NOT prove a human approved — the hook runs before the answer, so no
artifact of the hook can carry the verdict. The approval itself lives in the
harness, in a dialog this process never sees. Tickets are one-shot and
short-lived so a stale one cannot authorise a later run.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

#: A ticket older than this is not evidence about the call being made now.
TTL_SECONDS = 900


def _dir(run_root: str) -> Path:
    """A path the SANDBOXED hook can actually write to.

    `run_root` is deliberately ignored. The hook does not run as this process
    does: dsh executes it through the sandboxed shell, so writing into the
    checkout's `runs/` raised `[Errno 1] Operation not permitted: 'runs'`, the
    hook fell into its own error path, and every run was denied. Measured
    2026-09-21 — the gate failed CLOSED, which is the right direction, but it
    failed on everything.

    The temp directory is per-user and 0700. That bounds who can drop a ticket
    here to this account; it does NOT make a ticket unforgeable by something
    already running as this user, including the model's own shell. The ticket's
    job is to make an ABSENT gate fail closed, not to be a credential — the
    consent itself lives in the harness's approval dialog, which no process here
    can answer.
    """
    d = Path(tempfile.gettempdir()) / f"qmine-approval-{os.getuid()}"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


def issue(run_root: str, run_id: str, decision: str, reason: str) -> Path:
    """Written by the hook, once, immediately before it answers the harness."""
    p = _dir(run_root) / f"{run_id}.json"
    p.write_text(json.dumps({"run_id": run_id, "decision": decision,
                             "reason": reason, "at": time.time(),
                             "pid": os.getpid()}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def redeem(run_root: str, run_id: str) -> dict[str, Any] | None:
    """Consume the ticket for `run_id`. None when there is none, or it is stale.

    Consuming rather than reading is deliberate: one ticket authorises one
    launch, so a second `qmine_start_run` on the same id has to go through the
    gate again.
    """
    p = _dir(run_root) / f"{run_id}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent or unreadable is simply "no ticket"
        return None
    finally:
        p.unlink(missing_ok=True)
    if not isinstance(data, dict) or data.get("run_id") != run_id:
        return None
    if time.time() - float(data.get("at") or 0) > TTL_SECONDS:
        return None
    return data
