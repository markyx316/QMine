"""What a model driving this server may and may not set in motion.

`qmine chat` could enforce this in its own loop, because it owned the loop. Over
MCP it does not: the harness runs the agent, and a tool it can see is a tool it
can call. So the boundary has to hold HERE, on the server side, where it holds
no matter which harness is in front of it.

THREE TIERS.

* **read** — measures and reads. Runs freely. Asking permission to list a
  directory teaches people to approve without reading, and the one prompt that
  matters then gets the same reflex.
* **write** — produces files, spends no model money, and is reversible by
  re-running. Allowed by default, but only INSIDE permitted roots: a tool that
  can be pointed at an arbitrary path is a tool that can overwrite a delivered
  study.
* **spend** — makes paid model calls for hours. **Refused by default.** It
  returns the exact command instead, so the assistant does the thinking and a
  person presses go.

`QMINE_MCP_ALLOW_SPEND` chooses between three postures, and it is read from the
environment that launched the server because that is the one place a person can
set it that the model cannot reach:

| value | posture |
|---|---|
| unset / `0` | refuse; hand back the command for a person to run |
| `ask` | a run may start, but only after a preflight passes AND the caller echoes the run id |
| `1` | a run may start once the preflight passes — for scripted, unattended use |

WHAT `ask` IS AND IS NOT. Inside one tool call this server cannot tell "the
person asked for this" from "the model decided to". Every argument it sees was
written by the model, so the `confirm` echo is a STUMBLE GUARD — it stops a run
being started as an opening move, and it proves the preflight was in context —
and it is **not consent**. Real per-run consent comes from the harness in front:
`make chat` installs a `PreToolUse` hook that returns `ask`, and dsh then holds
the call at its own approval dialog until a human clicks. That dialog is not
model context and cannot be answered by anything the model emits.

So the two layers answer different questions. This one answers *may this
deployment spend at all, and is the run even viable* — the preflight runs on
BOTH `ask` and `1`, because a click should never be able to start a doomed run.
The harness answers *may this particular run start, now*.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Tier = Literal["read", "write", "spend"]


def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Authority:
    #: May this deployment spend at all? False is the default and the refusal.
    allow_spend: bool = False
    #: `ask` mode: each start must echo its run id. A stumble guard, not consent
    #: — see the module docstring. Harmless to leave on; the model can satisfy it.
    spend_confirm: bool = False
    allow_write: bool = True
    #: Roots a `write` tool may write into. Anything else is refused by path,
    #: before the tool runs.
    write_roots: tuple[Path, ...] = ()

    @classmethod
    def from_env(cls, *, run_root: str = "runs") -> "Authority":
        roots = [Path.cwd().resolve(), Path(run_root).resolve()]
        extra = os.environ.get("QMINE_MCP_WRITE_ROOTS", "")
        roots += [Path(p).expanduser().resolve() for p in extra.split(os.pathsep) if p.strip()]
        raw = (os.environ.get("QMINE_MCP_ALLOW_SPEND") or "").strip().lower()
        ask = raw == "ask"
        return cls(allow_spend=ask or _flag("QMINE_MCP_ALLOW_SPEND", False),
                   spend_confirm=ask,
                   allow_write=_flag("QMINE_MCP_ALLOW_WRITE", True),
                   write_roots=tuple(dict.fromkeys(roots)))

    # ------------------------------------------------------------------ checks
    def refusal(self, tier: Tier, tool: str, command: list[str] | None = None,
                why: str = "") -> dict[str, Any] | None:
        """None when the call may proceed; otherwise the payload to return instead."""
        if tier == "read":
            return None
        if tier == "write" and self.allow_write:
            return None
        if tier == "spend" and self.allow_spend:
            return None
        cmd = " ".join(shlex.quote(c) for c in (command or []))
        if tier == "spend":
            return {
                "status": "not_run_needs_a_person",
                "why": why or ("A mining run makes paid model calls and takes hours. This "
                               "server does not start one on a model's say-so."),
                "run_this_yourself": cmd,
                "to_allow_it_here_instead": (
                    "restart the harness with QMINE_MCP_ALLOW_SPEND=ask in its environment "
                    "(`make chat` does this, and installs the approval gate that holds each "
                    "run until a person clicks) — a deliberate act outside the conversation, "
                    "not something that can be granted inside it"),
                "what_you_can_do_now": [
                    "qmine_estimate_cost — what it would cost, spends nothing",
                    "qmine_plan_corpus — the preparation plan, spends nothing",
                ],
            }
        return {
            "status": "not_run_writes_are_disabled",
            "why": why or "This tool writes files and QMINE_MCP_ALLOW_WRITE is off.",
            "run_this_yourself": cmd,
        }

    def check_path(self, p: str | Path, *, what: str = "output") -> Path:
        """Resolve and refuse anything outside the permitted roots.

        `..` in a path is not a hypothetical: an output directory is the one
        argument a model fills in most freely, and the study it could overwrite
        took hours to produce.
        """
        target = Path(p).expanduser().resolve()
        for root in self.write_roots:
            if target == root or root in target.parents:
                return target
        raise PermissionError(
            f"refusing to use {target} as an {what}: it is outside the permitted roots "
            f"{[str(r) for r in self.write_roots]}. Set QMINE_MCP_WRITE_ROOTS to widen "
            "them deliberately.")

    def describe(self) -> dict[str, Any]:
        return {
            "read": "always allowed",
            "write": "allowed" if self.allow_write else "disabled (QMINE_MCP_ALLOW_WRITE=0)",
            "spend": (
                ("allowed, and every start is preflighted first; each one must echo its "
                 "run id, and the harness in front may also hold it for a human click"
                 if self.spend_confirm else
                 "allowed outright once the preflight passes (QMINE_MCP_ALLOW_SPEND=1)")
                if self.allow_spend else
                "refused — a run is proposed as a command for a person to run"),
            "write_roots": [str(r) for r in self.write_roots],
        }
