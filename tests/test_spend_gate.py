"""Starting a paid run: the preflight, the three postures, and the approval gate.

WRITTEN AFTER THE GATE SILENTLY WAS NOT THERE. On 2026-09-21 the `PreToolUse`
hook's command referenced an unquoted `${CLAUDE_PROJECT_DIR}`; this checkout's
path contains spaces, the shell split it mid-path, `qmine` never ran, and the
hook emitted no decision. A hook that emits no decision does not block anything
— dsh falls through to `next()`, which is ALLOW — so a paid run started with no
dialog and nothing anywhere said the gate was missing. Twice.

Both halves are pinned here: the decision the hook produces, and the server-side
refusal that makes a missing gate fail CLOSED instead of open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qmine.mcp import ticket
from qmine.mcp.authority import Authority
from qmine.mcp.server import QMineServer, pre_tool_use_hook

ROOT = Path(__file__).resolve().parents[1]
CORPUS = str(ROOT / "data" / "raw" / "k12_queries_50k.csv")


# ------------------------------------------------------------------ postures

def test_the_three_spend_postures_come_from_the_environment(monkeypatch):
    """`ask` is a THIRD state, not a synonym for on: it means a run may start
    only when an approval gate was consulted for it."""
    monkeypatch.delenv("QMINE_MCP_ALLOW_SPEND", raising=False)
    a = Authority.from_env()
    assert (a.allow_spend, a.spend_confirm) == (False, False)
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", "ask")
    a = Authority.from_env()
    assert (a.allow_spend, a.spend_confirm) == (True, True)
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", "1")
    a = Authority.from_env()
    assert (a.allow_spend, a.spend_confirm) == (True, False)
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", "0")
    assert Authority.from_env().allow_spend is False


def test_ask_mode_refuses_when_no_gate_was_consulted(tmp_path, monkeypatch):
    """THE ONE THIS FILE EXISTS FOR. A missing or broken hook must fail CLOSED.

    The model can satisfy `confirm` on its own — it is a stumble guard, not
    consent — so `confirm` alone must NOT be enough to start a run in `ask` mode.

    THE LAUNCHER IS STUBBED, AND THAT IS NOT COSMETIC. Without it, any mutation
    that disables the guard makes this test START A REAL PAID RUN: proving the
    mutant is caught by letting it spend the money the guard exists to protect.
    Measured — a mutation pass did exactly that on 2026-09-21. Asserting the
    launcher was never INVOKED is also the stronger claim.
    """
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", "ask")
    launched: list = []
    monkeypatch.setattr(QMineServer, "_launch",
                        lambda self, cmd, rid: launched.append(rid) or {"status": "LAUNCHED"})
    qm = QMineServer(str(tmp_path))
    out = qm.call("qmine_start_run", {"inputs": [CORPUS], "run_id": "gate-r1",
                                      "domain": "k12_zh", "fast": True,
                                      "confirm": "gate-r1"})
    assert not launched, "the run was launched despite no approval gate"
    assert out["status"] == "not_run_no_approval_gate", out


def test_a_ticket_is_one_shot(tmp_path):
    """One ticket authorises one launch: a second start on the same id has to go
    through the gate again, or an approved run id becomes a standing permit."""
    ticket.issue(str(tmp_path), "gate-r2", "ask", "why")
    assert ticket.redeem(str(tmp_path), "gate-r2") is not None
    assert ticket.redeem(str(tmp_path), "gate-r2") is None


def test_a_stale_ticket_is_not_evidence_about_now(tmp_path, monkeypatch):
    """A ticket from an hour ago says nothing about the call being made now."""
    ticket.issue(str(tmp_path), "gate-r3", "ask", "why")
    # ask the module where it put it: the directory is a per-user temp path the
    # SANDBOXED hook can write to, not something under run_root.
    p = ticket._dir(str(tmp_path)) / "gate-r3.json"
    d = json.loads(p.read_text())
    d["at"] -= ticket.TTL_SECONDS + 60
    p.write_text(json.dumps(d))
    assert ticket.redeem(str(tmp_path), "gate-r3") is None


# ---------------------------------------------------------------- the hook

def test_the_hook_denies_a_run_that_cannot_work(tmp_path):
    """Nobody should be asked to approve a run whose input does not exist."""
    out = pre_tool_use_hook(
        {"tool_name": "mcp__qmine__qmine_start_run",
         "tool_input": {"inputs": ["no-such-file.csv"], "run_id": "gate-r4"}},
        run_root=str(tmp_path))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "no-such-file.csv" in out["permissionDecisionReason"]


def test_the_hook_leaves_every_other_tool_alone(tmp_path):
    """Gating reads would teach people to approve without reading."""
    out = pre_tool_use_hook({"tool_name": "mcp__qmine__qmine_findings"},
                            run_root=str(tmp_path))["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"


def test_a_denied_run_issues_no_ticket(tmp_path):
    """A `deny` must not leave behind the evidence that would let the next call
    through: the ticket is issued only on the path that reaches the dialog."""
    pre_tool_use_hook(
        {"tool_name": "mcp__qmine__qmine_start_run",
         "tool_input": {"inputs": ["no-such-file.csv"], "run_id": "gate-r5"}},
        run_root=str(tmp_path))
    assert ticket.redeem(str(tmp_path), "gate-r5") is None


# ------------------------------------------------------- the hook's wiring

def _hooks_json() -> dict:
    return json.loads((ROOT / "integrations" / "dsh" / "hooks.json")
                      .read_text(encoding="utf-8"))


def test_the_hook_command_quotes_every_path_it_interpolates():
    """THE DEFECT ITSELF. `${CLAUDE_PROJECT_DIR}` is a real filesystem path and
    may contain spaces — this checkout's does. Unquoted, the shell split it
    mid-path, qmine never ran, and the decisionless hook failed OPEN."""
    cmd = _hooks_json()["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "${CLAUDE_PROJECT_DIR}" in cmd
    import re

    for m in re.finditer(r'\$\{CLAUDE_PROJECT_DIR\}', cmd):
        before = cmd[:m.start()]
        # every interpolation must sit inside a double-quoted word
        assert before.count('"') % 2 == 1, (
            f"unquoted ${{CLAUDE_PROJECT_DIR}} at offset {m.start()} in: {cmd}")


def test_the_hook_matches_the_tool_that_actually_spends():
    """A matcher naming a tool nobody serves is a gate on nothing."""
    matcher = _hooks_json()["hooks"]["PreToolUse"][0]["matcher"]
    served = {t["name"] for t in QMineServer("runs").specs}
    spend = {t["name"] for t in QMineServer("runs").specs if t["tier"] == "spend"}
    raw = matcher.rsplit("__", 1)[-1]
    assert raw in served, f"{matcher} names no served tool"
    assert raw in spend, f"{matcher} gates {raw}, which does not spend"


@pytest.mark.parametrize("posture,expect_gate", [("ask", True), ("1", False)])
def test_capabilities_states_the_posture_it_is_actually_in(posture, expect_gate,
                                                           monkeypatch, tmp_path):
    """The model decides whether to propose a run from this; it has to be true."""
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", posture)
    desc = QMineServer(str(tmp_path)).call("qmine_capabilities", {})
    blob = json.dumps(desc, ensure_ascii=False)
    assert ("run id" in blob) is expect_gate, blob
