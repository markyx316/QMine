"""What a run is doing RIGHT NOW, read off the files it writes as it goes.

A mining run takes hours, and for all of them the interesting question is not
"what did it find" but "where is it and is it still alive". Everything needed to
answer that is already on disk before the run ends:

    run.log          every phase start, every gate, every completion, live
    usage.json       model calls, tokens and elapsed, rewritten as it goes
    index.jsonl      one line per artifact the moment it is written
    findings.json    the run-level findings ledger
    dashboard.html   a full browsable page, written whatever the terminal is doing

`run_summary.json` is the only one that waits for the end — which is exactly why
its absence is the test for "not finished", and why nothing here will describe
results without it.

**A READING IS A SNAPSHOT AND SAYS SO.** The characteristic failure of a polled
progress surface is not a wrong number, it is a right number restated ten
minutes later as if it were current. Every payload carries `as_of` and an
instruction to call again rather than repeat.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: `12:23:56 INFO    qmine.graph: gate p0_provider: PASSED — real agents: ...`
_GATE = re.compile(r"gate\s+(\S+?):\s+(PASSED|WARNED|FAILED|SKIPPED)\s*—?\s*(.*)")
#: `12:23:53 INFO    qmine.graph: P0 foundation — run health-pool3, generation 1`
_PHASE = re.compile(r"qmine\.graph:\s+(P\d+[a-z]?)\s+(.+)")
#: `12:23:56 INFO    qmine.graph: ✔ p0_foundation completed in 3.2s`
_DONE = re.compile(r"✔\s+(\S+)\s+completed in ([\d.]+)s")

#: What becomes answerable once a given artifact exists. The point of a live
#: surface is not only "where is it" but "what can I ask about already".
ANSWERABLE: list[tuple[str, str, str]] = [
    ("data_audit", "qmine_partial with what='corpus'",
     "how big the corpus is, its length profile, its language mix, and which "
     "phrasing families were found"),
    ("domain_scout", "qmine_partial with what='domain'",
     "what the scout thinks this vertical is (HYPOTHESES ONLY — later phases "
     "test and overturn them)"),
    ("taxonomy", "qmine_partial with what='taxonomy'",
     "the L1 intent classes the architect wrote, with definitions"),
    ("taxonomy_v2", "qmine_partial with what='taxonomy'",
     "the redrawn intent classes"),
    ("representation", "qmine_partial with what='representation'",
     "which encoder and alpha won the bake-off, and by how much"),
    ("granularity", "qmine_partial with what='granularity'",
     "how K was located and what the sweep looked like"),
    ("tree_naming", "qmine_partial with what='tree'",
     "the delivered families and leaves with their names"),
    ("labels_full", "qmine_findings / qmine_class / qmine_examples",
     "every row's labels — the run has delivered"),
]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _phase_labels() -> dict[str, str]:
    try:
        from ..ui.live import PHASES

        return {p.key: f"{getattr(p, 'label_en', '')}".strip() for p in PHASES}
    except Exception:  # noqa: BLE001
        return {}


def read(run_dir: Path, *, log_lines: int = 12) -> dict[str, Any]:
    """A structured snapshot of one run in flight (or finished)."""
    now = time.time()
    gens = sorted((g.name for g in run_dir.glob("gen*") if g.is_dir()), reverse=True)
    gen = run_dir / gens[0] if gens else None
    summary = (gen / "run_summary.json") if gen else None
    finished = bool(summary and summary.is_file())

    out: dict[str, Any] = {
        "run_id": run_dir.name,
        "as_of": _iso(now),
        "generation": gens[0] if gens else None,
        "finished": finished,
        "how_to_use_this": (
            "This is a SNAPSHOT taken at `as_of`. A run moves; do not restate these "
            "numbers later as if they were current — call this tool again."),
    }

    log = run_dir / "run.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        age = now - log.stat().st_mtime
        out["log_last_written_seconds_ago"] = round(age, 1)
        out["log_tail"] = lines[-max(1, min(log_lines, 60)):]

        labels = _phase_labels()
        done = [(m.group(1), float(m.group(2))) for ln in lines
                if (m := _DONE.search(ln))]
        out["phases_completed"] = [
            {"phase": k, "seconds": s,
             "label": labels.get(k.split("_")[0], "")} for k, s in done]
        started = [(m.group(1), m.group(2)) for ln in lines if (m := _PHASE.search(ln))]
        if started:
            last_id, last_desc = started[-1]
            finished_ids = {k.split("_")[0] for k, _ in done}
            running = last_id.lower() not in finished_ids
            out["current_phase"] = {
                "phase": last_id, "description": last_desc.strip()[:160],
                "label": labels.get(last_id.lower(), ""),
                "still_in_it": bool(running and not finished)}

        gates: dict[str, list[str]] = {"passed": [], "warned": [], "failed": [], "skipped": []}
        for ln in lines:
            m = _GATE.search(ln)
            if m:
                gates[m.group(2).lower()].append(m.group(1))
        out["gates_so_far"] = {
            "passed": len(gates["passed"]), "skipped": len(gates["skipped"]),
            "warned": sorted(set(gates["warned"])), "failed": sorted(set(gates["failed"]))}
        if gates["failed"]:
            out["attention"] = (
                f"{len(set(gates['failed']))} gate(s) FAILED — a blocking gate halts the "
                "run rather than letting later phases build on a failed foundation.")

        if not finished:
            out["running"] = age < 900
            out["reading"] = (
                "The log was written to recently, so the run is alive."
                if age < 900 else
                f"No log activity for {age / 60:.0f} minutes and no run_summary.json — "
                "the run may have died. Read the tail.")
    elif not finished:
        out["reading"] = "No run.log yet — it is still starting up."

    usage = run_dir / "usage.json"
    if usage.is_file():
        try:
            u = json.loads(usage.read_text(encoding="utf-8"))
            out["spend_so_far"] = {
                "model_calls": u.get("calls"), "errors": u.get("errors"),
                "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
                "elapsed_minutes": round((u.get("elapsed_s") or 0) / 60, 1),
                # LISTS, not tuples: everything this module produces is destined
                # for JSON, and a tuple that only becomes a list further down is
                # a shape that differs between a direct call and a served one.
                "busiest_roles": [
                    [r, c] for r, c in sorted(
                        ((r, d.get("calls", 0)) for r, d in (u.get("by_role") or {}).items()),
                        key=lambda x: -x[1])[:5]],
            }
        except (json.JSONDecodeError, OSError):
            pass

    written: list[dict[str, Any]] = []
    idx = run_dir / "index.jsonl"
    if idx.is_file():
        for ln in idx.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                a = json.loads(ln)
            except json.JSONDecodeError:
                continue
            written.append({"name": a.get("name"), "producer": a.get("producer"),
                            "rows": a.get("rows")})
    if written:
        seen = {w["name"] for w in written}
        out["artifacts_written"] = len(written)
        out["latest_artifacts"] = written[-8:]
        out["what_can_be_answered_now"] = [
            {"ask_about": why, "with": how}
            for name, how, why in ANSWERABLE if name in seen]

    find = run_dir / "findings.json"
    if find.is_file():
        try:
            f = json.loads(find.read_text(encoding="utf-8"))
            out["open_findings"] = f.get("n_open")
            out["confirmed_open_findings"] = f.get("n_confirmed_open")
        except (json.JSONDecodeError, OSError):
            pass

    dash = run_dir / "dashboard.html"
    if dash.is_file():
        out["live_dashboard"] = {
            "path": str(dash),
            "how": "Open this file in a browser tab. It is written throughout the run, "
                   "whatever the terminal is doing, and refreshes far faster than "
                   "anyone would poll a tool.",
        }
    if not finished:
        out["do_not_report_results"] = (
            "No run_summary.json yet, so this run has NOT finished. Anything about its "
            "findings would be invented. What IS readable now is listed under "
            "`what_can_be_answered_now`.")
    return out


#: What `qmine_partial` can read, the artifact it needs, and the phase that
#: writes it. Asking for one that is not there yet gets told which phase it is
#: waiting on — a far better answer than "not found".
PARTIAL: dict[str, tuple[tuple[str, ...], str, str]] = {
    "corpus": (("data_audit",), "p1",
               "size, length profile, language mix, phrasing families"),
    "domain": (("domain_scout",), "p1",
               "the scout's guess at the vertical — HYPOTHESES ONLY"),
    "taxonomy": (("taxonomy_v2", "taxonomy"), "p2a",
                 "the L1 intent classes and their definitions"),
    "representation": (("representation",), "p3",
                       "which encoder and alpha won, and by how much"),
    "granularity": (("granularity",), "p456", "how K was located"),
    "tree": (("tree_naming",), "p7", "the delivered families and leaves"),
    "gold": (("gold_agreement",), "p2b", "annotator agreement on the gold set"),
}


def partial(run_dir: Path, what: str, *, limit: int = 40) -> dict[str, Any]:
    """Read ONE mid-run artifact, shaped small, while the run is still going."""
    key = str(what or "").strip().lower()
    if key not in PARTIAL:
        raise ValueError(f"unknown subject {what!r}. Available: "
                         + json.dumps({k: v[2] for k, v in PARTIAL.items()},
                                      ensure_ascii=False))
    names, phase, blurb = PARTIAL[key]
    gens = sorted((g.name for g in run_dir.glob("gen*") if g.is_dir()), reverse=True)
    if not gens:
        raise FileNotFoundError(f"{run_dir.name} has no generation yet")
    gen = run_dir / gens[0]
    obj = src = None
    for n in names:
        p = gen / f"{n}.json"
        if p.is_file():
            try:
                obj, src = json.loads(p.read_text(encoding="utf-8")), n
                break
            except (json.JSONDecodeError, OSError):
                continue
    if obj is None:
        return {"subject": key, "available": False, "waiting_on_phase": phase,
                "meaning": blurb,
                # NAME THE TOOL THAT EXISTS. A hint pointing at `qmine_progress`
                # — which is the module, not the tool — sends the model looking
                # for something it cannot call.
                "note": f"`{names[0]}` is written by {phase}. Call qmine_status to see "
                        "which phase the run is in."}

    out: dict[str, Any] = {"subject": key, "available": True, "source": f"{gens[0]}/{src}.json",
                           "meaning": blurb,
                           "mid_run_caveat": (
                               "This is an INTERMEDIATE artifact. Later phases revise it — "
                               "p8 rewrites the tree, and the taxonomy can be redrawn — so "
                               "it describes the run's state now, not what it will deliver.")}
    if key == "taxonomy":
        tx = obj.get("taxonomy", obj)
        nodes = [n for n in (tx.get("nodes") or []) if n.get("level") == 1]
        out["n_classes"] = len(nodes)
        out["classes"] = [{"code": n.get("code"), "name": n.get("name"),
                           "definition": (n.get("definition") or "")[:220],
                           "expected_share": n.get("expected_share"),
                           "risk": bool(n.get("risk"))}
                          for n in nodes[:limit]]
        out["truncated"] = len(nodes) > limit
    elif key == "tree":
        fams = obj.get("families_final") or obj.get("families") or []
        leaves = obj.get("namings") or []
        out["n_families"] = len(fams)
        out["n_leaves"] = len(leaves)
        out["families"] = [{"id": f.get("family_id"), "name": f.get("name_zh") or f.get("name"),
                            "coherent": f.get("coherent")} for f in fams[:limit]]
        out["leaves"] = [{"id": n.get("leaf_id"), "name": n.get("name_zh") or n.get("name"),
                          "coherence": n.get("coherence")} for n in leaves[:limit]]
    else:
        # Everything else is a flat-ish record; ship the scalars and name the rest.
        flat = {k: v for k, v in obj.items()
                if isinstance(v, (str, int, float, bool)) or v is None}
        big = sorted(k for k, v in obj.items() if k not in flat)
        out["fields"] = {k: (v[:400] if isinstance(v, str) else v)
                         for k, v in list(flat.items())[:60]}
        if big:
            out["larger_fields_not_shown"] = big[:30]
    return out
