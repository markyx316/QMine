"""QMine as an MCP server: the whole program as tools a chat harness can call.

Built for DeepSeek Harness (`dsh`), whose `dsh-mcp-client` plugin bridges an
external server's **Tools** capability and exposes them as
`mcp__<serverName>__<toolName>`. Resources and Prompts are not bridged as of the
August 2026 preview, so everything here is a tool — including the glossary,
which would otherwise have been a resource.

WHY MCP RATHER THAN A PLUGIN. `dsh` is TypeScript; this program is Python and
its value is in the Python — the statistics, the seven-layer quote guard, the
authority model. Re-expressing any of that in a plugin would mean maintaining
two copies of the rules that make the output trustworthy. Over MCP the harness
supplies the chat surface, the model adapter, the session log and the agent
loop, and this side supplies exactly what it already knows.

TOOL DESCRIPTIONS ARE PART OF THE PRODUCT. The model picks a tool from its
description and writes the answer from the payload, so each description carries
the trap that belongs to it — quote the verdict and not the bare zero, a
stratum axis is not a timeline, a fast run has an ABSENT kappa rather than a
perfect one. A correct number presented under the wrong caveat is the failure
mode this surface has, and prose is where it is prevented.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import answers, progress, store, ticket
from .authority import Authority

SERVER_NAME = "qmine"

#: (name, tier, description, input schema). One list, so the advertised surface
#: and the dispatch table cannot drift apart.
def _tools(run_root: str) -> list[dict[str, Any]]:
    S = lambda **p: {"type": "object", "properties": p}  # noqa: E731
    run_id = {"type": "string", "description": "Run id, e.g. `med-pool8`."}
    gen = {"type": "string", "description": "Generation (`gen01`). Default: the newest."}
    return [
        dict(name="qmine_capabilities", tier="read",
             description=(
                 "What this server may do right now — which tiers are allowed, where it "
                 "may write, and whether it can start a paid run. Call this first if the "
                 "user asks whether you can run something."),
             schema=S()),
        dict(name="qmine_list_runs", tier="read",
             description=(
                 f"Every mining run under `{run_root}/`, with its mode, provider, whether "
                 "it halted, and whether a cross-snapshot comparison has been built. Start "
                 "here when the user names no run."),
             schema=S()),
        dict(name="qmine_overview", tier="read",
             description=(
                 "What one run IS: mode, provider, phases completed, gate outcomes, the "
                 "shape it delivered. Read `read_this_first` if present — a `fast` run has "
                 "an ABSENT kappa (not a perfect one) and an `offline` run was written by "
                 "a deterministic stand-in and means nothing about the corpus."),
             schema=S(run_id=run_id, generation=gen)),
        dict(name="qmine_status", tier="read",
             description=(
                 "Where a run is RIGHT NOW — whether it is still going, which phase it "
                 "reached, how long it has been running, and the tail of its log. Use "
                 "this after `qmine_start_run`, which returns immediately because a run "
                 "takes hours. Returns the current phase, every gate so far, model calls "
                 "and tokens spent so far, which artifacts exist, open findings, and what "
                 "is already answerable (see `what_can_be_answered_now` — hand those to "
                 "`qmine_partial`). It also gives the path to a live dashboard the person "
                 "can open in a browser tab, which refreshes faster than anyone would "
                 "poll.\n\nTWO THINGS TO CARRY INTO YOUR ANSWER. This is a SNAPSHOT at "
                 "`as_of` — a run moves, so never restate it later as current; call "
                 "again. And a run with no `run_summary.json` has NOT finished: do not "
                 "report results for it."),
             schema=S(run_id=run_id, lines={"type": "integer",
                                            "description": "Log lines to return (default 20, max 80)."})),
        dict(name="qmine_partial", tier="read",
             description=(
                 "Read ONE intermediate artifact WHILE A RUN IS STILL GOING — the corpus "
                 "audit after p1, the intent classes after p2a, the encoder bake-off after "
                 "p3, how K was located after p456, the families and leaves after p7. This "
                 "is how you answer a question during a run instead of saying 'wait'. "
                 "Everything it returns is INTERMEDIATE: later phases revise it (p8 "
                 "rewrites the tree, the taxonomy can be redrawn), so describe it as the "
                 "run's current state, never as what it will deliver. Asking for one that "
                 "does not exist yet tells you which phase writes it."),
             schema=S(run_id=run_id,
                      what={"type": "string",
                            "description": "corpus | domain | taxonomy | representation | "
                                           "granularity | tree | gold"},
                      limit={"type": "integer", "description": "Max rows (default 40)."})),
        dict(name="qmine_findings", tier="read",
             description=(
                 "The measured headline of a run's cross-snapshot comparison: per-level "
                 "distance WITH its noise ceiling, the biggest significant differences, "
                 "per-level counts, and the privacy figures. This is the right first call "
                 "for 'what did we find'. Two things to carry into any answer: a distance "
                 "below its ceiling means 'this data cannot tell them apart', NOT 'they "
                 "are the same'; and on a `stratum` axis the snapshots are different "
                 "samples of ONE period, so differences are differences and never changes "
                 "over time."),
             schema=S(run_id=run_id, generation=gen)),
        dict(name="qmine_class", tier="read",
             description=(
                 "Everything the comparison knows about ONE class: its definition, its "
                 "share and confidence interval in each snapshot, its pairwise "
                 "differences, and any snapshot it is absent from. When it is absent, "
                 "quote the `判定` verdict and never the bare zero — zero of 950 rows is "
                 "consistent with a share up to 0.39%."),
             schema=S(run_id=run_id, generation=gen,
                      level={"type": "string",
                             "description": "td_l1 | td_l2 | bu_leaf | bu_family_final "
                                            "(or l1 / leaf / family / 意图 / 叶)"},
                      name={"type": "string", "description": "Class name or key."})),
        dict(name="qmine_examples", tier="read",
             description=(
                 "Real corpus rows for a class — selected by traffic and then at random, "
                 "and filtered by the seven-layer quote guard. A cell listed under "
                 "`cells_with_nothing_quotable` HAS rows but none that may be reproduced; "
                 "report it as a count, never as an absence. Do not quote any string this "
                 "tool did not return."),
             schema=S(run_id=run_id, generation=gen,
                      level={"type": "string", "description": "td_l1 or bu_leaf only."},
                      name={"type": "string", "description": "Class name or key."},
                      snapshot={"type": "string", "description": "Snapshot display name."},
                      n={"type": "integer", "description": "How many rows (default 6, max 30)."})),
        dict(name="qmine_table", tier="read",
             description=(
                 "Query any table behind the comparison, with filters, a column subset, a "
                 "sort and a row limit. Returns `rows_matched` and `rows_in_table` beside "
                 "the rows, so a truncated read is visible rather than mistaken for the "
                 "whole table. Call `qmine_tables` first to see what exists."),
             schema=S(run_id=run_id, generation=gen,
                      table={"type": "string", "description": "Table name without `.csv`."},
                      where={"type": "object",
                             "description": "Column → exact value, a list of values, or "
                                            "`>N` / `<N` for numeric bounds."},
                      columns={"type": "array", "items": {"type": "string"}},
                      sort={"type": "string", "description": "Column to sort by."},
                      descending={"type": "boolean"},
                      limit={"type": "integer", "description": "Default 40, max 200."})),
        dict(name="qmine_tables", tier="read",
             description="The tables a run's comparison produced, and what each one holds.",
             schema=S(run_id=run_id, generation=gen)),
        dict(name="qmine_document", tier="read",
             description=(
                 "One SECTION of a delivered document, by heading. Never returns a whole "
                 "report — they run to hundreds of kilobytes. Call with no `section` to "
                 "get the list of headings."),
             schema=S(run_id=run_id, generation=gen,
                      document={"type": "string",
                                "description": "File name or a fragment of it."},
                      section={"type": "string", "description": "Heading text or a fragment."})),
        dict(name="qmine_glossary", tier="read",
             description=(
                 "What a term means IN THIS METHODOLOGY — TVD, 同源噪声上界, 均衡指数, "
                 "缺席判定, pv_norm, 流量有效n, fast, stratum. Use it before explaining any "
                 "of them; several have a specific meaning here that the general one misses."),
             schema=S(term={"type": "string"})),
        dict(name="qmine_inspect_inputs", tier="read",
             description=(
                 "Measure raw export files before anything is decided: columns, duplicate "
                 "rate, length distribution, whether the weights look like a head or a "
                 "random sample, high-coverage prefixes that are the product's own wrapper, "
                 "and overlap between the files. Spends nothing and writes nothing."),
             schema=S(inputs={"type": "array", "items": {"type": "string"},
                              "description": "File paths."})),
        dict(name="qmine_plan_corpus", tier="read",
             description=(
                 "Propose how to pool several exports into ONE corpus: which column is the "
                 "query, which is traffic, what each snapshot is called, and an ordered list "
                 "of named cleaning operations. Writes nothing. The plan's `concerns` are "
                 "the part to read aloud — especially the comparison axis, which is inferred "
                 "unless declared and inverts the prose if it is wrong."),
             schema=S(inputs={"type": "array", "items": {"type": "string"}},
                      axis={"type": "string", "description": "time | stratum. Omit to infer."},
                      text_column={"type": "string"})),
        dict(name="qmine_preflight", tier="read",
             description=(
                 "Is this run going to WORK? Everything checkable before a penny is "
                 "spent: the run id is free, every input exists and has a query column, "
                 "the config and domain profile load, every agent role routes to a "
                 "reachable model, what it will cost, and whether there is disk for it. "
                 "Spends nothing.\n\nRun this BEFORE proposing a run and read the result "
                 "out. `verdict` is `go` or `no_go`; `blocking` is what makes it "
                 "impossible, `warnings` are what the person should know anyway — an "
                 "absent domain profile means NO risk screening, an undeclared axis means "
                 "the comparison will read as `stratum`, `fast` means kappa will be "
                 "ABSENT. `qmine_start_run` runs these same checks again and refuses on "
                 "any blocking one, so fixing them here is not optional."),
             schema=S(inputs={"type": "array", "items": {"type": "string"},
                              "description": "The raw export paths, in order."},
                      run_id={"type": "string", "description": "The id the run would take."},
                      domain={"type": "string", "description": "Domain profile, e.g. `med_zh`."},
                      config={"type": "string", "description": "Config path."},
                      fast={"type": "boolean", "description": "Price and check a FAST run."},
                      axis={"type": "string", "description": "time | stratum, if declared."})),
        dict(name="qmine_estimate_cost", tier="read",
             description="What a full run would cost and how many model calls it would make. Spends nothing.",
             schema=S(config={"type": "string", "description": "Config path."})),
        dict(name="qmine_prepare_corpus", tier="write",
             description=(
                 "Execute a preparation plan: pool the exports into one corpus, tiering out "
                 "what the interface printed rather than what a person typed. Nothing is "
                 "deleted — every removal assigns a tier and the removed rows ship beside "
                 "the corpus — and a rule that would remove more than a quarter of an input "
                 "is applied as a flag and recorded as refused. Makes no model call."),
             schema=S(inputs={"type": "array", "items": {"type": "string"}},
                      out={"type": "string", "description": "Output directory."},
                      axis={"type": "string"}, text_column={"type": "string"})),
        dict(name="qmine_build_comparison", tier="write",
             description=(
                 "Build (or rebuild) the cross-snapshot comparison for a finished pooled "
                 "run — tables, figures, report, workbooks. Makes NO model call, so it is "
                 "safe to re-run; this is how you refresh after renaming a snapshot or "
                 "adding a screened quote list."),
             schema=S(run_id=run_id, generation=gen, axis={"type": "string"},
                      title={"type": "string"})),
        dict(name="qmine_start_run", tier="spend",
             description=(
                 "Start the twelve-phase mining run. THIS COSTS REAL MONEY and takes "
                 "hours.\n\nWhat happens depends on how this deployment was launched — "
                 "call `qmine_capabilities` to see which. By DEFAULT the server does not "
                 "start anything and hands back the exact command for a person to run. "
                 "Where starting is allowed, this tool runs `qmine_preflight` again "
                 "itself and REFUSES on any blocking problem, and in `ask` mode it also "
                 "needs `confirm` set to the run id.\n\nNever call this as an opening "
                 "move. Preflight, show the person the cost and every warning, and let "
                 "them say go — and if the harness holds the call at an approval dialog, "
                 "that pause is the point, not an error."),
             schema=S(inputs={"type": "array", "items": {"type": "string"}},
                      run_id={"type": "string"}, domain={"type": "string"},
                      config={"type": "string"}, fast={"type": "boolean"},
                      prepare={"type": "boolean"},
                      axis={"type": "string",
                            "description": "time | stratum, for a pooled run."},
                      confirm={"type": "string",
                               "description": "In `ask` mode: the run id again, after the "
                                              "person has seen the cost and agreed."})),
    ]


TABLE_MEANING = {
    "matrix": "one row per class: its share, interval, traffic share, rank and balance "
              "index in every snapshot",
    "absence": "one row per (class, snapshot-with-zero-rows), with expected count, P(0) "
               "and the verdict",
    "newcombe": "one row per (class, snapshot pair): the share difference and its interval",
    "signature": "classes significantly higher (or lower) in one snapshot than in EVERY other",
    "confidence": "how securely each class was labelled, per snapshot",
    "surface": "the same comparison collapsed to the declared snapshot groups",
    "examples": "real corpus rows per (class, snapshot), already quote-guarded",
    "coverage": "per-snapshot profile: class count, entropy, HHI, head concentration",
    "topn_coverage": "how concentrated each snapshot's head is",
    "topn_members": "which classes are ranks 1..10 in each snapshot — the ONLY authority for that",
    "pairwise_tvd": "snapshot-to-snapshot distance with its bootstrap interval and noise ceiling",
    "exclusive_power": "whether a class unique to the smaller group would have been visible at all",
    "intent_leafmix": "same intent, different snapshots — does its internal leaf mix differ",
    "leaf_intentmix": "same leaf, different snapshots — does its internal intent mix differ",
}


def _opt(a: dict[str, Any], key: str) -> str | None:
    """An optional string argument, or None.

    `str(None)` is the four characters `None`, and over MCP an OMITTED optional
    argument arrives as the key present with a null value — so `str(a.get(k, ""))`
    yields `"None"` and every "did the caller name one?" test says yes. It only
    showed up through the real transport, because a direct call omits the key
    entirely and takes the `""` default.
    """
    v = a.get(key)
    if v is None:
        return None
    t = str(v).strip()
    return t or None


class QMineServer:
    def __init__(self, run_root: str = "runs") -> None:
        self.run_root = run_root
        self.authority = Authority.from_env(run_root=run_root)
        self.specs = _tools(run_root)
        self.tier = {t["name"]: t["tier"] for t in self.specs}

    # ---------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"no tool {name!r}",
                    "available": [t["name"] for t in self.specs]}
        try:
            # `native` before `clip`: clip serialises, and a numpy scalar or a
            # NaN from a pandas cell makes that raise from inside the tool.
            return store.clip(answers.native(fn(args)))
        except PermissionError as exc:
            return {"error": "refused", "detail": str(exc)}
        except (store.RunNotFound, FileNotFoundError) as exc:
            return {"error": "not_found", "detail": str(exc)}
        except ValueError as exc:
            return {"error": "bad_request", "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__, "detail": str(exc)[:500]}

    def _ref(self, a: dict[str, Any]) -> store.RunRef:
        rid = _opt(a, "run_id")
        if not rid:
            raise ValueError("run_id is required; call qmine_list_runs to see them")
        return store.resolve(str(rid), _opt(a, "generation"), self.run_root)

    # ------------------------------------------------------------------ read
    def _t_qmine_capabilities(self, _a: dict[str, Any]) -> dict[str, Any]:
        return {"server": SERVER_NAME, "run_root": self.run_root,
                "permissions": self.authority.describe(),
                "tools": [{"name": t["name"], "tier": t["tier"]} for t in self.specs]}

    def _t_qmine_list_runs(self, _a: dict[str, Any]) -> dict[str, Any]:
        runs = store.list_runs(self.run_root)
        return {"runs": runs, "count": len(runs),
                "note": "A run with has_cross_snapshot_comparison=false can get one from "
                        "qmine_build_comparison — it makes no model call."}

    def _t_qmine_overview(self, a: dict[str, Any]) -> dict[str, Any]:
        return answers.overview(self._ref(a))

    def _t_qmine_status(self, a: dict[str, Any]) -> dict[str, Any]:
        """Where a run is right now — read off the files it writes as it goes."""
        rid = _opt(a, "run_id")
        if not rid:
            raise ValueError("run_id is required; call qmine_list_runs to see them")
        root = Path(self.run_root) / rid
        if not root.is_dir():
            raise store.RunNotFound(f"no run {rid!r} under {self.run_root}")
        out = progress.read(root, log_lines=int(a.get("lines") or 12))
        if out.get("finished"):
            gens = out.get("generation")
            out.update(answers.overview(store.resolve(rid, gens, self.run_root)))
            out["finished"] = True
        return out

    def _t_qmine_partial(self, a: dict[str, Any]) -> dict[str, Any]:
        """One intermediate artifact, while the run is still going."""
        rid = _opt(a, "run_id")
        if not rid:
            raise ValueError("run_id is required")
        root = Path(self.run_root) / rid
        if not root.is_dir():
            raise store.RunNotFound(f"no run {rid!r} under {self.run_root}")
        out = progress.partial(root, _opt(a, "what") or "",
                               limit=int(a.get("limit") or 40))
        return answers.scrub(store.RunRef(rid, out.get("source", "gen01/x").split("/")[0],
                                          Path(self.run_root)), out) \
            if out.get("available") else out

    def _t_qmine_findings(self, a: dict[str, Any]) -> dict[str, Any]:
        return answers.findings(self._ref(a))

    def _t_qmine_class(self, a: dict[str, Any]) -> dict[str, Any]:
        name = _opt(a, "name")
        if not name:
            raise ValueError("name is required — the class to look up")
        return answers.class_detail(self._ref(a), _opt(a, "level") or "td_l1", name)

    def _t_qmine_examples(self, a: dict[str, Any]) -> dict[str, Any]:
        return answers.examples(self._ref(a), _opt(a, "level") or "td_l1",
                                _opt(a, "name"), _opt(a, "snapshot"),
                                int(a.get("n") or 6))

    def _t_qmine_tables(self, a: dict[str, Any]) -> dict[str, Any]:
        ref = self._ref(a)
        names = store.available_tables(ref)
        return {"run_id": ref.run_id, "generation": ref.generation,
                "tables": [{"name": n,
                            "holds": TABLE_MEANING.get(n.split("_")[0], "")} for n in names],
                "count": len(names)}

    def _t_qmine_table(self, a: dict[str, Any]) -> dict[str, Any]:
        ref = self._ref(a)
        name = _opt(a, "table")
        if not name:
            raise ValueError("table is required; call qmine_tables to see them")
        t = store.read_table(ref, name)
        out = store.shape(t, where=a.get("where"), columns=a.get("columns"),
                          sort=_opt(a, "sort"),
                          descending=bool(a.get("descending", True)),
                          limit=int(a.get("limit") or store.DEFAULT_LIMIT))
        out["source"] = f"runs/{ref.run_id}/{ref.generation}/pooled/tables/{name}.csv"
        out["columns_available"] = list(t.columns)
        return answers.scrub(ref, out)

    def _t_qmine_document(self, a: dict[str, Any]) -> dict[str, Any]:
        ref = self._ref(a)
        docs = sorted(list(ref.pooled_dir.glob("*.md")) + list(ref.gen_dir.glob("*.md")))
        if not docs:
            raise FileNotFoundError(f"{ref.run_id}/{ref.generation} has no markdown documents")
        want = _opt(a, "document")
        # DEFAULT TO THE COMPARISON REPORT, not whatever sorts first. `00_索引.md`
        # sorts ahead of every CJK filename, so "show me the document" landed on
        # the index rather than the study.
        default = next((d for d in docs if d.name.endswith(".zh.md")),
                       next((d for d in docs if "跨快照对比" in d.name), docs[0]))
        pick = next((d for d in docs if want and want in d.name), default if not want else None)
        if pick is None:
            raise ValueError(f"no document matches {want!r}. Documents: "
                             f"{[d.name for d in docs]}")
        text = pick.read_text(encoding="utf-8")
        heads = [ln.strip() for ln in text.splitlines() if ln.startswith(("## ", "### "))]
        sec = _opt(a, "section") or ""
        if not sec:
            return {"document": pick.name, "sections": heads[:120],
                    "chars": len(text),
                    "note": "Ask for one section by name; the whole document is too large "
                            "to read into a conversation."}
        lines = text.splitlines()
        start = next((i for i, ln in enumerate(lines)
                      if ln.startswith(("## ", "### ")) and sec in ln), None)
        if start is None:
            raise ValueError(f"no section matching {sec!r} in {pick.name}. "
                             f"Sections: {heads[:40]}")
        depth = len(lines[start].split(" ")[0])
        body: list[str] = [lines[start]]
        for ln in lines[start + 1:]:
            if ln.startswith("#") and len(ln.split(" ")[0]) <= depth:
                break
            body.append(ln)
        return answers.scrub(ref, {"document": pick.name, "section": lines[start],
                                   "text": "\n".join(body)[:store.MAX_CHARS],
                                   "source": str(pick.relative_to(ref.root.parent))})

    def _t_qmine_glossary(self, a: dict[str, Any]) -> dict[str, Any]:
        return answers.glossary(_opt(a, "term"))

    def _t_qmine_inspect_inputs(self, a: dict[str, Any]) -> dict[str, Any]:
        from ..prepare import profile_inputs

        paths = [str(p) for p in (a.get("inputs") or [])]
        if not paths:
            raise ValueError("inputs is required")
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"no such file(s): {missing}")
        profs = profile_inputs(paths, text_column=_opt(a, "text_column"))
        return {"inputs": [p.brief() for p in profs]}

    def _t_qmine_plan_corpus(self, a: dict[str, Any]) -> dict[str, Any]:
        from ..prepare import heuristic_plan, profile_inputs

        paths = [str(p) for p in (a.get("inputs") or [])]
        if not paths:
            raise ValueError("inputs is required")
        plan = heuristic_plan(profile_inputs(paths, text_column=_opt(a, "text_column")),
                              text_column=_opt(a, "text_column"), axis=_opt(a, "axis"))
        return {"plan": json.loads(plan.model_dump_json()),
                "read_the_concerns_aloud": plan.concerns,
                "note": "Nothing was written. qmine_prepare_corpus executes this."}

    def _t_qmine_estimate_cost(self, a: dict[str, Any]) -> dict[str, Any]:
        cmd = [sys.executable, "-m", "qmine.cli", "models"]
        if _opt(a, "config"):
            cmd += ["--config", str(_opt(a, "config"))]
        return self._shell(cmd, tier="read")

    # ----------------------------------------------------------------- write
    def _t_qmine_prepare_corpus(self, a: dict[str, Any]) -> dict[str, Any]:
        paths = [str(p) for p in (a.get("inputs") or [])]
        out = _opt(a, "out") or "prepared"
        cmd = [sys.executable, "-m", "qmine.cli", "prepare", ",".join(paths), "--out", out]
        if _opt(a, "axis"):
            cmd += ["--axis", str(_opt(a, "axis"))]
        ref = self.authority.refusal("write", "qmine_prepare_corpus", cmd)
        if ref:
            return ref
        if not paths:
            raise ValueError("inputs is required")
        target = self.authority.check_path(out)
        from ..prepare import execute, heuristic_plan, profile_inputs

        plan = heuristic_plan(profile_inputs(paths, text_column=_opt(a, "text_column")),
                              text_column=_opt(a, "text_column"), axis=_opt(a, "axis"))
        rep = execute(plan, target)
        return {"status": "prepared", "corpus": rep["corpus"],
                "rows_mined": rep["n_mined"], "snapshots": rep["snapshots"],
                "axis": rep["axis"], "concerns": rep["concerns"],
                "per_input": [{k: v for k, v in i.items() if k != "各步"}
                              for i in rep["inputs"]],
                "next": f"qmine run --input {rep['corpus']} --run-id <id>  (this COSTS money)"}

    def _t_qmine_preflight(self, a: dict[str, Any]) -> dict[str, Any]:
        from ..preflight import preflight

        return preflight(
            inputs=[str(p) for p in (a.get("inputs") or [])],
            run_id=_opt(a, "run_id"), domain=_opt(a, "domain"),
            config=_opt(a, "config"), fast=bool(a.get("fast")),
            axis=_opt(a, "axis"), run_root=self.run_root)

    def _t_qmine_build_comparison(self, a: dict[str, Any]) -> dict[str, Any]:
        rid = _opt(a, "run_id") or ""
        cmd = [sys.executable, "-m", "qmine.cli", "compare", rid,
               "--run-root", self.run_root]
        for flag, key in (("--axis", "axis"), ("--title", "title")):
            if _opt(a, key):
                cmd += [flag, str(_opt(a, key))]
        ref = self.authority.refusal("write", "qmine_build_comparison", cmd)
        if ref:
            return ref
        res = self._shell(cmd, tier="write")
        try:
            res["findings"] = answers.findings(self._ref(a))
        except Exception:  # noqa: BLE001
            pass
        return res

    # ----------------------------------------------------------------- spend
    def _t_qmine_start_run(self, a: dict[str, Any]) -> dict[str, Any]:
        paths = [str(p) for p in (a.get("inputs") or [])]
        cmd = [sys.executable, "-m", "qmine.cli", "run", "--input", ",".join(paths)]
        for flag, key in (("--run-id", "run_id"), ("--domain", "domain"),
                          ("--config", "config")):
            if _opt(a, key):
                cmd += [flag, str(_opt(a, key))]
        if a.get("fast"):
            cmd.append("--fast")
        if a.get("prepare"):
            cmd.append("--prepare")
        cmd += ["--run-root", self.run_root]
        ref = self.authority.refusal("spend", "qmine_start_run", cmd)
        if ref:
            return ref
        rid = _opt(a, "run_id")
        if not rid:
            raise ValueError("run_id is required to start a run")

        # THE PREFLIGHT RUNS HERE TOO, not only where the model chose to call it.
        # Whatever authorised the spending — an env var, or a person clicking
        # approve in the harness — none of it makes a doomed run worth starting,
        # and whoever clicked cannot be expected to have re-derived that the run
        # id is taken or that a pinned model routes nowhere.
        from ..preflight import preflight

        pre = preflight(inputs=paths, run_id=rid, domain=_opt(a, "domain"),
                        config=_opt(a, "config"), fast=bool(a.get("fast")),
                        axis=_opt(a, "axis"), run_root=self.run_root)
        if pre["blocking"]:
            return {"status": "not_run_preflight_failed",
                    "why": ("Not started: each of these would have wasted the money, or "
                            "produced output that means nothing about the corpus."),
                    "blocking": pre["blocking"], "warnings": pre["warnings"],
                    "run_this_yourself_once_fixed": " ".join(shlex.quote(c) for c in cmd)}

        # `ask` posture: the id has to be echoed back. A STUMBLE GUARD, not
        # consent — the model can satisfy it on its own — whose real value is
        # that the preflight cannot have gone unread. Consent, where it exists,
        # is the harness's approval gate in front of this call.
        if self.authority.spend_confirm:
            tkt = ticket.redeem(self.run_root, rid)
            if tkt is None:
                return {
                    "status": "not_run_no_approval_gate",
                    "why": ("Spending here is set to `ask`, which means a run may start "
                            "ONLY when an approval gate was consulted for it. No ticket "
                            "was found for this run id, so either the harness has no "
                            "PreToolUse hook installed or the hook failed — and a hook "
                            "that fails emits no decision, which would otherwise let this "
                            "through silently."),
                    "how_to_fix": [
                        "`make chat` installs the hook; check `qmine doctor`.",
                        "Run it yourself with the command below.",
                        "Or launch the harness with QMINE_MCP_ALLOW_SPEND=1 to accept no "
                        "gate at all — a deliberate choice, not a default.",
                    ],
                    "run_this_yourself": " ".join(shlex.quote(c) for c in cmd)}
        if self.authority.spend_confirm and _opt(a, "confirm") != rid:
            return {"status": "not_run_needs_confirmation",
                    "why": (f"Spending is allowed here in `ask` mode, so starting needs "
                            f"the run id echoed back: call again with confirm={rid!r} "
                            f"once the person has agreed."),
                    "preflight": {"verdict": pre["verdict"], "summary": pre["summary"],
                                  "warnings": pre["warnings"]},
                    "estimated_cost_usd": pre.get("estimated_cost_usd"),
                    "estimated_calls": pre.get("estimated_calls"),
                    "show_the_person_first": ("the cost and every warning above — they are "
                                              "what somebody needs in order to agree")}
        out = self._launch(cmd, rid)
        out["preflight"] = {"verdict": pre["verdict"], "summary": pre["summary"],
                            "warnings": pre["warnings"],
                            "estimated_cost_usd": pre.get("estimated_cost_usd")}
        return out

    # ---------------------------------------------------------------- launch
    def _launch(self, cmd: list[str], run_id: str) -> dict[str, Any]:
        """Start a run DETACHED and return at once.

        A tool call is request/response and a mining run takes hours — they do
        not fit inside one another. Blocking meant the client's tool timeout
        fired (dsh's default is 60 seconds; even a generous 15 minutes is a
        fraction of a run), the call came back as a failure, and the subprocess
        carried on unattached with nobody watching it: the worst of both, a run
        that is really happening and a conversation that believes it failed.

        So the process is started in its own session, its output goes to the
        run's own log, and the conversation gets a handle to poll.
        """
        root = Path(self.run_root) / run_id
        root.mkdir(parents=True, exist_ok=True)
        log = root / "mcp_launch.log"
        env = {**os.environ, "HF_HOME": os.environ.get("HF_HOME", str(Path.cwd() / ".hf"))}
        with log.open("ab") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=str(Path.cwd()), env=env,
                                    start_new_session=True)
        return {
            "status": "started",
            "run_id": run_id, "pid": proc.pid,
            "command": " ".join(cmd),
            "log": str(log),
            "this_takes_hours": True,
            "poll_with": f"qmine_status with run_id={run_id!r}",
            "note": ("The run is detached and will outlive this conversation. It "
                     "is NOT finished when this tool returns — say so, and poll "
                     "rather than reporting a result you do not have."),
        }

    # ----------------------------------------------------------------- shell
    def _shell(self, cmd: list[str], *, tier: str, timeout: float | None = 900
               ) -> dict[str, Any]:
        env = {**os.environ, "HF_HOME": os.environ.get("HF_HOME", str(Path.cwd() / ".hf"))}
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               env=env, cwd=str(Path.cwd()))
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "command": " ".join(cmd),
                    "detail": f"no result after {timeout}s — run it yourself to watch it"}
        tail = ((p.stdout or "") + (p.stderr or "")).splitlines()[-60:]
        return {"status": "ok" if p.returncode == 0 else "failed",
                "exit_code": p.returncode, "command": " ".join(cmd),
                "output_tail": "\n".join(tail), "tier": tier}


# ------------------------------------------------------------------- serving
#
# `mcp` 2.x renamed FastMCP to `MCPServer` and dropped the low-level decorators;
# `add_tool` derives each schema from a TYPED PYTHON SIGNATURE. So the tools are
# written out as real functions rather than as JSON Schema dicts — more lines,
# but the schema the client sees is the signature a reader can check, and the
# two cannot drift.


def build_app(run_root: str = "runs") -> tuple[Any, "QMineServer"]:
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise SystemExit(
            "the Model Context Protocol SDK is not installed — "
            "`.venv/bin/pip install 'mcp>=2.0'` (or `pip install -e '.[mcp]'`). "
            "Everything else in QMine works without it."
        ) from exc

    qm = QMineServer(run_root)
    app = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "QMine mines query logs into two independent label systems and compares "
            "snapshots of a corpus. Read tools are free; `qmine_build_comparison` and "
            "`qmine_prepare_corpus` write files; starting a run COSTS MONEY and is "
            "refused unless it was allowed from outside this conversation.\n\n"
            "Carry three things into every answer. A distance below its noise ceiling "
            "means the data cannot tell the snapshots apart, not that they are the "
            "same. On a `stratum` axis the snapshots are different samples of ONE "
            "period, so differences are never changes over time. A `fast` run has an "
            "ABSENT kappa, not a perfect one.\n\n"
            "Never quote a corpus row this server did not return: the example tools "
            "apply a seven-layer quote guard, and text obtained any other way has not "
            "been through it."),
    )

    def _j(name: str, args: dict[str, Any]) -> str:
        return json.dumps(qm.call(name, args), ensure_ascii=False, default=str)

    D = {t["name"]: t["description"] for t in qm.specs}

    def qmine_capabilities() -> str:
        return _j("qmine_capabilities", {})

    def qmine_list_runs() -> str:
        return _j("qmine_list_runs", {})

    def qmine_overview(run_id: str, generation: str | None = None) -> str:
        return _j("qmine_overview", {"run_id": run_id, "generation": generation})

    def qmine_status(run_id: str, lines: int = 12) -> str:
        return _j("qmine_status", {"run_id": run_id, "lines": lines})

    def qmine_partial(run_id: str, what: str, limit: int = 40) -> str:
        return _j("qmine_partial", {"run_id": run_id, "what": what, "limit": limit})

    def qmine_findings(run_id: str, generation: str | None = None) -> str:
        return _j("qmine_findings", {"run_id": run_id, "generation": generation})

    def qmine_class(run_id: str, level: str, name: str,
                    generation: str | None = None) -> str:
        return _j("qmine_class", {"run_id": run_id, "level": level, "name": name,
                                  "generation": generation})

    def qmine_examples(run_id: str, level: str = "td_l1", name: str | None = None,
                       snapshot: str | None = None, n: int = 6,
                       generation: str | None = None) -> str:
        return _j("qmine_examples", {"run_id": run_id, "level": level, "name": name,
                                     "snapshot": snapshot, "n": n,
                                     "generation": generation})

    def qmine_tables(run_id: str, generation: str | None = None) -> str:
        return _j("qmine_tables", {"run_id": run_id, "generation": generation})

    def qmine_table(run_id: str, table: str, where: dict[str, Any] | None = None,
                    columns: list[str] | None = None, sort: str | None = None,
                    descending: bool = True, limit: int = 40,
                    generation: str | None = None) -> str:
        return _j("qmine_table", {"run_id": run_id, "table": table, "where": where,
                                  "columns": columns, "sort": sort,
                                  "descending": descending, "limit": limit,
                                  "generation": generation})

    def qmine_document(run_id: str, document: str | None = None,
                       section: str | None = None,
                       generation: str | None = None) -> str:
        return _j("qmine_document", {"run_id": run_id, "document": document,
                                     "section": section, "generation": generation})

    def qmine_glossary(term: str | None = None) -> str:
        return _j("qmine_glossary", {"term": term})

    def qmine_inspect_inputs(inputs: list[str], text_column: str | None = None) -> str:
        return _j("qmine_inspect_inputs", {"inputs": inputs, "text_column": text_column})

    def qmine_plan_corpus(inputs: list[str], axis: str | None = None,
                          text_column: str | None = None) -> str:
        return _j("qmine_plan_corpus", {"inputs": inputs, "axis": axis,
                                        "text_column": text_column})

    def qmine_estimate_cost(config: str | None = None) -> str:
        return _j("qmine_estimate_cost", {"config": config})

    def qmine_prepare_corpus(inputs: list[str], out: str = "prepared",
                             axis: str | None = None,
                             text_column: str | None = None) -> str:
        return _j("qmine_prepare_corpus", {"inputs": inputs, "out": out, "axis": axis,
                                           "text_column": text_column})

    def qmine_build_comparison(run_id: str, axis: str | None = None,
                               title: str | None = None,
                               generation: str | None = None) -> str:
        return _j("qmine_build_comparison", {"run_id": run_id, "axis": axis,
                                             "title": title, "generation": generation})

    def qmine_preflight(inputs: list[str], run_id: str, domain: str | None = None,
                        config: str | None = None, fast: bool = False,
                        axis: str | None = None) -> str:
        return _j("qmine_preflight", {"inputs": inputs, "run_id": run_id,
                                      "domain": domain, "config": config,
                                      "fast": fast, "axis": axis})

    def qmine_start_run(inputs: list[str], run_id: str, domain: str | None = None,
                        config: str | None = None, fast: bool = False,
                        prepare: bool = False, axis: str | None = None,
                        confirm: str | None = None) -> str:
        return _j("qmine_start_run", {"inputs": inputs, "run_id": run_id,
                                      "domain": domain, "config": config,
                                      "fast": fast, "prepare": prepare,
                                      "axis": axis, "confirm": confirm})

    for fn in (qmine_capabilities, qmine_list_runs, qmine_overview, qmine_status,
               qmine_partial, qmine_findings,
               qmine_class, qmine_examples, qmine_tables, qmine_table, qmine_document,
               qmine_glossary, qmine_inspect_inputs, qmine_plan_corpus,
               qmine_preflight, qmine_estimate_cost, qmine_prepare_corpus,
               qmine_build_comparison, qmine_start_run):
        name = fn.__name__
        if name not in D:
            raise RuntimeError(f"{name} is served but not declared in _tools()")
        app.add_tool(fn, name=name, description=D[name])
    served = {fn.__name__ for fn in (
        qmine_capabilities, qmine_list_runs, qmine_overview, qmine_status, qmine_partial, qmine_findings,
        qmine_class, qmine_examples, qmine_tables, qmine_table, qmine_document,
        qmine_glossary, qmine_inspect_inputs, qmine_plan_corpus, qmine_preflight,
        qmine_estimate_cost, qmine_prepare_corpus, qmine_build_comparison,
        qmine_start_run)}
    undeclared = set(D) - served
    if undeclared:
        # A declared-but-unserved tool is a tool the model reads about in the
        # capabilities listing and can never call.
        raise RuntimeError(f"declared but not served: {sorted(undeclared)}")
    return app, qm


def pre_tool_use_hook(payload: dict[str, Any], run_root: str = "runs") -> dict[str, Any]:
    """Decide, for the harness, whether a `qmine_start_run` call may proceed.

    THIS IS THE ONLY REAL CONSENT CHANNEL. Every argument the server sees was
    written by the model, so nothing the server checks can distinguish "the
    person asked" from "the model decided". A `PreToolUse` hook can, because its
    `ask` answer is held by the harness at a dialog that is not model context
    and that only a human can click.

    So the hook does the one thing that makes that click meaningful: it runs the
    preflight and puts the verdict in the question. A `deny` when the run cannot
    work — nobody should be asked to approve a run whose id is already taken. An
    `ask` carrying the cost and the warnings otherwise, because "approve?" with
    no number attached is how people learn to click yes.

    Contract: Claude Code's `hooks.json` PreToolUse shape, which dsh bridges via
    `@deepseek-ai/dsh-hooks-claude-code` (`permissionDecision` of allow/deny/ask,
    folded most-restrictive-wins).
    """
    def out(decision: str, reason: str) -> dict[str, Any]:
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": decision,
                                       "permissionDecisionReason": reason}}

    name = str(payload.get("tool_name") or "")
    if not name.endswith("qmine_start_run"):
        return out("allow", "not a paid action")

    # A RELATIVE run root IS NOT RELATIVE TO US. dsh runs this hook from its own
    # working directory — the person's workspace, not the checkout — so `runs`
    # resolved there and the approval dialog told the operator about
    # `<their-workspace>/runs/.hf`. Anchor it to the checkout, which is where the
    # server that will actually launch the run resolves it.
    if not Path(run_root).is_absolute():
        run_root = str(Path(__file__).resolve().parents[3] / run_root)

    args = payload.get("tool_input") or {}
    rid = args.get("run_id") or "<unnamed>"
    try:
        from ..preflight import preflight

        pre = preflight(inputs=[str(p) for p in (args.get("inputs") or [])],
                        run_id=args.get("run_id"), domain=args.get("domain"),
                        config=args.get("config"), fast=bool(args.get("fast")),
                        axis=args.get("axis"), run_root=run_root)
    except Exception as exc:  # noqa: BLE001
        # FAIL CLOSED. A preflight that crashed has not cleared anything, and the
        # money is real.
        return out("deny", f"the preflight could not run ({type(exc).__name__}: {exc}), "
                           "so nothing is known about whether this run would work")

    if pre["blocking"]:
        lines = "; ".join(f"{c['name']}: {c['detail']}" for c in pre["blocking"])
        return out("deny", f"preflight NO-GO for {rid} — {lines}")

    # Issue the ticket BEFORE answering: it is the server's only evidence that
    # this gate ran at all for this call. Without one, `ask` mode refuses — which
    # is what turns a broken or absent hook from silent permission into a loud no.
    ticket.issue(run_root, str(rid), "ask", pre["summary"])

    cost = pre.get("estimated_cost_usd")
    calls = pre.get("estimated_calls")
    bits = [f"Start the mining run {rid}?"]
    if cost is not None:
        bits.append(f"Estimated ${cost:.2f} over {calls} model calls, several hours.")
    for c in pre["warnings"]:
        bits.append(f"• {c['name']}: {c['detail']}")
    bits.append("It runs detached and will outlive this conversation.")
    return out("ask", " ".join(bits))


def main(run_root: str = "runs") -> int:
    app, _ = build_app(run_root)
    asyncio.run(app.run_stdio_async())
    return 0
