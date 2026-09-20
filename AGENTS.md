# QMine — what is in this directory

Workspace instructions for an assistant whose working directory is this
checkout. The methodology, the vocabulary and the rules for stating a result
come from the QMine preset's own persona; this file is only the map of what is
on disk, so nothing has to be discovered by listing directories.

`CLAUDE.md` in the parent directory is for whoever is **changing** the program.
It is not guidance for answering questions about results.

## The four layers

| layer | command | costs | produces |
|---|---|---|---|
| Prepare | `qmine prepare A,B,C --out DIR` | nothing | one pooled corpus + every removed row with its tier |
| Mine | `qmine run --input CORPUS --run-id ID` | **money, hours** | two independent label systems over every row |
| Compare | `qmine compare ID` | nothing | per-class cross-snapshot tables, figures, report, workbooks |
| Ask | `qmine chat` · `qmine mcp` | nothing | answers about a finished study |

`qmine` lives at `.venv/bin/qmine`. Every Python invocation here wants
`HF_HOME=$(pwd)/.hf` in front of it, or model downloads land outside the project.

## Directories

| path | what it holds |
|---|---|
| `runs/<id>/genNN/` | every study: intermediate artifacts, deliverables, `run_summary.json`, and `pooled/` when a cross-snapshot comparison was built |
| `runs/<id>/run.log` | what the run did, whatever the UI was doing |
| `data/raw/` | bundled and supplied corpora |
| `configs/` | `live.yaml` is the default and routes to real models; `domains/` holds the per-vertical risk-screening profiles |
| `analysis/` | hand-built post-processing studies; `pooled5/` is the one the Compare layer generalises |
| `src/qmine/` | the program: `graph/` the phases, `pooled/` the comparison, `prepare/` the corpus planner, `mcp/` this chat surface |
| `integrations/dsh/` | the chat front door — the agent preset, its persona, and these skills |
| `tests/` | ~870 tests; each pins a defect that actually happened |
| `tools/verify_run.py` | mechanical checks over a finished run, against a known-broken control |

## Documents

| file | what it is | when to reach for it |
|---|---|---|
| `GUIDE.md` | the user-facing path from a pile of exports to results | someone asks what to do with their data |
| `README.md` | install, the commands, the chat front door | setup questions |
| `../Universal_Query_Mining_Playbook.md` | the method spec, in Chinese | "why is it done this way" |
| `../BottomUp_Approach_Final_Report.md` | the reference deliverable | "what does the output look like" |
| `HANDOFF.md` | a dated log of state and open questions | never authoritative about behaviour |
| `TOPDOWN_PATH_EXPLAINED.md` | the top-down route in detail | taxonomy questions |

## Reading a run without opening files

Prefer the `mcp__qmine__qmine_*` tools over the filesystem. They apply the quote
guard, the rounding and the caveats; the raw artifacts do not. `runs/` holds
dozens of studies and a single report runs to hundreds of lines.

Two things to check in `run_summary.json` before quoting anything from a run:
`mode` (`fast` means kappa is absent, and `fast_skipped` says what else) and
`llm_usage.provider` (`offline` means a deterministic stand-in wrote it, and it
says nothing about the corpus).

## Never

- Quote a corpus row that did not come back from `qmine_examples` or
  `qmine_table`. Those apply the seven-layer quote guard; nothing else does.
- Quote a row from a never-quote class — a named doctor, a private individual's
  details, sexual content, anything involving minors, debt collection, credit
  repair, cash-out — even when the guard returned it. The guard is a floor.
- Show a specimen of what you declined to show. "Rows like 〈x〉 were left out"
  publishes 〈x〉. Give the class, the count and the source instead.
- Edit anything under `runs/` or `analysis/`. Those are delivered evidence.
- Change source, configs or tests from this conversation. This is the front door,
  not the workshop — the person who wants a code change should open the project
  in a coding agent.
