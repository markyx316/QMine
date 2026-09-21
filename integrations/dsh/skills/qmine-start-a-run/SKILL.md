---
name: qmine-start-a-run
description: Plan, launch and watch a paid twelve-phase mining run — what must be settled first, why the tool may refuse, and what can honestly be said while it is still going.
whenToUse: Someone wants to mine a corpus, asks what a run costs, asks to start one, or asks how a run in flight is doing.
---

# Starting and watching a run

## 0. This is the layer that spends

A run makes real model calls and takes **hours**. Everything else in QMine is
free and re-runnable. Treat starting one as a decision the person makes with a
number in front of them, never as a step you take to be helpful.

## 1. Four things must be settled first

1. **The corpus is prepared.** One file, one `snapshot` column, the removed rows
   beside it. See the `qmine-prepare-datasets` skill. `qmine_start_run` can do
   this in the same call with `prepare: true`, and then the prepared corpus lands
   inside the run directory as evidence.
2. **The comparison axis**, if there is more than one snapshot. `time` and
   `stratum` produce identical arithmetic and opposite prose.
3. **A domain profile**, or the explicit acceptance that there is none.
   `configs/domains/` holds the risk-screening profiles. No profile means Chinese
   defaults with **no vertical risk screening** — say that out loud rather than
   letting it pass.
4. **The cost.** `qmine_estimate_cost` gives calls and dollars. Show it.

## 2. Full or fast

`fast` drops the **second-opinion layer** and nothing else. Grids and gold set
stay full size; every intermediate artifact is still written. The consequence is
specific and must always be stated: **kappa is ABSENT, not 1.0**, and
`fast_skipped` in `run_summary.json` names precisely what went unmeasured.

`fast` is the right choice when the question is "what is in this corpus". It is
the wrong choice when the answer has to defend its own reliability.

(`--smoke` is different again: it shrinks the *analysis* to check wiring. Its
output is never a result. It is not reachable from this conversation.)

## 3. Preflight, always

`qmine_preflight` answers "will this work" for free: the run id is free, every
input exists and has a query column, the config and domain profile load, every
agent role routes to a model somebody holds a key for, the cost, the disk.

`verdict: no_go` means stop — `blocking` lists what would burn the money and
return nothing. `warnings` are things to say out loud rather than reasons to
stop: no domain profile means no risk screening, `fast` means kappa will be
absent, an unpriced model means the estimate is a floor.

`qmine_start_run` runs the same checks again and refuses on any blocking one, so
skipping the preflight does not get a run started sooner — it just means the
person never saw the cost.

## 4. The tool will probably refuse, and that is correct

`QMINE_MCP_ALLOW_SPEND` decides, and it is read from the environment that
launched the server — nothing said in the chat can change it, including the
person saying it is fine.

| posture | what happens |
|---|---|
| unset / `0` | refused; you get the command for a person to run |
| `ask` | a run may start, but only after the preflight passes and an approval gate was consulted for it |
| `1` | a run may start once the preflight passes |

Under `ask` the harness holds the call at its own dialog, which shows the cost
and the warnings and which only a person can answer. **Waiting there is correct.**
A rejection is an answer: report it and stop.

When it refuses it hands back the exact command. Give that to the person
verbatim, with the cost estimate. Do not look for another route: there is a
reason permission lives outside a channel a model can talk its way through.

## 5. It returns immediately — that is not a failure

A run takes hours, so the tool starts it detached and returns a pid. Nothing has
been produced yet.

- `qmine_status` — the phase it reached, how long it has been going, gates so
  far, model calls and tokens spent so far, which artifacts exist, and
  `what_can_be_answered_now`.
- `qmine_partial` — one intermediate artifact, named by `what`. Take the names
  from `what_can_be_answered_now`; anything else is not written yet.
- `live_dashboard` — a path the person can open in a browser tab. It refreshes
  faster than anyone would poll, so hand it over early.

## 6. What may be said mid-run

Only what has actually been written, labelled as intermediate.

- A status is a **snapshot at `as_of`**. A run moves; never restate an old
  status as current. Call again.
- A run with no `run_summary.json` **has not finished**. It has no results. The
  taxonomy can still change, the tree is rewritten at p8, and the delivered
  partition does not exist until p10.
- `do_not_report_results: true` in the status payload means exactly that.

"Phase 2a, five researchers fanning out, three gates passed, one warned, 2,584
tokens" is a good mid-run answer. "It's finding that users mostly ask about X" is
not, however tempting the partial corpus makes it.

## 7. When it finishes

`qmine_overview` first — mode, provider, gates, shape. Then, for a pooled run,
`qmine_build_comparison` if `has_cross_snapshot_comparison` is false. Then read
it with the `qmine-answer-from-a-study` skill.

If a run halted, `qmine_overview` carries `halt_reason` and the failing gate. A
halted run's partial artifacts are evidence about the run, not findings about the
corpus.
