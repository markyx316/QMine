# QMine — 交接说明 / Handoff

> **How to maintain this file** (the contract `CLAUDE.md` refers to):
> - **§1 Status** — overwrite every session. It must always describe *now*.
> - **§2 Open questions** — **edit, never append.** Resolve an item → delete it here and
>   record the resolution in that session's section below.
> - **§3 Durable notes** — things worth not re-learning. Rarely changes.
> - **§4+ Session log** — append a new dated section per session. Never edit old ones.
>
> This file is a **log, not a specification.** When it disagrees with the code or the
> tests, the code and tests are right. Verify before relying on anything here.

---

## 1. Status — last updated 2026-08-21 (evening)

| | |
|---|---|
| Tests | **226 passing**; `ruff --select F` clean |
| Verified **live** | p0–p2a on the full 50k, five times |
| Verified **offline** | all 17 phases on the full 50k, real encoders |
| Never run live | **anything past p2b** — referee, adversary, l2, namer, reporter, maintainer |
| Live spend today | ~$15 across five runs, priced correctly for the first time |

**Paused mid-p2a, deliberately.** `live35` holds **59 cached calls** — all five
researchers (including both web ones), the architect, the critic, and all 48 pilot
and re-pilot annotations. Resuming replays the researcher fan-out in ~1 second
against the ~10 minutes it originally cost. Only the architect call that was
in flight when it stopped will regenerate (~8 min).

Nothing is broken and nothing is half-written: the cache is append-only and the
artifacts on disk are from the completed p2a of the previous attempt.

```bash
cd QMine
# Re-run an existing id: drop the CHECKPOINT, keep the CACHE. They are separate.
rm -f runs/live35/checkpoints.sqlite*
HF_HOME=$(pwd)/.hf caffeinate -i .venv/bin/qmine run \
  --input data/raw/k12_queries_50k.csv --domain k12_zh \
  --reference-columns legacy_l1,legacy_l2 \
  --config configs/live.yaml --provider router --run-id live35 --plain
```

`configs/live.yaml` carries the provider policy: labs excluded by LAB, and the three
reasoning roles pinned to `deepseek-v4-pro`. Pass `--domain k12_zh` **as well** — a
config file no longer swallows the domain, but the flag is what selects it.

---

## 2. Open questions — EDIT THIS SECTION, DO NOT APPEND

0. **The "`qmine models` under-prices by 5x" claim was WRONG and is withdrawn.**
   It came from comparing the planner's estimate (real catalogue prices) against
   `UsageLedger.estimated_cost_usd`, which hardcoded $3/$15 per million — frontier
   rates — while runs were on models costing $0.15–$1.32. Every cost this project
   ever reported was inflated ~7-11x; `live32` was reported as $4.68 and cost
   $0.67. The ledger now prices from the routing plan. **Re-derive the planner's
   accuracy against a corrected run before trusting or blaming it.**

1. **Does the guide repair help? First clean answer: no measurable effect.**
   `live20` produced the first uncontaminated comparison — both rounds at **n=596**,
   full annotator coverage on both sides, all 28 repair rules delivered:

   ```json
   [{"round": 1, "kappa": 0.8335, "n": 596, "raw_agreement": 0.849, "sample": "initial"},
    {"round": 2, "kappa": 0.8311, "n": 596, "raw_agreement": 0.849, "sample": "fresh"}]
   ```

   **Δκ = −0.002**, against a standard error of roughly 0.016 at this n — about
   0.1 SE, i.e. flat. Raw agreement is identical to four decimal places (0.849 both
   rounds), which is the more striking number: the two rounds are indistinguishable.

   So the machinery works — 8 of 8 open boundaries decided, 28 of 28 rules delivered,
   the guide rewritten, a fresh disjoint sample annotated — and **agreement does not
   move**. The honest reading is that marker-based boundary rules are not the binding
   constraint on annotator agreement here, which is consistent with the earlier
   finding that this corpus's disagreement is *diffuse* across ~40 pairs rather than
   concentrated in a few.

   Open for tomorrow: is the effect genuinely zero, or too small to see at 596? At
   3,000 rows the resolvable effect size drops to roughly 0.015. Worth one run to
   find out — and if it is still flat, the repair loop should probably be demoted
   from a gate remedy to a diagnostic that reports contested boundaries without
   claiming to fix them.
2. **The comparison design is still confounded.** Round 1 is scored on sample A and
   round 2 on sample B, so the delta mixes "guide improved" with "sample differed".
   The clean fix is a **control arm** — annotate sample B with the *old* guide too —
   at the cost of one extra annotation pass. Not built; the code now at least refuses
   to present the two numbers as comparable (`comparable: false` in the artifact).
3. **`_dedupe_rules` cannot see semantic contradictions.** It catches lexical
   duplicates; two rules that mean the same thing in different words slip through.
   The serialized referee is the mitigation, not the filter.
4. **Three domain profiles are untested on real data:** `finance_zh`, `sports_zh`,
   `politics_zh`.
5. **The redraw loop made agreement WORSE on its first real trial.** `live35`:
   kappa 0.8437 -> 0.8272 (-0.0165) after rewriting the six boundaries one
   annotator could not reproduce; the ceiling did not move (0.9243 -> 0.9239). It
   dropped no classes and added none — it rewrote definitions in place and the
   rewrite was worse. The revert guard caught it.

   This is evidence AGAINST the hypothesis the loop was built on: that a pair one
   annotator cannot reproduce means the boundary is not in the data, so redrawing
   is the only remedy. One trial at n=200, so not decisive — but it is the only
   controlled before/after this project has, and it points the wrong way. Do not
   describe the redraw as a fix until a second trial says otherwise.

6. **Is `annotator_b` worth its tokens? Still untested.** The reasoning model
   emits 6-20x the tokens of its partner for identical work and dominates
   wall-clock. `qmine_annotator_worth.py` (scratchpad) tests it: on every row the
   two split, the referee's verdict says which annotator was right. Needs a gold
   set, which no run has produced.

7. **Error classification reads PROSE, and it has bitten three times in one day.**
   A 402 hidden in `completion_tokens=4013`; a schema miss vs a truncation; a
   truncation vs a dead provider. Each fix was narrower than the last, but where
   the SDK raises typed exceptions the classification should key off the TYPE and
   use strings only as a fallback. Deliberate change, not a patch.

8. **The guide-repair question is half-answered and half-reframed.** The
   Δκ = −0.002 result above stands, but it was measured with the annotator seeing
   **one** adjudication rule: `_render_rules` rendered only the top-level list
   while 55 per-class rules sat unread in the artifact. Any conclusion about
   whether rules help was drawn from a near-ruleless condition. Re-open it once a
   run reaches P2b with the renderer fixed.
9. **`positive_examples` overlap cannot be detected mechanically.** A check for
   "do two classes claim the same example" scored **zero on both real taxonomies**
   — models make semantic overlaps, not syntactic ones. The pilot's
   self-consistency pass is the only detector we have. Do not rebuild this.
10. **The architect is high-variance and one probe does not predict a run.** A
   $0.52 probe on stored submissions produced 19 classes with a consistent basis
   of division; the live run on the same corpus with the same prompt produced 20
   classes across fourteen different bases. Treat a single probe as a smoke test,
   never as evidence a prompt change holds.

---

## 3. Durable notes — worth not re-learning

- `runs/*/llm_cache` is keyed on `(role, provider, model, system, user, schema)` and
  **not** on `max_tokens` — so token-budget changes do not invalidate it. Copying a
  cache directory into a new run is a legitimate way to skip replayable work.
  Web-research calls will still miss, because the fetched content differs each time.
- The offline stand-in produces a **degenerate taxonomy** (one node,
  `[offline-heuristic] code`), so `td_l1_name` is legitimately empty in offline runs.
  The code reports that rather than shipping a column of blanks.
- `completed_phases` uses an `operator.add` reducer — it cannot be pruned via
  `update_state`. Rewind by graph **position** (`as_node=<predecessor>`) instead.
- `qmine` has no `new-generation` command; `runner.new_generation()` exists but is
  not exposed on the CLI.
- Verify a live run really used live agents: `run_summary.json` →
  `llm_usage.provider` must read `routed`, not `offline`.

---

## 4. Session 1 (2026-08-18) — deliverables, providers, guide repair

### Deliverables now match the reference documents

- **`report/zh_figures.py` (new)** — the six-figure suite modelled on
  `K12_Embedding_Attempts_Comparison.ipynb`: K-sweep three-panel, α-decision with a
  phrasing-vote secondary axis, algorithm-battery scatter, embedding-space
  projections, per-intent split with `exp(H)` effective-family counts, and the
  uniform panel as bars with advisory metrics hatched. The notebook had **one**
  figure before; the reference carries six.
- **Figure production unified.** The report and the notebook each drew their own
  K-sweep, α, UMAP and panel charts — two different pictures of the same number in
  one deliverable. The notebook now executes first and its figures fill those slots,
  with `ops/viz` as fallback. Guarded by `test_one_figure_per_quantity`.
- **Both routes are equal citizens in `labels_full.csv`.** Bottom-up shipped 8
  columns including a margin and an ambiguity flag; top-down shipped 2 bare codes.
  Top-down now also has `td_l1_name`, `td_user_need`, `td_confidence`, `td_margin`,
  `td_ambiguous`, `td_decided_by` (`rule` vs `model`). `td_ambiguous` uses the same
  threshold as `bu_ambiguous` so the word means one thing.
- **`route_crosswalk.csv` (new)** — per bottom-up family: dominant top-down L1, its
  share, classes touched, `exp(H)` effective classes, and a verdict of
  `routes agree` / `partial overlap` / `routes disagree`. Neither route is an answer
  key, so the useful artifact is a map of where they concur.

### Gold-set sizing — a real deviation from the playbook, now fixed

`gold_sample_size` was hardcoded at **600**. The playbook says
`分层抽 3,000-5,000 条` (line 191) and separately `<1 万条 → 金标比例提高`
(line 119) — a small corpus needs a higher *proportion*, not the same absolute
count. A constant satisfied neither.

`config.gold_size_for()` now derives it: 3,000 on a 12k+ corpus, a rising share
below 10k, capped so gold can never become most of the corpus. Pin
`taxonomy.gold_sample_size` to an integer to force a cheap run.

**`p2a_pilot_agreement` now exists.** It was in the blocking-gates list and emitted
by no node. 50 queries, four LLM calls, hard stop below 85% agreement, reporting the
top confused pairs. This is what makes a 3,000-row gold set affordable — it catches
an ambiguous guide before the expensive annotation, per
`一致率 <85% 则回炉改指南/裁决规则, 而非直接开标`.

### The guide-repair loop (playbook `达不到先修指南再重标`)

Was absent entirely: κ was measured once, *before* the referee's rules existed, and
the gate's own `remediation` string said "fold the referee's rules into the guide and
re-annotate" — printed, never executed.

Now, when κ misses: find boundaries the referee resolved **inconsistently**, settle
each from rows **both annotators agreed on** (`discriminating_markers`, greedy set
cover, deterministic tie-break), add a `boundary_default` for marker-less queries
only where the agreed rows lean ≥75%, rewrite the guide, and re-annotate a **fresh
disjoint sample**.

**Known open question — see §4.** Whether this actually raises κ is still unproven.

### Provider and infrastructure fixes

| Defect | Effect | Fix |
|---|---|---|
| `resolve_base_url` cached a host after *every* probe errored | One transient failure pinned the process to a host that had just 401'd — 48 auth errors, an annotator lost 24 batches mid-run | Never cache a host we have evidence against; serialise the probe behind a lock |
| Global `max_tokens=16000` overrode per-role budgets | Taxonomy architect truncated at exactly 16001 tokens, twice | Role budget is authoritative, bounded only by the model's published ceiling |
| `_native_schema_is_broken` only matched `ValidationError` | DashScope's `response_format unavailable` and `must contain the word 'json'` were never learned; every call wasted an attempt | Both patterns added |
| Annotator batch failure → `return {}` | 400 rows became `UNLABELED` silently | Retry ×3 with backoff, then report the loss loudly |
| `agreement()` counted `UNLABELED` as disagreement | Charged infrastructure failures to the methodology; cost 0.008 κ | Excluded and counted as `n_unscored_unlabelled` |
| `p2b_kappa` reported a verdict at 33% coverage | "κ 0.813" read as a judgement on the guide; it described the survivors | `min_annotation_coverage` (0.90) → `MEASUREMENT UNSOUND`, neither pass nor fail |

### Correctness fixes found by running, not reading

- **`stratified_sample` returned index *labels* in its stratified branch and
  *positions* in the other two.** They coincide on a `RangeIndex` — which every
  caller had always passed — so passing `df.iloc[unseen]` blew up with
  `IndexError: index 11410 out of bounds for size 11400`. Fixed at the contract:
  positional in all three branches.
- **`select_active_learning_batch` called `len()` on a scipy sparse matrix.** Round-2
  active learning had *never once run* in this codebase; the exception was swallowed
  into a one-line log message. Fixed, plus the caller's `diversity_fraction=0.0`
  workaround removed — the diversity pass works fine on sparse, it was routing around
  a bug one line earlier.
- **The notebook executed against the wrong Python.** `kernel_name="python3"`
  resolved through the user's Jupyter kernelspecs to an unrelated project's venv
  lacking `pyarrow`; cell 1 raised and every figure below silently never drew. Now
  pinned to `sys.executable`.
- **`_dedupe_rules` was a rule-set shredder.** It compared rendered `when` sentences,
  where two markers for one boundary differ by ~2 characters in 45 (0.957 similar).
  It flagged every legitimate discriminating pair as a contradiction and withheld
  *both* halves — **9 of 41 rules survived**. Now compares structured
  `(class pair, trigger)` keys. Measured live tonight: **28/28**.
- **Rules could name a class that does not exist.** `R12 → 选 EXOD_INFO` against a
  taxonomy declaring `EXAM_INFO`, rendered verbatim into both annotators' prompts.
  Now validated at 2a and repaired by a dominance test.
- **A failed referee batch stamped `adjudicated=True`** on rows with no referee
  behind them, silently favouring annotator A. Now left visibly unresolved.
- **Resume could not tell a crash from a refusal.** Three separate gaps: it replayed
  the halt instead of retrying; clearing the flag still skipped the phase because the
  checkpoint stores *position*; and a cleared-but-unrewound checkpoint reported
  **success for a pipeline that stopped at phase three**. Now `halt_kind`
  (`crash`/`gate`/`review`) plus position derived from `completed_phases`.

---

### Live attempts — what each cost and taught

| run | outcome | cost | lesson |
|---|---|---|---|
| live01–02 | died at P2a | ~$7 | fenced JSON; 16k token truncation |
| live05 | crashed in the repair round | $6.72 | `stratified_sample` label/position contract |
| live10 | halted p2b, κ 0.813 on **199/600** | $7.39 | poisoned region cache; κ was an artifact, not a result |
| live20 | halted p2b, κ 0.831 on **596/596** — a sound measurement | $16.67 | first clean repair readout: **Δκ = −0.002**, i.e. no effect |

Roughly **$38** spent on live runs. Every failure was a distinct real defect, all
fixed with regression tests.

---

## 5. Session 2 (2026-08-19) — decision architecture, audit trail, portability

**Verdict on the playbook: mostly right, targeted repair — not redesign.** Five
independent audits agreed. The phase ORDER is correct and a joint grid search is
measurably *worse* (held-out ARI 0.653 vs greedy 0.739 at 3.4-4.7x the cost;
measured directly: 819 s for a 50-cell (alpha x K) surface on only 4k rows, and
greedy's alpha was already optimal at the final K). The defect was never the
sequence — it was **reading single noisy draws as truth**.

### K selection was noise
Replay stability's seed-to-seed sd is ~0.10; the gaps between adjacent K are ~0.05.
Four draws at one K gave 0.63 / 0.60 / 0.38 / 0.69. A tie-aware selector returns
**the whole grid**. It is also degenerate: K=2 scores ARI **1.0000** on both the 8k
and the real 50k corpus, so only the `k_sweep` list's hardcoded lower bound stood
between the pipeline and a two-way split. `expected_family_range` never constrained
anything.

**Now:** stability only *rejects* (a reproducibility floor — the role the literature
gives it); K is *located* by **AMI against the phrasing groups**, the one metric here
with a two-sided penalty and therefore an interior optimum (0.427 at K=2 → 0.724 at
K=25 → 0.664 at K=100), and **~10x more precise** (sd 0.005-0.023). It yields a
unique winner where stability ties everything. Runs now ship a **tie set** — the
honest answer, and the "several equally-good results" deliverable.

Dead ends not worth repeating: raw fragmentation is *structurally* monotone in K
(rho +0.97…+1.00 in all 13 runs) so it prefers K=1; the chance-adjusted version
(`adjusted_template_fragmentation`, implemented) prefers K=∞. Neither locates K.
Both are valid at *fixed* K, which is why alpha selection was always sound.

### Other confirmed defects, fixed
- **The panel measured the wrong object.** `stability_ari` was `replay_stability(X, k)`
  — corpus and cluster count only — so the "decisive" number on the delivered leaves
  described a fresh KMeans run and was pessimistic by ~0.25 ARI. Now
  `partition_stability`: half-sample centroid replay on the actual partition
  (leaves 0.893 ± 0.007 vs the 0.640 previously reported).
- **The algorithm battery was decorative** — `build_hierarchy` hardcodes KMeans, and
  the report announced `gmm_diag_k15` as the winner while arguing why KMeans is
  right. Now a falsification probe, and the report says so.
- **The pilot gate was dropped on the floor** — `deps.gate(...)` called without
  capturing the return, so it could never halt anything. My own
  `declared_gates_never_evaluated` diagnostic named it in five runs.

### The audit trail is now IN the deliverable
Previously the run recorded 7 decisions, 10 gates, 5 prescriptions and 332 agent
calls, and the report rendered roughly half the decisions and none of the rest.
Added as sections 9-11, in execution order:
- **§9 全流程决策链** — every decision with candidates, winner, who decided, decisive
  metric, full rationale, and the rejected options with their numbers. Plus
  `fig_decision_chain` (candidates → survivors per step).
- **§10 质量门总账** — every gate with observed vs threshold and its remediation, plus
  gates declared blocking that never fired. Plus `fig_gates` (headroom, normalised).
- **§11 治理台账** — every prescription's final disposition.

All authored rationale/remediation prose is translated via `report/i18n.prose()`,
guarded by `test_every_authored_rationale_reaches_the_reader_in_the_report_language`
which fails on any untranslated line reaching a Chinese report — it caught a string
I added minutes later.

**State:** 134 tests pass; report 15 sections / 664 lines / **0 lines of English**;
11 figures; notebook executes with 0 errors.

---

## 6. Session 3 (2026-08-20) — CLAUDE.md, portability, and the first live 50k run

### Pre-flight paid for itself
A **full-50k offline run** (`runs/preflight50k`) completed all 17 phases in 21.7
minutes, clean — the first time the whole pipeline has run at real scale with the
current code. Before that, an audit of `live20`'s per-role usage caught two defects
that would have failed the paid run:
- **`taxonomy_architect` at 99% of its cap** (23,759 of 24,000) — the role that had
  already truncated twice.
- **`annotator_b` exceeding its cap outright** (11,910 vs 4,500), because reasoning
  tokens are billed and capped as output while its partner emitted 1,439 for the
  same task.

### The live run: $3.02, halted at p2a, and worth it
`live30`, full 49,999 rows, real agents. The **annotator-ceiling pilot** — built
that morning on reasoning alone — returned its first real measurement:

```
pilot: kappa 0.761 (95% upper 0.814) on 200 queries
       annotator self-consistency kappa 0.8997 (0.8457 of ceiling reached)
top confusions: QUERY_POETRY_TEXT × QUERY_WORD_USAGE, EXPLAIN_WORD_MEANING × QUERY_POETRY_TEXT
```

Two annotators agree at 0.761; **one agrees with itself at 0.900**. So the 14-point
gap is guide ambiguity, not model noise — the *fixable* branch, established rather
than assumed. Yesterday the same situation cost $16.67 to reach a worse-informed
conclusion.

**Root cause:** the architect shipped **19 classes and 1 adjudication rule**. No
truncation (20,441 tokens against a 42,000 cap) — it simply followed
`"Aim for at least {{MIN_RULES}} rules"`, which is exactly the soft phrasing
Anthropic's own guidance says gets ignored.

### Three fixes, each of which exposed the next
Found by ~$0.05 probes against `live30`'s stored submissions, not by paid runs:
1. `"Aim for"` → **`YOU MUST`**. Result: 1 rule → 24. But classes went 19 → **2**,
   and the rules named a dozen classes that did not exist. Emphasis raises
   adherence to the emphasised instruction *and lowers it for what competes*.
2. Made the two requirements **joint**. Result: 20 classes, 22 rules — but only by
   emitting **42,001 tokens**, hitting the ceiling, and recovering via the
   plain-JSON repair path.
3. **Split the call**: architect writes classes, a new `RuleWriterAgent` writes
   rules *shown the finalised class list*. Probed: 19 classes, 22 rules,
   **22/22 naming a real class**, 731s. The invalid-rule failure is now structurally
   impossible rather than discouraged.

### Also this session
- **`timeout_seconds` is derived from the generation cap.** Raising the architect's
  cap to 42,000 left it 420s to emit them — at ~49 tok/s measured, about half what
  it needs. Two independently-tuned constants that only make sense together.
- **Three gates de-imported from K12.** Coverage gates on *rows* not a share (the
  e-commerce corpus was flagged for being more templated); held-out is bounded by
  the partition's own reproducibility; coherence reads the **weak tail**, not the
  mean (a tree with 16 good leaves and 4 incoherent ones passed on average).
- **`p2a_taxonomy_shape` now has three outcomes**: missing rules halt, a wildly-off
  class count halts, a near-miss warns.
- **`CLAUDE.md` written** (106 lines) plus three path-scoped `.claude/rules/` files.
  Verifying it caught two globs that matched **nothing**.

---

## 7. Session 4 (2026-08-21) — the operator's view, and a gate that could be won

Three live runs today (`live31`, `live32`, `live33`), 176 tests (was 141).

### The dashboard existed and had never been rendered
241 lines, on by default, zero tests, never once looked at. Rendering a real
`run.log` through it found six defects in twenty minutes: the two panes never
split (`Columns` sizes by content, so it stacked); Rich ate `researcher[log_reading]`
as markup so three agents showed as three identical lines; `P3a/b/c` were emitted
while `p3` was declared, so that row could never light up; gate notes were cut
mid-word at 70 chars; metric labels were sliced mid-token; and a *halted* run left
its last phase spinning at ◐, because a blocking gate returns rather than raises.

Then rendering the **live** log found a seventh the recording could not: eight
concurrent batches fail identically in one second, each wrapping to two display
lines, so one benign already-handled error filled all six activity slots and
pushed out the progress. **Build against a recording, re-render against live
traffic** — a clean run has no failure *concurrency*.

Also added: `qmine watch RUN_ID` (the panel reads `run.log`, so a run can be
launched detached and still watched), per-phase explanations, and an agent panel
showing role · model · elapsed · out-tokens · *what it returned*.

### `run.log` did not exist at all
The CLI quieted the **logger** to give the panel the screen, and there was no file
handler anywhere — so choosing the pretty view meant choosing to have no record.
Levels belong on handlers. Fixing it exposed that `open_run()` had **zero callers**:
34 lines duplicating the resource setup of the three functions that are real. My
first fix went into it and did nothing.

### The cost ledger was optimistic exactly where things go wrong
Three instances of one pattern: `ToolAgent.run` recorded a hardcoded
`output_tokens=0` for every tool loop — so the budget ceiling was blind to the one
path that *iterates*; `complete()`'s failure branch recorded zero for responses the
provider had already generated and billed; and `qmine models` still assumes one
output-tokens-per-call figure across roles (annotator_a 1,439, annotator_b 12,435).
First two fixed. Tool-loop turns also now write a cache entry and a transcript
entry — the web-researching agents were the only ones leaving no record of what
they said, which is a poor property for the agents citing pages nobody else saw.

### `TaxonomyNode.adjudication_rules` was write-only
Declared to hold rule *ids*, filled by the models with rule *text*, and read by
**nothing**. `_render_rules` rendered only the top-level list. Measured recovery:
`live30` would have shown the annotator **42** rules instead of 1; `live31` 70
instead of 46. This reframes `live30` retroactively — its κ 0.761 was achieved with
one visible rule, and the shape gate's "1 adjudication rules" was accurate about
what reached the annotator while 55 sat unread.

### The three taxonomy defects, measured
Replaying all 600 pilot labels out of `live31`'s cache separates intrinsic
ambiguity from fixable guide gaps — a query where annotator A disagrees with
*itself* is a boundary not in the data. That split 57 disagreements into **36
structural / 27 guide**, and attributed ~half to three corpus-independent defects:
overlapping siblings, siblings cut on different bases, and a catch-all defined by
content. The architect prompt was **requiring** the third ("a catch-all must be
defined by what it *is*") and simultaneously telling the architect both to write
and not to write adjudication rules.

A first hypothesis — that the LOOKUP/EXPLAIN *axis* was the problem — did not
survive its own significance test (z≈1.0). Recorded because it looked convincing.

### The κ gate was measuring its own confidence interval
The playbook's ≥0.9 came with "K12 达 0.966" — a floor beneath what *that* project's
annotators reached. Ours self-agree at 0.883. Worse, the gate tested the *upper*
bound, so the bar moved with the pilot size:

| pilot n | κ demanded |
|---|---|
| 50 | 0.801 |
| 200 | 0.857 |
| 3000 | **0.890 — above the ceiling, unwinnable** |

Now two independent conditions: **annotator fitness** (`ceiling ≥ 0.80`, the
conventional reliability threshold applied to the quantity it describes) and **no
significant recoverable slack**. 0.90 is reported as the playbook's aspiration.

### P2a can now redraw and re-pilot
`TaxonomyRedrawAgent` is shown the current taxonomy and the pairs one annotator
could not reproduce, and told to merge or re-cut *those* and leave the rest
byte-identical — not the architect, which rebuilds from evidence and re-rolls the
classes that were fine. Bounded at 2 rounds, reverts any redraw that lowers κ.

Extracting it into `_redraw_until_stable` to make it testable immediately found
two bugs that lint and the offline run both passed: the revert filtered the
*redrawn* nodes by the old codes (keeping new definitions under old names), and
the `return` sat inside the `for`, so the loop exited after one iteration on
success and returned `None` on any break.

### Results
| | live30 | live31 | live32 |
|---|---|---|---|
| pilot κ | 0.761 | 0.688 | **0.777** |
| ceiling | 0.8997 | 0.8023 | **0.883** |
| citable rules | 1 | 46 | **106** |
| prescriptions | 0 | 0 | **12** |

Predicted κ≈0.84 / ceiling≈0.90 from the defect attribution; got 0.777 / 0.883.
Direction right, magnitude about half. All three halted at `p2a_pilot_agreement`.

### Evening: OpenRouter, and five more live runs

An OpenRouter key was added mid-session. It changed more than it looked like it would.

**Every cost figure this project ever reported was fiction.**
`UsageLedger.estimated_cost_usd` hardcoded `in_rate=3.0, out_rate=15.0` per million —
frontier rates — while runs were on `deepseek-v4-flash` ($0.44/$1.32) and
`qwen3-next-80b` ($0.15/$1.20). `live32`: reported **$4.68**, actual **$0.67**. The
ledger now prices from the routing plan, and names any role priced by the fallback
rather than letting a guess read as a measurement. Fixing it exposed a second bug in
the fix: roles arrive suffixed (`researcher_log_reading`) while the plan is keyed on
the base role, so exact-match lookup dropped four roles back onto frontier rates.

**Independence was checked on the gateway, not the lab.** With an aggregator in the
pool `zhipu/zai/glm-5.1` and `openrouter/z-ai/glm-5.3` read as independent and are
one lab. `lab_of()` now resolves the originating lab, applied to the primary choice
AND the fallback chain — a fallback within one lab is one outage and one architecture.
The referee must now differ from BOTH annotators, not just annotator_b from annotator_a.

**Three things the price-as-capability proxy did, only one of which was intended.**
Removing price from `_assign_tiers` was tried and reverted three times: each attempt
let something worse win every role — a date stamp parsed as version 28
(`qwen-flash-2025-07-28`), then `:free` variants, then `openrouter/auto`, a
meta-endpoint. Price was also silently excluding those. Those exclusions are now
explicit in `_eligible` (`:batch`, `:free`, preview, unpriced) and the ordering stays
priced. The narrow fix for the real complaint — a newer model rejected for being
cheaper — is a same-LAB generation upgrade after scoring.

**Failover exists now**, and cost three attempts to get right. A `402 Insufficient
Balance` mid gold-annotation took twelve batches while two declared fallbacks sat
unused. Then the classifier killed a whole provider on
`CompletionUsage(completion_tokens=4013, prompt_tokens=8402)` — "4013" contains "401".
Then a truncation was misread as a dead provider when the remedy was more room.

**`glm-5.2` truncates in native structured-output mode and does not need to.**
Measured four times: truncates past 12,000 tokens natively, completes in ~5,300 on the
plain-JSON path — the same answer for less than half the tokens. A truncation now
raises the cap AND abandons native mode, both keyed by MODEL so one discovery serves
every role. Previously five researchers each paid ~180s to learn it separately.

**Latency: the gateway, not the model.** Same `deepseek-v4-flash`, same phase —
OpenRouter median 67.8s / max 429.9s, direct 83-98s with no tail. The router now
prefers the DIRECT route when the same bare model name is reachable both ways. Costs
~70% more on the estimate; buys back roughly two hours of wall-clock on a gold phase.

**The gate proceeds once its remedy is exhausted.** `live35` sat at kappa 0.844 with
0.080 of significant slack and a redraw that had run and failed. Halting there asks
the operator to do by hand what the pipeline just could not, while 0.844 is above the
reliability floor. It now passes — narrowly: the redraw must have RUN and FAILED, and
kappa must clear the floor. The message and `run_summary.json` both record that it
proceeded with residual slack.

**Re-running a run id: three separate traps, all new because we had never done it.**
The CHECKPOINT carries `halted=True` and exits in 3.1s without re-reaching the gate —
delete `checkpoints.sqlite`, keep `llm_cache/`. The TOOL path wrote cache entries and
never read them, so the two web researchers re-fetched live pages and cascaded a miss
through everything downstream (this is what defeated the `live33` resume too).
And `qmine watch` treated ANY `run_summary.json` as "finished", so it exited within
seconds of attaching to a re-run. All three fixed; the researcher fan-out now replays
in **one second** against ten minutes.

**Also:** `--config` silently discarded `--domain`, swapping k12_zh for `generic` and
halving template coverage in a deterministic phase with no error anywhere — caught
only because 18,298 had been read three times that day.

### Results across five live runs

| | live30 | live31 | live32 | live33 | live35 |
|---|---|---|---|---|---|
| pilot kappa | 0.761 | 0.688 | 0.777 | 0.839 | **0.844** |
| ceiling | 0.900 | 0.802 | 0.883 | 0.861 | **0.924** |
| citable rules | 1 | 46 | 106 | 92 | 58 |
| prescriptions | 0 | 0 | 12 | 12 | 12 |
| redraw fired | — | — | — | no | **yes, reverted** |

226 tests, up from 141 at the start of the day.

