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

## 1. Status — last updated 2026-08-27 (night)

# 552 tests passing. live41 abandoned after a RESUME dropped a branch; live42 relaunched fresh.

**The one thing to know:** the final report (`00_最终报告.md`) is now written by an
agent rather than assembled by Python, guarded by a fact-sheet check per section
*and* a coverage check over the run's own warnings. It has run end-to-end offline
only — the offline stand-in cannot produce a valid outline, so the **agent-planned
outline path has never executed**, and no section has ever been written by a real
model. Cheapest possible validation: run `narrate()` alone against live40's
artifacts (~15 calls) before committing to a full run.

| | |
|---|---|
| Tests | **552** passing; `ruff --select F src/qmine/` clean |
| `live40` gen01 | 17/17 phases, halted=False, **241.8 min**, `provider=routed` |
| Verification | **25/26 PASS, 0 FAIL** (live39 control: 19 PASS, **6 FAIL**) |
| Gates | 24 recorded — 20 passed, 4 warned, 0 failed |
| Findings ledger | **30 findings, 13 machine-confirmed** |

Every one of the 6 checks that fails on live39 is a mechanism built since it.

### The fork delivered exactly what the simulation predicted

    SS1   p2a 68 min  ∥  p3        12 min (then waits)
    SS2   p2b 79 min  ∥  p4+p5+p6  27 min (then waits)

All **39 min** of bottom-up work hidden — the simulation said 39.5. Without the
fork this run would have taken ~281 min instead of 241.8. It ran longer than
live39 only because it did two taxonomy redraws live39 never did, one of which
LOWERED kappa and was correctly reverted.

### The most important result: confirmed ≠ defective

13 findings were machine-confirmed. Independent re-verification (16 agents, each
reading the artifacts, then adversarial refutation) found **only 2 real
defects**. Eight were arithmetically correct and wrong anyway — almost all
because the two compared fields measured **different populations**: different
samples, different id spaces, different pipeline stages.

That is a property of the mechanism, not of this run. A `check` proves an
**assertion failed**; it says nothing about whether the observer's *conclusion*
holds. The panel section was headed "**这些不是观点**" — true of the arithmetic,
false of the conclusion — and a reader counting 12 confirmed blocking findings
would have been wrong about ten. Both the report framing and the observer prompt
now carry the measured rate.

### Defects found and fixed

**In what was built this week:**

- **The pre-delivery auditor found 4 real defects and I refused all 4.**
  - `check` semantics were **inverted for edits**. An observation asserts what
    *should* hold (failing = defect); an edit asserts what the artifacts *do*
    say (holding = sourced). The code refused every correctly-sourced edit. A
    test was *pinning* this bug.
  - The auditor is handed the gate ledger and told to cite its source, but
    citations resolved only against artifacts — gates live in `run_summary.json`.
    3 of 4 refusals were this.
- **`governance.py`: a merge that changed nothing was recorded as executed.**
  P011 targeted `[10, 11, 15]` carrying LEAF names while `merge_families` runs in
  the FAMILY namespace, where only 0..6 existed. Its map entries were no-ops, the
  §6 table said `executed`, and the three duplicate `pinyin_query` leaves it was
  meant to collapse are still in the delivered partition. Same leaf-id/family-id
  confusion that once made every family heading wrong.

**In older code:**

- **`cluster.py`: the falsification probe compared KMeans with itself.**
  `best_other = ranked[0]` is the best OVERALL, which is the reference whenever
  KMeans wins. `alternative_beats_reference_by = 0.0` by construction. Now the
  best *structurally different* arm: −0.0776 against `agglo_average_k15`.
- **`zh_topdown.py`: 高于 was hardcoded.** live40 shipped "对抗验证 (0.82)
  高于交叉验证 (0.8625)" — false — plus a causal story that only holds in the
  other direction. Both readings now written honestly.
- **`i18n.py`: "审计处方已全部执行"** with 8 of 17 executed. Replaced with what
  the gate actually guarantees: every prescription settled, executed or declined.
- 「可执行的触发式」 used for two different counts in one document; a hardcoded
  `112` beside the artifact's 123; the offline stand-in echoing a **schema field
  name** into a family heading.

### Not done

- **The two real confirmed findings and the fixes above have not been re-run
  live.** All verified offline plus by replay against live40's artifacts.
- The 8 false-positive findings remain in live40's shipped ledger. They are
  correctly *described* there now, but the run itself is not regenerated.

## 2. Open questions — EDIT THIS SECTION, DO NOT APPEND

**Resolved by live42 and deleted from this list:** "the final report has never met
a real model" (it has — 3/9 sections, see below), "the bottom-up selection changes
have never met a real model" (they have — K=18 via `legacy_l2`, split measurement
7/9, local-k spanning 1-8).

### P0 — FIXED 2026-08-28, verified end to end

1. ~~**The run crashes at TEARDOWN and `run_summary.json` is never written.**~~
   **Root cause was NOT the serializer.** `DecisionRecord` encodes fine; the
   message naming it was a symptom. `_checkpointer` (and `open_memory`, the same
   defect) wrapped a single `try/except` around its `yield` — and
   `@contextmanager` throws a WITH-BODY exception back in at that point, so both
   caught the ENTIRE pipeline's failures, relabelled them, and fell through to a
   SECOND `yield`, which a generator may not do. Hence
   "generator didn't stop after throw()", the original exception lost, and the run
   dead before `write_summary`.

   Fixed by building the saver/store inside `try/except` and yielding OUTSIDE it,
   exactly once, with no `except` around the yield. `write_summary` also moved
   into a `finally`, so a crash still records how far the run got. Verified: a
   full offline run writes a 13,370-byte summary with 17 phases and 15 gates, zero
   teardown errors, and `verify_run.py`'s falsely-failing checks now read real
   data. Mutation-tested — restoring the `except` reproduces live42's exact
   message, "SQLite checkpointer unavailable (the pipeline itself failed)".

   **live42's true verification score remains unknown** — its summary cannot be
   reconstructed after the fact. The next live run is the first fully verifiable one.

### P0 — none open
   live42 completed 17/17 phases and shipped every deliverable, then died:
   `DecisionRecord` is no longer msgpack-serializable, the SQLite checkpointer
   fell back to in-memory, and the store's context manager raised
   `RuntimeError: generator didn't stop after throw()` on exit.

   **This is worse than lost metadata: it blinds the verification harness.** Five
   of `verify_run.py`'s six failures on live42 are FALSE — "no observer gate
   reached state", "provider=None", "no completed_phases recorded", "0 observers",
   "the blocking delivery gate absent" — all of them read `run_summary.json`.
   A run cannot be checked at all without it. Fix the serializer registration and
   the teardown ordering; then re-verify live42, whose real score is unknown.

### P1 — real defects with a known cause

2. **The narrative report shipped 3 of 9 sections.** Three failed with EMPTY prose
   on all three attempts (`vector_choice_first`, `two_level_tree`,
   `samples_and_deployment`) — the model returned nothing, so retry feedback has
   nothing to act on. 10 `prompt block truncated` events fired during the report
   (sheets of 50-53k against a 40,000 budget), which is the obvious suspect and is
   NOT established. Reproduce one empty section offline against its recorded sheet
   before changing anything; ids and bundles are in `final_report_meta.json`.

3. **The narrative door has a numeric guarantee and NO attribution guarantee.**
   live42's §4 wrote 「交付的 K=18 是参照 phrasing_groups 的粒度锚点」 — the wrong
   reference, two lines after listing all three correctly. `check_numbers` is
   precision-only on NUMBERS; a wrong noun is invisible, and the must-cover anchor
   matched because every reference name appears somewhere. Same class as the
   `locator_reference` contradiction the p5 observer caught.

4. **Two English strings reach three Chinese reports each (6 lines).**
   "DIAGNOSTIC ONLY — do not choose a model from this number…" and "Match the two
   annotators…", both from the annotator-balance work, never routed through
   `prose()`. The static AST guard covers `deps.decision()` rationales only — it
   does not see gate messages or remediations, which is where these live.

5. **The resume rewind silently drops a concurrent branch.** Made LOUD (the join
   halts with `p2c_both_branches_arrived`), not fixed. Open: whether
   `update_state(as_node=...)` can restore a multi-superstep fan-out at all, or
   whether a gap in `phase_status` should force a clean re-run of the generation.
   Until answered: open a new generation and run it ONCE, never restart mid-flight.

### P2 — measured, disclosed, not acted on

6. **`challenger_beats_incumbent` has no production call site.** Four docstrings
   and a model-budget decision rested on it; they now say so. Needs a signature
   change (`propose_grid` returns a flat list, so selection cannot tell a proposed
   value from a configured one). Applying it as written flips live40 to a worse K.

7. **The spend ledger is wrong in both directions and neither is measured.**
   Provider prompt-cache discounts are not recorded (input is 7.3x output — 8.98M
   vs 1.24M on live42, annotators 94% of it — and the prompt is already ordered
   for prefix caching), so cost OVER-reports; `qwen3.7-plus` publishes no price so
   annotator_b counts as $0.00 and it UNDER-reports.

8. **The alpha decision sits inside its own noise.** Winners across 5 seed
   replicates: 0.1, 0.5, 0.1, 0.0, 0.1. `tie_band=0.05` is an unreachable default
   ~4.5x narrower than the metric's spread. **Do not widen the band** — at a
   measured 2-sd band live40 elects alpha=0.5, which its own k=7 panel shows
   fragments worse. The fix is replication.

9. **Annotator asymmetry, unattributed.** live42: annotator_a won 34.1% of 270
   contested rows, z=-5.2. NO baseline exists (the gate postdates live40). The
   parsimonious reading is the pairing — `deepseek-v4-flash` against
   `qwen3.7-plus`, flash tier against plus tier. Testing whether disabling
   reasoning contributed needs a paired re-run of the same rows.

10. **Observers cost ~2.5 hours of a 4-hour run.** 11 observers, median 809s on
    live41 against 285s on live40 for the SAME output volume — latency, not token
    inflation. They earn it (three real defects today), but the trade should be a
    decision, not an accident.

11. **Refinement has not converged on any run.** live40, live41, live42 all hit
    the 5-round limit. Honestly disclosed in the report, but it means the
    delivered leaf count depends on which round it stopped at.

12. **Two vacuous rule triggers found live and not repaired.**
    `academic_knowledge_qa x problem_solving` names 求/解/计算 with 0 of 19
    contested rows carrying any; `navigational x school_info` names 主页/入口/好不好
    with 0 of 8. The detector works; nothing acts on what it finds.

13. **Two of nine governance splits are geometrically unsupported** (leaf 19 -> 54
    at ARI 0.0595 with a near-duplicate name; leaf 0 -> 49 at 0.3876). Disclosed
    by design (measure-don't-veto). Nobody has looked at them.

### P3 — frozen debt, deliberately not growing

14. **22 `deps.gate()` messages are English f-strings** printed verbatim into
    Chinese deliverables. Frozen by gate NAME in `GATE_MESSAGE_DEBT`
    (`tests/test_pipeline.py`); the set may shrink, anything new fails the test.

15. **Three domain profiles are untested on real data:** `finance_zh`,
    `sports_zh`, `politics_zh`. `generic.yaml` ships 0 template seeds, so a corpus
    without a profile locates K against unvalidated mined groups — now gated by
    `p3_locator_reference_validated`, but never exercised.

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


---

## 8. Session 5 (2026-08-24) — past p2b for the first time, and why the referee was losing half its work

**`live36` gen02 passed the p2a gate and completed the 3,000-row gold set** — the
first time this pipeline has ever got past p2b. κ 0.814, raw agreement 0.829, 511
disagreements, annotator b short 11 rows (reported, not absorbed). It then died in
the referee and was halted deliberately; `live38` is the clean re-run.

### RESOLVED — the redraw loop has no demonstrated effect (was §2 item 5)

Three controlled before/after trials now exist, not one:

| run | κ before | κ after | Δ |
|---|---|---|---|
| live35 | 0.844 | 0.827 | −0.017 |
| live36 r1 | 0.781 | 0.806 | **+0.025** |
| live36 r2 | 0.806 | 0.795 | −0.011 |

se(κ) at n=200 is 0.027–0.031. **All three sit inside one standard error and the
signs disagree.** Resolving an effect this size needs a pilot of ~1,150 queries,
~6× the current 200 — more than the gold set it protects. The earlier "points the
wrong way" reading was over-confident: the honest finding is *unresolvable at this
n*, not *harmful*.

**Why, measured.** Replaying all three live36 pilots (class codes byte-identical
throughout, so pairs are comparable): 36 disagreements over **25 distinct pairs**,
19 seen exactly once, expected count per pair ≈1.4 — and the **top-6 overlap
between consecutive pilots was 3/6, twice**. `structural.most_common(6)` was
handing the architect a target list that was half resampling noise, at ~350s of
frontier model plus a full three-pass re-pilot per round. Now requires ≥2
observations, and logs what it dropped.

**The redraw does real work — it just never merges.** All 24 codes stayed
byte-identical while 24/24 definitions changed substantively, converting
inferential boundaries to surface-signal ones ("必须有公式/推导信号"). That is the
prompt's option 2 done properly. It never takes option 1, *merge*, which the
prompt ranks first and which is the only move that removes an unresolvable
boundary — the prompt states preservation twice and structurally against one line
of merge preference.

### The referee was truncating at exactly 24,001 tokens, ten times

10 of 15 referee calls died at *precisely* 24,001 output tokens; the successful
ones emitted 19,279–19,597. Three coupled defects:

1. the declared budget (4,000 → cap 12,000) was measured at 8,179 **on a model
   long since replaced**;
2. `_hit_length_limit` matched the error's **prose**, and glm-5.2 returns
   `no parseable structured output` for a truncation — so the bump never fired;
3. `_length_bump` was clamped at `max(current, 2)`, so a 2× floor that still
   truncated could never grow.

Fixed as: budget 12,000 (cap 36,000, from measurement); truncation detected by
`output_tokens >= cap_in_force`, which is ground truth on any provider; and a
second floor 2→4 on evidence of truncating while already bumped. **Confirmed live
on `live38` within minutes** — a researcher escalated 2× → 4× and completed.

A failed batch now **bisects and retries each half** instead of discarding 25
adjudications. This matters beyond waste: the referee adjudicates exactly the
contested rows, so dropping them strips the *hardest* cases and every downstream
number reads optimistically.

### The p2a gate punished a redraw that worked

`remedy_exhausted = redraw_attempted and not any(r["kept"] ...)`. On live36 gen01
redraw 1 improved κ and redraw 2 was reverted — the loop was out of moves, but
`any(kept)` was True, so the gate **failed a run whose agreement had gone up**. An
improving redraw was strictly worse for the gate than a failing one. Zero test
coverage before; five tests now, three of which fail against the old code.

Honest note: gen02's redraw did *not* help, so `redraw_helped` was False there and
the old code would have passed that gate too. The fix was necessary for gen01's
numbers, not gen02's.

### Four defects on the recovery path the halt message itself recommends

`new-generation` → `qmine run --resume` had never been run end to end:
both resume paths hardcoded **generation 1** (and the thread id is
`{run_id}-gen{generation}`, so it reopened the old halted thread); `new_generation`
writes no `config.resolved.yaml`, so config resolution found nothing and fell into
the refuse-existing-id guard; `--input` was required even when resuming; and a new
generation is a **new thread with no checkpoint**, so invoking with `None` raised
`Received no input for __start__`, which `open_memory` masked as "generator didn't
stop after throw()". Also `resume_run` wired no `registry.on_call`, so a resumed
run logged **zero** agent lines — the dashboard would have been empty for its whole
duration. Event wiring is now shared by both entry points.

**CLAUDE.md's re-run instruction was wrong** and has been corrected: the guard
fires on `llm_cache/` as well as `checkpoints.sqlite`, so "delete the checkpoint,
keep the cache" lands on the guard. Use `new-generation` + `--resume`.

### Budgets go stale because the MODEL changes, not the task

Re-measured with no prompt changed: architect **23,759 → 38,073**, referee
**8,179 → 19,597**. Both roughly tripled. The architect was at 91% of its cap and
failed this project's own `cap >= observed × 1.2` rule; raised to 48,000.
`test_output_budgets_cover_what_the_roles_actually_emit` now carries the live36
figures and the model they were taken on. The agent transcript now records
**per-call** input/output tokens, so the next budget question is answered by
reading a file instead of differencing cumulative totals out of `run.log` — which
only works for sequential roles and silently misleads for concurrent ones.

Related: the per-call agent line in `run.log` was printing the role's *cumulative*
output, which reads as per-call and is wrong by a factor of the call count. It sent
this session's own diagnosis down the wrong path for several minutes.

### Building the real routing plan (since `qmine models` cannot)

```python
from qmine.cli import _load_env; _load_env()
from qmine.config import QMineConfig
from qmine.llm.registry import ModelRegistry
from pathlib import Path
cfg = QMineConfig.load(Path("configs/live.yaml")); cfg.llm.provider = "router"
reg = ModelRegistry(cfg.llm, cache_dir=Path(".cache/preflight_llm"), run_cfg=cfg)
print((reg.usage().get("routing") or {}).get("assignments"))
```

Verified before launching `live38`: `provider: routed`, and annotator_a /
annotator_b / referee on **deepseek / qwen / zhipu** — three distinct labs.

### The confusion archetypes recur in a fifth taxonomy

live36 gen02's top confusions — `AMBIGUOUS_BARE_QUERY × LOOKUP_CLASSICAL_TEXT_TRANSLATION`,
`LIST_WORDS_BY_CONDITION × SOLVE_CHARACTER_PUZZLE` — are archetypes B and C again,
under yet another set of class names. Five independent taxonomies, same contested
regions. This belongs in the deliverable: it is a finding about the **corpus**, not
about any taxonomy we drew.

**RESOLVED same session — the referee bisect now has a test.** Extracted as
`_run_batch_with_bisect(run_batch, fold, chunk) -> (failed, covered)`, a pure
higher-order function, and covered by five tests; four fail against the old
drop-the-batch behaviour. Two properties they pin that are easy to lose: the
split is **not recursive** (bounded at two extra calls, so a systematically
failing referee cannot cause a retry storm), and `fold` runs **per call, not once
at the end** — a recovered half must bind the halves after it, or the rule set
acquires two rules that fire on the same trigger with opposite answers.

**A merge orphaned its own rules, and that confounds every merge trial.** The
redraw swaps `nodes` and leaves `rules` untouched, so a class merged away keeps
the rules that route to it. Measured on live36 gen02: dropping
`INTERPRET_LITERARY_MEANING` left **4 of 45 rules dangling, 2 routing directly to
the deleted code**, and both governed `INTERPRET_LITERARY_MEANING ×
LOOKUP_WORD_MEANING` — a pair that redraw had targeted. The annotator was
instructed to assign a label that was not in its class list, on precisely the rows
the before/after comparison is decided by.

Consequence for the redraw question: sort the trials by whether they merged.
live35 and both live36 gen01 rounds did **not** merge (24→24) and are clean.
live36 gen02 **did** (20→19) and is confounded. The only trials that exercised
merging — the remedy the prompt ranks first — are the unusable ones. `live38` is
merging too (22→21 with 50 rules), so its comparison is confounded as well; read
its `redraw 1:` line accordingly.

Fixed: rules whose `then` names a removed class are pruned before the re-pilot and
the count is announced. Two tests; one fails against the old behaviour.

**RESOLVED same session — `qmine models` now pre-flights the real configuration.**
It took `--config`, mirrors `ModelRegistry`'s own `route()` call argument for
argument (`prefer`, `budget_usd`, `prefer_chinese_native`, `excluded_labs`), and
prints the policy in force. Verified: it reports **$6.34**, the same figure
`live38` printed at launch, where before it silently planned against labs the
live config forbids.

It also now prints the annotator/referee **labs**, not just the gateways:

```
annotator/referee labs: a=deepseek, b=qwen, referee=zhipu — independent
```

The model column shows `qwen:dashscope/...` for BOTH annotator_b and the referee
because that is the gateway; their labs are qwen and zhipu. An operator reading
the table alone would reasonably conclude the referee shares a lab with an
annotator — the exact gateway-vs-lab confusion this repo already has a test for.
The independence rule is by lab, so it is now stated by lab.

### A multi-agent audit of the never-run phases, and what it found

Five read-only review lenses over everything after the referee — code the live
run reaches last and that has never executed with live agents — each finding then
handed to a separate agent instructed to REFUTE it. 15 candidates, 8 verified,
**7 confirmed, 1 refuted**. All seven are fixed with tests. Two would have ended
a paid run.

1. **`max_total_output_tokens = 6_000_000` was HALF what an honest run needs.**
   The rule (~2x the whole-run estimate) was right; its input was stale.
   Declared estimate 3,074,000 vs honest 12,096,234, so the ceiling was 2.0x the
   declared figure and **0.50x** the real one — a runaway guard that fires on
   correct behaviour. Root cause: `annotator_a` declared 5,000 output tokens per
   call and emits **21,975**. Annotator budgets → 22,000, ceiling re-derived to
   24,000,000. The pre-run estimate moved $6.34 → $17.15, against ~$14
   extrapolated from live38's actual spend — the old figure understated by half.

2. **Phase 7 naming could end a run on one blip.** `_name` returned `None` on the
   first exception while `_annotate` retries 3x with backoff, and a single
   unnamed leaf fails `p7_all_leaves_named`, which is BLOCKING and which resume
   refuses to overturn. Nothing anywhere re-names a lost leaf. Compounding it,
   the pool was a hard-coded `min(4, …)` while the shards run concurrently — 5 x
   4 = 20 calls on one provider against `max_concurrency: 8`, which made that
   knob inert in exactly the phase that cannot recover. Now retries, and 5 in
   flight.

3. **Guide repair discarded the referee's entire output.** With
   `repair_on_fresh_sample` (the default) round 2 annotates DISJOINT queries, and
   `rows = repair_meta["rows"]` swapped the list — throwing away ~3,000 round-1
   rows including every adjudication, and substituting a set whose only labelled
   rows are ones both annotators already agreed on, because the referee runs
   before repair and never sees round 2. **The gold set became agreement-only**,
   i.e. systematically the easy rows, so every classifier number computed from it
   read high for that reason alone. Now merges.

4. **`UNLABELED == UNLABELED` counted as agreement.** `agreement()` already
   excluded it from kappa, so the METRIC was safe while the GOLD SET was not: a
   row both annotators omitted was recorded agreed with `final="UNLABELED"`,
   skipped the referee, and passed p2c's non-empty filter into the classifier as
   a real class.

5. **An empty adversary response scored 1.000.** `n = max(len(verdicts), 1)` gave
   `1 - 0/1` — perfect accuracy manufactured by a provider failure. Denominator
   is now verdicts returned, coverage travels into the artifact and the Chinese
   report, and no verdicts means undefined. The chunk call was also unguarded, so
   a blip crashed the phase.

6. **L2 "visible to the embedding" used a flat 0.5** with no config path. kNN
   agreement means different things at different class counts — chance is ~4.5%
   at 22 classes, ~20% at 5 — so a flat bar calls large classes visible on their
   PRIOR and small ones invisible regardless of geometry. Now
   `max(floor, 2 x chance)` with the lift recorded. The test shows it: a dominant
   class embedded as pure noise clears 0.5 and fails the new bar. `n`/`share`
   renamed to `n_in_subsample`/`share_in_subsample` — they were subsample counts
   printed beside population-scale numbers.

7. **Calibration was in-sample, printed beside out-of-fold accuracy.** The report
   states phase 10 ROUTES on confidence, so a flattered ECE loosens a live
   threshold. Now out-of-fold, with `ece_basis` naming which path ran.

**The pattern across all seven: a number that stayed silent about its own
denominator or its own basis.** That is the same shape as the referee dropping
its hardest rows and as the stale budgets — and it is why "read `n` before
believing any metric" is the rule it is. Worth running this audit again against
any phase before trusting its output.

**Refuted, do not re-raise:** the adversary's 2,500-token budget. The claim's
premise ("attempt 0 truncates at 7,500") is false — the 4x floor and the
plain-JSON fallback both cover it.

### RESOLVED same session — the referee no longer bottlenecks the pipeline

Batching by row POSITION forced the phase to run strictly sequentially, for a
sound reason: the referee settles *boundaries*, and the same boundary appearing
in two batches gets decided twice, leaving the rule set with two rules that fire
on the same trigger and disagree. But that argument constrains ordering WITHIN a
boundary only, never across boundaries.

Now grouped so that **no class pair spans two groups**: groups are independent by
construction and run concurrently; a pair too big for one call is split into
sequential chunks that stay inside one group and thread their own earlier ruling
forward. `decided` is no longer threaded ACROSS groups — under concurrency that
would make each prompt depend on which siblings finished first, so a replay would
send different prompts and miss its own cache: the resume cascade, recreated
inside one phase.

Measured on live38's own 519 disagreements over 87 class pairs:

| | before | after |
|---|---|---|
| calls | 21 x 25 rows | 44 x ≤15 rows |
| largest call | 25 rows | 15 rows |
| critical path | 21 sequential | 4 sequential |
| wall clock | ~5-8h | **~1.0h** |

The smaller chunk also attacks the parse failure directly: live38's first 25-row
call emitted 34,099 tokens and failed to parse *with 144,000 tokens of room
available* — `_length_bump` is keyed by MODEL, and a researcher had already
escalated glm-5.2 to 4x, so the cap was never the constraint. Response SIZE was.
Cost is ~23 extra calls re-sending classes and rules, about +460k input tokens
(~$0.64) to save seven hours.

**A trap worth remembering:** the first version of this refused to split any pair,
which produced a 52-row group — LARGER than the 25-row calls already failing —
and on failure `bisect` would have halved it, splitting the pair anyway and
without the prior ruling. The measurement caught it; the synthetic tests had not.

### Can the top-down and bottom-up paths run concurrently? Yes — and the state layer already assumes it

Asked on 2026-08-24. Derived the real dependency DAG by extracting every
`deps.load/recover/embedding/has` and `state.get` in each phase node:

| phase | reads |
|---|---|
| p2a | data_audit, risk_screen, template_groups *(all p1)* |
| p2b | the taxonomy *(p2a)* |
| **p3** | **template_groups only — nothing from p2** |
| p2c | emb_base *(p3)*, gold *(p2b)* ← **the join** |
| p4/p5/p6 | emb_hybrid *(p3)* |
| p7 | emb_hybrid, leaf_centroids, leaf_labels *(p6)* |
| p8 | leaf_* *(p6)* + the TREE AUDITOR's prescriptions, not p2a's |
| p9 | gold_agreement *(p2b)* + leaf_family *(p6)* ← second join |

```
p1 ─┬─→ p2a → p2b ───────────────────────┐
    │                                     ├→ p2c → p2d → p2e ─┐
    └─→ p3 → p4 → p5 → p6 → p7 → p8 ──────┴───────────────────┴→ p9 → p10 → p11 → p12
```

**`p3` through `p8` — six phases, including p7's ~60 naming calls — have NO
dependency on p2 whatsoever.** The current order is already dependency-ordered
rather than path-ordered (which is why p3 sits between p2b and p2c), but
`build_graph` still wires one strict linear chain, so the two branches serialise
for no reason the data requires.

**The state layer already supports the fork.** Every field carries a
commutative reducer — `merge_artifacts`, `merge_prescriptions`, `merge_dict`,
`operator.add` — and `state.py`'s own docstring says the reducers are merges
"which matters because Phase 7" fans out. Only the edges are missing.

**Honest size of the win.** The bottom-up branch is CPU-bound and the top-down
branch is I/O-bound on provider latency, so they overlap almost perfectly. But
the magnitudes differ: `make full` does all 17 phases offline on 50k in ~25 min,
essentially all of it p3-p6 compute, while a live p2a+p2b+referee is ~3-3.5h.
So parallelising saves roughly **25-35 min of a ~3.5h run, about 10-15%** — real,
not transformative. Three things make it worth more than the percentage:

1. **Fail-fast.** An encoder download, an OOM, or a clustering bug currently
   surfaces at hour three. Concurrently it surfaces at minute twenty-five, before
   the expensive half is paid for.
2. **Blindness for free.** p3-p7 must be blind to the taxonomy (anti-anchoring).
   Running them alongside p2 rather than after it means the taxonomy does not yet
   exist while they run — the strongest form of that guarantee.
3. **It scales the right way.** Embedding and clustering cost grow with corpus
   size; the gold set is capped by config. On a 500k corpus the bottom-up branch
   dominates, and the saving grows with it.

**Not implemented, deliberately, and the ordering matters:** a fork-join changes
halt propagation (a p2a gate failure must stop a branch already in flight),
checkpoint/resume semantics, and concurrent artifact writes — in a pipeline that
has **never once completed end to end live**. Get one complete run with real
deliverables first; parallelise against a known-good baseline, where any
regression is attributable. Doing it in the other order means debugging
concurrency and a never-finished pipeline at the same time.

### Three more open questions closed this session

**`--reuse-taxonomy` is wired** (`RUN_ID`, `RUN_ID/genNN`, or a path). It skips
p2a entirely and reuses a finished `taxonomy.json`, which is the root fix for the
resume cascade: the web-using researchers are non-deterministic, so re-deriving a
taxonomy changes every annotator prompt and misses the cache on all 3,000 gold
rows. It RAISES on a bad spec rather than falling back to re-deriving — a run
that silently ignores the flag would pay for a full architect pass and miss the
cache it was pointed at, which is the very failure it exists to prevent.

**Error classification now keys off the SDK's `status_code`,** with the prose
scan kept only as a fallback for providers that raise plain exceptions. This was
the "bitten three times in one day" item. Both halves are now structured: a
truncation is `output_tokens >= cap_in_force`, and a dead provider is a status in
{401, 402, 403}. `completion_tokens=4013` can no longer read as a 401. (`bool` is
excluded explicitly — `isinstance(True, int)` is True in Python, so a flag would
otherwise have read as a status.)

**The planner's accuracy has been re-derived** now that the ledger prices at the
routed model's own rates. `qmine models --config configs/live.yaml` reproduced
`live38`'s launch figure exactly ($6.34 at the time), and after the budget
corrections reports $7.01 against ~$14 extrapolated from live38's actual spend —
under, but within the same order, where the pre-correction number was less than
half. Remaining gap is that `output_tokens_per_call` is now the pair AVERAGE for
the two annotators, since which one draws the reasoning model is not known until
routing has already happened.

### Silent prompt truncation, and the class of bug it represents

Four fixes, after finding that the referee's entire contribution was being cut
from the prompts meant to apply it:

1. **`budget_text` announces what it drops** — with the block's name, the share
   kept, and an explicit warning when the trim is HEAD-ONLY so anything appended
   is lost first. The in-band `… [truncated N chars]` marker told the MODEL it
   was reading an excerpt and told the operator nothing, because nobody reads the
   prompt. Every instance of this class hid behind that.
2. **Rules render newest-round-first.** A defence that does not depend on
   guessing the right budget: if the block is ever exceeded again, the rules lost
   are the oldest rather than the ones just written in response to observed
   disagreement.
3. **Rules budget 9,000 → 20,000**, sized against the 18,496 the pipeline
   actually produces. Verified 83/83 referee rules now survive, against 0 under
   the old budget and 29 under an intermediate 12,000.
4. **`taxonomy_v2` is persisted.** `taxonomy.rules.extend(new_rules)` and the
   repaired `labeling_guide` were applied to the in-memory object only and p2b
   returned no taxonomy artifact — so `gen02/taxonomy.json` showed 50 rules and a
   guide with no 边界裁定 while the run held 133 rules and a rewritten guide. A
   resumed run recovered the PRE-referee taxonomy. `deps.taxonomy()` already
   preferred `taxonomy_v2`; nothing had ever written it.

**The pattern, stated once:** this project's characteristic bug is *a number or a
payload that stays silent about what it left out* — the referee dropping its
hardest rows, adversarial accuracy shrinking its denominator, ECE changing basis
without changing label, a runaway guard derived from a stale projection, and now
a prompt block discarding the guidance it exists to carry. "Read `n` before
believing any metric" generalises: **make every mechanism say what it dropped.**

### Session 5 close — what to know before touching anything

**The pipeline reached p4 for the first time.** Every earlier run died at p2a or
p2b. The gold set that carried it through is **6,000 rows with 465 referee
adjudications**, and it exists only because of three fixes made today: the merge
that stopped guide repair destroying the refereed rows, the `UNLABELED`
exclusion, and the revert guard that undid a repair measuring 0.028 worse.

**First live results from phases that had never run:**

| phase | result |
|---|---|
| p2c classifier | CV accuracy **0.852**, macro-F1 **0.771**, ECE **0.023** out-of-fold, on 5,534 rows |
| p2d adversary | **0.953** survived attack, 7 wrong / 24 defensible, **coverage 100%** |
| p2e L2 | **5 of 21** classes rule-dependent |

Read all of them against the p2b gate's own caveat: κ 0.822 with ~0.10 of
residual slack, so these are accuracies against a gold set two annotators agreed
on 82% of the time — not against ground truth.

**Two claims this run falsified**, both now corrected in code:

1. The report asserted adversarial accuracy is *"lower and more trustworthy than
   cross-validated accuracy"*. Measured: **0.953 vs 0.852 — higher.** They are
   computed on different populations (the gold set is enriched with contested
   rows; the attack samples the corpus at random and draws mostly easy ones), so
   they are not comparable as levels at all.
2. My own L2 "chance-relative" bar never bound: `max(0.5, 2 x share)` with 22
   classes means the 0.5 floor always wins, so the flat constant still decided
   every verdict — and called a class separated **74x above chance**
   rule-dependent while passing one at 42x. Now `median - 1.0 x MAD`, which
   reproduces K12's five flagged classes exactly and flags nothing when every
   class clusters equally well.

**The habit to watch in this codebase, stated once more:** four separate
source-text assertions I wrote today misfired — matching inside a comment, inside
a longer identifier, on a legitimate read, and on a budget literal I later
changed. Test the behaviour, not the implementation text; where source inspection
is genuinely needed, match whole statements and assert the PROPERTY (a budget
exceeds what is measured) rather than the number.


## Session (2026-08-25, evening) — live39, and what only reading found

**The run.** `live39` gen01: 17/17 phases, 201 min, $5.52, provider=routed,
κ 0.8341, CV 0.8586, 11 families / 39 leaves, all named. First run carrying the
corrected reports, the phase observers, the interpreter and the grid proposer.

**Verification.** Built `qm_verify_run.py` (scratch) — 19 mechanical checks, one
per defect fixed since live38. Validated against live38 FIRST, where it scored
2/19: a harness that passes on the broken run proves nothing. live39: **18/19**.
The single failure is expected — p2a ran before `web_researched` existed.

Building it caught three bugs in the harness itself, one serious: a check that
called the FIXED `family_names()` against live38's artifacts and passed, on a run
whose every family heading was wrong. **A check that verifies today's code against
yesterday's output verifies nothing about the deliverable.**

**Five defects only reading found — three of them mine.**

| defect | note |
|---|---|
| `fig3_battery.png` MISSING | my colour fix used `matplotlib.cm.get_cmap`, removed in mpl 3.9 (env 3.11). I checked the cell PARSED and never rendered it. |
| the lost figure reported "0 cell errors" | the cell caught the exception and printed it, so nothing counted it |
| fig3 plots 15 of 18 configs | 3 HDBSCAN carry `stability_ari=NaN`; matplotlib drops NaN silently. The reason — **86-91% of rows came back as noise** — was invisible. Now stated on the chart. |
| `"chosen K (stability peak)"` in `viz.py` | my earlier fix searched for the KEY NAME, not the PHRASE. `grep "stability peak"` finds all four sites at once. |
| English `chosen_by` in the Chinese report | pre-existing; exposed when my longer text tripped the language test |

**Two things I reverted or re-did.**

* Replacing the α `tie_band` constant with the measured `noise_floor` — **tried and
  reverted.** The alpha sweep is non-monotone (1.98, 2.02, 2.42, 1.98, 2.41…), so
  its roughness is SIGNAL; the estimate widened the band from 2.08 to 2.42 and
  flipped the winner from α=0.1 to α=0.5 on nothing. Recorded in
  `.claude/rules/measurement.md`: `noise_floor` needs a smooth sweep, and
  separating noise from signal on a jumpy one needs REPLICATION.
* While fixing English-in-Chinese in the battery figure I **introduced the same
  defect** in the α rationale. `prose()` matches a literal prefix, so a
  translatable sentence must carry NO interpolated numbers — the shape that works
  is a stable sentence plus the numbers as separate DATA fields.

**A real defect the observer found, verified by hand.** The α decision rationale
said "Lowest template fragmentation with highest stability" — and α=0.1 had
NEITHER (2.0193 vs 1.9799 at α=0.0; α=0.5 more stable). The artifact's `chosen_by`
was right all along; only the sentence a reader sees was false. Now states the real
rule with this run's numbers.

**Not started:** the p6 `leaves_per_family` inconsistency (see §1), extending
triggers to all referee rules, and the three untested domain profiles.

---

## Session — 2026-08-26: agent authority, and what an agent may be trusted to do

The day's question was the observer's powerlessness on live39, and it turned into
a general one: **what is an agent allowed to do, and what makes each permission
safe?** The answer that held up across three separate mechanisms is the same one:

> An agent may supply the measurement that would settle its own claim. Only the
> measurement carries authority.

### The observer did not need permission — it needed a way to be proven right

Giving it write access would have put an unaudited LLM judgement in charge of the
run. But its live39 finding was *arithmetic over an artifact*, and nothing could
evaluate it. So an observation may now carry a `check`, and a confirmed one is an
assertion that failed rather than an opinion — which is what makes it safe to
block on. `severity` no longer decides anything; it is the model's own confidence.

The evaluator (`ops/checks.py`) is a security boundary: the string comes from a
model. `a.b` is a **dict lookup**, never `getattr`, so `__class__` resolves to a
missing key rather than to a Python object and the standard escape chain has no
first step. A mutation swapping one `return _MISSING` for a `getattr` survived the
whole suite at first — the hostile-expression tests only reached the dict branch.
Now each container branch is probed separately, and a parsed (not grepped) test
asserts `_lookup` contains no `getattr` call.

### Findings that cannot be forgotten

The other half of "nothing consumed the finding". `ops/findings.py` is run-level,
so a new generation inherits it like the LLM cache. The only automatic exit is a
measurement — the entry's own check passing again. A mutation relaxing that to
"the check is not failing" survived, and it mattered: a phase that stopped writing
an artifact would have silently closed every finding about it.

### The referee diagnosis was wrong, and measuring first is what caught it

Yesterday's plan was "make the referee emit a trigger". Reading its actual rules
first showed they are semantic conditions with no regex, 79 of 80. Holding them to
the referee's own verdicts found a defect nothing could see:

    OTHER × TEXT_INTERPRETATION — referee ruled TEXT_INTERPRETATION 15/21,
    and FIVE of the six rules say "no intent marker → OTHER"

Five rules restating one principle, each the opposite of what the referee had just
done on the rows in front of it. Reported per boundary, because a rule is
conditional and one at the minority is a legitimate exception — a per-rule score
would have flagged every honest exception as a defect.

### The one agent with write authority

`agents/audit_delivery.py` reads every gate, the ledger, the artifacts and the
finished documents together and may edit the reports. What makes that safe is not
trust, it is the shape of the operation: an anchored replacement, anchor proven
unique, every number sourced from **the artifact the edit cites** (which keeps the
pool small — `agents/verify.py` documents its own blind spot on large pools),
language checked, reason required, originals kept. `.md` only. Refusals are
printed beside the edits, because a report showing only successes is a sales
document.

### Method note

15 mutations across the new guardrails; 3 survived the first pass and each one was
a real hole in the tests rather than dead code. Two end-to-end offline runs, and
the second existed only because the first put three empty-claim rows in the ledger
— a defect that no test would have found and that reading the output did.

---

## Session — 2026-08-26 (late): the graph forks

The question was whether the bottom-up route really has to wait hours for p2b.
It does not: `p3_represent` reads only `template_groups`, from p1.

**What made this a design problem rather than an edge rewrite** was that
langgraph's scheduling is not what it looks like. Three behaviours, each measured
with a throwaway graph before any production code changed:

1. parallel branches lock-step per superstep — so the node GROUPING sets the
   cost, and 2-against-2 beats 1-against-3 by 22 simulated minutes;
2. a fan-in node fires once per incoming edge unless the edges arrive together;
3. a state field written by two branches in one superstep is a runtime error.

Guessing any of these wrong produces a pipeline that is slower, or that trains
its classifier twice, or that dies 107 minutes in. Measuring all three first cost
about ten minutes.

**The most valuable finding was not about scheduling.** Concurrency does not
create races; it reveals the ones a serial graph was hiding. Four shared
read-modify-writes were live, and the worst was `ops/findings.py` — added earlier
the same day, and measured here losing **6 of 8** concurrent filings. That is the
"nothing consumed the finding" failure the module exists to prevent,
reintroduced by the scheduler instead of by a missing consumer.

**Race tests lie.** Three of the four passed against a deliberately unlocked
implementation. A `RecordingLock` asserting the critical section is held is
deterministic and is what the invariant actually means; the stress test stays
beside it, because the structural check cannot see a section that is held but too
narrow — which is exactly what mutation testing then found in `deps.decision`.

---

## Session — 2026-08-26 (night): live40, and what a confirmed check is worth

**The mechanism found real defects and produced more false ones than true.**
13 machine-confirmed findings; 2 real. The dominant error was not bad arithmetic
but a bad inference from good arithmetic: two fields differ, and they were never
supposed to agree because they count different populations.

That is worth stating as a design property. `ops/checks.py` converts a claim into
a measurement, and the measurement it makes is narrow: *this assertion is false*.
Everything the observer wraps around that — which fields to compare, whether
they are comparable, what it means if they differ — is still an LLM judgement
with no guardrail on it. The confirmation makes the arithmetic trustworthy and
the conclusion no more trustworthy than before.

**The fleet found more outside the confirmed list than inside it.** Four false
statements reaching readers of shipped documents, and one executor no-op
touching the delivered partition — none of which any check had flagged, because
no observer thought to assert on them.

**And the auditor was right four times and refused four times.** Both causes were
mine: inverted check semantics for edits, and a citation namespace that excluded
the very evidence the auditor is handed. One of my own tests was pinning the
first bug — the suite was protecting the defect. That is the second time this
week a test encoded a wrong invariant rather than a right one.

---

## Session — 2026-08-26 (late): the reports were unreadable, and why

The user read live40's two markdown reports and could not follow them: "very hard
to follow", "can't grasp their overall logic and reasoning steps", "unnatural and
not that coherent and with patches everywhere". They asked whether script
generation caused it — explicitly saying scripts are *not* wrong and are important
for correctness — and whether an agent could write a genuinely synthesised final
report instead.

**Diagnosis: two independent causes, and the reader was right about both.**

*Mechanical defects in the templates.* §9 of the bottom-up report interpolated a
raw value into prose, so `p2e`'s per-class audit — a list of 13 dicts — reached a
Chinese deliverable as ~1,900 characters of `[{'class': 'OFFTOPIC_RISK_NOISE',
...}]` mid-sentence. Three rationales shipped in **English** inside Chinese
reports. 45 lines exceeded 400 characters. All fixed: `_short_value()` summarises
containers by size and leaves them in the artifact they came from.

*No narrative spine.* Fourteen sections, each generated by an independent
function, none referring to the previous. Reading order is generation order.
Every defect ever fixed added its caveat paragraph where it happened rather than
where a reader needs it. That is structural and no amount of template polish
fixes it.

**The guard that should have caught the English had TWO independent failures.**
`test_every_authored_rationale_reaches_the_reader_in_the_report_language` runs on
the offline fixture, which never takes the `p2e` audit branch or the HDBSCAN
screen — the known coverage gap. But its detector was also too weak to fire at
all: it asked for three consecutive ≥6-letter lowercase words, which real English
almost never contains, because function words break every run. **It would have
passed on the live report too.** Replaced with a CJK-absence detector
(`i18n.looks_like_english_prose`, shared by test and runtime so they cannot
drift), and backed by a *static* AST test over every `deps.decision()` rationale
literal in the source — coverage that no fixture's reach can hide.

Turning the detector up immediately exposed a larger class: **all 22
`deps.gate()` messages are English f-strings** and the ledger prints them
verbatim. That is a restructure, not a mapping, so it is frozen by gate NAME in
`GATE_MESSAGE_DEBT` — the set may shrink, and anything not in it fails. Visible
debt beats an invisible allowlist. **This is open work.**

### What was built: `agents/narrate.py` + `report/narrative_brief.py`

Research first, since the design question was real. Classic NLG separates content
determination → document planning → realization; the scripts do realization only,
which is *why* there is no spine. Two findings bound the design:

- **Ungrounded content drives fabrication.** RotoWire grounds only ~60% of its
  summary content in the box score, and that deficiency is what teaches a model
  to emit unconditioned facts. So a fact sheet must be scoped **and sufficient** —
  withholding a number a section must state induces invention rather than
  caution. This bit immediately: reading the taxonomy one level too shallow gave
  that bundle 2 facts and no class names.
- **Precision-only grounding permits selective reporting.** Coverage-aware
  evaluation exists because a system can report only favourable, easy-to-express
  facts and score perfectly on precision. `check_numbers` is precision-only.

**Design.** Two passes: a planner sees a *map* of the run (bundle titles + the
must-cover list, no numbers) and writes its own outline; a writer then does one
section at a time against evidence scoped to that section, carrying its own
outline and the previous section's tail. Structure is global and agent-authored,
grounding is local and mechanical. Four checks per section (numbers, figures,
language, length) plus a whole-document coverage check.

**The vestigial `ReporterAgent` was the natural home** — registered in
`ALL_ROLES`, never called, exactly the anti-pattern CLAUDE.md names. Its shape
(one call, whole report, 60k evidence blob) was precisely the configuration the
literature says fails. Repurposed into `StoryPlannerAgent` + `StoryWriterAgent`.

### Three defects found by running it, not by reasoning about it

1. **Coverage was satisfiable by boilerplate.** Run over the *assembled*
   document, the provenance banner — which names `自下而上聚类最终报告.md` —
   marked "both routes must be explained" covered on a run where no section
   explained either. Now scoped to authored, accepted sections only.
2. **The worked examples were empty.** `deployment.deterministic_exemplars` are
   phrasing-pattern samples carrying no labels; the bundle built to zero facts.
   Rebuilt from `labels_full.csv`, the only place a query carries both routes'
   labels — median-index row per family **plus the lowest-margin rows**, so the
   hard cases are in by construction rather than by the author's generosity.
3. **The panel bundle was 783 facts** — the artifact under a different name.
   Flattened to subject × metric.

### Deliberate: the fallback spine

If the planner fails structural validation three times, a code-chosen section
*order* is used and the document says so. This is the one thing a template may
decide about this report, and it decides only running order — every sentence is
still the agent's. Two reasons: delivering nothing on planner failure is worse,
and the offline stand-in cannot produce a valid outline, which would otherwise
leave the entire section-writing path unexercised by any test that runs the graph.
The stand-in was also taught to emit a number-free Chinese paragraph for
`markdown`, so offline runs traverse write → verify → assemble → coverage.

### Not done

- **Never run with a real model.** No agent-planned outline has ever executed.
- The 22 English gate messages (`GATE_MESSAGE_DEBT`).
- `verify_run.py` has no check for the final report.

---

## Session — 2026-08-27: how the bottom-up path actually decides, and what was wrong

The user asked how alpha, the algorithm, K and the leaf counts are really chosen,
why Phase 4 stopped sweeping algorithms, whether "7 families vs best kmeans k=15"
is a discrepancy, and whether silhouette is under-weighted. A 27-agent audit
raised 44 claims; 15 survived adversarial refutation. Several of the most
important findings came from measurements run here, not from reading.

### The three headline measurements

**1. Silhouette is precise and measures the wrong thing.** Subsample sd is 0.0007
(families) / 0.0040 (leaves) over 15 replicates — it is *not* noisy, and saying so
would be contradicted by anyone who measures. What it cannot do is rank across k:
**Spearman(k, silhouette) = -0.888**, peaking at the sweep's lower bound, so it has
no interior optimum. It also rises monotonically with alpha at every k (would elect
maximum surface weight, the template-twin failure) and, at fixed k, favours exactly
the geometry KMeans optimises (k=15: kmeans 0.070, agglo 0.009 — while agglo has
the second-best stability). **Prediction strength was tested as an alternative and
degenerates the same way** (Spearman -0.895, preferring k=2 at 0.98). All three
intrinsic criteria collapse toward small k; the external reference is a necessity,
not a shortcut. The fix is calibration (`lift_over_null`), not weighting.

**2. The K locator's reference is a decision nobody registered.** Holding
everything fixed and swapping only the reference partition: 6 trusted phrasing
groups -> peak k=12, all 12 groups -> k=12, the 25-class top-down L1 -> **k=25**.
AMI is also scored on only 33.4% of rows. live40 declared its 15-25 domain prior
wrong (`该修的是先验`) on a number that would have **agreed** with that prior under
a reference it already had.

**3. The two routes agree — at the layer nobody compared.** `route_crosswalk`
compared 7 families against 25 top-down classes and printed "routes disagree" on
every row, which 7-vs-25 forces arithmetically. At the leaf layer (25 vs 25):
AMI 0.5395 -> **0.6175**, median single-intent share 39.5% -> **80.3%**, 19/25
leaves majority-one-intent against 1/7 families. leaf 19 生僻字查询 is **100%**
BARE_TERM_LOOKUP. The prior was right about how many intent classes exist and wrong
only about which layer carries them.

### The largest defect: 36% of the delivered leaf layer was unmeasured

`choose_local_k` applies a null test and a stability floor to every leaf the
measured rule creates. `ops/governance.py:split_leaves` applied a `min_size` guard
and nothing else — and it made **9 of the 25 delivered leaves**. The two facts sat
in different artifacts and had never been read together.

Cause, found by measurement: ranking local k on **raw silhouette** hit the small-k
attractor — 5 of 7 families took k=2, the minimum, and none took 4-8. The Phase 7
audit noticed and prescribed 9 splits. **Replayed through `choose_local_k`'s own
tests, all 9 pass** (parent 6: lift 0.876 — a textbook-clean split the rule missed
entirely, and the very leaf that is 100% one top-down intent). The agent was
correctly compensating for a measurable defect in the geometry rule.

So splits are now **measured and not vetoed**, deliberately: a veto built on the
same biased geometry would reject the corrections to its own bias, and a split can
be semantically right and geometrically unsupported. The number ships beside the
split; a failing one is disclosed, not blocked.

### The "15 vs 7" verdict

Not a discrepancy — three different things wear the number 15. `battery_k` is a
diagnostic grid for a falsification probe that never selects K. But it *is* a real
gap that the probe runs at k in {15,20,30} and **never at the delivered k=7**,
below its own grid floor, so "this structure is a property of the corpus" is an
extrapolation across a >2x granularity gap stated as a measurement.

### Fixed this session

- `_rank_local_candidates` extracted and corrected: ranks on `lift_over_null`;
  both deltas measured against the same reference (they were not — `d_sil` vs
  `top`, `d_stab` vs the running `pick`, so the loop was order-dependent); a
  Pareto-dominated candidate can no longer ship (live40 family 3 shipped k=2 while
  k=3 dominated it on both axes). Replay: 16 -> 18 leaves, and **both halves of the
  fix do independent work** (ablated: metric alone fixes families 2 and 3, loop
  alone fixes only 3).
- Battery: `KMEANS_FAMILY` replaces a name-prefix filter that let `minibatch_*`
  and `bisecting_*` count as "structurally different" from KMeans; the margin is
  now **paired within k**, which flips live40's own sign at k=20 (`gmm_diag` ahead
  by 0.060) — invisible under the old unpaired comparison.
- `reference_profile()` records the locator's cardinality and coverage; p5 emits it.
- `route_concordance` artifact compares the routes at **every** bottom-up level.
- `stability_floor` is configurable and threaded (was a bare default in two places
  with no config entry and no test).
- HDBSCAN's claimed Phase-12 role removed from four places — the sentinel is a
  max-centroid cosine percentile and never touches HDBSCAN.
- Report fixes: the "K = 稳定性峰" mislabel (contradicted twice by the same
  report), the retracted "淘汰赛" sentence 23 lines above its own correction, a
  hardcoded noise claim that consulted no number, an English rule string in a
  Chinese figure title, and the English report path's "won a six-algorithm battery".
- The four docstrings claiming `challenger_beats_incumbent` protects the widened
  grid now say it is **not wired**. Deliberately documentation-first: applying the
  toll as written flips live40 to K=10, worse on both reported metrics.

### AMI's own noise, measured for the first time

K is located by argmax of `intent_alignment_ami`, and nobody had ever measured that
metric's noise — `noise_floor()` estimates it from the ROUGHNESS of a single-seed
curve, which is not a standard error over anything. Refitting KMeans with 5
independent seeds at k=7,8,10,12,15 on live40's full corpus:

| k | mean AMI | sd |
|---|---|---|
| 7 | 0.7508 | **0.0009** |
| 8 | 0.7114 | 0.0182 |
| 10 | 0.7495 | **0.0441** |
| 12 | 0.7562 | 0.0225 |
| 15 | 0.7248 | 0.0218 |

Pooled sd **0.0255**, against `noise_floor`'s 0.0129 — understated ~2x. The whole
live40 podium (k7 0.7495 / k10 0.7534 / k12 0.7507) spans **0.0039**, about 15% of
one sd, so **the argmax across k is noise**. Per-seed argmax was [7,10,10,10,10] and
the 5-seed mean argmax is k=12, not the k=10 that shipped as "best".

Two things follow. The tie band is `2 * se` = 0.0258, which lands within 2% of the
real 1-sd figure — so the tie set {7,10,12} is **correct, by an accident of doubling
an estimate that was half the true value**. Do not "fix" the factor without
re-measuring the band. And the noise is strongly heteroscedastic: k=7 is stable to
the fourth decimal while k=10 ranges 0.6728-0.7757 across seeds. **That is a far
better argument for the delivered K=7 than the "simplest tree" tie-break actually
used** — and nothing currently measures it. `noise_floor`'s docstring now carries
these numbers; its previous "validation" was one live38 figure agreeing with a
rounded note.

### Are the "trusted phrasing groups" a real reference, or our own priors?

The K locator scores AMI against them, so this is the question the whole selection
rests on. Findings, all measured on live40:

**They are hand-written.** `trusted = not is_discovered` (`templates.py:169`). All
6 trusted groups are seed regexes from `configs/domains/k12_zh.yaml`; **zero** mined
groups earned trust.

**But they are accurate.** Judged against the top-down taxonomy — an independent
methodology on the same corpus — median single-intent purity is **87.8%** against a
**14.2%** chance baseline, a ~6x lift. verse_continuation 95.8%, pronunciation
90.6%, meaning 88.6%, lexical_relation 86.9%, stroke_order 82.5%, word_formation
81.2%. So AMI is **not** measuring our priors back at us; the groups really are
same-intent. Mined groups are worse but not worthless (median 71.7%) with real
outliers both ways: `suffix:00字` 99.0%, `suffix:是什么` **42.0% across 7 intents**.

**The gate that vets them is anti-correlated with what matters.**
`validate_group_cohesion` scores mean pairwise cosine vs random — TOPICAL
tightness. An intent group spans topics by construction ("X的意思" for thousands of
X), so the gate penalises exactly the broad intent groups that make the best
references. Spearman(lift, purity) = **-0.60** (n=6, p=0.21): it PASSED
`word_formation` (lift 1.670, purity 81.2%, worst of the six) and REJECTED
`meaning` (lift 1.269, purity 88.6%, third best). `kept_because_seeded` — which
read like a courtesy to the human — was on this corpus **the more accurate call**.

**The portability path was silent.** `deps.template_masks(trusted=True)` falls back
to unvalidated mined groups under a comment reading "fall back loudly", with no
log, no gate, no artifact. `generic.yaml` ships **0 seeds**, so that is the default
for any corpus without a hand-written profile. Now emits and raises
`p3_locator_reference_validated`.

**Added:** `locator_reference_validation.json` at p10 measures each group's purity
against the top-down labels. **Limitation to keep in mind: it is retrospective.**
p5 runs concurrently with p2 under the fork, so no top-down labels exist when K is
chosen. It tells you whether the K you got rested on a good reference; it does not
improve the choice.

### The user asked to move p5 after p2 so K could use the top-down labels. Do not.

The instinct was right — the deciding reference is our own hand-written templates —
but the proposed fix is the one thing that breaks the project's headline result.
Measured on live40's full corpus before deciding:

1. **It makes the route-concordance result circular.** "Two independent routes found
   the same structure" (leaf-layer AMI 0.6175, 19/25 leaves majority-one-intent) is
   evidence ONLY because the tree was built without seeing the taxonomy. Locate K
   against `td_l1` and family-layer AMI moves 0.5748 -> 0.6308; that +0.056 is the
   fit, not agreement.
2. **The two-layer design collapses.** `td_l1` locates **K=18** while the delivered
   leaf layer is 25 — families and leaves at the same scale.
3. **`BlindnessFirewall.add_taxonomy` already forbids it.**
4. **It is unnecessary.** `ref_legacy_l1` — the corpus's own labelling, 9 classes,
   complete, external to BOTH routes, available at **p1** — locates **K=18 too**.
   The signal is obtainable free, with no serialisation and no circularity.

The premise also does not survive checking: `td_l1` on all 50k rows is a
**classifier's prediction** (cv_accuracy 0.8625, macro_f1 0.797) from a taxonomy
with κ=0.8427 and adversarial accuracy 0.82. A phrasing group is a deterministic
regex match. And the 87.8% purity figure for the phrasing groups was measured
AGAINST `td_l1`, so part of that 12.2% "impurity" is td_l1's own error — the groups
are plausibly better than the number says.

**Built instead:** `k_sweep` scores AMI against every declared reference column;
`reference_sensitivity()` reports where each one would locate K; `p5_k_references_agree`
fires when they disagree. Decision authority is unchanged — this is disclosure.

**The finding that matters for the next run:** on live40, phrasing groups locate
K=7 while BOTH non-template references locate K=18. The reference that decides is
the outlier. That is now visible in `granularity.json` and gated, but **the delivered
K did not change** — changing it is a methodology call for the user, not a silent fix.

### (3) Why the templates say K=7 — diagnosed, and it is COVERAGE not cardinality

**This corrects an earlier claim in this file and in two rule files.** I had said
the located K tracks the reference's *cardinality* (measured on a 15k subsample).
A full-corpus experiment with a coverage control refutes it.

Fixing the row set at the 33.4% the templates match and varying only class count:
6, 9 and 25 classes **all locate K=7**. Fixing the reference (`td_l1`) and varying
only the rows, all three sets identically sized:

| rows scored | locates K |
|---|---|
| rows our templates match | **7** |
| a random sample of the same size | **18** |
| rows our templates miss | **18** |

Not a subsample-size effect, not cardinality. **The six seed regexes select a
structurally atypical third of the corpus** — narrow lexical-lookup queries — and
K is located for that third, then applied to all of it.

**No label-free repair exists.** Background-as-one-class and downsampled-background
both still return K=7: the templates carry no information about rows they never
match, and reweighting cannot invent it.

### (2) Implemented — and the rule is portable, which was the user's real concern

Their objection to a hardcoded "prefer legacy" was correct: not every corpus has
legacy labels, and not every corpus admits accurate templates. So the rule names no
column. `locator_reach()` measures, **with no external labels**, what fraction of
clusters a reference holds a real share of — usable on any corpus:

- live40 @ k=18: phrasing groups **38.9%**, `ref_legacy_l1` **100%**
- reach is NOT row coverage: 33% of rows spread evenly reaches everything; 100% of
  rows concentrated in two clusters reaches almost nothing

`clustering.k_locator: auto` (default) gives the highest-reach reference the
locator role; `p5_locator_reaches_the_corpus` fires below 0.80 reach, saying the K
is scoped to the part of the corpus its reference could see.

**Consequence to expect on the next live run: K will change.** On live40 this hands
the locator to `ref_legacy_l1`, which locates K≈18 rather than 7 — a materially
different tree, cascading into leaves, naming and every downstream count. Nothing
was re-run, so this is untested against a real model. `k_locator: phrasing`
restores the old behaviour exactly.

### Not done — and one is now the top open question

- **`challenger_beats_incumbent` still has no call site.** Needs a signature change
  (`propose_grid` returns a flat list, so selection cannot tell a proposed value
  from a configured one) and a redesign, not a call.
- **The alpha sweep decides inside its own noise.** Winners across 5 seed
  replicates: 0.1, 0.5, 0.1, 0.0, 0.1. `tie_band=0.05` is an unreachable default
  ~4.5x narrower than the metric's own spread. **Do not simply widen it** — at a
  measured 2-sd band live40 elects alpha=0.5, which its own k=7 panel shows
  fragments worse. The fix is replication, not a wider band.
- The probe still does not run at the delivered K.
- Phase-8 metric deltas are one aggregate stamped on every prescription and
  computed on pre-split labels (+0.061 recorded vs +0.250 delivered).

---

## Session — 2026-08-27 (evening): a browsable dashboard, and three bugs it exposed

The terminal panel keeps `agents[-8:]`, `activity[-6:]`, `metrics[-8:]` because
that is what fits on a screen. live40 made **696 agent calls** over four hours, so
the panel showed roughly **1%** of the run and dropped the rest as it went. The
questions an operator actually has — what did that agent return, what exists so
far, what is queued — need scrolling, folding and search.

**`ui/web.py` renders `LiveDashboard`'s state as `runs/<id>/dashboard.html`.** One
model, two views, so they cannot disagree. A file rather than a server: no port,
no dependency, no lifecycle, and it keeps working after the run because the page
IS the record. Atomic `.tmp`-rename write, throttled to 3s, meta-refresh until the
run finishes, plus a daemon heartbeat every 10s.

### Three bugs found by replaying live40's own log

1. **The panel misreported the fork.** `current` was a single phase, so when P2a
   and P3a both start at 14:56:08 one branch was marked done while still running —
   p3 visibly flipped done → running. Completion is now **emitted** by `_wrap`
   (`✔ <node> completed in Xs`) instead of inferred from "a different phase
   started", which cannot be right for a forked graph. A branch phase never closes
   the other branch; a spine phase closes both, because reaching the spine means
   the join happened.
2. **`qmine watch` on a finished run showed every phase as 0s.** Timings used
   `time.time()`, and a replay reads 1000 lines in under a second — the follower
   built for re-watching timed nothing. `parse_log_clock` reads the log's own
   timestamp (midnight-safe). Replaying live40 now reproduces **241.8 min**,
   matching `run_summary.json` exactly, with p2a 72m / p2b 79m.
3. **Inferred durations were inflated** — live40's p3 rendered as 72 min when its
   work took ~12; the rest was the branch waiting at the superstep boundary.
   live41 measured it directly: **p3 = 1548.6s**.

Both fixes are mutation-tested.

### Two things found by looking rather than assuming

- Embedding all 696 full returns produced a **5 MB page that hung the browser's
  renderer**. Now 457 KB: newest `DETAIL_LIMIT=40` in full, older keep their
  result line and point at `agent_transcript.json`.
- The page **froze during quiet stretches** — it only rewrote on events, and a
  40-minute researcher fan-out emits almost none, so the meta-refresh reloaded the
  same stale elapsed time. Fixed with the heartbeat thread. **live41 started
  minutes before that landed, so its page ticks on events only.**

### live41 (running)

`--config configs/live.yaml --reference-columns legacy_l1,legacy_l2 --provider
router`. Pre-flight clean: annotator labs independent, est. $9.11, `reporter`
routed. Watch for **K landing near 18 rather than 7** if `legacy_l1` wins the
locator role on reach — that is the change from this morning and it has never run
live.

### Found and fixed DURING live41: a rule id that resolves to nothing

`p2a_observer` WARNED with a machine-confirmed check: 3 node-cited
`adjudication_rules` ids exist in no rule. Verified rather than assumed — 6
citing slots across 4 classes, referencing `RULE_POEM_TEXT_CHILD_OF_FULL_TEXT`,
`RULE_MATH_SCIENCE_OVER_FULL_TEXT`, `RULE_FULL_TEXT_RESOURCE_OVER_MATH_SCIENCE`.
The architect had invented a second id convention (SCREAMING_SNAKE with a `RULE_`
prefix) against the registry's lowercase snake_case.

**The consequence was concrete.** `_render_rules` treated a bare id as a
cross-reference only when it MATCHED a real rule; a dangling one fell through to
the free-text branch and reached the annotator's guide as

    - [POEM_TEXT_LOOKUP] RULE_POEM_TEXT_CHILD_OF_FULL_TEXT

— a line that looks like a rule, carries no adjudication content, and consumes a
budgeted section. So those boundaries had no tie-break **and** the guide gained
three lines of noise. Dropping is strictly better than passing through.

Fixed: `_is_bare_identifier` (ASCII-only, so an English rule's spaces and a
Chinese rule's CJK both survive) drops it, and `dangling_rule_references()`
reports every citing slot. live41's taxonomy renders 42 lines instead of 45 under
the fix. Mutation-tested.

**live41 is unaffected by the fix** — it passed p2a before the change, so its gold
set was annotated with the three noise lines present and those two boundaries
unruled. Worth checking whether POEM_TEXT/FULL_TEXT and MATH_SCIENCE/FULL_TEXT
show up as confused pairs in its referee log.

### live41 exercised the reach rule — and showed reach alone is NOT enough

The machinery worked exactly as designed and the design was incomplete. Reach
reproduced the offline measurement on fresh data (phrasing groups **40%** @ k=20
against 38.9% @ k=18 offline), `legacy_l1` won on 100% reach, and K moved 7 → 12.

**But 12 came out of an eight-way tie** — `[12,18,20,25,30,40,50,65]` — with the
raw argmax at 30. `min(tie_set)` returned the smallest of eight indistinguishable
values. Measured on live41's own sweep:

| reference | reach | range | range/se | peak K | tie set |
|---|---|---|---|---|---|
| phrasing groups | 40% | 0.2190 | 17.5 | 6 | 4 |
| **`legacy_l1`** ← chosen | 100% | 0.0373 | **5.9** | 30 | **8** |
| `legacy_l2` | 100% | 0.1072 | 17.5 | 18 | **3** |

Reach picked the LEAST discriminating of the three. `legacy_l1` is nine coarse
classes with two holding ~79% of the corpus, so its entire curve spans 0.037
against a 0.0126 tie band. The honest reading of its output is "this reference
cannot tell 12 from 65 apart", and it was reported as a located K.

**A locator must do two things: see the whole partition, and tell different K
apart.** `discrimination()` = range / `noise_floor`, and `choose_locator` now
requires reach ≥ 0.80 first, then maximises discrimination. On live41's data that
picks `legacy_l2` — same reach, 3x the signal-to-noise, a 3-way tie, locating
**K=18**, which is what every full-coverage reference gave on live40.
Mutation-tested against reach-only.

**live41 keeps K=12** — it passed p5 before the change.

### Also found by live41's observer: the decision record named the wrong locator

`evidence.locator` said `ami_vs_legacy_l1` while the same record's `rationale` and
`decisive_metrics` were hardcoded to the phrasing groups, left over from when they
were the only locator. Both now read from `tri`, and `deciding_reference` and
`locator_reach` are recorded beside them.

### Closed during the run

The observer machine-confirmed the alpha `chosen_by` mismatch flagged in the
audit: the string said "lowest template_fragmentation, broken on stability", and
the winner (alpha=0.1) is not the minimum (alpha=0.0 is). That is the tie band
working as designed and the sentence describing something else. `chosen_by` now
states the actual rule. **The selection itself is unchanged** — this was a
description defect, not a measurement one.

---

## Session — 2026-08-27 (night): three models silently became reasoning models

live41 gen01 was halted at **$28.14** against a $9.11 estimate, with the referee at
an **88% failure rate** and one batch of 11 rows abandoned outright. One root cause
explained all of it.

### The cause, probed directly rather than inferred

`deepseek-v4-flash`, `glm-5.2` and `qwen3.7-plus` all began returning
`reasoning_content` under the SAME model names since live40, and the tokens are
billed. One trivial prompt against each endpoint:

| model | completion tokens | of which reasoning | with `thinking: disabled` |
|---|---|---|---|
| deepseek-v4-flash | 505 | 497 | **9** |
| glm-5.2 | 237 | 227 | **10** |
| qwen3.7-plus | 586 | 573 | **10** |

Identical answers in every case. `enable_thinking: false` does NOT work on
DeepSeek (still 1,037 reasoning tokens); `thinking: {"type": "disabled"}` works on
all three.

### It caused three symptoms that looked unrelated

1. **Cost.** annotator_a went from 202 to 1,030 output tokens per label. Not the
   prompt — input HALVED (14,714 → 7,836/call) while output rose 5x, so the
   out/in ratio moved 0.35 → 3.25.
2. **The ValueError failures.** Reasoning consumed the output budget before the
   JSON was written, so responses truncated and surfaced as "no parseable
   structured output". **Plain-JSON mode could not fix this** — both models were
   ALREADY flagged `no_native_schema: true` in `.cache/model_quirks.json` from a
   previous run and failing anyway, which is why no switch message ever appeared.
   The schema wrapper was never the problem.
3. **The APIConnectionErrors.** Not the operator's network — proven by temporal
   segregation: qwen failed alone 20:15-21:15 while zhipu was clean, zhipu failed
   alone from 21:30 (the moment the referee fired 30 calls), and only **1** of 48
   failures had a cross-provider co-occurrence within 60s. glm-5.2's failure rate
   went 14% → 86% with load. Calls held connections open for 200-1000s because of
   reasoning; short calls should stop that.

### Fixed

`reasoning_kwargs(role, provider)` sends `thinking: {"type": "disabled"}` for
`NO_REASONING_ROLES` (annotators, referee, namer) on `REASONING_TOGGLE_PROVIDERS`
(deepseek, zhipu, qwen — an allowlist, because an endpoint that rejects an unknown
field fails the call). Deliberation roles keep their reasoning: an architect or an
observer is where those tokens earn their cost. Mutation-tested both ways.

**Output caps were deliberately NOT lowered** and **concurrency was deliberately
left at 8** — one variable at a time, and reasoning alone may fix the connection
errors by shortening calls ~20x. If they persist, lower `max_concurrency` next.

### Two defects of MINE, found by running it

**1. The reasoning field went in the wrong place.** I put `thinking` in
`model_kwargs`. LangChain forwards those as TOP-LEVEL arguments to the OpenAI SDK,
so every annotator call raised `TypeError: Completions.create() got an unexpected
keyword argument 'thinking'`. gen02 lost **24 annotation batches in one second**
and the pilot gate reported "kappa nan on 0 queries". Cost $0.62 — the calls failed
in 0.0s with nothing billed, and the gate caught it immediately.

The failure behind the failure: I verified the parameter was SET in the kwargs dict
and that `_build_routed` called the rule, and both were true while every call
failed. I had probed the raw HTTP API (which works) but never a call through the
actual client. Vendor body fields belong in **`extra_body`**. Now verified by real
calls: `annotator_a` → 9 completion tokens, reasoning None; `taxonomy_architect` →
124 tokens, 114 reasoning, both succeeding.

The test was strengthened to assert the field lands in `extra_body` specifically —
the original asserted the rule returned the right dict and that the builder called
it, which passed green while nothing worked.

**2. `new-generation --from-generation` defaulted to 1.** So running it on a run
already at gen02 created "the next after gen01" — gen02 AGAIN — and overwrote it
rather than advancing. That silently put a resumed run back into the generation
whose pilot had just failed, with nothing in the typed command to say so.
`artifacts.latest_generation` exists precisely because both RESUME paths had this
same hardcoded 1 (its docstring records that); this call site was missed. Now
defaults to the newest generation, verified by the command itself printing
"generation 3 inherits generation 2's config".

### Three more dashboard defects, all from one operator observation

The operator reported "my dashboard is still displaying old messages". Three
independent bugs, all mine from this session, all with that one symptom.

**1. A `--resume` run never had a dashboard at all.** The resume branch calls
`resume_run(...)` and RETURNS before the fresh-run setup — its own comment warns
"this branch returns before the fresh-run setup below ever sees it", and I walked
into exactly that. `resume_run` already accepted `on_event`; the CLI never passed
it. Both paths now go through `_attach_dashboard`.

**2. The page was gated on the terminal panel.** I built the HTML writer inside
`if use_dash:`, and `use_dash = dashboard and verbose` — both off without a TTY.
So a detached run, the case that most needs a browsable page because there is no
terminal to watch, wrote none. `enabled` now only switches the Rich view off.

**3. `qmine watch` fought the run for the same file.** Both wrote
`runs/<id>/dashboard.html`, and the follower always won on content because it
replays `run.log` from offset 0 — and `run.log` is append-only ACROSS generations,
so a run at gen03 had gen01 and gen02 events rendered into the page being read as
current. The follower now writes `dashboard.watch.html`.

**Plus a display bug the above exposed:** `_clock` ran `time.localtime` on a
REPLAY timestamp (seconds since midnight), so 22:03:48 rendered as "06:03:48" once
the timezone offset was applied. Durations were unaffected — differences cancel —
which is why it went unnoticed. Now discriminated on magnitude.

**The lesson, twice in one session:** I verified the writer was CONSTRUCTED rather
than that a page APPEARED, exactly as I earlier verified a kwarg was set rather
than that a call succeeded. Both times the check passed while the thing did not
work. Verified this time by watching the file appear, refresh, and contain only
the current generation.

### gen03 CONFIRMED every fix from today, on live data

**The reasoning fix.** annotator_a went 1,030 -> **67** output tokens per label
(15x), annotator_b 254 -> 70. The pilot completed **200/200 with zero rows lost**,
where gen02 lost all 200. Errors 53 -> 4, APIConnectionError 33 -> 2, batches lost
24 -> 0, spend $28.14 -> $1.38 at the same point.

**The battery fixes.** `best_alternative = agglo_average_k15` — correctly NOT a
KMeans-family variant, which the old name-prefix filter would have allowed. Paired
within-k margins expose the sign flip the unpaired comparison hid entirely:
k=15 **-0.0776**, k=20 **+0.0603** (gmm_diag ahead), k=30 **-0.1574**.

**The discrimination locator — and it vindicated the correction.** Measured live:

| reference | reach | discrimination | outcome |
|---|---|---|---|
| phrasing_groups | 0.40 | 17.48 | rejected on reach |
| `legacy_l1` | 1.00 | **5.88** | rejected on discrimination |
| **`legacy_l2`** | **1.00** | **17.19** | **locates K** |

**K = 18, tie set [18, 25, 40].** Against gen01's reach-only rule, which gave
K=12 out of an **eight**-way tie. 18 is the value every full-coverage reference
gave on live40. The two-stage rule (reach admits, discrimination decides) is doing
exactly what it was corrected to do.

### Small honesty fixes made during gen03

- `battery.note` still described the retracted "election" ("the winner is then
  fitted on the full corpus"), contradicting `verdict.role` in the same artifact.
  live41's p4 observer caught it.
- The probe's materiality bar (0.10 ARI) was applied and never recorded, so the
  conclusion "no alternative is MATERIALLY more reproducible" could not be checked.
  Now `materiality_threshold_ari` in the verdict.
- The p5 log said "K located by X (highest reach)" even when reach was TIED and
  discrimination decided — on live41 both legacy columns reach 1.0. Now names the
  criterion that actually broke the tie.

### A confirmed observation that was NOT a defect

`p2a_observer` WARNED: one query appears in two classes. Verified — it is a
`positive_example` of RIDDLE_BRAIN_TEASER and `source_evidence` for
RISK_COMPLIANCE_INTERCEPT. Different roles (one teaches a label, one records what
motivated the class), and **both classes are risk=True**, so handling is identical.
Checked the sharper property directly: 136 positive examples, 136 distinct
queries, **zero** duplicates and zero positive/negative contradictions. Another
instance of confirmed != defective.

### gen03

Clean branch from gen02, resumed off the 330-entry run-level cache — the five
researchers replay in under a second against gen01's 5+ minutes each. gen01 and
gen02 both kept as evidence.

**Note gen01's K=12 is superseded twice over** — it used the reach-only locator
AND ran before the discrimination fix.

---

## The resume rewind silently drops a concurrent branch

**The most serious defect of the day, and the hardest to see.** live41 gen03 ran
for 40 minutes through P4, P5 and P6 emitting healthy gates while the entire
top-down branch was missing.

| run | how it started | P4 | P2b |
|---|---|---|---|
| gen01 | fresh, no resume | 20:24:58 | **20:24:58** — same second |
| gen03 | resumed, then **rewound** | 23:45:23 | **never ran** |

`--resume` found a partial checkpoint and took the rewind path,
`graph.update_state(config, {...}, as_node="p1_audit")`. The fan-out survives for
the FIRST superstep — p2a and p3 both started — and not the second: `p456_tree`
was queued from p3's completion, `p2b_gold` never was. `phase_status` confirms it:
`{p0, p1, p2a, p3}` only, `halted: False`, no halting gate, `gold.csv` absent.

**This is NOT caused by the `--from-generation` fix** — that only chose a
directory. It is the pre-existing rewind path, which a fresh run never touches.
Restarting gen03 three times is what put the run on it.

**Why it is the worst shape of failure:** everything downstream still produced
artifacts and passed its gates, so nothing looked wrong. The run would have
reached the join and trained the classifier on a gold set that does not exist.

**Guard added at the join.** `_require_both_branches` reads `phase_status` and
halts with `p2c_both_branches_arrived` naming the phases that never ran. The gate
is RETURNED rather than registered — `deps.gate` only builds the record, and a
discarded gate cannot halt anything, which is exactly what
`test_the_delivered_leaves_gate_reaches_the_operator` was written against.
Mutation-tested: discarding the gate fails the test.

**The rewind itself is NOT fixed** — only made loud. Open question: whether
`update_state(as_node=...)` can restore a multi-superstep fan-out at all, or
whether a gap in `phase_status` should force a clean re-run of the generation
instead of a surgical rewind. Until then: **open a new generation and run it once,
without restarting mid-flight.**

## live42

live41 abandoned. Its three generations stay as evidence: gen01 (reasoning
tokens, 88% referee failure), gen02 (my `model_kwargs` TypeError), gen03 (the
dropped branch). live42 starts cold — no cache — which is the price of a clean
test of every fix at once.

---

## live42 results as they land (fresh run, all of today's fixes active)

**Gold set — better than live40 on two of three measures, with reasoning OFF:**
kappa **0.8928** on n=2999/3000 (live40: 0.8427), adversarial estimated accuracy
**0.9333** (live40: 0.82), classifier CV 0.857 (live40: 0.8625). The reasoning
disable did not cost label quality. NOT a controlled comparison — different
taxonomy, 21 L1 classes against 25 — so this is consistent evidence, not proof.

**`p2b_annotator_symmetry` fired for the first time ever:** annotator_a won only
34.1% of 270 contested rows, z=-5.2. There is NO baseline (the gate postdates
live40 and live41 gen01 never finished its referee), so this cannot be attributed
to the reasoning change. The parsimonious reading is the pairing itself —
`deepseek-v4-flash` against `qwen3.7-plus` is flash tier against plus tier.
Testing the reasoning hypothesis needs a paired re-run of the same rows.

**`p2b_rules_match_their_evidence` found 2 vacuous discriminators:**
`academic_knowledge_qa x problem_solving` names 求/解/计算 with **0 of 19**
contested rows carrying any; `navigational x school_info` names 主页/入口/好不好
with **0 of 8**. Rules naming a marker that appears in none of the rows they
adjudicate.

**The split measurement DISCRIMINATES — 7/9, not 9/9.** live40's retroactive
replay passed all nine, which made it look like a rubber stamp. live42:

| parent -> new | lift | stability | supported |
|---|---|---|---|
| 19 -> 54 | 0.0610 | **0.0595** | **no** |
| 0 -> 49 | 0.0778 | **0.3876** | **no** |
| 7 others | ok | 0.68-1.00 | yes |

Both failures are on STABILITY, not the null test — real structure that does not
reproduce. And the naming corroborates independently: leaf 19
「作文范文与写作指导查询」 split into leaf 54 「作文范文查询」, a near-duplicate of
its parent, at an ARI of 0.0595 (chance). The namer never saw the stability
number and the measurement never saw the name.

**This vindicates measure-don't-veto.** A veto would have blocked a split the
audit had semantic reason to want; disclosure lets a human weigh both.

**Tree:** 18 families / 49 leaves pre-governance, per-family k spanning 1-8
(`{1:1, 2:9, 3:4, 4:3, 8:1}`) against live40's 5-of-7-at-the-minimum. Half still
sit at k=2, so the small-k attractor is weakened, not gone.

**`converged: False` for the third run running.** The refinement loop hits its
5-round limit on live40, live41 and live42. Honestly disclosed in the report, but
a standing weakness: the delivered leaf count depends on which round it stopped at.

## The narrative report's FIRST live run: 3/9 sections, and why

The agent-written report ran against a real model for the first time. The
structural machinery worked: **the agent-authored outline passed on attempt 1**
(9 sections, the path the offline stand-in could never exercise), and **6 of 7
must-cover items were covered**, with the seventh disclosed in the document.

**But only 3 of 9 sections passed**, and the guardrails failed closed on the rest
— safe, and not useful. Two distinct causes, one of them mine.

### Cause B (fixed): the pool did not match the sheet

`sheet()` renders dotted paths, so a dict keyed by id displays numbers to the
author that `verify._flatten` — which pools only VALUES — refuses:

    execution.splits.32.new_leaf = 49      <- the agent reads "32"

`governance_and_risk` was rejected three times for citing `32, 40, 42, 43, 44,
45` — the leaf ids it had just been shown — and shipped as a hole. It was doing
exactly what it was told. `citable_numbers()` now pools every number the rendered
sheet SHOWS, and no more: an id absent from the text is still refused, verified
both ways and mutation-tested through `_reject` (a first version tested
`check_numbers` directly and the mutation passed — the test has to walk the path
the code walks).

### Cause A (OPEN): three sections returned EMPTY prose, three times each

`vector_choice_first`, `two_level_tree` and `samples_and_deployment` each failed
with `空白正文` on all three attempts — the model returned an empty `markdown`
field, not a wrong one. That is not a grounding failure and the retry feedback
cannot help it, because there is nothing to give feedback on.

Suspected but NOT established: 10 `prompt block truncated` events fired during
the report (fact sheets of 50-53k against a 40,000-char budget). Note the total of
all 16 bundles measured only 33k with EMPTY gates/decisions, so the live gates and
decisions bundles are what push a section over. Whether truncation causes the
empty returns is unverified — do not fix it as though it were established.

**Next step for this: reproduce one empty section offline against the recorded
sheet.** The section ids and their bundles are in `final_report_meta.json`.

## live42 finished — 17/17 phases, and what the narrative report is really like

All 17 phases completed and every deliverable shipped. **The run then crashed at
teardown**: `DecisionRecord` is no longer msgpack-serializable, so the SQLite
checkpointer failed, fell back to in-memory, and the store's context manager
raised `RuntimeError: generator didn't stop after throw()` on exit. Consequence:
**`run_summary.json` was never written** — the artifact `verify_run.py` and
several tests read. The deliverables themselves are intact.

### The pre-delivery auditor worked for the first time: 3 applied, 0 refused

On live40 it made 4 proposals and MY bugs refused all four. Here all three landed,
and they are good:

- `统一度量面板.md` — the prose claimed fragmentation "rises monotonically with
  cluster count" while the panel's own table falls from 1.8661 (k=18) to 1.827
  (k=24). It caught the report contradicting its own table.
- `自上而下类目体系最终报告.md` — a count mismatch around
  `n_triggers_rejected=39` with only 12 shown and 27 truncated.
- `00_最终报告.md` — the narrative report said the references "disagree"
  QUALITATIVELY without the numbers, so a reader could not follow the gate's own
  instruction to read the K together with its reference. It added the values.

One finding was dropped rather than applied: it cited `annotator_balance.n_contested`,
which does not resolve. The content was real (294 disagreements vs ~270 contested
rows adjudicated — two populations again), and the citation guard correctly
refused an edit it could not source.

### The report reads well where it passes — and has an error the checks cannot see

§4 explains why the locator is `ami_vs_legacy_l2`, names the competing indicators
(silhouette peak K=5, expert range 15-25, deep_aligned 28), reports the FULL tie
set 18/25/30/40 with every metric, shows the metrics disagreeing, embeds the
figure where it argues, and hands off to the next section.

**But it writes 「交付的 K=18 是参照 phrasing_groups 的粒度锚点」 — naming the
WRONG reference.** K=18 was located by `legacy_l2`; `phrasing_groups` located 10,
which the same paragraph states correctly two lines earlier.

`check_numbers` is precision-only on NUMBERS. A wrong noun is invisible to it, and
the must-cover anchor matched because every reference name appears somewhere in
the text. **The narrative door has a numeric guarantee and no ATTRIBUTION
guarantee** — same class as the `locator_reference` contradiction the p5 observer
caught. This is the top open item for the report, ahead of the empty-section
problem.
