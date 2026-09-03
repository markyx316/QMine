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

## 1. Status — last updated 2026-09-02

# 691 tests passing. `mode="fast"` is built and VERIFIED end-to-end on a paid run (`fin03`): 17/17 phases, verify_run **21 PASS / 6 N/A / 1 SKIP / 0 FAIL**.

(The six N/A are checks whose components fast mode skipped — a fast run's PASS
count is NOT comparable with a full run's.)

**The one thing to know:** `--fast` and `--smoke` are different things. `--smoke`
(was `fast_mode`) shrinks the analysis for a wiring test; `--fast` (`mode="fast"`)
keeps the analysis at full size and removes the second-opinion layer. `fin03`
confirmed the distinction on real models — full α grid, full derived 3,000-row
gold set, while kappa, the pilot and adversarial validation were ABSENT rather
than faked.

| | |
|---|---|
| Tests | **691** passing (was 665); `ruff --select F src/qmine/ tools/` clean |
| Newest run | `fin03` — fast, 10,000 finance rows, 17/17 phases, `routed`, 193 calls, **1.35 h**, est. **$3.19** |
| `fin03` quality | CV **0.914**, macro-F1 **0.839**, PV-weighted **0.941**, ECE 0.025, coherence 3.86 |
| `fin03` shape | 16 L1 intents / 21 delivered leaves / 15 families, all named |
| `fin03` gates | 16 recorded; 4 `skipped` (the fast-mode four), 1 `warned` (locator reach 20%), 0 failed |
| Verification | `verify_run`: `fin03` **21 PASS / 6 N/A / 0 FAIL / 1 SKIP**; `fin01` control 19/6/2/1 |
| Namer | pinned `deepseek-v4-pro` — over ALL namer roles at end of run: max **1701.6 s → 136.8 s** (12.4x), p50 74.9 → **31.9 s**, failures 3 → **0** |

**Everything that needed a live run got one.** The gate guards (a check that did
not run reports `skipped`, never `passed`), the solo-annotator schema check (no
phantom classes), the delivered-partition tree filter (no stale `家族 N`
headings) are confirmed against `fin03`, with `fin01` as the control that still
shows the two failures `fin03` clears. The store-resolution fix is NOT confirmed
by `fin03` and cannot be: it wrote its deliverables into gen01, where there is no
cross-generation lookup to exercise. Its confirmation is the `fin02` RENDER into
gen02 — 10,000 rows and 10 populated sheets where the pre-fix code produced 0 rows
and 8 empty ones.

**Cost/time, measured, for anyone sizing a run:** full 50k — `live39` 3.4 h/$5.52,
`live40` 4.0 h/$7.01, `live44` 9.8 h/$61.09 (841 calls vs 696, 8x the bill: one
expensive model). Fast 10k — **1.35-2.40 h / $3.19-$4.23** across the two complete fast runs
(`fin03`, `fin01`); quote the range, not the better one. The
closest like-for-like on 10,000 rows is `med04` (full) 7.9 h/$65.05 against
`fin03` (fast) 1.35 h/$3.19 — different corpora, so an order of magnitude rather
than a ratio.

**A full run ships 10 markdown documents, 6 CSVs and a notebook** (`med04`,
`live44`; `live42` shipped 9 — it varies with whether the narrative and audit
reports succeed). The docs and the fast-mode banner said "13" until today; the
banner no longer states a full-mode count at all, because a hard number there
goes stale silently inside every shipped document.

### What the live44 examination found, and where it landed

Seven defects, **none of which `verify_run.py` caught** — all came from reading
the deliverables. All seven are fixed:

| | defect | fix |
|---|---|---|
| D1 | `00_索引.md` claimed 21 L1 classes; the taxonomy has 20 | index counts derive from the taxonomy; `_WHAT_FOR` may not contain a digit |
| D2 | `heldout_reproduction` published under one name with two values (0.9853 pre-governance / 0.9748 delivered) | both now name their partition |
| D3 | `20 L1 intents across 1 axes` — an English sentence — on 7 lines of 3 Chinese deliverables | decision choice is symbolic (`L1 = 20, axes = 1`), like `alpha = 0.1` |
| D4 | `taxonomy.axes` empty while D003 counted 1 axis | registry derives from the nodes |
| D5 | delivery auditor shown **39%** of the deliverables | `budget_units` — whole documents, count in log and in-band |
| D6 | 2 of its 3 findings deleted on a citation technicality | `citable_namespace` — resolver AND check evaluator |
| D7 | maintainer failed 3x (44 min, `out 0`), `✔ completed`, disclosed nowhere | the failure is emitted |

**D1 and D2 are the sharpest result of the run.** The pre-delivery audit — the
last check before shipping — found both, and the pipeline discarded both. It did
that while having been shown 39% of the material. Whatever else is wrong is
likely in the 61% it never read.

### Family names and duplicate leaves (raised 2026-09-01, both fixed)

**Delivered families had no names.** `混合·主要成分「词语含义查询」45%` was being
used AS a family's title — in headings, table cells, a Mermaid node and a CSV
column. Cause: the tree auditor names the **Phase 7** tree, governance then
merged 18 families into 14 and isolated them back out to 23, and a delivered
family routinely spans several audit families, so no audit name is simply "its"
name. p8 now names the delivered partition directly (`FamilyNamerAgent`,
`families_final`), the same way it already re-names the leaves governance
changed. `_shape.family_names` prefers those; the composition label survives
only as the fallback when naming fails.

**The tree could only ever fragment.** `PrescriptionKind` had `merge_families`
but **no `merge_leaves`**, so the auditor's `duplicate_leaf_pairs` was a
write-only measurement. live44 listed **14 duplicate pairs** with cosines and
reasons — including leaves 12/14 ("汉字读音查询重复，任务无法区分") and 27/29
("偏旁部首查询重复，仅提问方向相反") — prescribed nothing on any of them, and
shipped a tree with **two leaves carrying byte-identical names in the same
family**. Governance even split leaf 30 into {30, 50} while the auditor had
flagged {25, 30} as duplicates, and both halves got the same name.

`merge_leaves` now exists with an executor (folds into the smallest id, so
re-runs are stable), and the auditor prompt requires every listed pair to get a
disposition — a merge or a documented `keep_as_is`.

**Still open on this:** the duplicate audit runs in p7, *before* p8 creates new
leaves, so a duplicate governance itself introduces (30/50) is never audited.
Same shape as every other "gate before the operation that breaks its invariant".
See §2.

## 2. Open questions — EDIT THIS SECTION, DO NOT APPEND

Resolved items are **deleted** here and their resolution recorded in that
session's log below. A struck-through entry is a maintenance failure, not a
record.

### P1 — worth doing next

0i. **`model_overrides` silently ignores a suffixed role.** Routing resolves
   `researcher_log_reading` to its BASE role `researcher` before looking up a
   model, so only `researcher` is ever consulted. An entry keyed on the suffixed
   role sits in the config doing nothing — no warning, no log line, and
   `qmine models` even ECHOES it back ("researcher_log_reading=deepseek-v4-pro"),
   which makes it look applied.

   Found by trying to route around 0h and watching med03 fail at 903.7s on the
   very model the override named away from. Per-angle pinning is not supported
   and currently cannot be discovered except by a run.

   Fix: either consult the suffixed role before falling back to the base (the
   `_prefix_route` longest-match rule already does this for `_routed`, so the two
   tables disagree), or REFUSE an override whose key resolves to no routable
   role, so dead config fails loudly at startup instead of being echoed as if
   live.

0h. **`researcher_log_reading` fails deterministically at ~903s — now THREE
   runs.** med01 903.5s, med02 902.9s, med03 903.7s. All `InternalServerError`
   with an HTML body and `out 0`, on glm-5.3-flash, reproduced to within 0.8s
   across three independent runs. Refuted explanation: not a duration ceiling —
   `risk_compliance` completed a single uninterrupted 1,024.8s call on the same
   model. Request-specific; `log_reading` reads raw corpus rows, the largest and
   least structured payload any researcher gets.

   The obvious workaround (pin the angle elsewhere) does NOT work — see 0i — and
   the obvious target is forbidden: the comment above the researcher pin records
   that deepseek-v4-pro 400s on `tool_choice` for researcher roles, returning
   parametric-knowledge candidates with zero tool calls. I proposed that target
   without reading the warning directly above the line I edited.

   Cost: ~15 minutes plus a retry, every run. Not data loss — the retry succeeds.
   Next step: fix 0i first, then pin this ONE angle to a provider that is neither
   zhipu (fails) nor deepseek-v4-pro (tool_choice 400) — e.g. the moonshot or
   qwen tier already in the plan — and confirm from the run log, not from
   `qmine models`, which echoes overrides it has not applied.

0g. **The alpha optimum can sit at the GRID'S EDGE with the metric still
   improving, and nothing says so.** Machine-confirmed by the p3 observer on
   med03: `alpha_sweep.chosen_alpha < max(grid_proposal.widened)` FAILED.

       alpha  0.0    0.1    0.2    0.3    0.5    0.7    1.0
       frag   2.487  2.433  2.719  2.623  2.204  1.771  1.481  <- best, at the edge
       stab   0.841  0.650  0.765  0.642  0.752  0.808  0.642  <- worst, at the edge

   Template fragmentation is still FALLING at alpha=1.0, the largest value
   searched, so the grid does not bracket the optimum — alpha > 1.0 may be
   better and nothing looked. At 1.0 the phrasing block controls **50%** of the
   cosine (`surface_vote_share = a^2/(1+a^2)`), which is a large representational
   commitment to make at an unexplored boundary.

   The winner is also the LEAST stable point in the sweep (0.642 against 0.841 at
   alpha=0). That is legitimate under the documented rule — fragmentation
   locates, stability only vetoes, and 0.642 clears the floor — but "best on the
   deciding metric, worst on the veto metric, and at the edge of the searched
   range" is three facts a reader should get together, and currently gets none of.

   NOT a reason to extend the grid automatically: `ops/propose.py` is blind to
   scores ON PURPOSE so its additions are pre-registered, and widening because
   the winner sits at the edge would use the scores it must not see.

   Fix (AFTER med03): a boundary DISCLOSURE, not an automatic extension. When the
   chosen value is the min or max of the swept grid AND the deciding metric is
   still improving in that direction, say so in the artifact and the report —
   "the optimum was not bracketed; the true optimum may lie outside the searched
   range". Same treatment as the singleton-agreement and cross-k fixes: the
   measurement stands, the overclaim goes.

0. **The duplicate audit runs BEFORE governance creates duplicates — PARTLY
   ADDRESSED.** p7 audits the tree, p8 then splits it, so a duplicate governance
   itself introduces was never audited (live44 split leaf 30 into {30, 50} and
   both halves were named `汉字笔画数查询`).

   `p8_leaves_are_distinguishable` now measures the DELIVERED partition after
   naming, and `DisambiguatorAgent` either names the difference or prescribes
   `merge_leaves`. The K12 demo proved it works: it caught `的拼音相关查询` on
   leaves [15, 16] on its first run.

   STILL OPEN: the gate is deterministic on EXACT name equality only. Semantic
   near-duplicates (live44's 27/29, 部首 vs 偏旁部首 — different strings, same
   concept) are caught only by the auditor's cosine list, which returned `null`
   for 4 of the 14 pairs it reported on that run. A geometric duplicate check on
   the delivered partition would close it; it needs a threshold, which is exactly
   what the exact-match gate was designed to avoid. p7 audits
   the tree, p8 then splits leaves — so a duplicate governance itself introduces
   is never audited. live44 split leaf 30 into {30, 50} and both halves were
   named `汉字笔画数查询`, byte-identical, in the same family. The split had real
   geometric support (lift 0.1565 over null, ARI 0.9887); it was **semantically**
   empty, and nothing measures that.

   `merge_leaves` (added 2026-09-01) lets the auditor act on duplicates it CAN
   see. It cannot see these. Options, none yet taken: re-run the duplicate check
   after governance; or refuse a split whose two halves would receive the same
   name, which is cheap because p8 already names both halves. Same shape as every
   other "a gate before the operation that breaks its invariant guarantees
   nothing".

1. **Does the narrative writer still return an empty JSON?** live42 lost three
   sections to `{"markdown": "", "covered": []}` on all three attempts. Today's
   `render --agents` against the same artifacts returned **9/9 sections**, so it
   did not recur — but the fact sheets changed too (the sign fix, the widened
   citable pool), so this is evidence, not a controlled test. Both real repros
   before it also failed to reproduce a blank. Do NOT add mitigation machinery
   until one reproduces: the three prior suspicions here (truncation, sheet size,
   prompt-block truncation) were each refuted by measurement.

2. **Two English strings reach three Chinese reports each (6 lines).** Confirmed
   still present today by `verify_run.py` on live42. "DIAGNOSTIC ONLY — do not
   choose a model from this number…" and "Match the two annotators…", both from
   the annotator-balance work, never routed through `prose()`. The static AST
   guard covers `deps.decision()` rationales only — it does not see gate messages
   or remediations, which is where these live.

   NOT closed by the 2026-09-01 work. That fixed a THIRD string of the same class
   (`{n} L1 intents across {m} axes`, `topdown.py`) by making the decision choice
   symbolic. These two are gate messages and still English. Note `prose()` cannot
   rescue an f-string: `PROSE_ZH` returns a fixed string, so it cannot carry the
   numbers. Either author them symbolically as well, or install a translator.

3. **The resume rewind silently drops a concurrent branch.** Made LOUD (the join
   halts with `p2c_both_branches_arrived`), not fixed. Open: whether
   `update_state(as_node=...)` can restore a multi-superstep fan-out at all, or
   whether a gap in `phase_status` should force a clean re-run of the generation.
   Until answered: open a new generation and run it ONCE, never restart mid-flight.

### P2 — measured, disclosed, not acted on

5. **The declared per-role token budgets are miscalibrated in both directions,
   and now quantified.** Measured against live42's own `usage.json`: the
   annotators are 500 of 702 calls, declared 12,000 output tokens, actual
   1,612-1,751 — a **7x over-estimate on the roles that dominate the run**.
   Observers, researchers and the delivery auditor run **2-6x UNDER** their
   declared budget. The declared number drives the cost estimate, the cap
   (`3x`) and the timeout, so recalibrating moves truncation risk — it deserves
   its own pass, not a drive-by edit.

6. **The spend ledger is wrong in both directions.** Provider prompt-cache
   discounts are not recorded (input is 7.3x output on live42, annotators 94% of
   it, and the prompt is already ordered for prefix caching), so cost
   OVER-reports; a pinned model with no published price counts as $0.00, so it
   UNDER-reports. `_pin_warnings` says so out loud; nothing corrects it.

7. **`challenger_beats_incumbent` has no production call site.** Four docstrings
   and a model-budget decision rested on it; they now say so, and the CLAUDE.md
   invariant row no longer states it as a live guarantee. Needs a signature
   change (`propose_grid` returns a flat list, so selection cannot tell a proposed
   value from a configured one). Applying it as written flips live40 to a worse K.
   Still open — only the overclaim was fixed, not the wiring.

0s. **FIXED 2026-09-02 — `merge_leaves` no longer voids a risk isolation.**
   med04 shipped leaves 14 and 24 merged away AND "isolated", so two risk
   clusters ([21,10,14] and [43,24,35]) were unisolated and ghost families 42/33
   held no rows — the artifact said 36 families where 34 had content.

   Two changes, both verified against med04's real conflict:
   - `execute_prescriptions` reconciles AFTER the prescription loop: a leaf that
     is both merged and isolated has its MERGE declined with a reason, because
     isolation is a SAFETY action and merging a QUALITY one. Redirecting the
     isolation to the survivor was rejected — it would isolate the survivor's
     other rows on a guess.
   - `isolate_leaves` now takes `leaf_labels` and REFUSES an empty leaf,
     recording why. Defence in depth for any other path that empties one.

   The cascade is right: both leaves stay live, the isolation moves real rows, no
   ghost family, and the surviving duplicate routes to
   `p8_leaves_are_distinguishable` — the safety action wins and the quality
   problem goes to the quality mechanism.
   Tests: `test_a_leaf_merge_never_voids_a_pending_risk_isolation`,
   `test_declining_a_merge_still_counts_as_settled`.

0n. **The observer's `decisions` channel is per-phase and narrow, so an agent
   that saw a decision id in an ARTIFACT cannot cite it.** med04 dropped four
   citations: `granularity.triangulation.k_tie_set`, `D004.decisive_metrics`,
   `D006.evidence`, `decisions[0].evidence.critic_verdict`.

   Callers pass `decisions=[decision]` (p4, p5) or `decisions=[]` (p6), so the
   id-indexing added for 10a only covers that phase's own decision. An agent
   reads D004 inside an artifact, cites it the way the record prints itself, and
   is refused. Residual of 10a, not a regression of it.

   Fix: index decision ids found in the ARTIFACTS too, or pass the full decision
   ledger to every observer. Prefer the latter — the ledger is small and an
   observer reasoning about a decision it can see is the point.

0f. **8 figures on med04 against live44's 11, unexplained.** Not investigated;
   `test_one_figure_per_quantity` passes, so it is not duplication. Check whether
   three figures are conditional on artifacts this corpus lacks (the untrusted
   template groups are the obvious candidate) — a figure that silently does not
   render is the same class as a section that silently does not ship.

0c2. **The 0c disclosure reaches NO artifact, and the fix is still unverified.**
   Two separate corrections to what I claimed on med04.

   FIRST: the `1.0344 of ceiling reached` in med04's `p2b_kappa` gate is NOT the
   0c fix. There are two ratios and two gates:
   - the PILOT gate prints `share_of_ceiling_reached`, which comes from
     `headroom` — the variable 0c changed
   - `p2b_kappa` prints `share_of_ceiling` (topdown.py ~974), a DIFFERENT
     variable that was never clamped
   med04's pilot ratio was 0.9603, below 1.0, so the clamp never engaged and the
   fix was never exercised. Verifying it needs a run whose PILOT kappa exceeds
   the PILOT ceiling — med01 (0.930 vs 0.9188) and med02 (0.922 vs 0.9033) did;
   med04 did not.

   SECOND, and larger: **no artifact on disk contains `self_consistency` at
   all.** The pilot dict lives only in LangGraph state, so
   `self_consistency_is_lower_bound`, `kappa_exceeds_self_consistency` and the
   explanatory note are computed and then reach no artifact, no report and no
   reader. That is the same write-only pattern as `decisions` and
   `risk_isolated` — a value nothing consumes.

   Fix: persist the pilot block into an artifact (it is the evidence behind the
   ceiling argument, which the reports already discuss), then verify 0c on a run
   that reproduces the pilot-level inversion. Until both are done, treat 0c as
   WRITTEN BUT UNPROVEN — it was nearly recorded as verified on the strength of a
   number produced by different code.

0d2. **A SECOND batch-loss signature the three-tier key matching does not
   rescue: ALL keys unmatched, not one.** med04:

       ⚠ annotator[b] batch lost 25 rows after 3 attempts:
         ValueError: returned 25/25 labels (25 queries unlabelled)

   med02's case was `1 queries unlabelled` — one key with a trimmed space or a
   full-width comma, which exact -> NFKC -> whitespace-stripped matching now
   resolves. Here NONE of the 25 matched, so the model applied a SYSTEMATIC
   transformation to every echoed key (a paraphrase, a truncation, a translation
   — unknown, because the retry discards the response).

   Not caused by the fix and not made worse by it: this batch would have been
   lost before it too. The fix rescues per-character noise; it cannot rescue a
   wholesale rewrite, and it should not try — matching 25 rewritten strings back
   to 25 queries by similarity would risk assigning labels to the wrong rows,
   which is worse than losing the batch.

   Impact is bounded and DISCLOSED: 1 batch of 120, `annotator[b] labelled
   2975/3000` against annotator[a]'s 3000/3000, so 0.83% coverage on one side and
   the number says so. Contrast live43, where the same underlying failure
   produced `1500/3000` with zero warnings.

   Worth doing before the next fix attempt: LOG THE FIRST FEW RETURNED KEYS on a
   total mismatch. The diagnosis needs the actual transformation, and every
   attempt so far has had to infer it from a count. One `deps.emit` of
   `sorted(got)[:3]` would settle what the model is doing.

8b. **The TAXONOMY is not stable across runs, and neither is its
   annotatability.** Same corpus, same method, same five angles:

   | run | classes | rules | pilot kappa | ceiling | ordering |
   |---|---|---|---|---|---|
   | med01 | 22 | 51 | 0.930 | 0.9188 | inverted |
   | med02 | 20 | 28 | 0.922 | 0.9033 | inverted |
   | med04 | **18** | 47 | **0.835** | 0.869 | correct |

   Kappa spans **0.835-0.930** on identical data. It is NOT explained by:
   - the 0h fix — `log_reading` contributed 12 candidates in ALL THREE runs (it
     succeeded on retry in med01/med02), so making it succeed faster changed the
     wall clock, not the input. Hypothesis checked and refuted.
   - class count — med04 has the FEWEST classes (18, coarser distinctions) and
     the LOWEST agreement, which is backwards if granularity drove it.

   What differs is the partition itself: three genuinely different carvings of
   the same space, each internally coherent, each differently annotatable. This
   is the top-down analogue of item 8 — there the GRID decided alpha; here the
   architect's draw decides the taxonomy, and a headline kappa inherits that
   variance.

   Consequence for reading any single run: kappa is a property of THIS
   taxonomy-and-annotator pair, not of the corpus or the method. Comparing kappa
   across runs — or citing one run's kappa as "the" agreement for this corpus —
   is comparing different objects. n=3, so this is an observation with a
   direction, not a measured distribution.

   Worth doing before trusting any kappa as a methodology result: replicate the
   taxonomy draw on one corpus and report the spread, the same way the alpha
   noise floor was established.

8. **The alpha decision is decided by the GRID, not by the corpus — three
   answers from identical data.** Stronger evidence than the seed-replicate note
   this replaces. The medical sweep rows are byte-identical across med01/02/03
   (alpha=0 is always frag 2.4868, stab 0.8405), yet:

   | run | grid top | tie band | contenders | chosen |
   |---|---|---|---|---|
   | med01 | 0.85, 1.0 | 1.5548 | 0.85 (stab .775), 1.0 (stab .642) | **0.85** |
   | med02 | 0.57 | 2.0239 | 0.57 only | **0.57** |
   | med03 | 1.0 | 1.5548 | **1.0 only** — 0.85 was not proposed | **1.0** |

   med03 chose 1.0 BY DEFAULT: with 0.85 absent from the blind proposer's grid,
   only one value cleared the band. Same corpus, same metrics, three answers.

   **The selection rule is NOT the problem and has not been changed**:
   `contenders = frag <= band` then `max(stability)`. Fragmentation defines the
   band; stability picks inside it. The problem is that the band is computed from
   `min(frag)` over WHATEVER the grid contains, so a proposer that omits one
   value moves the band and changes the winner.

   **Open question worth deciding deliberately** (raised by the user 2026-09-01,
   and the evidence supports it): alpha=0 has the BEST stability in the whole
   medical sweep (0.8405) while the chosen alpha=1.0 has the WORST (0.6419), so
   the delivered tree is the least reproducible option available. At 1.0 the
   phrasing block controls 50% of the cosine, which on a medical log risks
   clustering by question form — `黄精的功效` beside `布洛芬的功效` because both
   are `X的功效与作用`. Alpha=0 never enters contention because fragmentation
   alone defines the band and alpha=0 has the worst fragmentation.

   Counter-evidence, and it matters: `template_fragmentation` is NOT merely
   circular with alpha. On K12 it moves the OPPOSITE way — 1.9799 at alpha=0
   rising to 2.5713 at alpha=1.0 — so higher phrasing weight makes it worse
   there. The metric responds to corpus structure, not mechanically to alpha.

   Candidate resolutions, none applied: (a) require the winner to clear a
   stability floor relative to the sweep's best, not just the absolute veto;
   (b) make the tie band fixed rather than derived from the grid's own minimum;
   (c) report the alpha decision as a range when the grid is sparse near the
   optimum. Do NOT let the proposer see scores — that is the pre-registration.

9. **Annotator asymmetry, unattributed.** live42: annotator_a won 34.1% of 270
   contested rows, z=-5.23. NO baseline exists (the gate postdates live40). The
   parsimonious reading is the pairing — a flash tier against a plus tier. Note
   the pins have since changed (`annotator_b` is now `qwen:qwen3.8-flash`), so a
   re-test would not reproduce live42's pairing. Testing whether disabling
   reasoning contributed needs a paired re-run of the same rows.

11. **Observers cost ~2.5 hours of a 4-hour run.** 11 observers, median 809s on
    live41 against 285s on live40 for the SAME output volume — latency, not token
    inflation. They earn it (three real defects), but the trade should be a
    decision, not an accident.

12. **Refinement converges on some corpora, not others** (was: "has not converged
    on any run" — falsified 2026-09-02 by `hierarchy_meta.converged`:
    `fin01`/`fin02`/`fin03` and `ecom01`/`ecom02` are all `true`; every K12 and
    medical run is `false`). live40, live41, live42 all hit
    the 5-round limit. Honestly disclosed in the report, but it means the
    delivered leaf count depends on which round it stopped at.

13. **Two vacuous rule triggers found live and not repaired.**
    `academic_knowledge_qa x problem_solving` names 求/解/计算 with 0 of 19
    contested rows carrying any; `navigational x school_info` names 主页/入口/好不好
    with 0 of 8. The detector works; nothing acts on what it finds.

14. **Two of nine governance splits are geometrically unsupported** (leaf 19 -> 54
    at ARI 0.0595 with a near-duplicate name; leaf 0 -> 49 at 0.3876). Disclosed
    by design (measure-don't-veto). Nobody has looked at them.

### P3 — small, known, deliberately not growing

15. **22 `deps.gate()` messages are English f-strings** printed verbatim into
    Chinese deliverables. Frozen by gate NAME in `GATE_MESSAGE_DEBT`
    (`tests/test_pipeline.py`); the set may shrink, anything new fails the test.

16. **Two domain profiles are untested on real data:** `sports_zh`,
    `politics_zh`. (`finance_zh` was exercised on 2026-09-02 by `fin01`/`fin02`/
    `fin03` and works — its risk categories, template seeds and
    `expected_l1_range` [15,22] all held, and all three runs landed 16-20 L1
    classes.) `generic.yaml` ships 0 template seeds, so a corpus without a profile
    locates K against unvalidated mined groups — now gated by
    `p3_locator_reference_validated`, but never exercised.

18. **`zh_panel`'s `fixed` and `waived` lists are unsorted** (`entries.values()`),
    unlike `open_findings`, which sorts blocking-first. Their caps are disclosed
    now, so nothing hides silently, but the order is arbitrary.

19. **The reference shelf is Chinese-only.** `builder.build_all_reports` emits
    `类目清单` / `标注规范与裁定规则` / `家族与叶层级` / `00_索引` under `zh` only, so
    an English run delivers the six reports and none of the reference documents.

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
- Verify a live run really used live agents: `run_summary.json` →
  `llm_usage.provider` must read `routed`, not `offline`.

---

## 4. Session (2026-09-01) — live44, and the queue cleared

`live44` completed: 17/17 phases, `provider=routed`, 841 calls, $61.09, 9.81h,
κ 0.880 on n=3000, 53 leaves / 23 families delivered, `verify_run` 26/0/2 against
live42's 20/6/2. Then the whole defect queue was fixed. **625 → 642 tests.**

**Every defect was found by reading, not by the harness.** `verify_run.py` scored
live44 clean on all 26 applicable checks while seven real defects sat in the
deliverables. Two of them the pipeline had already found and thrown away.

### The compounding failure

The delivery auditor was shown **39%** of the deliverables (`budget_text` cut
142,957 of 232,957 chars out of the MIDDLE), found 3 defects in that third, and
had **2 of the 3 deleted** because the citation resolver only knew `artifacts`
while the auditor had also been handed `gates` and `findings`. The two deleted
were both real: the index claiming 21 L1 classes against a taxonomy of 20, and
one quantity published with two values. A last line of defence reading a third
of the evidence and losing two thirds of its conclusions.

### Five of my own diagnoses were wrong before they were fixed

Recorded because the pattern repeated: **measure one thing, assert a cause about
another.** Each was checkable in under a minute.

1. "Researchers cost most and retrieve least" — they are *deliberately tool-free*
   (`RESEARCH_ANGLES`, `web: False`), and returned the MOST candidates (12 each).
2. "P7 drops risk findings" — all ten `risk_report` clusters were isolated by p8.
   They travel via the risk sentinel, not `audit.risk_findings`. The log said so.
3. "A shared client-side deadline" — no shared deadline. `max_repair+1 = 3` outer
   attempts x SDK `max_retries=2` = **9 HTTP requests**, and 9 x 292s = 2,628s
   against the 2,638s observed.
4. "Reasoning made rules verbose" — density is identical (204.7 vs 207.5
   chars/rule). live44 simply produced more rules, and live42 sat at 86% of the
   budget unnoticed.
5. "The referee block would drop 85%" — wrong density applied. It is at 72%.

The ECE null landed the same lesson: a perfectly calibrated model at n≈5,800
still shows ECE ≈ 0.0074 ± 0.0028, so live42's 0.023 was ~5 sd out and was never
the clean baseline I had been comparing live44's 0.065 against.

### Fixed

| area | change |
|---|---|
| truncation | `budget_units` — whole rules/documents, count in log AND in-band, plus an explicit "do not describe this as the complete set" |
| attribution | the `label=` test found **9** unlabelled budget calls, incl. the narrator's fact sheet and the observer's own decisions/gates |
| observation door | `citable_namespace` feeds resolver AND check evaluator — widening one alone would demote a measurable claim to advisory |
| retries | SDK `max_retries` 2 → 0; per-tier throughput (researcher 585s → 1560s, maintainer 292s → 780s) |
| calibration | `ece_noise_floor` + `p2c_calibration` gate, against the run's own null, never a constant |
| kNN flags | nearest-first, full neighbourhood published — the 6-of-k sample manufactured false confirmations |
| safety booleans | `coherent` / `risk` / `risk_isolated` tri-state; they defaulted to the reassuring answer |
| family names | `FamilyNamerAgent` names the DELIVERED partition; `混合·主要成分「X」N%` is now only a fallback |
| duplicates | `merge_leaves` prescription + executor; the auditor must give every listed pair a disposition |
| `doctor` | `detect()` across all providers — it checked `ANTHROPIC_API_KEY` while the project routes to DeepSeek/Zhipu/Qwen |

### Deliberately not done

- **Did not loosen the English-prose detector** for the third English string. Its
  table-row and code-span exclusions are deliberate; widening it would flag every
  identifier — the documented grounding-false-positive trap. Fixed the source.
- **`p2c_calibration` is warn-only.** A new blocking gate on a metric nothing has
  ever gated would halt the next run on a pre-existing condition.

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

1. **Fail-fast.** An encoder download, an OOM, or a clustering bug currently
   surfaces at hour three. Concurrently it surfaces at minute twenty-five, before
   the expensive half is paid for.
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

## Session (2026-08-27 night → 08-28) — live41 abandoned, live42 run end to end

> These six subsections were appended as top-level headings, which broke the
> one-dated-section-per-session contract at the top of this file. Grouped here
> without editing their content. The exact night/morning boundary is not
> recoverable, hence the range.

### The resume rewind silently drops a concurrent branch

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

### live42

live41 abandoned. Its three generations stay as evidence: gen01 (reasoning
tokens, 88% referee failure), gen02 (my `model_kwargs` TypeError), gen03 (the
dropped branch). live42 starts cold — no cache — which is the price of a clean
test of every fix at once.

---

### live42 results as they land (fresh run, all of today's fixes active)

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

### The narrative report's FIRST live run: 3/9 sections, and why

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

### live42 finished — 17/17 phases, and what the narrative report is really like

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

### Translation: from 34 hardcoded prefixes to a guarded model call

`PROSE_ZH` maps an English PREFIX to fixed Chinese at 20 call sites. Two holes no
diligence closes: a newly authored string is English until a human notices (three
separate leaks in one day), and an f-string can never be matched by a fixed prefix
— which stranded all 22 `deps.gate()` messages permanently.

`report/translate.py` adds a third tier to `prose()`, below the curated mapping
and above the English fallthrough. What makes it safe is that **nothing is
trusted** — every result is verified before use:

* **numbers** — the numeral multiset must match both ways. A rounded value is a
  changed value.
* **identifiers** — every backticked span survives verbatim. Numerals INSIDE
  backticks are excluded from the number count, because the "2" in
  `p2b_annotator_symmetry` is part of a name; counting it made a translated
  identifier report as a changed NUMBER, which a test caught.
* **actually translated** — a result with no CJK is the model echoing the source.

Any failure keeps the ENGLISH, exactly the old behaviour, so this cannot make a
report worse than the mapping it extends. Results cache by content hash in
`.cache/translations.json`: a string is paid for once and renders identically on
every future run — wording that drifts between runs for no measured reason is its
own defect.

**Verified against a real model** on the two strings that leaked into live42:
both translated cleanly, and `kappa 0.8928 on 2999 rows` returned as
`2999 行上的 kappa 0.8928` with both numbers intact. A gate message carrying five
interpolated values also translates — a class previously unreachable.

`GATE_MESSAGE_DEBT` is now OFFLINE-ONLY (the fixture installs no translator). Off
in `offline`; `cfg.translate_prose` disables it.

---

## Session (2026-08-28, late) — model pins, reasoning, and why the report was two-thirds empty

### The empty sections: measured, not inferred

live42's `00_最终报告.md` delivered **3 of 9 sections**. The document's own
placeholders under-report this — three of them say only 「空白正文」 three times,
and all of them say 「未通过校验」. `final_report_meta.json` and `run.log` carry the
real picture:

| section | what actually happened |
|---|---|
| `question_and_two_routes` | rejected: `90, 99, 10` |
| `vector_choice_first` | **blank body** ×3 |
| `topdown_taxonomy_and_labels` | rejected once (`5.23, 5.2`), then passed |
| `bottomup_k_not_single` | rejected once (`0.014048, 0.007024`), then passed |
| `two_level_tree` | **blank body** ×3 |
| `governance_and_risk` | rejected: `0.0169`, then `32, 40, 42, 43, 44, 45` |
| `unified_panel` | rejected once (`0.0162`), then passed |
| `samples_and_deployment` | **blank body** ×3 |
| `audit_and_limits` | rejected: `21, 5.23`, then `5.23, 5.2` |

So the number check touched **6 of 9** sections and killed 3 outright. Reading the
cached drafts (`runs/live42/llm_cache/`, `meta.role == "reporter"`) shows what the
author had actually written:

- `z_vs_even=-5.23` → extracted as `+5.23`. **`_NUMBER` had no sign.** A negative
  fact was uncitable: the author copied the sheet exactly and was told the number
  was not in the sheet. No retry can satisfy that.
- `裁判模型是 qwen:glm-5.2` → a phantom claim of `5.2`. A hyphen before a digit
  read as a discarded minus, so naming the model that did the work was a
  fabrication.
- `第 90 百分位为 13`, `家族 32、40、42` → the sheet SHOWS these (`length.p90`,
  dict keys) but the value-only pool did not carry them.

This selects against the sections that matter: negative numbers are where the
**warnings** live, so it deletes governance, audit-and-limits and the panel.

Verified with `check_numbers(text, {"z_vs_even": -5.23})` → unsupported, control
passing. Fixed in `verify._NUMBER` with two lookbehinds separating a minus from an
identifier hyphen by what precedes it. **Every one of live42's number-rejections
now passes**, replayed against the run's own artifacts; the three cases that must
still fail (fabrication, sign flip, miscount) still do. Mutation-tested.

Two further fixes from the same reading:

- **`annotator_balance` was in no bundle.** It is measured and lives in
  `taxonomy_v2.json`, which `build_catalogue` never read. Shown `n_contested=274`
  and `annotator_a_won=92` through other bundles, the narrator DERIVED the rest —
  `178 = 270-92`, `0.3407 = 92/270` — and every one was correctly refused. A
  starved sheet induces derivation, not caution. Added to `topdown_gold`, where
  `lopsided` belongs on the merits anyway.
- **Everything the writer was shown is now citable** (`_reject(shown=...)`):
  must-cover items, figure captions, previous section. A must-cover arrives under
  「必须原样包含这句话」, so a number inside one ORDERS the author to write what the
  check forbids. The outline is excluded — `_plan` verifies nothing numeric, so
  pooling the thesis would launder a number into every section — and so is the
  rejection notice, which prints the offending values.

The blank-body residue is unresolved and is now §2 item 2. It did not reproduce in
two real calls; both spent 93-94% of their output tokens on reasoning.

### Model pins

`glm-5.3-flash` and `qwen3.8-flash` are real and answer on their providers' direct
endpoints (verified by real calls). `qwen3.8-flash-next` **does not exist** (404).
Neither flash model is in the 1,930-model catalogue, because the catalogue is
fetched and they are newer than the price feed.

A bare pin with no card became `provider="explicit"`, a sentinel nothing handles —
so the run died on that role's **first real call**, after `qmine models` printed a
clean plan. Now: `resolve_pin` accepts `provider:model`, and an unresolvable pin
raises `UnroutablePin`, which `_build_routing_plan` re-raises rather than degrading
to the static tiers (its blanket `except` is for a missing catalogue, not a config
error). `configs/live.yaml` pins `zhipu:glm-5.3-flash` (referee, researcher) and
`qwen:qwen3.8-flash` (annotator_b); all three verified with real calls.

`glm-5.3-max` does not exist either — error 1214 on both endpoints with **both**
the user's abroad-registered and China-registered keys, so the China key changes
nothing and can be ignored. Note qwen IS region-split: an abroad key 401s on
`dashscope.aliyuncs.com` and works on `dashscope-intl`.

### Reasoning re-enabled for `referee` and `namer`

Both left `NO_REASONING_ROLES`. Neither is bulk classification: the referee
adjudicates the residue the annotators disagreed on and drafts rules that reach
them; the namer authors names that appear in the deliverable.

**This required raising `namer`'s budget.** The trace shares the role's output cap,
and measured here it runs 8-10x the content. `namer` had
`output_tokens_per_call=1200` → a 3,600 cap, which is inside a single trace — the
same shape as the referee's 88% failure rate on live41. Raised to 3,000 (cap
9,000); the declared budget moves too, or the ledger under-reports. `referee` had
36,000 already. Now pinned by test.

### State

- **571 tests passing**, `ruff --select F src/qmine/` clean.
- Verified live: the three pins route direct and answer; `referee` emits reasoning
  tokens (160) while `annotator_b` stays at `None`.
- Not yet run end to end. The next live run is the first to exercise the pins, the
  widened citable pool and the two reasoning roles together.

---

## Session (2026-08-28, later) — the reference shelf, and a dashboard that was showing the wrong call

### Budgets for the two roles that now reason — measured, and my first sizing was wrong

I sized `namer` by extrapolating the REPORTER's reasoning ratio (93-94% of
completion tokens). That does not transfer. Measured directly, on the real
prompts:

| role | completion | reasoning | content |
|---|---|---|---|
| namer | 696 / 1,149 | 531 / 994 | 165 / 155 |
| referee | 6,145 / 2,497 | 3,873 / 598 | 2,272 / 1,899 |

Reasoning is a roughly FIXED cost here, not a ratio — a few hundred to a few
thousand tokens, not 8-10x the content.

**The namer raise was still right, for a different reason than I gave.** live42's
namer spent 2,542-3,039 output tokens per call with reasoning OFF, against a
3,600 cap — 20% headroom. Adding a measured 531-994 exceeds it. Now 3,000/9,000.
**The referee needed nothing**: 1,486/call on live42 against a 36,000 cap.

**Bonus finding, from live42's own `usage.json`:** the declared budgets are badly
calibrated in both directions and this is HANDOFF item #7, now quantified. The
annotators are 500 of 702 calls, declared 12,000, actual 1,612-1,751 — a **7x
over-estimate on the dominant roles**. Observers, researchers and the delivery
auditor run 2-6x UNDER their declared budget. Not changed: recalibrating moves
caps, which moves truncation risk, and that deserves its own pass.

### The dashboard was showing a different call's output

Four defects, all verified in code AND against the user's screenshots:

1. **The agent detail was mispaired.** `raw_log` and `on_call` are two streams
   from the same `_store` with no join key, so `_agents()` paired them by index —
   a global reversed index against a per-role chronological list. The user's
   screenshot proves it: the row headed `reporter … 04:42:11` (attempt 1 at
   `audit_and_limits`) opened onto the top-down taxonomy section. Both streams
   now carry `cache_key`.
3. **The detail was `str(dict)`** — a Python repr of Chinese prose.
4. **`agent_transcript.json` never existed for live42** (killed by the teardown
   bug, since fixed), so the fallback pointed at a missing file.

Plus, found by audit and verified here: the artifact column read `key` where
`index.jsonl` writes `name` (blank for every run); replay elapsed printed
**496,632h**; and §8 of the top-down report shipped six rows of `| ? |  | — |`
because `_failure_history` reads `option`/`why_rejected` while the architect's
dropped candidates carry `name`/`why_dropped`.

**The event log** is now faceted: severity (from `logging`'s own level when
replaying, glyph otherwise) and phase, both captured at emit time, both with
counts, plus a 「怎么读这一栏」 line. Replaying live42 gives 99 warnings / 3 edits /
227 info across 12 phases — the log level alone caught 45 warnings the glyph
convention missed, including the two teardown failures.

### The reference shelf — the run was producing this and delivering none of it

`zh_reference.py`, wired into `builder.py`, four documents plus three CSV twins:

- **类目清单.md** — the 21 L1 classes with definition, `user_need`, positive and
  negative examples, **measured delivered size** beside the architect's
  prediction, and how many rules route to each. Symmetric to 叶清单.md.
- **标注规范与裁定规则.md** — the labeling guide VERBATIM (it appeared zero times in
  the whole delivery) and all 139 rules, one section each, grouped by target
  class, marked 架构师预判 vs 裁判补充.
- **家族与叶层级.md** — the delivered two-level tree, with a Mermaid top level.
  Reads `leaf_*_final`, and says out loud that the audit describes 20 families
  where 24 were delivered.
- **00_索引.md** — the reading order. Ten files landed in one directory with no
  index; every `put_markdown` call already passed a `summary` and all of them
  were thrown away.

Format follows the evidence (W3C DWBP BP 12; GitHub renders MD to 400KB and CSV
as a searchable table to 512KB; these are 7-38KB): Markdown is the reading
surface, CSV is the machine twin, and rules get one section each rather than a
table because a row cannot hold `when → then` plus rationale and examples.

Also fixed: the referee's rules shipped in English (`drafted by the referee to
close a gap…`) on 100 of the 139 — our own hardcoded template, now through
`prose()`, with the disagreeing query left verbatim as evidence.

Two further gaps from the same audit, closed in the same pass:

- **叶清单.md now shows the evidence each name was made FROM.** `naming_cards.json`
  holds the exact sample the blind namer saw — 15 centroid, 10 random, 5 edge per
  leaf — and none of the 1,470 sampled queries reached any deliverable. The EDGE
  samples carry the weight: leaf 1 is named 「2026年中小学暑假放假时间查询」 and its
  edges are `目瑙纵歌2026年时间表`, `退潮赶海时间表` — queries about times in general.
  A reader sees the boundary immediately. The sampling is mechanical, which is
  what makes it admissible rather than a flattering selection.
- **家族与叶层级.md now carries the cross-route mapping per family.**
  `route_crosswalk.csv` is the only artifact that says how the two routes line
  up and was named in no document. It is keyed by delivered family, so it belongs
  beside the family rather than in a table a reader has to join by hand. On
  live42 it reads well: families 5/7/11 agree with the intent taxonomy at 91-96%,
  while family 8 (14,171 rows) has 8.33 effective classes and 「routes disagree」.

### Five more, found by the workflow's synthesis re-checking current source

It correctly identified everything already fixed, and surfaced five live defects.
All five verified here before acting; all five fixed.

1. **The family definitions in my OWN new document were borrowed.** 14 of the 24
   delivered families carried a definition shared with one or two others —
   family 8 (17 leaves, 14,171 rows) and family 10 (1 leaf) got the identical
   sentence. This is this project's own delivered-partition trap, documented in
   `report-generators.md` and then walked into. A definition is now shown only
   when the family's leaves come from ONE audit family AND that audit family
   backs no other delivered family; the other 16 are told plainly whose
   definition it is and which delivered families share it.
3. **§2.1 L2 子意图 had never rendered.** It read a LIST from `subintents` or
   `groups`; the artifact carries a DICT under `subdivision`, keyed by L1 code.
   Neither key has ever existed. 54 sub-intents across 19 of 21 classes, and the
   panel's strongest comparative claim rests on them. Now rendered from the real
   shape, stating that they are UNNAMED and disclosing where silhouette disagreed.
4. **The escape hatch pointed at a file that need not exist** — unconditional
   "full return in agent_transcript.json".
5. **`qmine watch` hardcoded `provider=""`**, so every replayed page read
   "provider ?" while `usage.json`, already loaded for the KPIs, says `routed`.
   That is the exact field this project uses to decide whether a run was real.

### Still open from the synthesis, NOT done

- `fig_gates.png` plots 7 of 26 gates and treats `True` as `1` (`viz.py:310-314`).
- `zh_panel.py:187-200` slices findings `[:12]`/`[:8]` with no "showing N of M",
  hiding two BLOCKING findings, and cuts claims mid-token at `[:90]`/`[:150]`.
- Broken link to `Report_Uniform_Panel.md` (`zh_topdown.py:89`) and eight wrong
  `§9` cross-references (`zh_bottomup.py:827`).
- The `<details>` rule table in the top-down report should become a link to
  `标注规范与裁定规则.md` rather than have its truncation fixed twice.
- **No CLI path regenerates reports from finished artifacts.** Everything above
  was verified through a scratch harness; a `qmine render-reports RUN_ID`
  writing into a new generation would make it a one-command, zero-LLM operation.
  This is the highest-value remaining item: the reference shelf exists and has
  never been delivered by the pipeline itself.

### State

- **594 tests passing**, `ruff --select F src/qmine/` clean.
- Verified by replaying live42: the reference documents build from its real
  artifacts (14,395 / 33,252 / 10,992 chars), and the dashboard renders with
  correct elapsed, artifact names, facet counts and per-call detail.
- **Not yet exercised by a live run.** `live42/gen01` was deliberately left
  byte-identical to what it delivered; the new documents were built into
  `/tmp/qmine_refs` instead.

---

## Session (2026-08-31) — `qmine render`, and the five deferred items

### 1. Are the reference documents produced by a run? YES — verified end to end

`make demo` now emits all of them: 类目清单.md, 标注规范与裁定规则.md,
家族与叶层级.md, 00_索引.md and the three CSV twins, alongside the six original
documents. They were absent from `runs/live42/gen01` only because that generation
was deliberately left byte-identical to what it delivered.

**`qmine demo` was itself broken** and had to be fixed to check this. It calls
`run()` as a plain function while naming only 13 of its 16 parameters, so
`reuse_taxonomy`, `resume` and `dashboard` arrived as `typer.models.OptionInfo`
sentinels and one reached a Pydantic model:
`PydanticSerializationError: Unable to serialize unknown type`. `make demo` is
what CLAUDE.md points at to check wiring and it could not run. It now reads the
declared defaults out of `run`'s signature, so a new option arrives with its own
default instead of reintroducing the bug silently.

### 2. The five deferred items — all done

- **`fig_gates.png`** paired the FIRST numeric in `observed` against the FIRST in
  `threshold` with nothing tying them together, so a gate observing
  `{"n": 600, "kappa": 0.8928}` against `{"min_kappa": 0.70}` was drawn as 600
  versus 0.70. And `isinstance(True, int)` is True, so a boolean assertion
  contributed a value of 1. Both now go through `records.paired_gate_metric`,
  one definition shared with `_passed_below_threshold` so a figure and a table
  cannot disagree; gates with no numeric bar are counted under the axis instead
  of vanishing (live42: 7 of 26 plotted, all green, none of the four warned).
- **Panel findings** were capped at `[:12]`/`[:8]` with no "showing N of M" and
  claims cut mid-token. Both fixed. **Correction to the audit that raised it:**
  it claimed the cap hides two blocking findings — it does not.
  `FindingsLedger.open_findings` sorts blocking-first, so they survive the cut.
- **Two broken cross-references**: the panel link named `Report_Uniform_Panel.md`,
  the English build's filename, in every Chinese delivery; and `§9 失败史` pointed
  at the decision chain (the failure history is §12). The section is now cited by
  NAME, which is correct in both documents that share the helper.
- **The two `<details>` blocks** in the top-down report are gone. The class table
  and the referee's rules now link to 类目清单.md and 标注规范与裁定规则.md, which
  carry them un-truncated.

### 3. `qmine render RUN_ID [--agents]`

Rebuilds a finished run's deliverables into a NEW generation. Verified on live42:
gen01 → gen02, nine documents, gen01 byte-identical afterwards. Every fix above
is visible in the re-rendered output — 0 `?` rows (was 6), 0 `<details>` (was 2),
§2.1 L2 子意图 rendering for the first time, the findings cap disclosed.

Four defects found and fixed while building it, each recorded in
`report-generators.md`: the auditor had no config gate so `--no-agents` still ran
it against the offline stand-in and wrote `[offline-heuristic] file` into a
deliverable; `render` did not call `_load_env()` so `--agents --provider router`
silently ran offline and reported "10/10 sections verified" for a report no model
wrote; a rendered generation had no `run_summary.json` so re-rendering FROM it
lost the gate ledger; and `GateResult.model_validate(value)` raises because
`name` is the dict key, which would lose the ledger to an `except` two frames up.

### A regression the suite caught, and it was mine

Unifying the gate pairing put a `break` in `gate_metric_pairs`, so one threshold
naming several observed values (`min_kappa` matches `kappa` AND
`self_consistency_kappa`) kept only the first. `_passed_below_threshold` then
stopped flagging a gate whose LATER number is under its bar — and the Chinese
「带保留通过」 prefix that flag adds was the only CJK on a line whose message is
authored in English, so an untranslated gate conclusion reached a Chinese report.
Caught by `test_every_authored_rationale_reaches_the_reader_in_the_report_language`,
localised by stashing files one at a time. Now pinned by its own test.

### Still open

- **The checkpointer degrades mid-run.** The clean demo wrote 5 checkpoints for
  17 phases and ended with `TypeError: Type is not msgpack serializable:
  DecisionRecord` — with no "SQLite checkpointer unavailable" line, so the sqlite
  saver WAS in use. Both `_serializer()` and a default `JsonPlusSerializer`
  round-trip a top-level `DecisionRecord`, all 16 record models are allowlisted,
  and every model reachable from `PipelineState`'s annotations is declared. NOT
  isolated. It matters: `render`'s best state source is the checkpoint (live40
  has 17 rows and recovers everything; live42 has none and loses `observations`
  and `metrics`).
- `zh_panel`'s `fixed` / `waived` lists are still unsorted (`entries.values()`).
- The reference shelf has no English build — `builder` emits it only under `zh`.

### State

- **605 tests passing**, `ruff --select F src/qmine/` clean.
- `runs/live42/gen01` remains byte-identical to what it delivered.

---

## Session (2026-08-31, later) — CLAUDE.md and HANDOFF audited against the code

Both files were audited claim by claim rather than edited from memory. Every
command in CLAUDE.md was run, every test name checked against the suite, every
file path resolved.

### CLAUDE.md — what was wrong

- **`448 tests, ~4 min`** — measured 605 in 3 min. Replaced with `~600`: an exact
  count rots on every session that adds a test, and a stale number in the file
  that loads into *every* session is the exact failure it warns about.
- **`26 mechanical checks` / `live39 18/19, live38 2/19`** — the harness has **28**
  checks and the denominators were wrong. Re-measured today: live40 25 pass / 1
  fail / 2 skip, live39 19/7/2, live38 2/9/17. The scores were replaced with the
  PRINCIPLE (always pass a known-broken control; read PASS/FAIL/SKIP, never PASS
  alone) because scores rot too.
- **`live42 lost 6 of 9 report sections to one [grounding false positive]`** —
  overstated. 3 were lost to a false rejection, **3 to empty model returns**.
- `make demo ~3 min` — measured 3.7. Now `~4 min`.
- One over-long line left by an earlier mid-paragraph insertion, and four
  area-specific invariant rows moved to `measurement.md` to hold 200 lines.

Verified sound and left alone: all 37 test names exist; every file path resolves;
`run --resume --run-id` is correct; `16 leaves vs 25 on live40` is correct
(`hierarchy_meta.n_leaves=16`, delivered 25).

### HANDOFF.md — what was wrong

- **§1 was four days stale and 88 lines of session narrative**, contradicting its
  own contract ("overwritten each session; must describe *now*"). It claimed 552
  tests, `25/26 PASS 0 FAIL`, and that no narrative section had ever been written
  by a real model — all false. Rewritten to 38 lines that describe now, plus a
  reading order for a new session. Its fork and confirmed-≠-defective narrative
  was already duplicated in the session log, so it was dropped rather than moved.
- **§2 kept resolved items struck through** where the contract says delete them,
  left orphaned prose under a `### P0 — none open` heading, and listed the same
  English-strings item **twice** — once struck, once open. Rebuilt: 0 struck-through
  entries, renumbered, and today's measurements folded in (the budget
  miscalibration is now quantified rather than asserted).
- **Six top-level headings were undated fragments**, breaking the one-dated-
  section-per-session contract. Grouped under
  `## Session (2026-08-27 night → 08-28)` and demoted, content untouched. The
  exact night/morning boundary is not recoverable, hence the range.
- `(2026-08-28, late)` sorted *before* `(2026-08-28, evening)`, contradicting file
  order. The second is now `later`.

### Two defects the audit surfaced in `render`

1. **A render overwrote the run's own spend record.** `_wire_events` wrote
   `root/usage.json` unconditionally, so re-rendering live42 replaced 702 calls /
   $29.69 with 11 calls / $0.78 — unrecoverably, because live42's teardown bug
   meant no `run_summary.json` held a second copy. `_wire_events` now takes a
   `usage_path` and a render writes into its own generation.
1. **`configs/live.yaml` is the default config** (`cli._load_config`) and it
   declares `provider: router`. Forgetting the routing policy was the one launch
   mistake nothing caught: the router then picks on price and discards lab
   independence, the capability list and every pin.
3. **`auto` asked only `_has_anthropic_credentials()`.** With deepseek, qwen,
   zhipu and openrouter all configured it still resolved to the OFFLINE stand-in,
   because none of them was Anthropic — on a project whose own live config
   EXCLUDES Anthropic. So `auto` could never route at all. It now asks
   `detect().usable`, and warns loudly when the answer is empty.
4. **`p0_provider` gate.** A warning in `run.log` is not enough: with the stand-in
   the pipeline emits a full set of deliverables, every gate passes, and no
   document says the prose was not written by a model. The question "was this run
   real?" now has an answer in the artifacts, recorded either way.

Verified end to end: with keys, a bare `qmine run` resolves to `routed`; with
every key removed, to `offline` with the warning.

**`demo` and `full` are now explicitly offline.** They are the documented ~4-minute
and ~25-minute checks, and the routing default would have quietly turned them into
paid runs on 8,000 and 50,000 rows.

### Correction to the previous entry

The `make live` target no longer hardcodes the corpus: `LIVE_INPUT`,
`LIVE_DOMAIN`, `LIVE_TEXT` and `LIVE_REFS` are overridable, and `LIVE_REFS=`
emits no flag at all — correct for a corpus with no legacy labels. Calling it
"the only way to launch" was wrong, and contradicted the project's own first
line: this runs on **any** query dataset.

### live43 halted at 80 minutes — half the gold set was silently unlabelled

`annotator[b] labelled 1500/3000` with **zero** lost-batch warnings. The cause was
not a lost batch and not batch size:

    annotator_a  136 cached batches, 3400 labels -> 25.0 per batch  {25: 136}
    annotator_b  128 cached batches, 1650 labels -> 12.9 per batch  {0: 62, 25: 66}

Bimodal — 0 or 25, never partial. Probing the real model with live43's own prompt
showed every empty return was byte-identical: `finish_reason=stop`, 309 output
tokens, 833 characters beginning `{"$defs": {"QueryLabel": {"properties": ...`.

**`qwen3.8-flash` was returning the JSON SCHEMA instead of the data**, and
`AnnotationBatch.model_validate()` accepted it: pydantic ignores unknown keys and
`labels` defaults to `[]`, so a schema echo becomes a valid empty batch. No
exception, so `_one`'s three-attempt retry never ran, and 25 rows disappeared per
occurrence with nothing logged.

This is the same shape as `SectionDraft.markdown` defaulting to `""` — a
permissive default turning a failed generation into a successful empty one. It
cost the narrative report six sections in August; here it cost half the gold set,
which is the artifact everything in the top-down route rests on.

Fixed in three places: `_is_schema_echo` rejects the echo before validation in
both `_plain_json_call` and `_salvage`; `_one` raises when a batch returns fewer
labels than queries so the existing retry fires; and the plain-JSON ask now says
not to return the schema. **Verified on the real model** — 4/4 complete batches at
sizes 25, 12 and 6, against 3/4, 3/4 and 1/4 before.

Also fixed while reading the log: the repair message announced "repairing via
plain-JSON mode" for a model ALREADY in that mode, so six lines claimed a mode
switch that had happened three days earlier.

**Why a fresh run and not a resume.** `llm_cache/` is run-level and stores parsed
values, so the 62 empty batches replay on any resume AND on a new generation of
live43 — the fix would never fire. Independently, p2b sits inside the forked
region, which is the case §2 warns never to restart mid-flight. live43 stopped at
$11.75 / 295 calls and stays as evidence.

---

## Session 2026-09-02 — fast mode

**What was asked:** a mode that returns results faster by skipping the
double-checking, delivering three files (two per-route reference documents and
one fully-labelled dataset) without reducing the evidence a user can audit.

**The collision found first.** `fast_mode` already existed and meant "shrink the
grids for a wiring smoke test" — α grid to 3 values, k sweep to 5, gold to 120
rows, researchers to 3. That degrades the ANALYSIS. The requested mode is the
opposite: full analysis, no checking. Shipping both under `--fast` would mean a
user asking for quick results silently getting a degraded smoke test wearing a
production label. Renamed the old one to `smoke_mode` / `--smoke` (its own
docstring already called it that) across 32 sites; `--fast` is now the new mode.
The rename was done with a **word-boundary** regex — `fast_mode` is a substring of
`fast_model`, and a naive replace renames the model tier.

**What fast mode removes** (all second opinions; none decides a parameter):
dual annotation, kappa, the pilot and its self-consistency ceiling, guide repair,
boundary redraw, phase observers, adversarial validation, the narrative report,
the delivery audit, result interpretation. **What it does not touch:** every grid,
the corpus, the gold size, the researcher panel, `propose_grids` (a widener, not
a check), and every `store.put_*` call.

**Three design decisions worth keeping:**

1. **`_annotate_both` returns `(labels, None)`, never `(labels, labels)`.** The
   copy would let all four call sites run unedited and write `kappa: 1.000` — a
   perfect score for a measurement nobody took. The `None` forces each caller to
   state what it does with one reading. `GoldRow.n_annotators` was added so
   `agreed` cannot be misread as agreement.
2. **`deps.gate(skipped=True)`.** `GateStatus` already had `"skipped"` and
   `deps.gate` could not produce it. `p2a_pilot_agreement`, `p2b_kappa` and
   `p2b_annotator_symmetry` now record `skipped`; `passed=True` would have left a
   ledger entry identical to a full run's.
3. **One list drives every banner.** `_fast_mode_drops_the_second_opinion`
   populates `cfg.fast_skipped` as it turns each component off, and
   `fast_deliver._banner()` renders that list. A skip cannot exist without
   appearing in all three deliverables. An unknown key degrades to the raw key
   rather than being dropped — `test_an_unknown_skip_key_is_still_disclosed`.

**Verified by mutation, not by passing.** Three guardrails were deliberately
broken (banner drops unknown keys; `_annotate_both` returns a copy; fast mode
shrinks the α grid) and each was caught by exactly the test that describes it.

**Also changed:** `verify_run.py` takes `@check(..., needs=[...])` and reports
`N/A` — never PASS — for a check whose component was skipped; it also knows which
fast deliverable absorbed each full-mode document, which turned 8 SKIPs into real
checks (17 PASS vs 12). `run_summary.json` records `mode` and `fast_skipped`.
`scaled_requirements` zeroes the silent roles so `qmine models --fast` prices what
will actually run. `make fast RUN=x` added.

**One defect of mine, caught by the suite:** `getattr(ctx.cfg.taxonomy, ...)`
assumed `cfg.taxonomy` exists; the seven annotator-concurrency tests build a
`SimpleNamespace` with only `cfg.llm`. Fixed by fetching `cfg.taxonomy`
defensively too — the same shape as the `getattr(ctx.cfg.llm, ...)` beside it.

**Two defects found by running the render, not by reading it:**

1. **`qmine render` upgraded a fast run to a full one.** `render` builds its
   config from the CLI, where `mode` defaults to "full", so re-rendering
   `/tmp/fastrun/f1` produced the thirteen full-mode documents — 叶清单.md,
   类目清单.md, 统一度量面板.md — with **no banner anywhere in them**. The one
   command whose purpose is re-deriving deliverables was the one that could strip
   the disclosure off them. Fixed by `runner.inherit_mode`, which reads the source
   generation's own `config.resolved.yaml`; the RECORDED `fast_skipped` wins over
   the validator's rebuilt list so a banner describes the run, not today's code.
2. **Every render lost the domain — pre-existing, and not mine.** A full-mode
   render of a `k12_zh` run also wrote "**领域**: `generic`". Cosmetic until fast
   mode, whose deliverable filenames carry the domain key: the render deposited
   `generic_自上而下_….md` beside `k12_zh_自上而下_….md`, one document under two
   names. Fixed in the same function.

**Two more defects, found by running `make demo` after adding a parameter to
`run()` — both mine, both in the p8 delivered-leaf collision check:**

3. **The collision gate passed on a check that had crashed.** `cents` was bound
   only inside `if new_labels is not None and not array_equal(...)`, the branch
   that runs when governance actually rewrote the partition. On every run where
   governance changed nothing, `_resolve_indistinguishable_leaves` raised
   `UnboundLocalError`, its own `except` swallowed it, `still_colliding` stayed
   `[]`, and the gate reported "every delivered leaf is distinguishable from its
   siblings by name" having compared nothing. Present in every offline run this
   session; absent from `med04`, where governance did rewrite the tree — which is
   why no live run had shown it. Fixed by binding `cents` before the branch (when
   governance is a no-op the delivered partition IS the pre-governance one) and by
   giving the gate `skipped=not collision_check_ran` so a crashed check can never
   read as a pass.
4. **That gate never reached state.** `deps.gate(...)` at what is now
   naming.py:711 was called without assignment, so the gate was logged and
   dropped: `run_summary.json` did not contain `p8_leaves_are_distinguishable` on
   any run. Invisible to the router, unable to halt anything, unreadable
   afterwards — the same mistake `topdown.py` documents having made once before
   and found five runs later. Now captured and returned.

   The error handler that hid #3 printed only `(UnboundLocalError)` — no file, no
   line, no message. It now names the frame, which is how #3 was found at all.

**The finance corpora, and two more defects the dry run caught before spending.**
`data/raw/金融query-{250701,260701}.xlsx` — 10,000 rows each, one year apart, same
schema: `original_query`, `wise_pv` (full-log frequency), and a
`query_1st_category` that is CONSTANT ("金融"), i.e. the slice that produced the
file rather than a label. A `finance_zh` domain profile already existed.

5. **A CLI default silently overruled the config file.** `cfg.data.text_column =
   text_column` assigned unconditionally, so Typer's default `"query"` overwrote
   a config saying `original_query`, and p1 halted with `KeyError: 'query'`
   before reading a row. `reference_label_columns` had the same bug, failing more
   quietly — declared columns silently become none. The identical defect had
   already been found and fixed for `provider` eleven lines below, with a comment
   saying "a config option that the command line always wins is not an option";
   these two were left standing. Both now guard on the flag being given, and
   `--text-column` takes a `None` default so "unset" is representable.
6. **`--config` REPLACES the default config; there was no way to extend it.**
   `_load_config` loads exactly one file, so a corpus config stating only a text
   column would have discarded the whole of `live.yaml` — the role pins, the
   excluded labs, and the lab-independence requirement double-blind annotation
   rests on. `_load_config`'s own docstring calls that "the one launch mistake
   nothing catches", and I nearly made it. `QMineConfig.load` now honours
   `extends:` (resolved relative to the file, recursive, extending file wins per
   key), and `configs/live_finance.yaml` uses it.

**`fin01` — the first live fast run** (`金融query-250701.xlsx`, finance_zh,
`configs/live_finance.yaml`). Estimated $4.53 fast vs $8.80 full; both
under-report, because three pinned models publish no price and `qmine models`
says so. Early confirmation from the log: `p0_provider PASSED — provider=routed`,
`p1_reference_columns_declared PASSED — no reference label columns, and the
corpus offers none`, and the alpha sweep ran the FULL seven-value grid
`[0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8]` — fast mode shrank nothing.

Corpus notes for whoever reads `fin01`: template coverage is **18.6%**, below the
20% floor, so template fragmentation rests on a small base — a property of this
corpus, reported not gated. `researcher[legacy_audit]` returned no candidates,
correctly: there are no legacy labels to audit.

**Class CODES do not reproduce across runs; the structure largely does.**
Measured on the two finance runs (same source, one year apart, independently
designed taxonomies):

| | |
|---|---|
| classes | `fin01` 20, `fin02` 19 |
| **exact code overlap** | **0** — 0% of the union |

Zero. Yet the Chinese names line up pair for pair: `LOOKUP_FX_RATE` /
`FX_RATE_LOOKUP` (汇率查询), `CONVERT_CURRENCY` / `CURRENCY_AMOUNT_CONVERSION`,
`STOCK_FORUM` / `STOCK_FORUM_NAVIGATION`, `DAILY_ANSWER_RETRIEVAL` /
`DAILY_QUIZ_ANSWER`, `FIND_INSTITUTION_CONTACT` /
`CUSTOMER_SERVICE_PHONE_LOOKUP`, `VERIFY_PLATFORM_LEGITIMACY` /
`TRUST_VERIFICATION`, `OTHER` / `UNKNOWN_OR_OTHER`, and so on — plus two genuine
SPLITS (`LOOKUP_SECURITY_QUOTE` → `STOCK_QUOTE_LOOKUP` + `FUND_NAV_LOOKUP`;
`LOOKUP_COMMODITY_QUOTE` → `FUTURES_COMMODITIES_QUOTE_LOOKUP` +
`PRECIOUS_METAL_PRICE_LOOKUP`).

**STRONGER EVIDENCE, no corpus confound:** `fin02` and `fin03` ran the SAME file
(`金融query-260701.xlsx`) under the same config, and still share **0 of 35 codes**
(19 vs 16 classes). The fin01/fin02 comparison below was confounded by being
different corpora; this one is not. The architect re-invents its naming
convention every run.

Two `fin03` differences are structural, not cosmetic:
* it MERGED pairs `fin02` split (futures + precious metals -> one
  `COMMODITY_CRYPTO_QUOTE`; opinion + commentary -> `SECURITY_INFO_OPINION`);
* it produced **no catch-all class at all** — `fin01` had `OTHER`, `fin02`
  `UNKNOWN_OR_OTHER`, `fin03` none. Gold is unaffected (unfittable rows go to
  `UNLABELED` and are dropped), but at INFERENCE every row is forced into a real
  class with no escape hatch. Worth watching in that run's delivered
  distribution.

**The 0% is measured; the correspondence is EYEBALLED from the Chinese names and
is not a measurement.** Making it one would mean labelling one corpus under both
taxonomies and computing agreement — worth doing, not done here.

Two consequences that matter now:

1. **Never diff two runs by class code.** It reports that nothing reproduced when
   most of it did. This is the concrete evidence behind CLAUDE.md's existing
   advice to reuse the TAXONOMY rather than the run when cross-run comparability
   matters — `--reuse-taxonomy` is the only thing that holds codes fixed.
2. A few differences look like real corpus change rather than naming drift:
   `fin01` carries `GRAY_APP_DOWNLOAD`, `CREDIT_REPORT_CHANNEL` and
   `CALLER_ID_FRAUD_CHECK`, which `fin02` does not; `fin02` adds `CHART_LOOKUP`
   and `SECURITY_CODE_IDENTIFICATION`. Whether those are drift or a genuine
   year-over-year shift is undetermined and would need the same joint labelling.

**Rule EXECUTABILITY swung 72% -> 0% between the two finance runs, and it is
the architect varying, not a bug.** Same config, same source, one year apart:

| | rules | executable | rejected |
|---|---|---|---|
| `fin01` | 46 | **33** | 13 |
| `fin02` | 52 | **0** | 52 |

The rejection reason is identical in both (`does not fire on the rule's own
example or originating query`). The difference is what the architect wrote:
`fin01` named literal phrases (`还会涨/跌吗、未来走势、亏不亏`), `fin02` named
CATEGORIES (`裸数字代码`, `主观/推荐词`, `具体金额换算`). A category is not a
test, and `rules_against_evidence` correctly refuses to pretend otherwise. **No
code was changed for this** — the check is right and the variance is upstream of
it. Open question: an architect prompt that demands literal markers would make
the rules mechanically checkable, but see
[[qmine-prompt-emphasis-is-zero-sum]] — hardening one requirement here has broken
a competing one before.

I chased two wrong hypotheses first (a quoting-style mismatch, then an extractor
bug) and the measurement killed both before either became an edit. It did surface
one real gap on the way: `_QUOTED`'s character class had 「」 but not 『』, while
`usable_markers` already stripped both. Four `fin02` rules gain genuinely usable
markers from the fix (『净值』, 『主连/合约/期货/连续』, 『k线图/图表/走势图』,
『净值/基金』). That is a real defect and a small one; it explains 4 rules, not 52,
and the test says so explicitly so nobody later reads it as the cause.

**Template COVERAGE and locator REACH move independently — do not read one as a
proxy for the other.** Measured across the two finance runs:

| | template coverage (share of ROWS) | locator reach (share of CLUSTERS) |
|---|---|---|
| `fin01` (2025) | 18.6% | 24% |
| `fin02` (2026) | **36.3%** | **12%** |

Coverage doubled and reach halved. The grid proposer's own note on `fin02` says
why: its 12 phrasing groups are built around stock-code prefixes (600/300/60)
plus 走势图/行情/股吧, so the covered rows pile into a few clusters. A phrasing
group that matches many rows in ONE cluster raises coverage and does nothing for
reach — and reach is what decides whether the K located on the reference frame
generalises to the corpus it is then applied to.

Both runs therefore WARN on `p5_locator_reaches_the_corpus`, and `fin02` warns
harder despite looking better on the headline number. Anyone using coverage to
predict reach will get the sign wrong; I did.

**`fin02` was damaged by a mid-run macOS file-access revocation, and the damage is
instructive.** The grant was pulled while the run was in flight. My session
recovered on restart; the run's own process (pid 91953) never did, so from
17:16 it could not open any file it had not already read.

| phase | state |
|---|---|
| p1-p7 | CLEAN — finished before the revocation |
| p8 | DEGRADED — `families_final: 0` (family naming could not read its prompt), leaf disambiguation skipped |
| p11 | FAILED — `ModuleNotFoundError: qmine.report.builder` (lazy import, first touched after the revocation) |

**Three fixes from this session were validated by the accident, not by design:**

1. `p8_leaves_are_distinguishable` reported **SKIPPED** — naming the failing frame
   (`PermissionError ... disambiguator.md — at pathlib.py:1013 in open`). Before
   this session it would have reported "PASSED — every delivered leaf is
   distinguishable from its siblings by name" on a check that had crashed, with
   no file, no line and no message. This is the exact failure the fix was written
   against, reproduced by an accident nobody could have staged.
2. `qmine render fin02` recovered all three deliverables from artifacts —
   **10,000 rows x 19 cols, 10 sheets**, and the `原始档案位置` table resolved down
   to 2 unresolved entries. Before the store-resolution fix the same render
   produced 8 empty sheets, 0 rows, and 13 x "未生成". Verified on a real run.
3. The render carried `领域: finance_zh` and `模式: fast` — mode and domain
   inheritance both holding on a live run rather than a fixture.

**`fin02` IS a valid verification run and is NOT a reference delivery.** Its
p1-p7 results stand (19 intents, 33 leaves, coherence 3.79, held-out
reproduction 99.3%, no phantom classes in 3,200 gold rows). Its family names do
not: `families_final` is empty, so the documents fall back to
`树审计未覆盖 (治理新建) · 主要叶「…」`. No resume path repairs this — p8
"completed", so `--resume` restarts at p11 and re-runs nothing that was degraded.

`verify_run`: `fin02/gen02` **17 PASS / 6 N/A / 3 FAIL / 2 SKIP** against
`fin01/gen01` 19/6/2/1. All three failures are the damage, correctly named:
the unnamed families, the p11 halt, and the known render limitation on
gold-set provenance.

**The phantom-class fix is CONSISTENT but not independently proven.** `fin02`
carries zero off-schema labels across 3,200 gold rows — but the guard never
fired, and `fin01` hit the condition once in 3,200 (0.03%), so zero is equally
consistent with luck. `test_a_solo_annotator_cannot_invent_a_class` remains the
actual verification.

**Two §2 open questions closed today** (removed there, recorded here per the
file's contract):

* **#17 `verify_run.py` on a rendered generation** — taught it the store's
  cross-generation resolution. It now searches DOWN through generations for
  artifacts but deliberately NOT for documents: an older generation's report is
  the thing a re-render replaces, and reading it would silently verify the
  previous version. Measured on `fin01/gen03`: 8 PASS / 12 SKIP before,
  **19 PASS / 2 SKIP** after.
* **#16 `finance_zh` untested** — narrowed, not deleted. Three real runs
  exercised it; its risk categories, template seeds and `expected_l1_range`
  [15,22] all held (16-20 L1 classes across the three). `sports_zh` and
  `politics_zh` remain untested.

**A wrong number was shipping inside every fast deliverable.** The banner said
"交付文档从 13 份减为 3 份". A full run ships **10** markdown documents (`med04`,
`live44`; `live42` shipped 9). The count appeared in 8 places including the
banner and the `--fast` help text. The banner now states no full-mode count at
all — a reader of a fast deliverable needs to know what they hold and that
nothing was withheld, and a hard number there goes stale silently inside every
shipped document. The docs carry the real figure with its composition.

**TEMPORAL DRIFT: pool the snapshots, never diff two runs.** Two runs do not
produce comparable labels — `fin02` and `fin03` ran the SAME 10,000 rows under the
same config and shared **0 of 35 class codes**. So a year-over-year comparison has
to happen INSIDE one run.

* `tools/pool_snapshots.py` stacks snapshots with a `_snapshot` tag.
* `tools/drift_report.py` joins `labels_full.csv` back **positionally** (verified
  20,000/20,000) — NOT on query text, which fans out on repeated queries.
* `_snapshot` is deliberately NOT passed as a reference label column: reference
  columns are the frame the K locator scores against, so declaring it would ask
  the clustering to find a K separating 2025 from 2026.
* Shares are WITHIN-SNAPSHOT, never raw. 医疗 PV falls 9.74M -> 5.21M (-47%), so
  raw PV reports every class as declining. Row share answers "did the variety of
  asks change", PV share "did traffic change". Row shares get a z-test; PV gets
  none by design — traffic is the population, not a sample.

**The degeneracy check is the one that matters**, and `fin-pool` passes it: of 54
leaves and 17 intents, **0 are >95% one snapshot** (share-of-2025 median 0.51).
The clusters span both years, so the frame really is shared. Run this check on
every pooled run before believing its drift table.

**`fin-pool` measured:** 21 PASS / 6 N/A / 0 FAIL (control `fin03` identical),
17/17 phases, 2.11 h, 252 calls, $4.49, 54 delivered leaves, 34/34 families named.
Pooling was CHEAPER than feared and better in two ways: gold is capped at 3,000
regardless of corpus size, ECE 0.0159 **PASSED** where every single-snapshot
finance run warned (0.0247-0.0289), and 0/59 leaves fell below the coherence
floor. Finding: the 2026 finance mix consolidated onto `LOOKUP_MARKET_QUOTE`
(66.2% -> 79.0% of PV) and stock-forum browsing, while every service and
informational intent receded — corroborated by regexes that never touch the
pipeline's labels (stock-forum +0.7pp in both, exactly).

**Domain profiles: `medical_zh` had ZERO template seeds**, which is how the first
pooled medical run reported "NO template group passed the cohesion check (0/12)"
and rested its alpha decision on untrusted masks. Now 8 seeds, 54.5%/60.1%
coverage across the two snapshots, worst overlap 11-12%. Three decisions, each
measured: `有哪些` REJECTED as a seed (a phrasing, not an intent — it enumerates
symptoms, foods, types and hospitals alike); `禁忌/副作用` REJECTED (67% of its
matches also matched efficacy — a facet, not a family); efficacy SPLIT by dosage
form into drug/substance, disjoint at 835 + 1,917 = 2,752 exactly.
`people_zh`, `film_tv_zh`, `education_zh` added the same way.

**An empty text cell halted a launched run.** Under pandas 3.0 `.astype(str)` no
longer turns NA into "nan", so one blank in 20,000 rows reached `char_profile` and
died with `object of type 'float' has no len()`. `edu-pool` halted on row 17,717
AFTER launch. p1 now drops empty-text rows with a loud count and REFUSES a corpus
that is >5% empty, because that is a broken export and analysing what survives
hides it.

**Open, and deliberately not resolved here:** no paid fast run has been made, so
the single-annotator gold set is untested against a real annotator. `med04`'s
40.3% vs `live38`'s 78.3% annotator-a win rate says which annotator is better
flips by corpus and model, so `primary_annotator` is a recorded default, not a
finding.
