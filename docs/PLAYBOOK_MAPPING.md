# Playbook → code

Where every principle, phase, and trap from the playbook lives in this
repository, and — for the principles — what makes it *enforced* rather than
merely intended.

## The twelve principles

| # | principle | where it lives | enforcement |
|---|---|---|---|
| 1 | Two routes, one axis each | [`nodes/topdown.py`](../src/qmine/graph/nodes/topdown.py), [`nodes/bottomup.py`](../src/qmine/graph/nodes/bottomup.py) | `labels_full.csv` carries `td_l1` and `bu_family_final` as separate columns; neither node writes the other's |
| 2 | The data — and the reviewer — hold a veto | [`build.py::_make_review_node`](../src/qmine/graph/build.py) | `interrupt()` at review points; a rejection writes to the `rejections` namespace and routes to a **new generation**, never a patch |
| 3 | Metrics must not betray the objective | [`records.py::METRIC_AUTHORITY`](../src/qmine/records.py), [`panel.py`](../src/qmine/ops/panel.py), [`cluster.py::choose_local_k`](../src/qmine/ops/cluster.py) | `decisive_ranking()` **raises** on an advisory metric, so silhouette can select nothing *through the panel*. It does rank at one place — k inside a single fixed representation, where its phrasing bias is a constant offset — and there `choose_local_k` requires the split to beat a structureless reference first, and overrules silhouette when a lead inside its noise costs real reproducibility |
| 4 | Stability is a **veto**, not a ranker | [`cluster.py::replay_stability`](../src/qmine/ops/cluster.py), [`cluster.py::choose_local_k`](../src/qmine/ops/cluster.py) | **Measured: stability's seed-to-seed sd (~0.10 ARI) is twice the gap between adjacent K (~0.05), so ranking by it reads noise — and the noise leans coarse.** It therefore only *vetoes* a K that cannot be reproduced at all; K itself is located by intent alignment (AMI against phrasing groups), the one metric here with a two-sided penalty and thus an interior optimum. Silhouette has no vote across representations, because its bias tracks the very quantity being varied; within one representation (local k) it ranks and stability vetoes bad trades. von Luxburg 2010 positions stability as a *screen* — this is now consistent with that, and the old "stability peak" wording was the overreach |
| 5 | Blind review defeats anchoring | [`memory/context.py`](../src/qmine/memory/context.py), [`nodes/naming.py`](../src/qmine/graph/nodes/naming.py) | `Send` payload isolation (structural) **+** `BlindnessFirewall.assert_blind()` (lexical); both tested |
| 6 | Governance executed, not recorded | [`ops/governance.py`](../src/qmine/ops/governance.py) | `assert_all_settled()` raises on any `proposed` prescription, and runs before reports are written |
| 7 | Uniform panel, deterministic display | [`panel.py::UniformPanel`](../src/qmine/ops/panel.py), [`determinism.py::median_index_exemplar`](../src/qmine/determinism.py) | `comparison_table()` refuses to mix `panel_id`s; exemplars are a pure function of the hit set |
| 8 | Everything reproducible | [`determinism.py`](../src/qmine/determinism.py), [`artifacts.py`](../src/qmine/artifacts.py) | seed policy in the manifest; append-only generations; content-addressed memo cache |
| 9 | Granularity by triangulation | [`cluster.py::triangulate_k`](../src/qmine/ops/cluster.py) | three independent estimators; **on disagreement it takes the intent-alignment optimum and records the dissent** — not the stability peak, which has no resolving power here. Ties are decided against a measured noise floor and the whole tie set is reported |
| 10 | Risk isolated, always | [`ops/audit.py::screen_risk`](../src/qmine/ops/audit.py), `risk_sentinel` role | pre-screen + independent sentinel + `isolate_leaves()` executed as a lookup remap |
| 11 | Every category is a `user_need` sentence | [`records.py::TaxonomyNode`](../src/qmine/records.py), [`LeafNaming`](../src/qmine/records.py) | required field on both label systems; `叶清单.md` lists all of them, and flags any delivered leaf that has none |
| 12 | Report the model's limits | [`report/builder.py`](../src/qmine/report/builder.py) | mandatory "what these numbers do not mean" section; asserted by a test |
| 13 | Agents describe; measured quantities decide | [`ops/checks.py`](../src/qmine/ops/checks.py), [`ops/findings.py`](../src/qmine/ops/findings.py), [`ops/edits.py`](../src/qmine/ops/edits.py), [`ops/propose.py`](../src/qmine/ops/propose.py), [`agents/verify.py`](../src/qmine/agents/verify.py) | Four doors, each with a mechanical guardrail, none able to change a parameter: prose checked value-by-value against a fact sheet; an observation must cite a *resolving* artifact path and may carry a machine-evaluable assertion (confirmed = the assertion failed, and only then may it block); grid proposals made blind to every score, so pre-registered; deliverable edits anchored, sourced from the cited artifact, language-checked — and refusals printed beside the applied edits. Findings live at the **run** root and close only when their own assertion holds again |

## The twelve phases

> The two routes now run **concurrently** — `build_graph` forks at p1 and joins at p2c. See `.claude/rules/graph-and-state.md` for the superstep constraints that decide how much that saves.

| phase | node | core module | key artifacts |
|---|---|---|---|
| P0 foundation | `p0_foundation` | [`determinism.py`](../src/qmine/determinism.py) | `run_manifest.json`, `config.resolved.yaml` |
| P1 audit + templates | `p1_audit` | [`ops/audit.py`](../src/qmine/ops/audit.py), [`ops/templates.py`](../src/qmine/ops/templates.py) | `data_audit.json`, `template_groups.json`, `risk_screen.json` |
| P2a taxonomy | `p2a_taxonomy` | [`agents/roles.py`](../src/qmine/agents/roles.py) | `taxonomy.json` |
| P2b gold | `p2b_gold` | [`ops/classify.py::agreement`](../src/qmine/ops/classify.py) | `gold.csv`, `gold_agreement.json` |
| P2c classifier | `p2c_classifier` | [`ops/classify.py`](../src/qmine/ops/classify.py) | `topdown_model.joblib`, `topdown_metrics.json` |
| P2d validation | `p2d_validate` | `adversary` role | `adversarial_validation.json` |
| P3 representation | `p3_represent` | [`ops/represent.py`](../src/qmine/ops/represent.py) | `emb_base.npy`, `emb_hybrid.npy`, `representation.json` |
| P4 battery | `p4_battery` | [`ops/cluster.py`](../src/qmine/ops/cluster.py) | `battery.json` |
| P5 granularity | `p5_granularity` | `k_sweep`, `deep_aligned_estimate` | `granularity.json` |
| P6 hierarchy | `p6_hierarchy` | `build_hierarchy`, `refine` | `leaf_labels.npy`, `leaf_centroids.npy` |
| P7 naming + audit | `p7_prepare` → `p7_name_shard` ×5 → `p7_audit` | [`ops/cards.py`](../src/qmine/ops/cards.py) | `naming_cards.json`, `tree_naming.json` |
| P8 governance | `p8_governance` | [`ops/governance.py`](../src/qmine/ops/governance.py) | `leaf_family_final.npy`, `governance.json` |
| P9 panel | `p9_panel` | [`ops/panel.py`](../src/qmine/ops/panel.py) | `metrics_panel.json` |
| P10 deployment | `p10_deploy` | [`ops/classify.py::CentroidClassifier`](../src/qmine/ops/classify.py) | `labels_full.csv`, `centroid_classifier.joblib` |
| P11 reports | `p11_report` | [`report/`](../src/qmine/report/) | 4 markdown reports, 6 figures, executed notebook |
| P12 maintenance | `p12_maintain` | [`nodes/delivery.py`](../src/qmine/graph/nodes/delivery.py) | `maintenance.json` |
| P2d/P12 referee upgrade | `qmine promote` | [`ops/promotion.py`](../src/qmine/ops/promotion.py) | `<label>_v1` column, `label_source` |

## The fifteen traps

| # | trap | how this codebase avoids it |
|---|---|---|
| 1 | silhouette as an interpretability proxy | `METRIC_AUTHORITY` marks it advisory; `decisive_ranking()` raises on it |
| 2 | assuming a bigger encoder is better | `encoder_bakeoff()` ranks on clustering metrics, not leaderboard scores |
| 3 | GBDT on raw embedding coordinates | linear head only, with the reason in the docstring |
| 4 | BisectingKMeans for a cheap hierarchy | in the battery as a *candidate*; loses on stability and the loss is recorded |
| 5 | auto-tuning HDBSCAN on a composite score | screened by `(noise_rate asc, n_clusters desc)` for human review, never auto-selected |
| 6 | trusting kNN label-correction flags | flags are reported, never auto-applied |
| 7 | cross-interpreter pickle | parquet/npy/json as primary formats; the manifest pins the interpreter |
| 8 | missing CJK fonts | `viz.setup_matplotlib()` sets the family and silences the warning |
| 9 | governance recorded but not executed | `assert_all_settled()` raises before reports are written |
| 10 | cherry-picked display samples | `median_index_exemplar()` — a pure function of the hit set |
| 11 | one global fine-grained K | two-level `build_hierarchy()`; `heldout_reproduction` gate catches regressions |
| 12 | distillation accuracy read as human accuracy | metric carries the disclaimer; report section asserts it |
| 13 | expecting clustering to find pragmatic intents | `pragmatic_only` flag on taxonomy nodes; profile lists them per domain |
| 14 | namers seeing existing labels | `Send` isolation + firewall, both tested |
| 15 | shipping an unexecuted notebook | `nbclient` executes before delivery; the artifact summary records the error count |

## Domain migration (Part IV)

| must be re-derived | file |
|---|---|
| phrasing seeds, risk categories, tokenizer, n-grams, encoder candidates, expected sizes | `configs/domains/*.yaml` |
| **α** | re-run the sweep; `configs` deliberately has no `alpha` field to inherit |
| family K | re-run the K-sweep and re-locate (AMI optimum; stability vetoes only) |

Five profiles ship: `k12_zh`, `finance_zh`, `sports_zh`, `politics_zh`,
`ecommerce_en`. See the [`qmine-new-domain`](../skills/qmine-new-domain/SKILL.md)
skill for writing a sixth.


## Running a phase outside the pipeline

| capability | command | why it exists |
|---|---|---|
| blind naming by an external panel | `qmine export-cards` / `qmine import-namings` | naming is the step where a stronger judge pays for itself; the firewall check travels with the exported briefs so an outside panel is under the same constraint as the in-process agents |
| statistically honest gates | [`ops/stats.py::proportion_gate`](../src/qmine/ops/stats.py) | a blocking gate that fires on sampling noise teaches people to lower thresholds; gates compare a confidence interval and report `met` / `missed` / `underpowered` |

## Phase coverage — every step in Part III, and where it runs

Sub-phases are listed individually because the playbook's Phase 2 is really five
steps, and an audit that only checks "Phase 2: done" hides the ones that are not.

| playbook step | graph node | implementation |
|---|---|---|
| P0 engineering foundation | `p0_foundation` | [`nodes/foundation.py`](../src/qmine/graph/nodes/foundation.py) — manifest with seeds, versions, prompt hashes, config hash |
| P1 audit + template mining + risk pre-screen | `p1_audit` | [`ops/audit.py`](../src/qmine/ops/audit.py), [`ops/templates.py`](../src/qmine/ops/templates.py) — coverage gate on the 20-40% window |
| P2a taxonomy design (research fan-out) | `p2a_taxonomy` | 5 disjoint research angles → architect → critic, [`agents/roles.py`](../src/qmine/agents/roles.py) |
| P2b gold: double-blind + κ + referee | `p2b_gold` | [`nodes/topdown.py`](../src/qmine/graph/nodes/topdown.py); referee drafts new rules from the gaps it finds |
| P2b step 4 active-learning round 2 | `p2b_gold` | `_active_learning_round` — lowest-margin rows under a round-1 sparse model |
| P2c rules + sparse⊕dense classifier + ECE | `p2c_classifier` | [`ops/classify.py`](../src/qmine/ops/classify.py) — linear head by design, not default |
| P2d step 1 kNN label-error scan | `p2d_validate` | `knn_label_scan` — flags are **review-only**, never auto-applied |
| P2d step 2 adversarial validation | `p2d_validate` | agents instructed to *disprove* each label |
| P2d step 3 referee upgrade protocol | `qmine promote` | [`ops/promotion.py`](../src/qmine/ops/promotion.py) — blind, order-randomised, significance-gated |
| P2e L1 geometric audit + sub-intents + L2 classifier | `p2e_subintents` | [`ops/subintent.py`](../src/qmine/ops/subintent.py) — reports which classes the embedding cannot see |
| P3a encoder bake-off | `p3_represent` | judged on *your* clustering task, not a leaderboard |
| P3b char TF-IDF + SVD | `p3_represent` | [`ops/represent.py`](../src/qmine/ops/represent.py) |
| P3c hybrid + α-sweep | `p3_represent` | α² algebra documented; silhouette explicitly vote-less |
| P4 algorithm battery | `p4_battery` | [`ops/cluster.py`](../src/qmine/ops/cluster.py) — density methods screened separately |
| P5 K triangulation | `p5_granularity` | intent-alignment optimum + DeepAligned survival + domain prior, judged separately; stability vetoes, never ranks |
| P6 two-level tree + refinement + held-out | `p6_hierarchy` | gate uses a confidence interval, not a point estimate |
| P7 blind naming + tree audit | `p7_prepare` → `p7_name_shard` (Send) → `p7_audit` | [`nodes/naming.py`](../src/qmine/graph/nodes/naming.py) + firewall |
| P8 governance executed | `p8_governance` | [`ops/governance.py`](../src/qmine/ops/governance.py) — run fails on any unexecuted prescription |
| P9 uniform panel | `p9_panel` | [`ops/panel.py`](../src/qmine/ops/panel.py) — refuses to mix panel ids |
| P10 deployment + margin routing + live demo | `p10_deploy` | both label systems side by side, plus `td_l2` |
| P11 reports + figures + executed notebook | `p11_report` | [`report/`](../src/qmine/report/) — notebook executed, 0 errors required |
| P12 baseline + novelty sentinel + drift diff | `p12_maintain`, `qmine diff` | maintenance analyst runs when an earlier baseline exists |

### Deliberate deviation from the playbook's ordering

The playbook draws Phase 2 and Phase 3 as parallel branches. They are
conceptually independent, but P2c's feature recipe concatenates the dense
embedding that P3a selects, so the graph runs
`p2a → p2b → p3 → p2c → p2d → p2e`. Taxonomy design and gold annotation need no
embedding and stay first; only the classifier waits.
