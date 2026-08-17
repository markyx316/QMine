# Implementation plan

The plan this repository was built from, with what actually happened recorded
against each stage. Kept as-written rather than tidied afterwards, because the
places where execution diverged from the plan are the useful part.

---

## Stage 0 — Study before building

**Goal.** Do not write a line of LangGraph from memory. Verify every API, every
version, every model id against the live packages and current documentation.

**Method.** Nine research agents in parallel over disjoint assignments —
LangGraph core, multi-agent orchestration, memory and context engineering, the
TradingAgents teardown, agent/tool/skill design, production observability, the
2026 clustering and embedding stack, LLM taxonomy-induction methods — followed
by an adversarial completeness pass whose only job was to find where the other
eight were wrong.

**Outcome.** ~550 KB of dossiers, in [`docs/research/`](research/). The critic
pass paid for the whole stage:

| finding | consequence if missed |
|---|---|
| `temperature` is a **400** on Opus 5 / Fable 5 — the parameter was removed | every deep-tier call fails |
| model ids carry **no date suffix** (`claude-opus-5`) | every call fails |
| Haiku 4.5's minimum cacheable prefix is **4096 tokens** | the taxonomy prefix silently never caches on the bulk-labelling tier |
| `GenericFakeChatModel.with_structured_output()` raises `NotImplementedError` | the entire offline test path is impossible as usually recommended |
| LangGraph's default `recursion_limit` is **10007**, not 1000 | an unbounded refinement loop spends five figures before erroring |
| `Send` payload isolation verified empirically | the anti-anchoring guarantee is structural, and can be *tested* |
| `cohen_kappa_score(replace_undefined_by=0.0)` exists in sklearn 1.9 | a degenerate annotator returns NaN and poisons the gate instead of failing it |

Two dossiers contradicted each other on `SqliteStore`'s existence; the critic
resolved it by importing the module, which deleted a whole planned workstream
(writing a custom `BaseStore`).

---

## Stage 1 — Foundation

**Goal.** Make reproducibility and artifact handling structural before any
science depends on them.

- `determinism.py` — seed policy (0 numbers / 42 pictures / (0,1) replay),
  content hashing, `median_index_exemplar` (the anti-cherry-pick rule)
- `artifacts.py` — append-only generations, `ArtifactRef`, content-addressed
  memo cache keyed on `(op, params, input)`
- `records.py` — typed records, and `METRIC_AUTHORITY` as a **table** so
  Principle 3 becomes a type-level fact
- `config.py` — the invariant/re-derive split from Part IV
- `state.py` — graph state holding pointers only, with reducers on every channel
  a parallel branch writes

**Acceptance.** An artifact round-trips; a second identical `memoize` call is a
cache hit; opening a new generation leaves the old one untouched. ✅

---

## Stage 2 — LLM layer

**Goal.** Two-tier routing, response caching, budget ceilings, and — the
load-bearing piece — an offline mode good enough that the whole graph runs in CI.

**Divergence from plan.** The plan said "use `GenericFakeChatModel` for tests".
The critic pass had already shown that cannot work. So `OfflineHeuristicModel`
was written instead: it walks the requested Pydantic schema and fills it by
*actually computing something* — cluster names from the card's own top n-grams,
labels from regex evidence — stamping every record `offline-heuristic`.

This turned out to be worth more than a test double. It made the full
twelve-phase pipeline runnable with no credentials, which is how most of the
integration bugs below were found.

**Acceptance.** All eleven agent schemas synthesise and validate offline; a
repeated call returns byte-identical output. ✅

---

## Stage 3 — The science

Deterministic ops, each a pure function of its inputs so the whole layer is
cacheable and testable without a graph.

| module | phases | the piece that mattered |
|---|---|---|
| `ops/audit.py` | P1 | shape-defined vs catch-all legacy buckets, reported separately |
| `ops/templates.py` | P1, P3, P9 | fragmentation as `exp(H(p))` — perplexity, so stragglers barely register and a real three-way split is loud |
| `ops/represent.py` | P3 | the α² algebra, and a sweep whose selection key excludes silhouette |
| `ops/cluster.py` | P4-P6 | replay stability; two-level hierarchy; refinement converging on movement |
| `ops/panel.py` | P9 | `panel_id` stamping; `decisive_ranking()` raising on advisory metrics |
| `ops/governance.py` | P8 | prescriptions as a state machine with an evidence pointer |
| `ops/classify.py` | P2c, P10 | linear head by argument; κ with the degenerate case handled |
| `ops/cards.py` | P7 | centre + random + **edge** members, so coherence scores mean something |

**Acceptance.** 19 unit tests, including a numerical verification that
`cos(H,H') = (cos_sem + α²·cos_surf)/(1+α²)` holds to 1e-5. ✅

---

## Stage 4 — Agents

Eleven roles, each a thin subclass over a versioned prompt file whose hash lands
in the run manifest.

The design work was in what each role is **denied**: researchers get disjoint
angles and never see each other's findings; annotators never see each other's
labels; namers see member queries and nothing else at all; the adversary is told
to *attack* rather than verify, because an agent asked "is this right?" agrees
with whatever it is shown.

**Acceptance.** Every role runs offline and returns a validated object. ✅

---

## Stage 5 — Memory

Three tiers: checkpointer (working), artifact store (data), `SqliteStore`
(long-term, namespaced semantic / episodic / procedural).

The **blindness firewall** was written here rather than in the naming phase, on
purpose — it belongs to the memory/context layer because its job is to police
what enters a prompt, and putting it at the call site would mean a future call
site could forget it.

**Acceptance.** Memory survives a process restart; the firewall raises on a
taxonomy name, a legacy label, and a peer agent's output; a clean card passes. ✅

---

## Stage 6 — The graph

17 nodes, SQLite checkpointing, a gate router that halts on blocking failure,
`interrupt()` at reviewer sign-off points, and `Send` fan-out for Phase 7.

**Divergence from plan.** The plan followed the playbook's diagram, where the
two routes are parallel. The first run failed with
`KeyError: no artifact named 'emb_base'` — Phase 2c's feature recipe concatenates
the dense embedding that Phase 3a chooses, so the top-down classifier genuinely
cannot precede the bottom-up encoder choice. Order became
`p2a → p2b → p3 → p2c → p2d`, with the dependency documented in the source.

**Acceptance.** The graph compiles and every phase completes offline. ✅

---

## Stage 7 — Reports

Reports as *projections of recorded state*, never prose written from memory:
the failure-history section renders the `rejected` list inside each
`DecisionRecord`; the governance section renders the prescription ledger; the
notebook is assembled with `nbformat` and executed with `nbclient` before
delivery, because an unexecuted notebook is not a deliverable.

**Acceptance.** Tests assert the provenance note and the "what these numbers do
not mean" section are present. ✅

---

## Stage 8 — Validation on real data

This stage found the bugs that mattered, because the smaller runs could not.

| # | found | fix |
|---|---|---|
| 1 | `fast_mode` assigned after construction, so the model validator never shrank the grids — a "fast" run silently ran full-size | re-validate the config after CLI overrides |
| 2 | Two offline annotators returned identical labels (κ = 1.000) because the response cache keyed on prompt but not role | role is part of the cache key; κ became 0.801 |
| 3 | A failed blocking gate halted the run with an empty `halt_reason` | a terminal node that names the gate and prints its remediation |
| 4 | Agglomerative clustering on 50k rows materialises a 20 GB distance matrix | the battery ranks on a capped sub-sample; the winner is fitted on the full corpus |
| 5 | **Mined affix families poisoned the decisive metric.** `是什么` scored 8.07 fragmentation — it attaches to every intent in the corpus, so its spread measures the marker, not the partition | `trusted` flag: seeded families judge representations, mined ones drive coverage and display only |
| 6 | **The α rule let noise outvote signal.** Lexicographic sorting picked α=0.3 on a fragmentation lead of 0.13 (noise) while discarding a stability lead of 0.22 (real) | fragmentation differences inside a 5% tie-band are ties, broken on stability |

| 7 | **Derived state lived only in process memory.** A resumed run started with an empty cache, so Phase 8 computed its metric deltas over zero phrasing families and reported no fragmentation change | `Deps.template_masks()` / `.taxonomy()` / `.leaf_family_final()` rebuild from artifacts |
| 8 | `qmine resume` rebuilt a *default* config, dropping `reference_label_columns` — so the blindness firewall armed with **zero** terms and the anti-anchoring guarantee lapsed silently | resume loads the run's own `config.resolved.yaml` |
| 9 | **The firewall's own design was wrong.** A purely lexical scan flagged the legacy label `作文` inside the genuine query `我的自画像作文350字`, and the shard failure dropped ten clusters from naming | field whitelist as the primary check; lexical scan restricted to non-corpus-derived fields |
| 10 | A failed naming shard was a warning, so ten unnamed clusters shipped and the mean coherence averaged over the hole | blocking `p7_all_leaves_named` gate |
| 11 | Refinement oscillated — merge joined two leaves, the split probe saw the seam and split them back, forever | leaves created by this round's merge are exempt from splitting |

| 12 | **The auditor declared its own prescription `executed`** with an empty evidence pointer — the exact "recommended but never applied" state the Phase 8 gate exists to catch, arriving through the gate's own front door | status and evidence are reset on ingest; only the pipeline may execute |
| 13 | `split_leaf` prescriptions were accepted and never implemented, so they silently disappeared | splits implemented as a real local re-partition; declined **with a stated reason** when the data to apply them is absent |
| 14 | The split overwrote `leaf_labels`, destroying the pre-governance partition Phase 9 compares against | written as `leaf_labels_final`; the original survives, as the append-only rule requires |

Items 5, 6 and 9 are methodological rather than mechanical, and none is visible
on synthetic data or a 2,000-row sample. Item 9 is the one worth dwelling on: the
first firewall was *too strict*, and being too strict made it silently destructive
rather than safely conservative. They are the argument for running the real
corpus before declaring the thing finished.

---

## Stage 9 — Skills and documentation

Five Agent Skills (`qmine-run`, `qmine-new-domain`, `qmine-review-gate`,
`qmine-diagnose`, `qmine-interpret-tree`), the architecture document, the
playbook mapping, and the research dossiers.

Each skill carries the honesty requirement explicitly: check
`llm_usage.provider` before describing any agent-produced artifact, and say so
if the run was offline.

---

## What this does not do

Stated plainly, because a plan that only lists successes is not a plan.

- **No live-LLM run was performed.** This environment has no API key, so every
  end-to-end run used the offline heuristic stand-in for judgment steps. The
  clustering, embeddings, and metrics are real and were computed on the full
  50k corpus; the taxonomy and cluster names produced here are heuristic
  placeholders. Supply `ANTHROPIC_API_KEY` and the same graph runs the real
  agents — that path is exercised by unit tests against the registry but has not
  been run end to end.
- **The reviewer loop is implemented and tested, not exercised by a human.**
  `interrupt()`, veto recording, and new-generation routing all work; nobody has
  actually vetoed a tree here.
- **No LangSmith tracing.** Deliberate: the system must run with zero network
  egress. Tracing switches on through the standard environment variables.
- **Batch API not wired.** For a 5,000-row gold set the Message Batches API
  would halve the cost; the research dossier covers it and the integration is
  not written.
- **The referee upgrade protocol is implemented and unit-tested, not exercised
  end to end.** `qmine promote` judges disagreements blind with randomised side
  order and promotes only on a significant win; running it needs two real label
  sets from two real model generations, which this environment cannot produce.
