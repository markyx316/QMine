# Architecture

## The problem this shape solves

The playbook is twelve phases of mostly-deterministic computation — encoding,
clustering, metrics — punctuated by a handful of genuine judgment calls, and
gated by a human who holds a veto. Three properties follow from that shape, and
they drove every structural decision below.

**Judgment is rare and expensive; computation is frequent and cheap.** So the
agents are not in charge. A supervisor that re-decides the phase order on every
step would spend model calls rediscovering a sequence fixed by the methodology.
The graph is mostly a chain; agents are called at the specific points where a
human would have had to think.

**The data is large and the context window is not.** A 50k × 768 embedding is
150 MB. If it enters graph state, every checkpoint serialises 150 MB and every
agent prompt risks carrying a slice of it. So state holds *pointers*, and agents
receive purpose-built evidence packets — thirty member queries, not a matrix.

**Several methodological guarantees are only as good as their enforcement.**
"The naming agents did not see the existing labels" is either a structural fact
or an unverifiable claim. This system makes it structural.

---

## The three planes

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONTROL PLANE — LangGraph StateGraph                               │
│  17 phase nodes · SqliteSaver checkpoint after every node           │
│  gate router (halts on blocking failure) · interrupt() for review   │
└───────────────┬─────────────────────────────────────────────────────┘
                │ state = pointers + small records (kilobytes)
┌───────────────▼─────────────────────────────────────────────────────┐
│  DATA PLANE — ArtifactStore                                         │
│  runs/<id>/gen01, gen02, …   append-only generations                │
│  cache/  content-addressed memo (op, params, input) → skip recompute │
│  ArtifactRef{name, path, sha256, shape, producer} lives in state    │
└───────────────┬─────────────────────────────────────────────────────┘
                │ evidence packets (cards, tables, samples)
┌───────────────▼─────────────────────────────────────────────────────┐
│  JUDGMENT PLANE — 11 agent roles                                    │
│  two-tier model routing · response cache keyed by (role, prompt)    │
│  blindness firewall on every blind-review payload                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why state holds pointers

```python
class PipelineState(TypedDict, total=False):
    artifacts: Annotated[dict[str, ArtifactRef], merge_artifacts]   # ~400 bytes each
    metrics:   Annotated[dict[str, MetricSet], merge_metrics]
    decisions: Annotated[list[DecisionRecord], operator.add]
    prescriptions: Annotated[list[Prescription], merge_prescriptions]
```

Every channel written by more than one node in the same super-step has a
reducer. This is not stylistic: LangGraph raises `InvalidUpdateError` on
concurrent writes to a channel without one, and the Phase 7 fan-out writes
`namings` from five workers at once.

`merge_prescriptions` is the interesting one — it keys by id, so an `executed`
record supersedes its own `proposed` predecessor instead of sitting beside it and
double-counting.

---

## The agent roster

| role | tier | job | what it is denied |
|---|---|---|---|
| researcher ×5 | fast | one angle each on the corpus | the other researchers' findings |
| taxonomy_architect | deep | synthesise one taxonomy | — |
| taxonomy_critic | deep | break the draft | — |
| annotator ×2 | fast | independent gold labels | each other's labels |
| referee | deep | adjudicate, and draft the missing rule | — |
| adversary | fast | prove labels wrong | — |
| namer ×5 | fast | name clusters blind | every label in the system |
| tree_auditor | deep | build families, find twins, prescribe | — |
| risk_sentinel | deep | independent compliance sweep | what the namers flagged |
| reporter | deep | write deliverables | — |
| maintainer | deep | diff against the previous baseline | — |

The **denied** column is the design. Five researchers given the same brief
produce five rediscoveries of the obvious categories; five given disjoint angles
span the problem. Two annotators who can see each other's work agree by
construction and their κ measures nothing.

Two tiers, borrowed from TradingAgents' deep/quick split: the architect, the
referee and the auditor get the strong model because their mistakes propagate
into every downstream artifact; the six hundred annotation calls get the fast one.

---

## How blind review is enforced

Two independent mechanisms, because anchoring is the default outcome rather than
an edge case.

**Structural.** The fan-out uses `Send`, and a `Send` worker's state contains
exactly the keys in its payload. Parent state is unreachable — not undisclosed,
unreachable. A future edit that tries to pass the taxonomy into a naming shard
would have to change the payload, visibly.

```python
return [Send("p7_name_shard", {"shard_id": i, "leaf_ids": ids}) for i, ids in enumerate(shards)]
```

`tests/test_principles.py::test_send_payload_is_the_workers_entire_state` asserts
this against LangGraph itself, so a regression in the library surfaces as a test
failure rather than as a quietly anchored tree.

**Structural, at the payload level.** `BlindnessFirewall.assert_card_blind()`
enforces a field whitelist: a card may carry `leaf_id`, `size`, `share`, the
three sample lists, `top_ngrams`, and `length_stats`. Nothing else. A smuggled
`legacy_label` or `taxonomy_hint` field cannot pass, whatever it contains.

**Lexical, on the non-corpus fields.** Anything not derived from the corpus is
scanned against the forbidden vocabulary — taxonomy names and codes, every
legacy label value, every peer agent's output — and a hit raises.

The split between those last two is not a detail; it is the difference between a
firewall that works and one that gets switched off. The first version scanned
*everything* lexically, and on the real corpus it dropped ten clusters from the
naming pass because the legacy label `作文` appears inside the genuine query
`我的自画像作文350字`. Good category names come from their domain's own
vocabulary, so label strings and ordinary query words overlap heavily — and a
member query cannot anchor a namer anyway, because it is the thing being judged.
What anchors is label vocabulary arriving as *annotation*, and that is exactly
what the whitelist blocks.

Together these make one specific finding credible: when a namer that was never
told about gambling flags the gambling cluster anyway, it is evidence. Without
the firewall it would be a claim.

---

## How metric authority is enforced

```python
METRIC_AUTHORITY = {
    "stability_ari": "decisive",
    "template_fragmentation": "decisive",
    "silhouette": "advisory",        # reported, never given a vote
    "distill_accuracy": "diagnostic",
}
```

`UniformPanel.decisive_ranking()` raises `ValueError` when handed an advisory
metric. Silhouette is not merely discouraged from selecting a representation —
it is structurally incapable of it.

The reason is specific. Silhouette rewards tight, well-separated clusters, and
queries sharing a phrasing template form the tightest possible cluster. Optimise
it and you systematically select representations that split one intent into
several phrasing-shaped families — precisely the failure the phase exists to
prevent. Every sweep records what silhouette *would* have chosen, because the
disagreement is itself evidence.

---

## How governance is enforced

A `Prescription` is a state machine: `proposed → executed` (with an evidence
pointer naming the artifact and column that changed, plus the metric deltas) or
`proposed → declined` (with a stated reason). `assert_all_settled()` raises on
anything still `proposed`, and it runs *before* the reports are written — so a
report claiming a fix the data does not contain cannot ship.

Merges are a leaf→family lookup-table remap. Leaf assignments and centroids do
not move, no re-clustering happens, and the pre-merge column is retained in the
delivered table, so every merge is reversible and auditable.

---

## Memory: three tiers

| tier | mechanism | lifetime | what breaks without it |
|---|---|---|---|
| working | `SqliteSaver` checkpoint | one run | a crash costs the whole run |
| artifact | `ArtifactStore` + `ArtifactRef` | forever | state balloons, context dies |
| long-term | `SqliteStore` namespaces | across runs | the team re-learns lessons |

Long-term memory is namespaced by *kind of knowledge*:

- `decisions` — semantic. What we chose, why, what we rejected. The report's
  failure-history section is a projection of these.
- `lessons` — episodic. Situation → action → outcome → lesson, written when a
  gate fails or a reviewer vetoes, retrieved by similarity into later prompts.
  This is TradingAgents' reflection loop pointed at methodology instead of P&L.
- `rules` — procedural. Adjudication rules that *grow* when the referee finds a
  case the guide does not cover. The taxonomy behaves differently as this fills,
  which is what makes it procedural rather than merely stored.
- `glossary`, `domain_priors`, `rejections`.

---

## Determinism, and its honest limit

Reproducible: seeds (0 for numbers, 42 for pictures, (0,1) for replay),
content-addressed artifact caching, prompt files hashed into the manifest,
append-only generations.

Not reproducible in the strict sense: **LLM calls**. `temperature=0` is a hard
400 on the current frontier models — the parameter was removed. So determinism
for judgment steps is **replay, not regeneration**: responses are cached by
`sha256(role, model, system, user, schema)`, and a re-run replays the identical
judgments rather than sampling fresh ones. The manifest records this distinction
rather than glossing it.

Role is part of the cache key on purpose. The cache exists so the *same* agent
replays itself, not so two agents sharing a prompt collide — two annotators
sharing a cache entry would agree perfectly by construction.

---

## Offline mode

`OfflineHeuristicModel` replaces every agent with a deterministic function that
*actually computes something*: cluster names come from the card's own top
n-grams, labels from regex evidence. It is not a language model and does not
pretend to be one — every record it produces is stamped `offline-heuristic`, and
`ModelRegistry.provenance_note()` puts a paragraph saying so at the top of every
report.

This matters more than it sounds. It means the entire twelve-phase graph — every
node, gate, artifact write, and report — is exercised in CI with no key and no
network, so a wiring bug surfaces in a two-minute test rather than in a
two-hour run. The stock LangChain fakes cannot do this:
`GenericFakeChatModel.with_structured_output()` raises `NotImplementedError`, and
both Phase 2 labelling and Phase 7 naming depend on structured output.

---

## Phase order: one deviation from the playbook diagram

The playbook draws the two routes as parallel branches. They are conceptually
independent, but Phase 2c's feature recipe concatenates the dense embedding
chosen by Phase 3a — so the top-down classifier genuinely cannot be built before
the bottom-up route has picked an encoder. The implemented order is therefore:

```
p0 → p1 → p2a → p2b → p3 → p2c → p2d → p4 → … → p12
                       ↑     ↑
              encoder chosen │ classifier consumes it
```

Taxonomy design and gold annotation need no embedding and stay first.
