# QMine — a query-intent mining agent team

A production-grade implementation of the *Universal Query Mining & Clustering
Playbook*: twelve phases that take a raw query log and return an intent
taxonomy, a cluster tree, a fully labelled table, a deployable classifier, and a
report that shows its work — including the parts that failed.

Built on LangGraph. Runs on a laptop. Runs offline.

```bash
pip install -e ".[all]"
qmine doctor
qmine demo                     # bundled 50k K12 corpus, end to end
```

---

## What it produces

One command over a CSV of queries yields, in `runs/<run-id>/gen01/`:

| deliverable | what it is |
|---|---|
| `Report_BottomUp_Approach.md` | representation bake-off, tree, governance, deployment |
| `Report_TopDown_Approach.md` | taxonomy, gold set, classifier, adversarial validation |
| `Report_Uniform_Panel.md` | every candidate compared under one measurement harness |
| `Report_Naming_Panel_Comparison.md` | when an external panel names the tree: what changed and why |
| `Leaf_Catalogue.md` | every cluster with a checkable `user_need` sentence |
| `labels_full.csv` | both label systems side by side (`td_l1`, `td_l2`, `bu_family_final`, `bu_leaf`) plus confidence margins |
| `Walkthrough.ipynb` | executed — every number computed in-cell, nothing pasted |
| `centroid_classifier.joblib` | the deployable model, a few hundred KB |
| 6 figures | K-sweep, α decision, template spread, panel, refinement, UMAP |

Plus the evidence: `run_manifest.json` (seeds, versions, prompt hashes),
`governance.json` (every audit finding and what it changed), `battery.json`,
`granularity.json`, `metrics_panel.json`.

---

## Verified on two domains

| | K12 Chinese (50k real queries) | E-commerce English (7.2k, synthetic) |
|---|---|---|
| phases completed | 18/18 nodes | 17/17 nodes |
| held-out reproduction | 0.991 (95% CI 0.989-0.992) | 0.991 (95% CI 0.985-0.995) |
| blind coherence | 4.27/5 | 4.02/5 |
| governance | 4 executed, 1 declined | 15 executed |
| notebook | executed, 0 errors | executed, 0 errors |

The English corpus is synthetic and labelled as such — it exists to exercise the
paths that differ by language (whitespace tokenisation, char 3-5 grams,
prefix-style templates, Latin-script profiling, no CJK fonts), which is the part
of portability that code can get wrong. The K12 corpus is real.

## The twelve phases

```
P0  foundation ── seeds, manifest, environment pinned
P1  audit ─────── corpus profile, phrasing families, risk pre-screen
     │
     ├── P2a taxonomy ── 5 researchers (disjoint angles) → architect → critic
     ├── P2b gold ────── 2 blind annotators → κ → referee drafts missing rules
     │
P3  representation ── encoder bake-off · char TF-IDF+SVD · α sweep
     │
     ├── P2c classifier ── rules + linear head on dense ⊕ sparse ⊕ flags
     ├── P2d validation ── agents instructed to *disprove* the labels
     │
P4  battery ────── 6 algorithms, one identical harness
P5  granularity ── stability peak × over-clustering survival × domain prior
P6  hierarchy ──── stable families, locally chosen leaves, refine to convergence
P7  naming ─────── 5 blind agents in parallel + auditor + independent risk sweep
P8  governance ─── every prescription executed against the data, or declined
P9  panel ──────── everything re-measured by one code path
P10 deployment ─── centroid classifier, margin routing, delivered table
P11 reports ────── markdown + executed notebook + deterministic exemplars
P12 maintenance ── drift baseline, novelty sentinel, rerun contract
```

---

## What makes it different from a pipeline that does the same steps

Four of the playbook's principles are the kind that erode silently. Here they are
enforced by code rather than by discipline.

**Silhouette cannot decide anything.** Every metric carries an `authority`, and
`decisive_ranking()` raises when handed an advisory one.

```python
>>> panel.decisive_ranking("silhouette")
ValueError: 'silhouette' has authority 'advisory' and cannot decide anything.
```

The reason is specific: silhouette rewards tight clusters, and identically-phrased
queries form the tightest cluster there is. Optimise it and you reliably select a
representation that splits one intent into several phrasing-shaped families —
exactly the failure the α sweep exists to prevent. Every sweep records what
silhouette *would* have chosen, because the disagreement is itself evidence.

**Blind naming is structural, not promised.** The five naming agents fan out via
`Send`, so a worker's state contains only its own payload — parent state is
unreachable. On top of that, a firewall enforces the card contract: a namer sees
member queries, n-grams, and size, and *no other field*. A smuggled
`legacy_label` cannot pass, and any label vocabulary in a non-corpus field
raises.

Notably, member queries themselves are exempt from the lexical scan — the first
version was not, and on the real corpus it dropped ten clusters because the
legacy label `作文` appears inside the genuine query `我的自画像作文350字`. A
member query cannot anchor a namer; it is the thing being judged. That is what
makes one finding credible: when an agent that was never told about gambling
flags the gambling cluster anyway, it is evidence rather than bookkeeping.

**Governance reaches the data or the run fails.** An audit prescription is a
state machine — `proposed → executed` (with the artifact and column it changed,
and the metric deltas it caused) or `→ declined` (with a reason). Anything left
`proposed` raises, *before* the reports are written. A report cannot claim a fix
the CSV does not contain.

**Rejected options are kept.** Every decision records what it beat and why, and
the reports' failure-history sections are projections of those records rather
than recollections. Generations are append-only, so a rejected tree stays on
disk — in the source project, a discarded 107-leaf tree later became the
phrasing-pattern library.

---

## Running it

```bash
qmine run \
  --input queries.csv \
  --domain k12_zh \
  --text-column query \
  --reference-columns legacy_l1,legacy_l2
```

| flag | effect |
|---|---|
| `--sample N` | first pass on an unfamiliar corpus |
| `--fast` | shrink every grid; wiring check in minutes |
| `--offline` | no network, no key — see below |
| `--human-review` | pause for sign-off after taxonomy, tree, and panel |

Everything checkpoints after every node. `qmine resume <run-id>` picks up where a
crash left off; `qmine inspect <run-id> --what panel` reads a finished run
without recomputing anything.

### Command reference

| command | what it does |
|---|---|
| `qmine run -i data.csv -d <profile>` | the full twelve-phase pipeline |
| `qmine run … --resume` | continue an interrupted run from its last checkpoint |
| `qmine inspect <run> --what panel\|gates\|leaves\|governance` | read a finished run without recomputing |
| `qmine export-cards <run>` / `qmine import-namings <run> f.json` | run blind naming with an external panel |
| `qmine promote --old A.csv --new B.csv` | referee protocol: let a challenger label set earn its place |
| `qmine diff <run-a> <run-b>` | drift vs method change between two quarters |
| `qmine doctor` | environment, credentials, models, fonts |

### Handing Phase 7 to a stronger reviewer

Naming is the one step where a better judge is worth the cost — a mis-named
family propagates into the catalogue, the report, and everything downstream. The
protocol is therefore exportable, and the blindness guarantee travels with it:

```bash
qmine export-cards <run-id> --shards 5    # firewall-checked briefs, one per reviewer
# hand the shard_*.md files to any agent panel, or to real humans
qmine import-namings <run-id> verdicts.json --named-by "your panel"
```

On the bundled K12 corpus this was run with five independent Claude agents plus a
tree auditor. They recovered the phrasing-driven split of three pinyin clusters
from the `user_need` sentences alone, explicitly forbade merging the lottery-probe
riddle cluster into the idiom cluster on topical similarity, and surfaced four
risk categories the domain profile had never heard of — which were then written
back into the profile. See `Report_Naming_Panel_Comparison.md` in the run.

### Long runs on a laptop

A full 50k run takes tens of minutes, and laptops sleep. Two things make that
survivable:

```bash
caffeinate -i qmine run -i queries.csv --run-id q3 ...   # no idle sleep
qmine run -i queries.csv --run-id q3 --resume            # continue after any interruption
```

`caffeinate -i` blocks *idle* sleep; closing the lid on battery still suspends
the machine, and losing the network still kills a foreground process. That is
what the checkpoints are for — `--resume` restarts at the node that was running,
restores the run's own resolved config, and replays nothing that already
finished. Encoding and clustering results additionally come back from the
content-addressed cache, so a resumed run skips the expensive parts even when
the node itself re-executes.

### Promoting a new model's labels

A model change is not evidence of improvement, so replacing labels requires
winning an argument:

```bash
qmine promote --old runs/q1/gen01/labels_full.csv --new runs/q2/gen01/labels_full.csv
```

It judges only the rows where the two systems **disagree** — agreements carry no
information about which is better — presents each to a referee blind and with
randomised side order (LLM judges have documented position bias), and promotes
only on a statistically significant win. The old labels move to a `_v1` column
either way, and every overturned row is stamped `label_source='referee'`.

### Domains

Five profiles ship — `k12_zh`, `finance_zh`, `sports_zh`, `politics_zh`,
`ecommerce_en` — each carrying the settings the playbook says must be re-derived
per vertical: phrasing seeds, risk categories, tokenizer, n-gram ranges, encoder
candidates. **α is not among them.** It is re-derived by sweep on every run,
because the K12 value is a fact about K12's phrasing ecology rather than a
constant.

### Offline mode

With no `ANTHROPIC_API_KEY`, the provider resolves to `offline` and every agent
is replaced by a deterministic heuristic that actually computes something —
cluster names from the card's own top n-grams, labels from regex evidence.

The clustering, embeddings, and metrics are **entirely real**. The judgments are
not model judgments, every record they produce is stamped `offline-heuristic`,
and a paragraph saying so appears at the top of every report. This is what lets
the whole twelve-phase graph run in CI with no credentials, so wiring bugs
surface in a two-minute test rather than a two-hour run.

---

## Layout

```
src/qmine/
  determinism.py     seeds, content hashing, the anti-cherry-pick exemplar rule
  artifacts.py       generation store + content-addressed memo cache
  records.py         typed records; METRIC_AUTHORITY lives here
  state.py           graph state (pointers only) and its reducers
  config.py          invariants vs the per-domain pile
  llm/               two-tier routing, response cache, offline stand-in, budget
  memory/            three-tier memory; the blindness firewall
  ops/               the science — audit, templates, represent, cluster, panel,
                     governance, classify, cards, promotion, viz
  agents/            11 roles + versioned prompt files
  graph/             nodes per phase, gates, human review, assembly
  report/            markdown builders + programmatic notebook
configs/domains/     5 vertical profiles
skills/              5 Claude Code Agent Skills
docs/                architecture, playbook mapping, 9 research dossiers
tests/               64 tests — principles, ops, durability, end-to-end
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — the three planes and why each guarantee holds
- [Playbook → code](docs/PLAYBOOK_MAPPING.md) — every principle, phase and trap, mapped
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md) — the plan this was built from
- [Research dossiers](docs/research/) — the nine-agent study that preceded the code
- [Skills](skills/) — driving QMine from Claude Code

## Tests

```bash
pytest tests/ -q          # 64 tests, fully offline, ~5 minutes
```

`tests/test_principles.py` is the set worth reading first: each test names the
playbook principle it guards. A clustering bug produces worse numbers; a failure
there produces numbers that look fine and are not trustworthy.

`tests/test_durability.py` covers the defects that only appeared on the real
corpus — chiefly state that lived in a process's memory rather than in an
artifact, which looked correct in a single run and silently degraded across a
resume.
