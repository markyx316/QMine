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

Deliverables are written in the corpus's own language. On a Chinese corpus:

| deliverable | what it is |
|---|---|
| `00_最终报告.md` | **the one document not assembled by Python** — an agent writes the outline and every sentence over the same artifacts; the numbers are checked value-by-value against a per-section fact sheet and the run's own warnings become a must-cover list checked over the whole text |
| `自下而上聚类最终报告.md` | representation bake-off, tree, governance, deployment — including what was rejected |
| `自上而下类目体系最终报告.md` | taxonomy, gold set, classifier, adversarial validation, and whether the guide's rules agree with the referee's own verdicts |
| `统一度量面板.md` | every candidate re-measured by one code path, plus the open-findings ledger |
| `叶清单.md` | every delivered cluster with a checkable `user_need` sentence; unnamed leaves listed as defects |
| `交付前审核报告.md` | what the pre-delivery auditor changed, **what it was refused**, and what it read and dismissed |
| `labels_full.csv` | both label systems side by side (`td_l1`, `td_l2`, `bu_family_final`, `bu_leaf`) plus confidence margins |
| `自下而上聚类全流程.ipynb` | executed — every number computed in-cell, nothing pasted |
| `centroid_classifier.joblib` | the deployable model, a few hundred KB |
| figures | K-sweep, α decision, template spread, panel, refinement, gates, decision chain |

Plus the evidence: `run_manifest.json` (seeds, versions, per-role model and
spend), `governance.json` (every audit finding and what it changed),
`battery.json`, `granularity.json`, `metrics_panel.json`,
`delivery_audit.json` — and, one level up at the **run** root,
`findings.json`: the open-findings ledger, which a new generation inherits and
which only closes an entry when that entry's own assertion holds again.

---

## Verified on two domains

| | K12 Chinese (50k real queries, `live40`) | E-commerce English (7.2k, synthetic) |
|---|---|---|
| phases completed | 17/17 | 17/17 |
| wall clock | 241.8 min, 696 agent calls, `provider=routed` | offline |
| held-out reproduction | 0.989 (95% CI 0.987-0.991, n=10,000) | 0.991 (95% CI 0.985-0.995) |
| classifier | CV 0.863, macro-F1 0.797, ECE 0.037 | — |
| gates | 24 recorded — 20 passed, 4 warned, 0 failed | — |
| mechanical verification | **25/26 checks pass** (`tools/verify_run.py`) | — |
| notebook | executed, 0 errors | executed, 0 errors |

The English corpus is synthetic and labelled as such — it exists to exercise the
paths that differ by language (whitespace tokenisation, char 3-5 grams,
prefix-style templates, Latin-script profiling, no CJK fonts), which is the part
of portability that code can get wrong. The K12 corpus is real.

## The twelve phases

The two routes run **concurrently**. P3 consumes only P1's output, so nothing in
the bottom-up branch waits on the taxonomy or the gold set; they rejoin at P2c,
which needs the gold set *and* the encoder the α sweep picked.

```
P0  foundation ── seeds, manifest, environment pinned
P1  audit ─────── corpus profile, phrasing families, risk pre-screen
     │
     ├─────────────── FORK ───────────────┐
     ▼                                    ▼
 P2a taxonomy                         P3  representation
   5 researchers → architect → critic    encoder bake-off · char TF-IDF+SVD · α sweep
   → pilot → redraw-until-stable            │
     ▼                                    ▼
 P2b gold                             P4  battery ─ is the structure an artefact
   2 blind annotators (different labs)     of KMeans? a falsification probe
   → κ → referee (a third lab)          P5  granularity ─ K located by intent
   → rules → guide repair                   alignment; stability only VETOES
     │                                  P6  hierarchy ─ refine to convergence
     └─────────────── JOIN ───────────────┘
                      ▼
P2c classifier ── rules + linear head on dense ⊕ sparse ⊕ flags
P2d validation ── agents instructed to *disprove* the labels
P2e sub-intents ─ L2 where the embedding can see it
P7  naming ─────── 5 blind agents via Send + auditor + independent risk sweep
P8  governance ─── every prescription executed against the data, or declined
P9  panel ──────── everything re-measured by one code path
P10 deployment ─── centroid classifier, margin routing, delivered table
P11 reports ────── markdown + executed notebook + pre-delivery audit
P12 maintenance ── drift baseline, novelty sentinel, rerun contract
```

Measured on `live40`: 39 minutes of bottom-up compute disappear entirely into the
107 minutes of taxonomy design and gold annotation in front of it. `qmine run
--config` with `concurrent_branches: false` restores the strict chain.

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

## Agents describe; measured quantities decide

Agents now touch nearly every phase, which forces the question: *on what basis
may anything an agent says reach the deliverable?* The answer is not "trust it"
but **give it a way to prove itself** — an agent may supply the measurement that
settles its own claim, and only the measurement carries authority.

Four doors, each with a mechanical guardrail. **None of them can change a
parameter.**

| door | guardrail | on failure |
|---|---|---|
| **prose** | the author gets a fact sheet; every number it writes must be in it | rejected and re-asked, with the offending values quoted |
| **narrative** | the same fact-sheet check per section, **plus** a coverage check: the run's warned gates, ties and open findings become a must-cover list verified against the agent's own prose | a section that cannot pass ships as a marked hole; an uncovered point is disclosed in the document |
| **observation** | must cite a *resolving* artifact path; may carry a machine-evaluable assertion | an unresolvable citation is dropped before anyone reads it |
| **grid proposal** | proposed *blind to every score*, so additions are pre-registered; capped, additions-only, graded each run | a score-shaped token in the payload aborts the call |
| **deliverable edit** | anchored replacement: anchor unique, every number sourced from the artifact the edit **cites**, language checked, reason required | refused — and refusals are printed beside the applied edits |

An observation's assertion is three-valued, and the asymmetry is the point:

- **confirmed** (the assertion fails) — now a *measurement*, and the only kind
  that may fail a gate
- **refuted** (it holds) — dropped; the agent's own false-positive filter
- **unverifiable** — advisory, as before

> **Confirmed is not the same as defective, and the report says so.** Measured on
> `live40`: of 13 machine-confirmed findings, independent re-verification found
> **2 real defects**. Eight were arithmetically correct and wrong anyway — almost
> all because the two compared fields measured *different populations*. A check
> proves an assertion failed; which fields to compare and what a difference means
> are still unguarded judgement.

**A finding nobody acts on must at least be unable to disappear.** The ledger
lives at the run root beside the LLM cache, so a new generation inherits it, and
an entry closes only when its own assertion holds again. This exists because the
opposite already happened: a critic agent identified a κ defect *before* the run
that shipped it, wrote the finding to an artifact, and nothing read it.

**One agent may write.** The pre-delivery auditor reads every gate, the ledger,
the artifacts and the finished documents together, and edits the reports. It is
bounded to `.md` files — a report *describes* a measurement; an artifact *is*
one — and every edit it makes, and every one it is refused, appears in
`交付前审核报告.md`.

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

### Deliverables are written in Chinese

`report_language` defaults to `zh`. The bottom-up report and the walkthrough
notebook follow the structure of the reference K12 deliverables: an executive
summary whose table is the argument, every metric **derived on screen** rather
than quoted (`① 模板群 → ② 家族分布 p → ③ H = −Σp·ln p → ④ exp(H)`), the full
family→leaf tree, real queries traced through it, `user_need` definition
sentences with ★ coherence ratings, and a mandatory 「这些数字不代表什么」 section.

A `user_need` sentence is simultaneously the annotation guideline, the acceptance
criterion and the downstream product spec — it can only do those jobs in the
language the team that owns the corpus works in. Set `report_language: en` to
switch.

### Agents can research the web

The Phase 2a literature and risk-compliance researchers get `web_search` and
`fetch_url` tools and run a real tool loop. Without them, "ground this taxonomy
in published work" produces citations the agent cannot have checked.

DuckDuckGo is the default and needs **no key**; `TAVILY_API_KEY` or
`BRAVE_API_KEY` are used when present. The log-reading and pragmatic-intent
angles deliberately get *no* tools — their value is direct observation of the
rows, and a search box invites recall to replace it.

### Live dashboard

`qmine run` shows a Rich panel: phase list with elapsed times, current agent
activity, metrics as they land, gates as they fire, and running spend. Falls back
to plain lines with `--plain`, `--quiet`, or no TTY.

### Multi-provider model routing

Supply whatever API keys you have; the system picks a model per agent role from a
**live catalogue** (1,917 callable models across 18 providers, refreshed from
LiteLLM and OpenRouter, no key needed to read) and balances capability against
cost. Nothing is hardcoded, so it does not go stale.

```bash
# keys go in QMine/.env (see .env.example) or the environment
# DEEPSEEK_API_KEY=...  ZHIPU_API_KEY=...  QWEN_API_KEY=...
qmine models --prefer-chinese-native --budget 5    # inspect before spending
qmine run -i queries.csv -d k12_zh --provider router
```

High-volume roles get cheap-but-capable models, fallbacks span **labs**, and the
two gold annotators and the referee are routed to three *different labs* — not
merely different providers, since two labs reach you through one gateway and look
identical in a provider column. κ is supposed to measure the labelling guide; if
the annotators share an architecture it measures a shared prior instead, and if
the referee shares one with an annotator it sides with that annotator in a
direction nobody would think to check.

**The router cannot judge capability, and says so.** Tier is derived from a
**price percentile**, which works only while price tracks capability. It does not
across the Chinese labs: after excluding the Western ones, *not one* model rates
`frontier`, so asking for that tier silently relaxes and buys nothing — and
within a tier the cheapest always wins. That handed the referee a lightweight
model whose adjudication was measurably near chance.

So capability is **stated, not inferred**. `capable_models` in the config is a
human judgement — a curated list of ids — and it gates the candidate pool for
roles whose errors are expensive; price only breaks ties *inside* it, and still
governs the high-volume contained roles where cheap-and-adequate is right. The
plan is printed **before the first call**, with each role's model, its lab, its
estimated calls and spend, and every warning the router attached.

Full design and its known limits: [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md).

### Mixed languages and unknown domains

A minority language is the dangerous case: measured on a real corpus, at 2%
English **97% of all English queries collapse into one cluster** — and swapping
in a multilingual encoder does not fix it. Phase 1 measures the script mix and
warns; Phase 6 resolves minority intents in a script-appropriate space and ships
them as a column rather than faking tree leaves the deployed classifier cannot
represent.

For an unknown vertical, `--domain generic` carries universal risk categories and
zero template seeds — the phrasing families are mined from the corpus and must
each earn trust by being measurably tighter than random. On the K12 corpus with
all seeds removed, that rediscovered five of the six hand-written families and
correctly rejected "是什么", which attaches to every topic.
Details: [docs/LANGUAGE_AND_DOMAIN.md](docs/LANGUAGE_AND_DOMAIN.md).

### Command reference

| command | what it does |
|---|---|
| `qmine run -i data.csv -d <profile>` | the full twelve-phase pipeline |
| `qmine run … --resume` | continue an interrupted run from its last checkpoint |
| `qmine inspect <run> --what panel\|gates\|leaves\|governance` | read a finished run without recomputing |
| `qmine export-cards <run>` / `qmine import-namings <run> f.json` | run blind naming with an external panel |
| `qmine promote --old A.csv --new B.csv` | referee protocol: let a challenger label set earn its place |
| `qmine diff <run-a> <run-b>` | drift vs method change between two quarters |
| `qmine models` | reachable providers, per-role model choice, estimated run cost — spends nothing |
| `qmine new-generation <run> --reason '…'` | re-run an id and reuse its paid work; the old generation stays as evidence |
| `qmine watch <run>` | attach the dashboard to a run, live or finished |
| `qmine run … --plain` | disable the live dashboard (CI, pipes) |
| `qmine doctor` | environment, credentials, models, fonts |
| `python tools/verify_run.py runs/ID/genNN [runs/OLD/genNN]` | 26 mechanical checks over a finished run; pass an older run as a **control** — a harness that passes on a known-broken run proves nothing |

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
                     governance, classify, cards, promotion, viz, and the
                     agent-authority mechanics: checks, findings, edits,
                     propose, select, rule_conflict, annotator_balance
  agents/            18 roles + versioned prompt files; observe/verify/interpret/
                     propose_grid/audit_delivery carry the authority contracts
  graph/             nodes per phase, gates, human review, assembly
  report/            markdown builders + programmatic notebook
configs/domains/     5 vertical profiles
skills/              5 Claude Code Agent Skills
docs/                architecture, playbook mapping, 9 research dossiers
tests/               508 tests — principles, ops, durability, concurrency,
                     agent authority, routing, end-to-end
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — the three planes and why each guarantee holds
- [Playbook → code](docs/PLAYBOOK_MAPPING.md) — every principle, phase and trap, mapped
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md) — the plan this was built from
- [Research dossiers](docs/research/) — the nine-agent study that preceded the code
- [Skills](skills/) — driving QMine from Claude Code

## Tests

```bash
pytest tests/ -q          # 508 tests, fully offline, ~3.5 minutes
```

`tests/test_principles.py` is the set worth reading first: each test names the
playbook principle it guards. A clustering bug produces worse numbers; a failure
there produces numbers that look fine and are not trustworthy.

`tests/test_durability.py` covers the defects that only appeared on the real
corpus — chiefly state that lived in a process's memory rather than in an
artifact, which looked correct in a single run and silently degraded across a
resume.
