# Multi-provider model routing

The goal: the user supplies whatever API keys they have, and the system picks a
suitable model for each of the fourteen agent roles — balancing capability
against cost — without anyone hand-editing a table of model IDs every few weeks.

---

## Why not a table of model names

The obvious implementation is a config file listing providers and their models.
It is also what makes most multi-provider code quietly obsolete: the table is
correct on the day it is written and wrong within a month.

The research gathered for this feature demonstrates the problem rather than
solving it. The model-landscape dossier in `docs/research/model-landscape.md`
opens by flagging that **its own price tables are secondary-source and were not
confirmed against any vendor pricing page**, and it records a direct
contradiction between two sources on OpenAI's mid-tier pricing. That is not a
failing of the research; it is the nature of a market that reprices weekly.

So no price or model ID from that document is embedded in the running system.

**The adversarial pass makes the point sharper.** A seventh agent re-checked the
six dossiers against the live catalogues and found them contradicting each other
on prices that were only hours old:

- Claude Sonnet 5 — one dossier said $3/$15, the live feeds said **$2/$10**. That
  dossier's entire cost model was inflated ~50%.
- GPT-5.6 Luna — one dossier said $1.00/$6.00, the live feeds said **$0.20/$1.20**.
  A 5× error, and it caused that dossier to dismiss on price a model with
  confirmed structured output and a 1.05M context — a genuine annotator candidate.
- DeepSeek — one dossier reported two model IDs *retired* and V4-Pro pricing
  *quadrupled*, and escalated it to "audit your config now". The live catalogue,
  at a commit dated after both claimed events, showed neither had happened.

Every one of those corrections is something the running system gets right for
free, because it reads the same feeds the critic used to adjudicate. That is the
argument for this design in one paragraph: research goes stale between being
written and being read, and the only durable answer is to not depend on it.

## Where the facts come from instead

Two public feeds, both maintained by other people, both verified live and
neither requiring an API key to read:

| source | what it is | why it is here |
|---|---|---|
| **LiteLLM** `model_prices_and_context_window.json` | 3,039 models, 124 providers, 1.7 MB | Primary. The only one covering **embedding** models (124), carries `deprecation_date` on 334 entries, serves an ETag for cheap conditional refresh. |
| **OpenRouter** `/api/v1/models` | 413 models, all priced | Secondary. The only free source of a per-model `structured_outputs` flag (335 of 413) and live provider uptime. |

Normalised into one `ModelCard` shape: **1,917 callable chat models**, 994 with
structured output confirmed.

Refresh degrades in order — fresh disk cache → network → *stale* disk cache →
pinned snapshot → a built-in floor — and whichever rung it lands on is recorded.
The pipeline must run air-gapped, so "no catalogue" is a degraded mode, never an
error.

**Reproducibility.** Every run records the catalogue's source, fetch time, and a
content hash, so a routing decision can be explained months later even though
live prices have moved. `catalog_pinned` replays an old snapshot exactly.

---

## How a model is chosen

### Roles declare requirements, not models

The thirteen-plus roles differ along two axes that a `deep`/`fast` switch
conflates: **how hard the judgment is**, and **how much of it there is**.

| role | tier needed | blast radius | calls | cost weight |
|---|---|---|---|---|
| `annotator_a` / `annotator_b` | standard | contained | 240 each | 0.85 |
| `namer` | standard | phase | 60 | 0.55 |
| `referee` | strong | phase | 30 | 0.25 |
| `taxonomy_architect` | frontier | **run** | 1 | 0.05 |
| `tree_auditor` | frontier | **run** | 1 | 0.05 |

An annotator makes an easy judgment 240 times; the architect makes one very hard
judgment whose error every later artifact inherits. Requirements are properties
of the task, so they stay stable while the model world churns.

### Capability is estimated from price — and the estimate knows its own failure mode

There is no free, current, cross-provider benchmark. Price is the available
proxy, and within a generation it tracks capability closely because labs price
against each other. Models are tiered by price **percentile across the whole
catalogue** (not the reachable subset — percentiles over four models are
meaningless).

The proxy's known failure is a genuinely strong *and* cheap model. This is not
hypothetical: DeepSeek at **$0.28/$0.42 per Mtok with 131k context and confirmed
structured output** lands in a low price band, and a naive tier gate would
exclude it from exactly the high-volume roles it suits best.

So the tier is a **hard gate only where an error is unrecoverable** — the
`run`-blast-radius roles, where we pay for certainty — and a **scoring term
everywhere else**, letting a cheap capable model win on the roles that dominate
the bill. Capability credit is also capped at what the role needs, so nobody
pays frontier prices for headroom the task cannot use.

### Where this breaks, and the fix: price is not capability

The tier is a **price percentile over the reachable set**. That is a proxy, and
it fails exactly where this project runs. After excluding the Western labs,
**not one Chinese model rates `frontier`** — `deepseek-v4-pro`, `glm-5.2`,
`qwen3.8-max` and `kimi-k3` all land in a single `strong` band. Two consequences,
both observed:

- **Asking for a higher tier does nothing.** `frontier` finds nothing reachable
  and silently relaxes back to `strong`.
- **Within a tier the cheapest always wins**, because capability credit is capped
  at the requirement so every candidate at or above the bar ties on capability.
  The referee — whose verdicts *become* the gold set — was handed an "air"
  lightweight over `glm-5.2` on **$0.30/M of input**. Measured on the same
  annotator pair: the strong referee chose the stronger annotator on 78.3% of
  contested rows (z=+12.1); the lightweight one on 55.1% (z=+2.2), which is near
  chance and is what an adjudicator that cannot discriminate looks like.

Removing price from the ranking was tried and reverted **three times** — price is
also what keeps free tiers, previews and meta-endpoints out of the roles that
matter. So price stays, and **capability is supplied explicitly**:

`capable_models` in the config is a curated list of bare model ids, each entry a
human judgement from an independent evaluation or from a run that exercised it.
It gates the candidate pool for `run`/`phase` blast-radius roles; price breaks
ties *inside* it and still governs the high-volume `contained` roles. It is
gated on **hard** constraints only — checking the derived tier would let a
`frontier` role find nothing and fall back to the price-decided choice the list
exists to replace — and it falls back **loudly** when nothing on it is eligible.

Pin a role on top of that only for something the list cannot express: a verified
quirk (`researcher` needs tools, and `deepseek-v4-pro` rejects the `tool_choice`
LangChain sets when a response format is bound) or a lab-independence constraint
(`referee`). A pin **bypasses `_eligible` entirely**, including its exclusion of
unpriced models — so a pinned model with no published rate estimates as $0.00 and
under-reports spend, and the router now says so rather than showing a confident
zero.

`blast_radius` is the switch that decides whether a role is capability-gated, so
it must describe **what a wrong answer costs**, not how many calls the role
makes. Two were wrong: `referee` (its verdicts *are* the gold set) and
`adversary` (its output is the accuracy estimate quoted for the whole taxonomy,
and its own rationale already said "a cheap adversary produces a flattering
number").

### Hard filters, all learned from failures observed while building this

- **no structured output** → excluded (every role in this pipeline needs it)
- **context below the role's floor** → excluded
- **`deprecation_date` in the past** → excluded
- **emits images** → excluded. Image models price and tier like frontier text
  models while being tuned for another job; before this filter existed the
  router put a `gpt-5.4-image` variant on the tree auditor.
- **fine-tune price templates (`ft:…`)** → excluded at catalogue level. They are
  price rows for a model you would have to train first, they are cheap, and a
  cost-aware router therefore picks them for everything — which it did.

### Fallbacks span providers

Three models from one provider is one outage, not a fallback chain.
Alternatives are drawn from different providers wherever the catalogue allows.

### The two annotators are routed apart

Cohen's κ between two annotators is only evidence about the *labelling guide* if
their errors are independent. Give both the same model and shared architecture
makes them agree on the same mistakes — inflating κ and hiding the guide
ambiguity the gold set exists to expose. So `annotator_b` is routed to a
different provider from `annotator_a`, dropping down the ranking to get it. With
only one provider configured it shares, and says so in a warning rather than
accepting it silently.

---

## Using it

```bash
export ANTHROPIC_API_KEY=...      # any subset
export DEEPSEEK_API_KEY=...
export OPENROUTER_API_KEY=...

qmine models --prefer-chinese-native --budget 5     # inspect before spending
qmine run -i queries.csv -d k12_zh --provider router
```

`qmine models` shows what is reachable, what each role would use, the fallback
chain, and the estimated total for a full run. A real plan across three keys:

```
annotator_a          deepseek:deepseek-v4-pro          standard  240 calls   $0.94
annotator_b          openrouter:qwen3-next-80b         standard  240 calls   $0.94
referee              openrouter:gemini-3.5-flash       strong     30 calls   $0.54
taxonomy_architect   anthropic:claude-opus-4-5         frontier    1 call    $0.23
tree_auditor         anthropic:claude-opus-4-5         frontier    1 call    $0.23
                                              estimated total: $3.76 per run
```

Eighteen providers are known, including the Chinese-native labs (DeepSeek, Qwen,
Moonshot, Zhipu, MiniMax) that are often strongest per dollar on Chinese text —
`--prefer-chinese-native` nudges multilingual-critical roles toward them, as a
scoring bonus the measured capabilities can still override.

`model_overrides` wins outright over the router. It is a default, not an
authority: a user who knows a specific model is right for a role should not have
to argue with a heuristic.

## What this does not do

- **It does not benchmark models.** Tiering is a price proxy with stated failure
  modes, not a quality measurement. If you have real evaluation data for your
  corpus, `model_overrides` is how you use it.
- **It does not verify that a key works** unless you ask. `detect()` reads
  environment variables; `probe()` makes the network call. The plan records
  which claim it relied on.
- **It cannot rescue an empty catalogue.** With no network, no cache and no
  pinned snapshot, routing falls back to the statically configured deep/fast
  models and says so.
