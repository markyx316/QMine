# Capability/Cost Routing Strategies

> Gathered 2026-08-18. Facts marked verified were fetched live; the model-landscape
> dossier in particular flags its prices as secondary-source and unconfirmed —
> which is why the running system fetches prices from a live catalogue rather
> than embedding any table from this document.

> **All prices and model IDs below were verified by live web search on 2026-08-18.** Anthropic figures are cross-checked against the bundled `claude-api` skill catalog (cached 2026-06-24) *and* August-2026 pricing pages. `platform.claude.com`, `docs.litellm.ai`, and `arxiv.org` were **not fetchable** from this environment (domain-verification block), so third-party aggregators and search snippets are the sources where noted. Treat every price as "re-verify before you bill against it."
>
> **Runnable artifact:** `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/8075e8db-1d8b-4b54-b04d-7d995fbae90d/scratchpad/routing.py` (535 lines, stdlib only, `python3 routing.py` runs the full demo). The scratchpad is session-scoped — copy it out. Full source is reproduced in §8.

---

## 1. The headline result: routing is the wrong lever for this pipeline

I built the router, priced QMine's actual 13 roles at three scales, and measured it against QMine's current two-tier table. **Routing is worth less than the batch API, at every scale I tested.**

Mid-scale run (600 gold rows, 60 clusters, 461 LLM calls), all costs predicted by the cost model in §5:

| Strategy | Cost | vs. today |
|---|---:|---:|
| QMine today (`deep=claude-opus-5`, `fast=claude-sonnet-5`, no batch) | **$16.03** | — |
| QMine today **+ Batch API** (no routing at all) | **$11.40** | **−29%** |
| Full routing, Anthropic key only | $16.93 | **+6% (worse)** |
| Full routing, all 7 providers | $14.54 | −9% |
| **Hybrid: Opus 5 for low-volume roles, routed+batched fan-out** | **$12.96** | **−19%** |

Large run (5000 rows, 200 clusters, 1572 calls): today $39.64 → +batch $24.21 (−39%) → routed-7 $31.33 (−21%) → hybrid $29.76 (−25%).

Two things make routing weak here:

**(a) The cost is concentrated in fan-out, not in the expensive roles.** At 5000 rows/200 clusters, spend attribution under today's config:

```
annotator_a   $5.08  12.8%  (208 calls)     namer_1..5  $3.72 each,  47.0% combined (200 calls each)
annotator_b   $5.08  12.8%  (208 calls)     referee     $2.37   6.0%  (69 calls)
```

**78% of the money is in annotate + name + adjudicate.** The high-stakes roles everyone wants to optimize — architect, tree_auditor, reporter — are 1–3 calls each and cost cents. Routing them *down* saves nothing and risks the whole run; routing them *up* is nearly free.

**(b) The cost–quality frontier is flat.** Sweeping the willingness-to-pay parameter λ over a 3000× range moves total spend only 2.15×:

```
lambda      USD   architect / annotator_a / namer_1 / reporter
     1    24.71   fable-5 / fable-5 / fable-5 / fable-5
    30    19.81   fable-5 / opus-5  / qwen3.6-flash / fable-5
   300    13.35   opus-5  / opus-5  / qwen3.6-flash / opus-5
  3000    11.48   opus-5  / opus-5  / qwen3.6-flash / opus-5
```

The knee is at λ≈100–300. Below the knee you buy Fable 5 for everything; above it, nothing further is available to cut because capability floors bind. Compare the literature's headline numbers (RouteLLM: 85% cost reduction on MT-Bench; FrugalGPT: up to 98%) — those come from routing a *homogeneous stream of user queries* between a frontier model and a much weaker one. QMine has no such stream. It has 13 heterogeneous roles with hard capability floors, and the floors do the work that a router would otherwise do.

**So: the real value of routing here is not cost. It is correlated-error avoidance and reliability.** That reframes the deliverable, and §3 is the most important section.

## 2. Verified model catalog (2026-08-18)

| Model ID | Provider | In $/MTok | Out $/MTok | Context | Max out | Cache min | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `claude-fable-5` | Anthropic | 10.00 | 50.00 | 1M | 128K | 512 | No prefill; thinking always on; **requires 30-day retention** (ZDR orgs get 400 on every request) |
| `claude-opus-5` | Anthropic | 5.00 | 25.00 | 1M | 128K | **512** | Thinking **on by default**; `max_tokens` caps thinking+text together |
| `claude-opus-4-8` | Anthropic | 5.00 | 25.00 | 1M | 128K | 1024 | |
| `claude-sonnet-5` | Anthropic | 3.00 | 15.00 | 1M | 128K | 1024 | **Intro $2/$10 through 2026-08-31** — expires in 13 days |
| `claude-haiku-4-5` | Anthropic | 1.00 | 5.00 | 200K | 64K | **4096** | Still the latest Haiku as of Aug 2026; no Haiku 5 announced |
| `gpt-5.6-sol` | OpenAI | 5.00 | 30.00 | 1.05M | 128K | ~1024 | `gpt-5.6` aliases sol |
| `gpt-5.6-terra` | OpenAI | 2.00 | 12.00 | 1.05M | 128K | ~1024 | Some sources quote $2.50/$15 — **conflicting** |
| `gpt-5.6-luna` | OpenAI | 0.20 | 1.20 | 1.05M | 128K | ~1024 | |
| `gemini-3.1-pro` | Google | 2.00 | 12.00 | 1,048,576 | 65K | ~2048 | |
| `gemini-3.7-flash` | Google | 0.75 | 3.75 | 1,048,576 | 65K | ~2048 | Intro; **$1.50/$7.50 from 2027-01-01** |
| `deepseek-v4-pro` | DeepSeek | 0.435 | 0.87 | 256K | 32K | — | **Peak/off-peak change eff. 2026-08-17 — yesterday. Re-verify.** JSON mode, not native schema |
| `deepseek-v4-flash` | DeepSeek | 0.14 | 0.28 | 256K | 32K | — | Cheapest credible option |
| `qwen3.6-flash` | Alibaba | 0.19 | 1.13 | 1M | 32K | — | |
| `glm-5.2` | Zhipu | 1.40 | 4.40 | 200K | 32K | — | |
| `kimi-k3` | Moonshot | 3.00 | 15.00 | 256K | 32K | — | K2.6 is $1.20/$4.50 |

Universal cost multipliers on Anthropic: **batch 0.50×, cache read 0.10×, cache write 1.25× (5m) / 2.0× (1h)**. Break-even for 5m caching is 2 requests; for 1h caching, 3.

**The cache-minimum column is the single most actionable number in this table**, and it is non-monotonic across generations: Opus 5 and Fable 5 cache at **512** tokens; Sonnet 5 / Opus 4.8 at 1024; Opus 4.7 at 2048; **Opus 4.6, Opus 4.5, and Haiku 4.5 at 4096**. QMine's `LLMConfig` docstring already discovered this empirically ("Haiku 4.5's minimum cacheable prefix is 4096 tokens, and the taxonomy prefix… sits below that"). It is right, and the consequence generalizes: **cache-prefix feasibility must be a hard eligibility gate, not a cost adjustment.** A model whose cache minimum exceeds your stable prefix does not cost 10% more — it costs 10× more on every call after the first, silently, with no error. My router encodes this as a gate; §8 shows the diagnostic firing.

Corollary worth acting on: Opus 5's 512-token minimum is *half* Sonnet 5's. There now exist prefixes that cache on Opus 5 and silently do not cache on Sonnet 5 — which inverts the usual "downgrade to save money" intuition for short-prefix roles.

## 3. Independence groups — the finding that actually matters

QMine has three sets of roles whose entire purpose is to *disagree*:

- `annotator_a` vs `annotator_b` → Cohen's κ, a **blocking gate** at 0.90 (`gates.blocking` contains `p2b_kappa`)
- `namer` ×5 → "five blind agents independently named this cluster"
- `taxonomy_architect` vs `taxonomy_critic` → adversarial review

Today all of these route through one `ROLE_TIER` table to one of two models. So:

**κ between two annotators on the same model, same prompt, same temperature is not an inter-rater reliability statistic.** It measures within-model sampling variance and prompt-order sensitivity. A κ of 0.93 obtained that way tells you the model is self-consistent, which you already knew. The gate is passing on a measurement of the wrong quantity. The same argument kills "5 blind namers" — five samples from one model at `temperature=None` are near-identical by construction, so "all five agreed" is evidence of nothing. And `referee` sharing a model with `annotator_a` is systematically biased toward a's labels when adjudicating a-vs-b disagreements.

My router encodes this as `independence_group` and forces distinct **model families** within a group (family, not model — `claude-opus-5` and `claude-opus-4-8` share training lineage and will correlate). It reports achieved diversity and warns when it fails.

**And it does fail, for a reason that is itself the most interesting technical result here.** Running the annotator's floors (`structured ≥ 85`, `zh ≥ 88`, `reasoning ≥ 72`) against the full catalog:

```
claude-opus-5        ELIGIBLE          gpt-5.6-sol      zh 86 < 88
claude-opus-4-8      ELIGIBLE          gemini-3.1-pro   zh 85 < 88
claude-fable-5       ELIGIBLE          qwen3.6-flash    structured 82 < 85
claude-sonnet-5      zh 87 < 88        glm-5.2          structured 84 < 85
claude-haiku-4-5     cache minimum 4096 > stable prefix 3200; zh 76 < 88
deepseek-v4-pro      structured-output mode json_mode below native_schema
```

**The Chinese requirement and the schema requirement are anti-correlated across the market, and their intersection is one vendor.** Chinese-native models (Qwen, GLM, Kimi, DeepSeek) win on `zh` and lose on `structured`; Western frontier models (GPT-5.6, Gemini) win on `structured` and lose on `zh`. Only Anthropic's top three clear both — which means **at these floors, annotator diversity is unobtainable, and κ cannot be made a genuine inter-rater statistic by routing alone.**

Three responses, in order of preference:

1. **Lower the structured floor to 82 and let the repair loop absorb the difference.** QMine already has `max_repair=2` re-prompting in `registry.complete()`. Measured: annotators become `qwen3.6-flash` / `claude-fable-5`, families across the rater group become `{claude, qwen}`, total cost $14.57. You buy real independence for ~nothing. **Cost of the repair loop must be measured, not assumed** — a 15% first-attempt schema-failure rate on Qwen adds 15% to that role's calls.
2. Keep one family but use `claude-opus-5` vs `claude-sonnet-5` and **re-derive the κ threshold**. The 0.90 floor was calibrated on a same-model run; cross-model κ will be lower and a 0.90 gate will false-fail. Do not change the models without re-deriving the threshold — that is a gate-breaking change, and QMine's own README treats re-derivation as mandatory per domain.
3. Accept same-model annotators and **stop calling the statistic inter-rater reliability** in the report. This is the honest do-nothing option.

For **namers the constraint is much weaker** (floors: `structured ≥ 82`, `zh ≥ 90`), and the router happily spreads five namers across **4 families — `claude-fable-5`, `glm-5.2`, `kimi-k3`, `qwen3.6-flash`** — while *cutting* cost, because naming is short-output and the Chinese-native models are both cheaper and better at `zh`. **This is the single highest-value change available: it makes the blind-naming evidence real and saves money simultaneously.** It is also where the "route Chinese work to a Chinese model" advice is correct — for free-text Chinese generation, not for schema-constrained classification.

## 4. Bugs found in QMine's existing LLM layer

Verified against the source at `/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/llm/`.

**(1) `budget.py` — cost estimate uses one flat rate for both tiers.**
```python
def estimated_cost_usd(self, in_rate: float = 3.0, out_rate: float = 15.0) -> float:
    return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000
```
Those are Sonnet 5's rates, applied to Opus 5 deep-tier tokens too. Opus 5 is $5/$25. **Deep-tier spend is under-reported by 40%.** The ledger already breaks down `by_role`; the fix is to price per role using `ROLE_TIER`.

**(2) `registry.py::_account` counts cached tokens at full price.** I read the installed `langchain_anthropic` **1.5.6** source (`chat_models.py:2700–2760`). Its docstring is explicit: *"Anthropic's `input_tokens` excludes cached tokens, so we manually add `cache_read` and `cache_creation` tokens to get the true total."* It sums base + `cache_read` + `cache_creation` into `usage_metadata["input_tokens"]`. QMine reads only that field:
```python
usage = getattr(msg, "usage_metadata", None) or {}
self.ledger.record(role, input_tokens=int(usage.get("input_tokens", 0) or 0), ...)
```
So cache reads — which bill at **0.10×** — are ledgered at 1.0×. Combined with (1), you have one error understating cost 40% and another overstating it, in different roles, not cancelling. The fix is available in the same object: `usage["input_token_details"]` carries `cache_read` and `cache_creation` separately.

**(3) `registry.py` cache key records the *configured* model, not the *served* one.** `cache_key` includes `"model": self.model_name(tier)` and `_store` writes `{"model": self.model_name(tier)}` into the metadata. Today that is always true. **The moment you add a fallback chain it becomes false** — a call served by the fallback gets cached under the primary's name, and a re-run replays a judgment attributed to a model that never made it. For a pipeline whose selling point is reproducibility, this silently corrupts provenance. Any fallback implementation must thread the *actual* served model into both the cache key and `provenance_note()`.

**(4) No content-policy fallback path.** `ModelRefused` is raised and the run dies. `risk_sentinel` and `researcher_risk` handle compliance content and are the most refusal-prone roles in the system. My router flags exactly this: at the current floors *both* roles have an Anthropic-only eligible set, so `dispatch("refusal", ...)` returns **ABORT** — there is no cross-provider escape, and a same-provider retry will refuse for the same reason. Anthropic's own answer is the server-side `fallbacks` parameter (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`), which routes by refusal category and is one line.

**(5) Retry layering.** `LLMConfig.max_retries=2` + SDK retries + LangGraph node policies — the docstring already flags "27 calls for one logical request." Adding a 3-deep fallback chain multiplies this again. **A fallback must replace a retry layer, not stack on it.**

**(6) The five researchers share one `ROLE_TIER` key** (`"researcher": "fast"`), but `researcher_risk` feeds the blocking gate `require_risk_independently_found`. A gate-critical role is on the cheap tier. Same structural issue for the five namers.

## 5. Cost model (predict before you run)

For role *r* on model *m*, with stable prefix *S*, variable prefix *V*, output *O*, calls *N*:

```
cache_eligible = r.needs_cache AND m.min_cache_prefix <= S AND N > 1
if cache_eligible:  in_tokens = S*1.25  +  S*0.10*(N-1)  +  V*N
else:               in_tokens = (S+V)*N
usd = (in_tokens*m.usd_in + O*N*m.usd_out) / 1e6
if r.batchable and m.batch_mult: usd *= 0.50
```

Call counts come from the workload, not from guesses: `annotator_calls = ceil(n_gold/25) + ceil(AL_batch/25)`, `namer_calls = n_leaves × 5`, `referee_calls ≈ disagreement_rate × annotator_calls`. Token counts should come from **`client.messages.count_tokens()`** against the target model, never from `tiktoken` (wrong tokenizer for Claude) and never from a multiplier carried over from another model — Opus 4.7 introduced a new tokenizer that Opus 4.8/5 and Fable 5 share, and Sonnet 5's produces ~30% more tokens than Sonnet 4.6 for the same text.

Budget enforcement has two layers: **pre-flight** (refuse to start a run whose predicted cost exceeds the ceiling, and name the levers — `gold_sample_size`, `n_leaves`, batch, floors) and **in-flight** (QMine's existing `UsageLedger.check()`, which correctly raises *before* the call rather than reporting after). Add a per-role sub-budget so one runaway role cannot consume the global ceiling.

## 6. Allocation algorithm

Budget is a **ceiling, not a target** — this was a real bug in my first draft, which spent the entire budget whenever one was available and put five namers on Fable 5. The fix is a Lagrangian:

> maximize Σ_r [ blast_radius_r × quality(m_r, r) ] − λ × Σ_r cost(m_r, r)

For fixed λ this is **separable**, so each role's choice is an independent argmax — exact, not greedy. λ is `max(LAMBDA_DEFAULT, λ_that_fits_the_budget)`, found by bisection. Generous budget → willingness-to-pay binds (sane spend, headroom left unspent). Tight budget → budget binds. The router reports which one bound.

`blast_radius` is the key weight: 1.0 for an annotator (one row), 100 for the taxonomy architect (every downstream artifact inherits it). It is what makes "spend more on the 3-call role than the 200-call role" fall out of the arithmetic rather than being asserted.

## 7. Fallback chain design

The rule everyone gets wrong: **a fallback must satisfy the same hard gates as the primary.** Falling back from a 1M-context model to a 200K one, or from native structured output to JSON mode, converts a rate-limit into a data-corruption. My `fallback_chain()` filters from the same eligibility pool.

Failure classes route differently — this is LiteLLM's key design lesson (it ships three distinct fallback lists: general `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`):

| Failure | Policy |
|---|---|
| 429 / overloaded / timeout | Retry same model with backoff, then next in chain |
| context exceeded | Skip to first chain entry with **strictly larger** context |
| **refusal** | Skip same-provider entries entirely — a same-provider retry refuses again |
| schema invalid | Repair-prompt ×2 (QMine has this), then first entry with `native_schema` |
| auth | Do not retry; drop the provider from the pool for the whole run |

Chain ordering inverts for refusal-prone roles: normally same-provider-peer first (survives a per-model 429 with one hop), but for refusal risk, cross-provider **first**.

## 8. Prior art, assessed

**Gateways/routers.** *LiteLLM Router* — open source, self-hosted, free; routes on operational signal (`simple-shuffle`, `least-busy`, `usage-based-routing-v2`, `latency-based-routing`, `cost-based-routing`); `allowed_fails` + `cooldown_time` isolate a failing *deployment* rather than the model group; three fallback classes; per-deployment budgets that eject an exhausted deployment and re-admit it on window reset; Redis required for multi-instance cooldown state. **This is the closest fit for QMine and the one to copy** — it needs no training data and is orthogonal to structured output. *OpenRouter* — routing is two independent decisions (which model, which provider); default provider load-balance is weighted by **inverse square of price** with recent-outage filtering; `:nitro` = `provider.sort: "throughput"`, `:floor` = `provider.sort: "price"`. Useful as a fallback *transport*; its auto-router routes on the prompt, which is the wrong signal for a fixed-role pipeline. *NotDiamond, Martian, Requesty, Portkey* — all proprietary managed services; Martian and NotDiamond predict per-prompt quality/latency/cost. **All of them route on prompt content. QMine's routing decision is known statically from the role, so per-prompt prediction adds latency and a dependency for zero information gain.**

**Literature.** *RouteLLM* (LMSYS, ICLR 2025) — four routers trained on Chatbot Arena preference data (similarity-weighted, matrix factorization, BERT, causal LLM); MF router hit 95% of GPT-4 performance using 26% GPT-4 calls (~48% cheaper than random), improving to 14% of calls with LLM-judge augmentation; headline 85% cost reduction on MT-Bench at 95% quality, but only 45% on MMLU and 35% on GSM8K — **the savings collapse as tasks get harder and more homogeneous, which is QMine's regime.** *FrugalGPT* (Chen/Zaharia/Zou, Stanford, TMLR) — three strategies (prompt adaptation, LLM approximation, LLM cascade); up to 98% cost reduction matching GPT-4, or +4% accuracy at equal cost. *Hybrid LLM* (ICLR 2024) — trains a router on the predicted **quality gap** between small and large model; up to 40% fewer large-model calls at no quality drop. *BEST-Route* (Microsoft, ICML 2025) — selects model *and* number of samples; up to 60% cost cut at <1% quality drop. *Cascade routing* (ETH SRI, ICML 2025) — unifies routing and cascading, iteratively selecting the best model at each step and allowed to skip or reorder; provably dominates either alone. *RouterArena* (2026) — 8,400 queries; **the diagnostic oracle scores 80.72 while the best real router scores 72.08**, and the paper attributes the gap specifically to routers being *bad at recognising when a cheap model suffices*. That oracle gap is the honest ceiling on what any learned router buys you.

**Capability tiers.** There is no defensible general mapping from (benchmark, price, context, params) to "can this model do task X." *LLMStructBench* (2026) found **prompting strategy matters more than model size** and semantic errors persist even when structural validity holds; *ExtractBench* (2026) found frontier models at only **4.6% field-level pass rate** on 12,867 fields. Model size does **not** predict structured-output quality — Phi-4 (14B) scored 0.798 value accuracy vs GPT-5's 0.795. And naive grammar-constrained decoding **degrades reasoning** while guaranteeing syntax. The defensible move is therefore **task-specific measurement, not benchmark inference** — which is why my catalog's scores are labelled elicited priors and ship with `calibrate()` (~200 calls per model-axis, cents at Flash rates).

**The Chinese angle.** Qwen3 is reported to outperform alternatives at every comparable size on Chinese/Japanese/Korean, trained on the largest Chinese corpus of any open-weight model. But a 2025–26 study ("Do Chinese models speak Chinese languages?") finds **homogenization in multilingual performance across models** from different linguistic and political contexts, attributed to shared benchmarks and training resources — i.e. the gap is narrower than vendor claims. My conclusion stands as a *hypothesis to test on your corpus*: route free-text Chinese generation (naming, L2 interpretation) to Chinese-native models, keep schema-constrained Chinese classification on whichever model your own `calibrate()` run shows clears both bars.

## 9. Recommended implementation order

1. **Turn on the Batch API** for `annotator_*`, `namer_*`, `adversary`, `l2_interpreter`. −29% at mid scale, −39% at large scale, zero routing complexity, no capability risk. Do this first and alone.
2. **Fix the two ledger bugs** (§4.1, §4.2) so you can measure anything at all.
3. **Split `ROLE_TIER` into per-role entries** with `independence_group`. Route the five namers across ≥3 families. This is the highest-value quality change and it is cost-negative.
4. **Add the fallback chain** with per-failure-class dispatch and Anthropic's server-side `fallbacks="default"` for the two refusal-prone roles. Thread the served model into the cache key (§4.3) *in the same change* — not after.
5. **Run `calibrate()`** on `structured` and `zh` before trusting any cross-vendor decision.
6. Only then consider annotator diversity, and re-derive the κ threshold when you do.

## 10. Full source

The complete module is at the path in the header. Core structures:

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str; provider: str; family: str      # family = the independence unit
    usd_in: float; usd_out: float
    context: int; max_output: int
    so_mode: SOMode                          # native_schema|tool_only|json_mode|none
    min_cache_prefix: int                    # HARD gate, not a cost adjustment
    cache_read_mult: float = 0.10
    cache_write_mult: float = 1.25
    batch_mult: float | None = 0.50
    scores: dict[str, int] = field(default_factory=dict)

def gate_failures(m, r, *, ctx_safety=1.5, min_so="native_schema") -> list[str]:
    """Why this model cannot serve this role. Empty list = eligible."""
    out = []
    need = int((r.prompt_stable_tokens + r.prompt_variable_tokens) * ctx_safety) + r.output_tokens
    if m.context < need:            out.append(f"context {m.context} < required {need}")
    if m.max_output < r.output_tokens: out.append(f"max_output {m.max_output} < {r.output_tokens}")
    if SO_RANK[m.so_mode] < SO_RANK[min_so]:
        out.append(f"structured-output mode {m.so_mode} below {min_so}")
    if r.needs_cache and m.min_cache_prefix > r.prompt_stable_tokens:
        out.append(f"cache minimum {m.min_cache_prefix} > stable prefix "
                   f"{r.prompt_stable_tokens} (prompt cache would silently never engage)")
    for axis, floor in r.floors.items():
        if m.score(axis) < floor:   out.append(f"{axis} {m.score(axis)} < floor {floor}")
    return out

def _pick(r, pool, lam):   # separable given lam -> exact, not greedy
    return max(pool, key=lambda m: r.blast_radius * quality(m, r) - lam * predict_cost(m, r))
```

`route(roles, providers, budget_usd, lam=None)` returns `(assignments, summary)` where the summary carries `predicted_usd`, `min_feasible_usd`, `lambda`, `binding_constraint`, `families_per_group`, and `warnings`. It raises with named remediation levers when the minimum feasible cost exceeds the budget. `frontier()` prints the cost–quality curve; `calibrate()` is the harness that replaces the elicited scores with measurements.

---

## Recommendations

- Turn on the Batch API for the fan-out roles (annotator_a/b, namer_1-5, adversary, l2_interpreter) BEFORE doing any routing work. Measured −29% cost at 600 rows/60 clusters and −39% at 5000/200, versus −9% for full 7-provider routing. Batch is 0.50x on Anthropic, needs no new providers, and carries no capability risk. Routing is not the cost lever on this pipeline.
- Split ROLE_TIER into per-role entries with an `independence_group` field, and route the five namers across at least three distinct model FAMILIES (measured: claude-fable-5 / glm-5.2 / kimi-k3 / qwen3.6-flash all clear the namer floors). Five samples from one model at temperature=None are not five blind agents, so today's 'five namers agreed' evidence is near-vacuous — and fixing it is cost-NEGATIVE because Chinese-native models are both cheaper and stronger on free-text Chinese naming.
- Make cache-prefix feasibility a HARD eligibility gate (`model.min_cache_prefix <= role.stable_prefix_tokens`), not a cost adjustment. Cache minimums are non-monotonic across generations — Opus 5 / Fable 5 cache at 512 tokens, Sonnet 5 / Opus 4.8 at 1024, Opus 4.7 at 2048, Haiku 4.5 / Opus 4.6 / Opus 4.5 at 4096 — and a violated minimum produces no error, just 10x cost on every call after the first. QMine's LLMConfig docstring already discovered this for Haiku; generalize it into the router.
- Fix the two ledger bugs before measuring anything: (a) budget.py `estimated_cost_usd(in_rate=3.0, out_rate=15.0)` applies Sonnet 5 rates to Opus 5 deep-tier tokens, under-reporting deep-tier spend by 40%; (b) registry.py `_account` reads only `usage_metadata['input_tokens']`, which langchain-anthropic 1.5.6 defines as base + cache_read + cache_creation summed together (verified in chat_models.py:2700-2760) — so cache reads that bill at 0.10x are ledgered at 1.0x. Read `usage['input_token_details']` instead.
- Do NOT change annotator_a/annotator_b to different models without re-deriving the kappa>=0.90 blocking gate. At the current floors (structured>=85, zh>=88) only claude-opus-5, claude-opus-4-8 and claude-fable-5 are eligible — the Chinese and schema requirements are anti-correlated across the market and their intersection is one vendor. Relaxing the structured floor to 82 admits qwen3.6-flash and buys real independence at $14.57 total, but cross-model kappa will be lower than the same-model threshold was calibrated against and a 0.90 gate will false-fail.
- Add per-failure-class fallback dispatch (rate_limit / context_exceeded / refusal / schema_invalid / auth route differently), require fallbacks to clear the SAME hard gates as the primary, and thread the ACTUALLY SERVED model into the response cache key in the same change. registry.py currently keys the cache on `self.model_name(tier)` — the configured model — so the first fallback silently caches a judgment under a model that never made it, corrupting the reproducibility guarantee that is QMine's selling point.
- Skip the learned/commercial routers (RouteLLM, NotDiamond, Martian, Requesty). All of them route on prompt content, but QMine's routing decision is fully determined statically by the role — per-prompt prediction adds latency and a dependency for zero information gain. Copy LiteLLM Router's operational design instead (three fallback classes, per-deployment cooldowns with allowed_fails, per-deployment budgets), which needs no training data and is orthogonal to structured output.

## Unverified

- All capability scores in the catalog (reasoning / long_context / structured / zh / critique) are ELICITED PRIORS, not measurements. The central finding in §3 — that the zh>=88 AND structured>=85 conjunction admits only Anthropic models — is a direct consequence of those priors and could invert under real measurement. The `calibrate()` harness is included precisely because this must be tested on the actual corpus (~200 calls per model-axis, cents at Flash rates) before any cross-vendor decision is made.
- platform.claude.com, docs.litellm.ai, and arxiv.org were all unreachable from this environment (domain-verification block on WebFetch). Anthropic pricing rests on the bundled claude-api skill catalog (cached 2026-06-24) cross-checked against third-party aggregators; LiteLLM and paper details rest on search snippets rather than primary sources. Verify LiteLLM config shapes against the live docs before writing code against them.
- gpt-5.6-terra pricing is reported inconsistently: $2.00/$12.00 in some August 2026 sources and $2.50/$15.00 in others. I used $2/$12. This changes the terra-vs-gemini-3.1-pro comparison but not any recommendation.
- DeepSeek announced a peak/off-peak pricing change effective 2026-08-17 — one day before this research. The $0.435/$0.87 figure for V4 Pro may already be stale, and off-peak pricing could make DeepSeek materially cheaper for batchable overnight roles (a lever I did not model).
- claude-sonnet-5's $2/$10 introductory pricing expires 2026-08-31, 13 days out. Every Sonnet-5 figure I quote uses the $3/$15 list rate. If you are budgeting against current invoices you are seeing intro pricing that is about to rise 50%.
- Cache-minimum values for OpenAI, Google, and the Chinese providers are estimates; only the Anthropic column is sourced (from the claude-api skill's prompt-caching table). Since cache minimum is a hard gate in my router, a wrong value silently changes eligibility for cache-dependent roles.
- The repair-loop cost of routing annotators to a weaker-schema model is unmodelled. If qwen3.6-flash fails first-attempt schema validation on, say, 15% of calls, that role's effective call count rises 15% and the cost advantage shrinks. QMine's existing max_repair=2 loop absorbs the correctness risk but not the cost — measure the first-attempt validation rate before committing.
- The referee's disagreement rate is assumed at ~1/3 of annotation calls. The real rate depends on annotator agreement, which is exactly what changes if you diversify the annotators — so referee cost and annotator diversity are coupled in a way my static model does not capture.
- I did not verify whether QMine's LangGraph node retry policies actually stack with LLMConfig.max_retries and the SDK's own retries in practice; the 27-calls-per-request figure comes from QMine's own docstring, not from an execution trace.

## Sources

- https://benchlm.ai/anthropic/api-pricing — Claude API pricing, August 2026
- https://www.aipricing.guru/anthropic-pricing/ — Anthropic pricing: Fable, Opus, Sonnet
- https://devtk.ai/en/blog/claude-api-pricing-guide-2026/ — Claude 5 / Opus / Sonnet / Haiku rates
- bundled `claude-api` skill catalog + prompt-caching table (cached 2026-06-24) — model IDs, cache minimums, batch/cache multipliers, refusal + fallbacks semantics
- https://www.aipricing.guru/openai-pricing/ — OpenAI pricing August 2026 (GPT-5.6)
- https://devtk.ai/en/blog/openai-api-pricing-guide-2026/ — GPT-5.6 Sol/Terra/Luna
- https://techjacksolutions.com/ai-tools/chatgpt/gpt-5-6-pricing/ — GPT-5.6 tier pricing and context window
- https://benchlm.ai/google/api-pricing — Gemini API pricing August 2026
- https://kunavo.com/guides/gemini-api-pricing-2026 — Gemini 3.7 Flash / 3.1 Pro rates and model IDs
- https://benchlm.ai/deepseek/api-pricing — DeepSeek V4 Pro / Flash rates
- https://www.orcarouter.ai/blog/deepseek-v4-pro-pricing — DeepSeek V4 Pro $0.435/$0.87
- https://yage.ai/share/ollama-cloud-vs-api-vs-subscriptions-en-20260428.html — GLM-5.1/5.2, Kimi K2.6/K3, DeepSeek V4 comparison
- https://www.morphllm.com/llm-api — 12 LLM APIs compared by price per 1M tokens
- https://docs.litellm.ai/docs/routing — LiteLLM Router strategies, cooldowns, fallbacks (via search snippets; domain not fetchable)
- https://docs.litellm.ai/docs/proxy/reliability — LiteLLM fallbacks / context_window_fallbacks / content_policy_fallbacks
- https://openrouter.ai/blog/insights/model-routing/ — OpenRouter provider routing, inverse-square price weighting
- https://www.datastudios.org/post/openrouter-model-variants-explained-free-extended-nitro-floor-exacto-thinking-and-provider-sp — :nitro / :floor variants
- https://www.lmsys.org/blog/2024-07-01-routellm/ — RouteLLM framework and benchmark results
- https://arxiv.org/pdf/2406.18665 — RouteLLM: Learning to Route LLMs with Preference Data (ICLR 2025)
- https://arxiv.org/abs/2305.05176 — FrugalGPT (Chen, Zaharia, Zou)
- https://lingjiaochen.com/papers/2024_FrugalGPT_TMLR.pdf — FrugalGPT, TMLR version
- https://proceedings.iclr.cc/paper_files/paper/2024/file/b47d93c99fa22ac0b377578af0a1f63a-Paper-Conference.pdf — Hybrid LLM (ICLR 2024)
- https://github.com/microsoft/best-route-llm — BEST-Route (ICML 2025), arXiv 2506.22716
- https://arxiv.org/abs/2410.10347 — A Unified Approach to Routing and Cascading for LLMs (cascade routing, ICML 2025)
- https://github.com/eth-sri/cascade-routing — cascade routing reference implementation
- https://arxiv.org/html/2510.00202v1 — RouterArena: open platform for comparing LLM routers (oracle gap)
- https://arxiv.org/html/2603.04445v2 — Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey
- https://github.com/MilkThink-Lab/Awesome-Routing-LLMs — curated routing-LLM literature list
- https://arxiv.org/html/2604.25359v1 — The Structured Output Benchmark (LLMStructBench)
- https://openreview.net/forum?id=FKOaJqKoio — JSONSchemaBench: evaluating constrained decoding
- https://arxiv.org/pdf/2605.02363 — When Correct Isn't Usable: structured output reliability in small LMs
- https://arxiv.org/html/2504.00289v3 — Do Chinese models speak Chinese languages? (multilingual homogenization)
- https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Mandarin-Chinese — Qwen3 Chinese performance
- https://evalscope.readthedocs.io/en/latest/benchmarks/cmmlu.html — CMMLU benchmark definition
- https://www.requesty.ai/blog/best-llm-routing-platforms-compared-2026-requesty-portkey-litellm-openrouter — commercial router comparison (vendor-authored)
- https://simorconsulting.com/blog/llm-gateway-comparison-litellm-portkey-martian/ — LiteLLM vs Portkey vs Martian
- local: /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/.venv/lib/python3.12/site-packages/langchain_anthropic/chat_models.py:2700-2760 (v1.5.6) — usage_metadata cache-token summation
- local: QMine src/qmine/llm/registry.py, budget.py, config.py — existing routing/ledger implementation