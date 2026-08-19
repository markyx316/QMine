# Model Landscape & Pricing (Aug 2026)

> Gathered 2026-08-18. Facts marked verified were fetched live; the model-landscape
> dossier in particular flags its prices as secondary-source and unconfirmed —
> which is why the running system fetches prices from a live catalogue rather
> than embedding any table from this document.

> **READ THIS FIRST — verification status.** Direct page fetches were **blocked** in this environment: every `WebFetch` against `platform.claude.com`, `platform.openai.com`, `ai.google.dev`, and `api-docs.deepseek.com` returned `Unable to verify if domain … is safe to fetch. This may be due to network restrictions or enterprise security policies`. There is also **no API key on this machine** (a persisted memory note confirms QMine runs here use the offline heuristic stand-in), so I could not verify a single price against a live API either.
>
> Everything below therefore comes from **`WebSearch` result summaries run on 2026-08-18**, i.e. *secondary* sources (pricing-aggregator sites, vendor blogs, OpenRouter model pages, Wikipedia). Treat every number as **"reported on 2026-08-18, not independently confirmed against a vendor pricing page or a live API call."** Where two searches disagreed I report both and flag the conflict rather than picking a winner. **Re-verify anything load-bearing against the vendor's own pricing page before you route production traffic or sign a budget.**
>
> The one exception: the **Anthropic** numbers are cross-checked against the `claude-api` skill's own cached model table (cached 2026-06-24), which is a first-party artifact.

---

# 1. Anthropic (Claude)

Verified 2026-08-18 via search; cross-checked against the bundled `claude-api` skill table (cached 2026-06-24).

| Model ID | Context | Max out | Input $/1M | Output $/1M |
|---|---|---|---|---|
| `claude-fable-5` | 1M | 128K | $10.00 | $50.00 |
| `claude-mythos-5` (Project Glasswing only) | 1M | 128K | $10.00 | $50.00 |
| `claude-opus-5` | 1M | 128K | $5.00 | $25.00 |
| `claude-opus-4-8` | 1M | 128K | $5.00 | $25.00 |
| `claude-opus-4-7` | 1M | 128K | $5.00 | $25.00 |
| `claude-opus-4-6` | 1M | 128K | $5.00 | $25.00 |
| `claude-sonnet-5` | 1M | 128K | **$2.00** | **$10.00** |
| `claude-sonnet-4-6` | 1M | 128K | $3.00 | $15.00 |
| `claude-haiku-4-5` | 200K | 64K | $1.00 | $5.00 |

**Material change since the skill's cached table:** the skill lists Sonnet 5 at `$3.00 / $15.00` with a `$2.00 / $10.00` *introductory* rate expiring 2026-08-31. Search on 2026-08-18 reports that **the $2/$10 rate is now permanent and the 2026-09-01 increase has been cancelled** ([aipricing.guru](https://www.aipricing.guru/anthropic-pricing/), [benchlm.ai](https://benchlm.ai/anthropic/api-pricing)). This is 13 days before the old expiry — worth confirming directly, because if true it changes the Sonnet-5-vs-alternatives math permanently. **Flagged: single-source, not confirmed on platform.claude.com.**

- **Endpoint:** `https://api.anthropic.com/v1/messages` (from SDK docs, not re-verified today).
- **Structured output:** best-in-class for QMine's needs. `output_config: {format: {type: "json_schema", schema: {...}}}` gives *constrained decoding*, plus `strict: true` on tool definitions. Python SDK exposes `client.messages.parse(..., output_format=PydanticModel)` returning `response.parsed_output` as a validated instance. Supported on Fable 5, Opus 5, Opus 4.8, Sonnet 5, Haiku 4.5.
- **Batch:** `POST /v1/messages/batches`, **50% off all tokens**, ≤100,000 requests / 256 MB per batch, most complete <1h, hard max 24h, results retained 29 days. Results arrive **out of order — key by `custom_id`**.
- **Caching:** `cache_control: {type:"ephemeral"}`, 5m default TTL or `ttl:"1h"`. Reads ≈**0.1×** base input; writes **1.25×** (5m) or **2×** (1h). Max 4 breakpoints. **Minimum cacheable prefix is model-dependent and non-monotonic: 512 tok on Opus 5 / Fable 5 / Mythos 5; 1024 on Opus 4.8, Sonnet 5, Sonnet 4.6; 2048 on Opus 4.7; 4096 on Opus 4.6 and Haiku 4.5.** Also: a 20-content-block lookback window — a single agentic turn emitting >20 blocks silently misses the previous cache.
- **Rate limits (reported):** per-org RPM / ITPM / OTPM by tier Start / Build / Scale. Sonnet 5 and Opus 5 share brackets: **Start 1,000 RPM / 2M ITPM / 400K OTPM; Build 5,000 / 5M / 1M; Scale 10,000 / 10M / 2M** ([requesty.ai](https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek), [standardcompute.com](https://standardcompute.com/rate-limits/anthropic)). **Cached input does not count toward ITPM** — at 80% hit rate a 2M ITPM limit passes ~10M input tok/min. That is a bigger throughput win than a tier upgrade and is directly relevant to a 5,000-call annotation job. **Flagged: tier numbers are third-party, not from platform.claude.com.**
- **Gotchas that will bite QMine:** `temperature`/`top_p`/`top_k` are **removed** (400) on Opus 5 / 4.8 / 4.7 / Fable 5 / Mythos 5 — the repo already encodes this in `_NO_TEMPERATURE` at `src/qmine/llm/registry.py:326`. `budget_tokens` is 400 on those models. Assistant-turn prefills 400 on all 4.6+ Opus/Sonnet. On Opus 5 thinking is **on by default** and `max_tokens` caps thinking + text together (the repo's `max_tokens: 16000` comment already reflects this). Refusals arrive as **HTTP 200 with `stop_reason:"refusal"`** — the repo's `_check_refusal` handles it.

# 2. OpenAI

Verified 2026-08-18. Current family: **GPT-5.6**, released 2026-07-09, three tiers named **Sol / Terra / Luna** (capability tiers, not generations).

| Model ID | Context | Max out | Input $/1M | Output $/1M | Long-ctx (>272K) in/out |
|---|---|---|---|---|---|
| `gpt-5.6-sol` | 1.05M | 128K | $5.00 | $30.00 | $10 / $45 |
| `gpt-5.6-terra` | 1.05M | 128K | $2.50 | $15.00 | $5 / $22.50 |
| `gpt-5.6-luna` | 1.05M | 128K | $1.00 | $6.00 | $2 / $9 |

Knowledge cutoff **2026-02-16** on all three. Long-context billing is **2× input / 1.5× output applied to the whole request** once the prompt crosses 272K tokens. Sources: [Wikipedia GPT-5.6](https://en.wikipedia.org/wiki/GPT-5.6), [simonwillison.net](https://simonwillison.net/2026/Jul/9/gpt-5-6/), [OpenRouter gpt-5.6-terra](https://openrouter.ai/openai/gpt-5.6-terra), [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing).

⚠️ **Direct conflict.** A second search reported **Terra $2.00/$12.00 and Luna $0.20/$1.20** ([aipricing.guru](https://www.aipricing.guru/openai-pricing/), [cloudzero](https://www.cloudzero.com/blog/openai-pricing/)). I favour the $2.50/$15 and $1/$6 figures because they scale exactly to the independently-reported 2×/1.5× long-context rates ($5/$22.50, $2/$9), which the other set does not. **Do not budget on either without checking the OpenAI pricing page.** Older gens still listed: GPT-5.5 at $5/$30, GPT-5.4 at $2.50/$15 (long-ctx $5/$22.50), GPT-5 Nano ~$0.03 input.

- **Endpoints:** `https://api.openai.com/v1/responses` (current) and `/v1/chat/completions` (compat). Not re-verified today.
- **Structured output:** strict JSON-Schema via `response_format: {type:"json_schema", json_schema:{strict:true, schema:{...}}}`, plus SDK `client.responses.parse(text_format=PydanticModel)`. Strong — comparable to Anthropic.
- **Caching:** automatic prefix caching. Reported billing on GPT-5.6: **cache writes 1.25×** uncached input, **cache reads at 90% discount** ($0.50/M on Sol), 30-minute reuse window. The **entire prefix is cacheable — messages, images, audio, tool definitions, and structured-output schemas**. For GPT-5.6 you must set **`prompt_cache_key`** to get reliable matching for both implicit and explicit caching ([developers.openai.com/api/docs/guides/prompt-caching](https://developers.openai.com/api/docs/guides/prompt-caching), [aihubmix](https://aihubmix.com/blog/gpt-5-6-is-live-prompt-caching-billing-changes-explained)). The 1.25× write charge is **new** relative to older OpenAI caching, which was free-to-write — re-run your cost model.
- **Batch:** Batch API historically 50% off / 24h SLA. **Could not confirm the discount still applies to GPT-5.6 — flagged as unverified.**
- **Rate limits:** usage tiers 1–5 by cumulative spend. **No current per-tier RPM/TPM numbers verified today — flagged.**

# 3. Google (Gemini)

Verified 2026-08-18. The Gemini lineup is the **messiest** to pin down; multiple overlapping Flash generations were reported.

| Model | Context | Input $/1M | Output $/1M | Notes |
|---|---|---|---|---|
| Gemini 3.1 Pro | **2M** | $2.00 (≤200K) / $4.00 (>200K) | $12.00 / $18.00 | Largest context on the market |
| Gemini 3.7 Flash | 1M | **$0.75 intro** → $1.50 (2027-01-01) | **$3.75 intro** → $7.50 | Released **2026-08-13** |
| Gemini 3.6 Flash | 1M | $0.75 intro → $1.50 | $3.75 intro → $7.50 | Same intro deal |
| Gemini 3.5 Flash | 1M | $1.50 | $9.00 | Launched 2026-05-19 |
| Gemini 3.5 Flash-Lite | — | $0.30 | $2.50 | |
| Gemini 2.5 Flash-Lite | — | $0.10 | $0.40 | Cheapest listed |

Sources: [ai.google.dev/gemini-api/docs/latest-model](https://ai.google.dev/gemini-api/docs/latest-model), [deepmind model card 3.7 Flash](https://deepmind.google/models/model-cards/gemini-3-7-flash/), [verdent.ai Gemini 3.1 Pro pricing](https://www.verdent.ai/guides/gemini-3-1-pro-pricing), [benchlm.ai](https://benchlm.ai/google/api-pricing), [metacto](https://www.metacto.com/blogs/the-true-cost-of-google-gemini-a-guide-to-api-pricing-and-integration). **Intro pricing on 3.6/3.7 Flash expires 2026-12-31** — do not build a 2027 budget on $0.75/$3.75.

⚠️ **I could not confirm a Pro-tier model newer than 3.1 Pro.** A 3.7-generation Flash coexisting with a 3.1-generation Pro is plausible (Google ships Flash faster) but I did not verify it. **Flagged.**

- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`. Not re-verified.
- **Structured output:** `response_mime_type: "application/json"` + `response_schema` (OpenAPI subset). Explicitly listed as supported on 3.7 Flash alongside caching, code execution, function calling, search grounding.
- **Thinking:** `thinking_level` = `LOW` / `MEDIUM` (default) / `HIGH` on 3.7 Flash. **`MINIMAL` is rejected with a validation error on 3.7 Flash** — a real migration trap if you carried config forward from 3.5/3.6.
- **Batch:** 50% off every model, 24h SLA.
- **Caching:** implicit caching on by default (no code change); explicit context caching also available. Pro cached reads reported at **$0.20–$0.40/M** (~90% off).

# 4. xAI (Grok)

Verified 2026-08-18 ([aipricing.guru/xai-pricing](https://www.aipricing.guru/xai-pricing/), [costgoat](https://costgoat.com/pricing/grok-api), [mem0](https://mem0.ai/blog/xai-grok-api-pricing)).

| Model | Input $/1M | Cached in $/1M | Output $/1M |
|---|---|---|---|
| **Grok 4.6** (released **2026-08-12**) | $2.00 | $0.50 | $6.00 |
| Grok 4.5 | $2.00 | $0.30 | $6.00 |
| Grok 4.3 / Grok 4.20 variants | $1.25 | — | $2.50 |
| Grok Build 0.1 (coding) | $1.00 | — | $2.00 |

**Grok 4.6 doubles the whole request** to $4.00 / $1.00 / $12.00 once a prompt reaches 200K tokens. Endpoint `https://api.x.ai/v1/chat/completions` (OpenAI-compatible). Structured outputs: OpenAI-style `response_format` json_schema reported supported; **not independently verified — flagged.** No batch-discount or rate-limit-tier data found. **For QMine specifically, Grok is uninteresting: no Chinese-language advantage, no batch discount, and Grok 4.5 has a *better* cache rate than 4.6.**

# 5. DeepSeek

Verified 2026-08-18 — and this one changed **two days ago**.

- **`deepseek-v4-pro`** — DeepSeek-V4-Pro-0813, **GA as of 2026-08-13**. 1.6T total / 49B active MoE.
- **`deepseek-v4-flash`** — DeepSeek-V4-Flash-0731, **public beta**. 284B / 13B MoE.
- Both **1M-token context, 384K max output**, text-only. V4 family launched 2026-04-24.
- **`deepseek-chat` and `deepseek-reasoner` were RETIRED on 2026-07-24.** They were aliases of V4-Flash non-thinking/thinking. **If QMine or any config still names them, those calls are dead.**

**Pricing — effective 2026-08-16 16:00 UTC, peak/off-peak went official:**

| Model | Peak in / out $/1M | Off-peak (50%) in / out |
|---|---|---|
| `deepseek-v4-pro` | $1.32 / $3.96 | $0.66 / $1.98 |
| `deepseek-v4-flash` | $0.14 in (cache-miss) / $0.28 out | 50% of peak |

**Peak hours: 01:00–04:00 and 06:00–10:00 UTC. The other 17 hours bill at 50%.** Pre-2026-08-16 V4-Pro was $0.435 / $0.87 — so the price roughly **tripled-to-quadrupled** ([androidheadlines](https://www.androidheadlines.com/2026/08/deepseek-v4-pro-ai-model-quadruples-prices-remains-cheaper.html), [deepseek.ai/pricing](https://deepseek.ai/pricing), [aipricing.guru](https://www.aipricing.guru/deepseek-pricing/)). One search still quoted the old $0.435/$0.87 — that is **stale**; the $1.32/$3.96 figure is corroborated by three sources including the price-change news story.

⚠️ **Suspect number:** V4-Flash cache-hit input reported at **$0.0028/M**. That is 2% of the $0.14 cache-miss rate, whereas DeepSeek's documented caching discount is 90% (which would be $0.014). One of the two is wrong. **Flagged — verify before modelling.**

- **Endpoint:** `https://api.deepseek.com`. Notably, **the API speaks both OpenAI ChatCompletions *and* Anthropic Messages format**, so it drops into Anthropic-shaped clients without a proxy — relevant because QMine's registry currently builds a `ChatAnthropic`.
- **Structured output:** ⚠️ **This is DeepSeek's weak point for QMine.** Sources describe "JSON responses and tool requests combined with **application-side validation**" — i.e. JSON *mode*, not constrained strict-schema decoding. Every one of QMine's 13 roles uses Pydantic structured output. **Assume you need retry-on-parse-failure, not a guarantee. Flagged as the single biggest adoption risk.**
- **Batch:** **no separate Batch API tier.** The off-peak window *is* the batch discount — schedule the annotation job for the 17 off-peak hours.
- **Caching:** automatic prefix caching, no configuration.
- **Rate limits:** **DeepSeek publishes no RPM and no TPM.** It documents **concurrency only: 500 concurrent for V4-Pro, 2,500 for V4-Flash** (account level). Under load it **slows responses rather than returning 429**. For a 5,000-call job this is actually favourable — you will not get rate-limited, you will get latency. Budget wall-clock, not retries.

# 6. Alibaba Qwen

Verified 2026-08-18.

| Model ID | Context | Input $/1M | Output $/1M |
|---|---|---|---|
| `qwen3.8-max` (GA **2026-08-03**, 2.4T total / 95B active, multimodal MoE) | **1M** | $2.00 | $6.00 |
| Qwen3.7-Max / Qwen3-Max | 262K (1M ext.) | $1.20–$3.00 (tiered by prompt length) | $3.75–$6.00 |
| Qwen3.5 397B | — | $0.60 | $3.60 |
| Qwen3.5 Plus | — | $0.40 | $2.40 |
| **Qwen3.5 Flash** | — | **$0.10** | **$0.40** |

Sources: [ofox.ai](https://ofox.ai/blog/qwen-3-8-max-price-context-window-api-access-open-weights-2026/), [qwencloud](https://www.qwencloud.com/models/qwen3.8-max), [benchlm.ai/alibaba](https://benchlm.ai/alibaba/api-pricing), [felloai](https://felloai.com/qwen-pricing/). Qwen3-Max's tiered-by-prompt-length pricing is a genuine budgeting hazard — reported range $1.20→$3.00 input across its context.

- **Endpoints — three surfaces per model:** OpenAI-compatible at **`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`** (key from `DASHSCOPE_API_KEY`), an **Anthropic-compatible** route, and native DashScope. Regions: Beijing, **Singapore (`dashscope-intl`, the default for international users and where the free quota lives)**, Tokyo, Frankfurt, US Virginia. The platform is now branded **Alibaba Cloud Model Studio** (formerly DashScope).
- **Structured output:** `response_format: {"type": "json_object"}` — **JSON mode, documented as working in thinking mode too, for Qwen3.7-Max series** ([alibabacloud.com/help/en/model-studio/qwen-structured-output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)). Again: **JSON mode ≠ strict schema.** Plan for validation + retry.
- **Batch:** **50% off input and output**, and **batch requests are exempt from real-time rate limits**. Excellent fit for QMine's annotator sweep.
- **Caching:** two modes — **implicit (automatic) at 20% of input price**, **explicit (user-managed) at 10%**. Documented effective cost with implicit caching ≈ **60% of non-cached** in the modelled mix.
- ⚠️ **Critical interaction: batch discount and context-cache discount CANNOT be combined on the same request.** You must pick one. For QMine's annotators (huge shared taxonomy prefix, low latency sensitivity), run the numbers both ways — with a 4K-token shared prefix and only 800 fresh tokens per call, explicit caching at 10% may beat flat 50% batch.

# 7. Moonshot / Kimi

Verified 2026-08-18.

| Model | Context | Input $/1M | Cache-hit in | Output $/1M |
|---|---|---|---|---|
| `kimi-k3` | **1,048,576** | $3.00 | **$0.30** | $15.00 |
| `kimi-k2.7-code` | 262,144 | $0.95 | $0.19 | $4.00 |
| `kimi-k2.6` | 262,144 | $0.95 | $0.16 | $4.00 |
| `kimi-k2.5` | 262,144 | $0.60 | $0.10 | $3.00 |

**⚠️ `kimi-k2.5` and `moonshot-v1` are being RETIRED on 2026-08-31 — 13 days away.** Migration path is `kimi-k2.7-code` or `kimi-k3` ([nxcode](https://www.nxcode.io/resources/news/kimi-k2-5-pricing-plans-api-costs-2026), [benchlm.ai/moonshot](https://benchlm.ai/moonshot/api-pricing), [trilogyai](https://trilogyai.substack.com/p/kimi-k3-is-live-pricing-benchmarks)). **K3 has no long-context surcharge** — flat pricing across the full 1M window, which is unusual and valuable.

- **Endpoints:** `https://api.moonshot.cn/v1` (CN) / `https://api.moonshot.ai/v1` (intl), OpenAI-compatible. Not re-verified.
- **Caching:** automatic prefix matching, **no cache ID or parameter to configure**, cached token = 1/10 fresh.
- **Rate limits:** tiered by **cumulative recharge, Tier0 ($1) → Tier5 ($3,000)**. Moonshot has **not published per-tier RPM/TPM for K3** — one source explicitly warns to treat any cited number as unverified. **Flagged.**
- **Batch:** none found. **Flagged as absent, not confirmed absent.**
- K3 at $3/$15 is **50% more expensive than Claude Sonnet 5** at $2/$10. K2.6/K2.7-Code at $0.95/$4.00 is the interesting tier.

# 8. Zhipu / GLM (Z.ai)

Verified 2026-08-18.

| Model | Context | Max out | Input $/1M | Cached in | Output $/1M |
|---|---|---|---|---|---|
| **GLM-5.2** (open weights `zai-org/GLM-5.2`) | **1M** | 128K | $1.40 | **$0.26** | $4.40 |
| GLM-5.1 (via DeepInfra) | — | — | $1.05 | — | $3.50 |
| GLM-4.5 | — | — | $0.60 | — | $2.20 |
| GLM-4.5-Air | — | — | $0.20 | $0.03 | $1.10 |
| **GLM-4.7-Flash / GLM-4.5-Flash** | — | — | **$0** | — | **$0** |

Sources: [docs.z.ai/guides/llm/glm-5.2](https://docs.z.ai/guides/llm/glm-5.2), [siliconflow](https://www.siliconflow.com/blog/glm-5-2-api-guide), [openrouter z-ai/glm-5.2](https://openrouter.ai/z-ai/glm-5.2), [venturebeat](https://venturebeat.com/technology/z-ais-open-weights-glm-5-2-beats-gpt-5-5-on-multiple-long-horizon-coding-benchmarks-for-1-6th-the-cost), [aipricing.guru/z-ai](https://www.aipricing.guru/z-ai-pricing/). One source lists cached input as "limited-time free" on GLM-5.2, another as $0.26 — **conflict, flagged**.

- **Endpoints:** `https://api.z.ai/api/paas/v4/` (intl) / `https://open.bigmodel.cn/api/paas/v4/` (CN), OpenAI-SDK drop-in. Z.ai has also shipped an **Anthropic-compatible route** in past releases; I did **not** confirm it for GLM-5.2 today — **flagged**.
- **Structured output: GLM is the standout among Chinese labs here.** Multiple independent sources: *"the most reliable tool calling among Chinese models with Berkeley Function Calling scores close to GPT-5, and dependable structured JSON output"*; *"the strongest Chinese pick for agentic and tool-use workflows in 2026"*; one guide recommends *"start with GLM-5.1 and test it against your specific schema requirements"* ([checkaimodels](https://checkaimodels.com/en/articles/china-ai-models-landscape-2026/), [nextfuture](https://nextfuture.io.vn/blog/2026-chinese-llm-stack-qwen-deepseek-minimax-kimi-glm-compared)). **For a 13-role all-Pydantic pipeline, this matters more than price.**
- **Caching:** prompt caching supported, up to ~90% off repeated context.
- **Batch:** not found. **Flagged.**
- **Subscription alternative:** GLM Coding Plan — Lite $30/quarter, Pro $90/quarter, Max $240/quarter. Irrelevant to API batch work but relevant if QMine is developer-driven.

# 9. MiniMax

Verified 2026-08-18.

| Model | Context | Max out | Input $/1M | Cache read | Cache write | Output $/1M |
|---|---|---|---|---|---|---|
| **MiniMax M3** (launched **2026-06-01**, open-weight) | **1,048,576** | 262,144 | $0.30 (≤512K in) | **$0.06** | — | $1.20 |
| MiniMax M2.7 | 204,800 | 196,608 | $0.30 | $0.06 | $0.375 | $1.20 |
| MiniMax M2 (legacy) | — | — | $0.255–$0.30 | — | — | $1.00–$1.02 |

Sources: [openrouter minimax/minimax-m3](https://openrouter.ai/minimax/minimax-m3), [codersera](https://codersera.com/blog/minimax-m3-developer-guide/), [benchlm.ai/models/minimax-m3](https://benchlm.ai/models/minimax-m3), [pricepertoken](https://pricepertoken.com/pricing-page/provider/minimax).

- **M3's $0.30/$1.20 is a "Permanent 50% off" promotional rate** — reported as permanent, but promotional pricing is exactly the kind that reverts. **Flagged.**
- **Tier variants:** M3 **Priority** = 1.5× standard ($0.45/$1.80). M2.7 **`-highspeed`** = 2× standard ($0.60/$2.40).
- **M2.5, M2.1, M2 are in MiniMax's legacy catalog.**
- **Architecture note worth knowing:** M3 uses **MiniMax Sparse Attention (MSA)**, cutting per-token compute at long context to roughly **1/20** of the previous generation at 1M tokens. That is why the 1M-context price is so low.
- **Endpoints:** `https://api.minimax.io/v1` (intl) / `api.minimaxi.com` (CN). Not re-verified.
- **Structured output:** no specific evidence found. **Flagged as unknown.**
- **Batch:** not found. **Flagged.**

# 10. ByteDance Doubao / Volcengine Ark

Verified 2026-08-18. Pricing is **published natively in CNY** on the Volcano Engine console; USD figures below are third-party conversions and inherit FX risk.

| Model | Context | Max out | Input | Output |
|---|---|---|---|---|
| `doubao-seed-2.1-pro` (June 2026) | — | — | **¥6/MTok** (~$0.83) | **¥30/MTok** (~$4.15) |
| `doubao-seed-2.0-pro` (Feb 2026) | 256K | 128K | ~$0.47 | ~$2.37 |
| `doubao-seed-1.6-flash` | — | — | **$0.022** | **$0.219** |
| Doubao Seed 2.0 Mini | — | — | ~$0.030 | — |

Sources: [llmreference seed-2.1-pro](https://www.llmreference.com/model/seed-2.1-pro/volcengine), [cloudprice](https://cloudprice.net/models/bytedance-doubao-seed-2-pro), [dev.to Doubao API setup](https://dev.to/tokenmixai/doubao-api-setup-2026-19-bytedance-models-0022m-floor-python-in-5-min-2akn), [china-llm.com](https://china-llm.com/provider/bytedance-doubao). Catalog is ~17–19 models; input floor **$0.022/M**, the cheapest credible rate found anywhere in this survey.

- **Endpoint:** `https://ark.cn-beijing.volces.com/api/v3`. An **international platform accepts email signup and USD/EUR by credit card** — historically the hardest Chinese provider to onboard from outside CN, now reportedly easier. **Not verified by me — flagged.**
- **Structured output / batch / caching / rate limits:** **no data found on any of the four. Flagged across the board.** This is the least-documented provider in the survey from outside China.
- **Chinese-language strength is Doubao's headline claim** — see §13.

# 11. Mistral

Verified 2026-08-18. **Mistral Large 3: $0.50 / $1.50 per 1M** — down from Large 2's $2.00/$6.00, a ~4× cut. Mistral Medium 3.5 also current; family floor reported around $0.04–$0.10/M ([benchlm.ai/mistral](https://benchlm.ai/mistral/api-pricing), [cloudzero](https://www.cloudzero.com/blog/mistral-api-pricing/), [aipricing.guru](https://www.aipricing.guru/mistral-pricing/)). Endpoint `https://api.mistral.ai/v1`. Cache-read discounts available on most models. **No Chinese-language advantage; no exact model-ID strings or context windows verified — flagged.** For QMine, Mistral is a Western-cheap option with no reason to prefer it over Sonnet 5 or GLM-5.2.

# 12. Meta Llama — via which hosts

Meta does not meaningfully sell first-party inference; Llama is a **host-shopping exercise**. Verified 2026-08-18 ([amnic](https://amnic.com/blogs/llama-api-pricing), [aipricing.guru/meta](https://www.aipricing.guru/meta-pricing/)):

**Llama 3.3 70B** — DeepInfra **$0.23/$0.40** · Groq **$0.59/$0.79** · Together **$0.88/$0.88** · Fireworks **$0.90/$0.90**.
**Llama 3.1 8B** — DeepInfra **$0.06** · Together $0.10–$0.18.
**Llama 405B** — ~$0.80 to $9.50 across providers.
Llama 4 Maverick appears in open-weight leaderboards but I found no current price. **Flagged.**

**Assessment for QMine: skip Llama.** The models quoted are 3.x-generation, and on Chinese-language work every Chinese-native open-weight model in this survey beats them at comparable or lower cost.

# 13. Cloud aggregators — Bedrock, Azure, Vertex

⚠️ **The search results for this section were visibly stale and I do not trust them.** One result quoted "Claude Opus 4.7 costs $15 input / $75 output" (Anthropic's own rate is $5/$25 — off by 3×) and "Gemini 1.5 Flash on Vertex" (a model generations out of date). **Treat this entire section as low-confidence.**

What is probably still structurally true:

- **Amazon Bedrock** — broadest catalog: **the full GPT-5.5 family** plus Claude, Llama, Mistral, Cohere, AI21, Stability, Amazon Titan/Nova. Pricing modes: on-demand per token, **batch at ~50% of on-demand**, and provisioned throughput by the hour. Nova Lite ~$0.06/M input, Nova Pro ~$0.80/$3.20. Two current-looking corroborations: AWS announced **GPT-5.6 Sol/Terra/Luna now support 1M-token context on Bedrock** (Aug 2026), and **explicit prompt caching for GPT-5.6 models on Bedrock**. Claude on Bedrock uses **`anthropic.`-prefixed model IDs** (`anthropic.claude-opus-5`) and is **partner-operated with separate pricing** from Anthropic first-party.
- **Azure / Microsoft Foundry** — deepest OpenAI access; GPT-5 family is **first-party only on Azure AI Foundry and OpenAI's own API**. Per the `claude-api` skill, **Claude on Microsoft Foundry bills at standard Anthropic API rates** through the Microsoft Marketplace — i.e. no partner markup, unlike Bedrock/Vertex.
- **Google Vertex AI** — Gemini is **exclusive to Vertex**; also carries Claude via Model Garden at partner pricing. Batch 50%.
- **Feature masks matter more than price.** From the skill's first-party availability table: on **Bedrock**, web search / web fetch / code execution / Message Batches / Files API / Models API / MCP connector are **not supported** for Claude. On **Vertex**, web fetch, code execution, Batches, Files, and Models API are **not supported**, and web search is limited to the older `web_search_20250305`. **Automatic prompt caching is unsupported on both Bedrock and Vertex** — you must place `cache_control` manually. For QMine that last one is decisive: your annotator caching strategy behaves differently on Bedrock/Vertex than on the first-party API.
- **Claude Platform on AWS** is a distinct, **Anthropic-operated** offering with same-day API parity and **bare** (unprefixed) model IDs — not the same thing as Bedrock.

# 14. Serverless aggregators — OpenRouter, Together, Fireworks, Groq, DeepInfra, Cerebras

Verified 2026-08-18 ([blog.alephant.io comparison](https://blog.alephant.io/openmodels-vs-openrouter-together-fireworks-deepinfra-2026/), [openrouter blog June 2026](https://openrouter.ai/blog/insights/the-open-weight-models-that-matter-june-2026/), [pricepertoken comparisons](https://pricepertoken.com/endpoints/compare/deepinfra-vs-openrouter)).

**DeepInfra is the price leader on open weights** and hosts the widest current-generation catalog (Kimi K2 family, Qwen3.5 family, GLM-5, DeepSeek V4):

| Model | DeepInfra | Fireworks/Baseten | Together | Novita |
|---|---|---|---|---|
| DeepSeek V4 Pro | **$1.30 / $2.60** | $1.74 | $2.10 | — |
| DeepSeek V4 Flash | **$0.10 / $0.20** | — | — | — |
| Kimi K2.6 | **$0.75 / $3.50** | $0.95 | $1.20 | $0.80 |
| GLM-5.1 | **$1.05 / $3.50** | — | — | — |
| Qwen3-235B-A22B-Instruct | **$0.09 / $0.10** | — | — | — |

**Note that DeepInfra's DeepSeek V4 Pro at $1.30/$2.60 now UNDERCUTS DeepSeek's own peak-hour rate of $1.32/$3.96** — a direct consequence of the 2026-08-16 first-party price rise. DeepSeek V4 Flash at $0.10/$0.20 on DeepInfra also beats first-party $0.14/$0.28.

- **OpenRouter** — 319 models vs DeepInfra's 73; broadest availability, routing/failover, but markup and less control. Endpoint `https://openrouter.ai/api/v1`.
  ⚠️ **A structured-output landmine for QMine:** OpenRouter supports `response_format: {type:"json_schema"}` including with `stream:true`, and `method="json_schema"` uses native enforcement **where the backend model supports it**. But **when multi-model routing is active (a `models` list, or `route:"fallback"`), it always degrades to `functionCalling` because the actual backend model's capabilities are unknown at request time** ([openrouter structured-outputs docs](https://openrouter.ai/docs/guides/features/structured-outputs), [instructor integration](https://python.useinstructor.com/integrations/openrouter/)). Set **`provider: {require_parameters: true}`** to keep requests only on providers that honour what you sent. **Pin a single model, or accept degraded schema enforcement.**
- **Groq** — LPU speed play: **280–1,000 tok/s**, Llama 3.3 70B at ~750 tok/s. Floor $0.05/M (Llama 3.1 8B) up to $0.84/M (Qwen3.6 27B). Lists **Kimi K2 at $1.00/$3.00** with prompt caching. Endpoint `https://api.groq.com/openai/v1`.
- **Cerebras** — Llama 3.3 70B at ~**2,100 tok/s** (WSE-3), ~$0.85/$1.20. More per token, more tokens per second; cost-per-second-of-output is comparable to Groq.
- **Together** — `https://api.together.xyz/v1`. **Fireworks** — `https://api.fireworks.ai/inference/v1`. Both consistently priced above DeepInfra on identical weights.

# 15. Chinese-language evidence — the part that actually decides this

This is the strongest **structural** finding in the survey, and also the one with the **weakest numeric backing**. The published C-Eval/CMMLU figures I could find are for models a generation or two behind the current lineup.

**Direct benchmark numbers found (2026-08-18):**
- **Doubao Seed 1.6: CMMLU 91.5, C-Eval 92.1** — the only concrete C-Eval/CMMLU pair I could verify for a 2026-generation Chinese model ([index.dev](https://www.index.dev/blog/chinese-ai-models), [explore.n1n.ai](https://explore.n1n.ai/blog/chinese-ai-model-benchmarks-2026-deepseek-glm-kimi-qwen-2026-06-28)). Note this is Seed **1.6**, not the current Seed 2.0/2.1.
- **Qwen3 "clearly ahead on Chinese quality, top tier on both C-Eval and CMMLU."** Qwen3-Max leads **Arena-Hard at 90.5** (vs DeepSeek V3.2 at 87.1).
- **Qwen2.5-72B-Instruct scored 68.90 on SuperCLUE**, 2.34 points below the average of the world's top-5 closed models — the cleanest available quantification of the open-vs-closed gap in Chinese.
- **Kimi K2.6 leads SWE-Bench Pro at 58.6%**, ahead of GPT-5.4 (57.7%), Gemini 3.1 Pro (54.2%), Claude Opus 4.6 (53.4%) — coding, not Chinese, but it establishes these are not toy models.
- **Chatbot-arena-style Elo for top open weights: GLM-5 1451, Kimi K2.5 1447, DeepSeek-R1 1436** — all Chinese labs.

**The C-SimpleQA finding is the most decision-relevant one for QMine.** The Chinese SimpleQA paper ([arxiv 2411.07140](https://arxiv.org/html/2411.07140)) reports that Chinese-community LLMs (Doubao-pro, GLM-4-Plus, Qwen-Max, DeepSeek) are **significantly better than GPT/o1 on the Chinese Culture (CC) subtopic**, while o1 holds a significant advantage on **science subtopics (Engineering/Technology/Applied Sciences, Natural Science)**. Critically: **introducing RAG collapses the gap** — for GPT-4o vs Qwen2.5-3B the spread fell from **42.4% to 9.3%**.

**What that means for a search-query-mining pipeline, concretely:**
1. The Chinese-native advantage is **strongest exactly where QMine lives** — culturally-grounded, entity-dense, colloquial short text. Search queries are the CC subtopic, not the NS subtopic.
2. But the advantage is **largest when the model must supply the knowledge from parameters**. QMine's annotators receive the taxonomy and the query in-context — structurally closer to the RAG condition, where the gap narrows to single digits. **Do not assume the headline CC gap transfers to your annotation task.** It is a hypothesis to test, not a result to bank.
3. Older, non-2026 evidence points the same way: a peer-reviewed ACM TMIS comparison found **GPT-4 Turbo leads in English contexts whereas Chinese LLMs stand out in Chinese contexts** — the effect is durable across generations.

⚠️ **I found NO benchmark specifically covering Chinese search-query intent classification or Chinese short-text understanding.** One search targeted exactly this and returned nothing on-point. **This is the gap you have to close yourself** — and QMine already has the instrument: a prior finding in memory records that a blind 5-agent panel rediscovered the template-twin split and 4 unseeded risk categories. **Run that same blind-panel protocol with a Chinese model in the annotator seat and measure inter-annotator agreement against the existing Claude baseline.** That is a stronger signal than any public benchmark and you can produce it in one run.

**SuperCLUE** ([www.SuperClueAI.com](https://github.com/CLUEbenchmark/SuperCLUE), site refreshed 2026-02-06) is the right public leaderboard — four dimensions (language understanding/generation, professional knowledge, AI agents, safety) across 12 capabilities. It reports **Doubao in the global top tier**. I could not retrieve the current 2026 table (fetch blocked). **Flagged — check it directly.**

# 16. Cross-cutting comparison

| Provider | Strict JSON schema | Batch | Caching | Rate-limit shape |
|---|---|---|---|---|
| **Anthropic** | ✅ constrained decoding + strict tools | ✅ 50%, ≤24h | 0.1× read / 1.25×–2× write; **min prefix 512–4096 by model** | RPM+ITPM+OTPM tiers; **cache excluded from ITPM** |
| **OpenAI** | ✅ strict json_schema | ⚠️ 50% historically, unconfirmed for 5.6 | 90% read, **1.25× write**, 30 min; needs `prompt_cache_key` | Tiers 1–5, numbers unverified |
| **Google** | ✅ `response_schema` | ✅ 50%, 24h | implicit on by default; ~90% off | unverified |
| **xAI** | ⚠️ reported | ❌ none found | explicit cached-input rate | unverified |
| **DeepSeek** | ❌ **JSON mode + app-side validation** | ❌ — **off-peak 50% instead** | automatic, ~90% | **concurrency only: 500 Pro / 2500 Flash; no 429, just latency** |
| **Qwen** | ❌ `json_object` mode | ✅ **50%, exempt from rate limits** | implicit 20% / explicit 10% — **mutually exclusive with batch** | batch bypasses |
| **Moonshot** | ⚠️ unknown | ❌ none found | automatic, 1/10 | **recharge tiers Tier0–Tier5, numbers unpublished** |
| **Zhipu GLM** | ✅ **best of the Chinese labs** (BFCL near GPT-5) | ❌ none found | up to 90% | unverified |
| **MiniMax** | ⚠️ unknown | ❌ none found | read $0.06/M, write $0.375/M | unverified |
| **Doubao** | ⚠️ unknown | ⚠️ unknown | unknown | unknown |
| **OpenRouter** | ⚠️ **degrades to function-calling under fallback routing** | — | passthrough | — |

# 17. QMine-specific analysis

Grounded in the actual repo (`/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine`). Current state at `src/qmine/config.py:211-212`: `deep_model = "claude-opus-5"`, `fast_model = "claude-sonnet-5"`. Tier assignment at `src/qmine/llm/registry.py:46-63` — 7 deep roles (`taxonomy_architect`, `taxonomy_critic`, `referee`, `tree_auditor`, `risk_sentinel`, `reporter`, `maintainer`), 6 fast (`researcher`, `annotator_a`, `annotator_b`, `adversary`, `namer`, `l2_interpreter`), default fast.

**The repo already encodes the single most important caching fact in this whole report**, in the `LLMConfig` docstring: *"The fast tier is Sonnet, not Haiku. Haiku 4.5's minimum cacheable prefix is 4096 tokens, and the taxonomy prefix we resend on every annotation batch sits below that — so on Haiku the prompt cache silently never engages."* **Carry this constraint to every candidate.** The relevant question for each new fast-tier model is not "what does it cost" but "does my ~4K-token taxonomy prefix actually cache."

**Modelled cost, 5,000-row run** — assumptions: ~4,000-token taxonomy prefix resent per annotation call, 25 rows ≈ 800 fresh input tokens, ~1,200 output tokens/batch, 200 calls × 2 annotators = 400 calls → **1.92M input (1.6M cacheable) + 0.48M output** on the fast tier. Deep tier estimated at ~50 calls, 1.5M input, 300K output.

| Fast-tier candidate | Annotation cost |
|---|---|
| Claude Sonnet 5 ($2/$10, cache 0.1×) | **~$5.76** |
| Kimi K3 ($3/$15, cache $0.30) | ~$8.64 |
| GLM-5.2 ($1.40/$4.40, cache $0.26) | **~$2.98** |
| MiniMax M3 ($0.30/$1.20, cache $0.06) | ~$0.77 |
| Qwen3.5-Flash ($0.10/$0.40) + batch 50% | ~$0.19 |
| DeepSeek V4-Flash off-peak | ~$0.09–0.18 |
| Doubao Seed 1.6-Flash ($0.022/$0.219) | ~$0.15 |

**Deep tier on Opus 5: ~$15.** Total all-Anthropic run **≈ $21**.

**The load-bearing conclusion: the deep tier is ~72% of run cost while being ~11% of call volume.** Swapping the fast tier from Sonnet 5 to the cheapest Chinese model saves **~$5.60 per run (27%)**; the fast tier cannot save you more than $5.76 no matter what you do, because that is all it costs. **Optimising the annotators is optimising the wrong end.** If cost is the objective, the lever is the deep tier — and the deep roles (`taxonomy_architect`, `referee`, `tree_auditor`) are exactly the ones whose mistakes propagate into every downstream artifact, so that is also where you least want to economise.

**This mirrors a finding already in memory** — *"a 13x-faster clustering estimator matched the answer by luck (Spearman 0.43)"*. The same failure mode applies here: a cheap annotator that agrees with Sonnet on aggregate label distribution may be agreeing by luck. **Measure per-row agreement (Cohen's κ against the existing referee-adjudicated gold), not aggregate distribution.**

**Recommended experiment, in priority order:**
1. **GLM-5.2 in the fast tier.** Best structured-output reputation among Chinese labs (the binding constraint for 13 Pydantic roles), 1M context, cached input $0.26, ~48% cheaper than Sonnet 5 on annotation. Open weights (`zai-org/GLM-5.2`) mean you can pin a version forever.
2. **Keep the deep tier on `claude-opus-5`.** Strict constrained decoding, the `tree_auditor`'s all-clusters-at-once call needs long context with reliable schema adherence, and refusal handling is already wired.
3. **Qwen3.5-Flash + Batch API for the `namer` role only.** Namers are one call per cluster, 20–200 clusters, blind, latency-insensitive, narrow output schema. Batch is 50% off *and* exempt from rate limits. Remember batch and context-cache discounts are **mutually exclusive** on Qwen — for namers (no big shared prefix) batch wins.
4. **Do not put DeepSeek in any role that needs strict schema** until you have measured parse-failure rate. Its JSON support is app-side-validated, not constrained. Its concurrency-not-429 behaviour is otherwise ideal for a 5,000-call sweep.
5. **Do not route through OpenRouter with fallback enabled** — schema enforcement silently degrades to function-calling.

# 18. Runnable code

Every Chinese provider in this survey is OpenAI-compatible, so one shim covers all of them. Verified against the documented endpoint shapes; **endpoint hostnames are from SDK docs and my own knowledge, NOT re-verified today.**

```python
# qmine_multiprovider.py — OpenAI-compatible shim for the fast tier.
# pip install openai>=1.0 pydantic>=2
import os, json
from typing import Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# base_url, env var for key, and whether the provider does STRICT json_schema
# (True) or only JSON *mode* (False -> you must validate + retry yourself).
PROVIDERS = {
    "zhipu":    ("https://api.z.ai/api/paas/v4",                        "ZHIPU_API_KEY",     True),
    "qwen":     ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1","DASHSCOPE_API_KEY", False),
    "deepseek": ("https://api.deepseek.com",                            "DEEPSEEK_API_KEY",  False),
    "moonshot": ("https://api.moonshot.ai/v1",                          "MOONSHOT_API_KEY",  False),
    "minimax":  ("https://api.minimax.io/v1",                           "MINIMAX_API_KEY",   False),
    "doubao":   ("https://ark.cn-beijing.volces.com/api/v3",            "ARK_API_KEY",       False),
    "deepinfra":("https://api.deepinfra.com/v1/openai",                 "DEEPINFRA_API_KEY", True),
    "groq":     ("https://api.groq.com/openai/v1",                      "GROQ_API_KEY",      True),
    "xai":      ("https://api.x.ai/v1",                                 "XAI_API_KEY",       True),
}

# Model IDs verified 2026-08-18 via web search only.
FAST_TIER = {
    "zhipu":    "glm-5.2",
    "qwen":     "qwen3.5-flash",
    "deepseek": "deepseek-v4-flash",   # deepseek-chat/-reasoner RETIRED 2026-07-24
    "moonshot": "kimi-k2.6",           # kimi-k2.5 + moonshot-v1 RETIRE 2026-08-31
    "minimax":  "MiniMax-M3",
    "doubao":   "doubao-seed-1.6-flash",
}

def client_for(provider: str) -> OpenAI:
    base, env, _ = PROVIDERS[provider]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"{env} is not set; refusing to call {provider}")
    return OpenAI(base_url=base, api_key=key)

def structured(provider: str, model: str, system: str, user: str,
               schema: Type[T], *, retries: int = 3) -> T:
    """Structured call that degrades gracefully from strict schema -> JSON mode.

    QMine's 13 roles all use Pydantic output. Anthropic and OpenAI give
    CONSTRAINED DECODING; DeepSeek/Qwen/Moonshot give JSON *mode* only, so a
    validate-and-retry loop is mandatory, not optional, on those providers.
    """
    _, _, strict_ok = PROVIDERS[provider]
    cli = client_for(provider)
    if strict_ok:
        fmt = {"type": "json_schema",
               "json_schema": {"name": schema.__name__, "strict": True,
                               "schema": schema.model_json_schema()}}
    else:
        fmt = {"type": "json_object"}
        system += ("\n\nReturn ONLY a JSON object matching this schema, no prose:\n"
                   + json.dumps(schema.model_json_schema(), ensure_ascii=False))

    last = None
    for attempt in range(retries):
        r = cli.chat.completions.create(
            model=model, response_format=fmt,
            messages=[{"role": "system", "content": system},   # keep prefix FIRST -> cacheable
                      {"role": "user", "content": user}],
        )
        txt = r.choices[0].message.content
        try:
            return schema.model_validate_json(txt)
        except ValidationError as e:
            last = e
            user += f"\n\nYour previous reply failed validation: {e}. Return valid JSON only."
    raise RuntimeError(f"{provider}/{model} failed schema validation {retries}x: {last}")


# --- QMine annotator smoke test: 25 Chinese queries, one call ---------------
class Row(BaseModel):
    query: str
    intent: str
    confidence: float

class Batch(BaseModel):
    rows: list[Row]

if __name__ == "__main__":
    TAXONOMY = "..."          # ~4000 tokens, IDENTICAL every call -> cache prefix
    queries = ["附近的火锅店", "iphone 17 参数", "怎么退税", ...]  # 25 rows
    out = structured("zhipu", FAST_TIER["zhipu"],
                     system=f"你是查询意图标注员。分类体系:\n{TAXONOMY}",
                     user="\n".join(f"{i+1}. {q}" for i, q in enumerate(queries)),
                     schema=Batch)
    print(out.model_dump_json(indent=2, ensure_ascii=False))
```

**Cache-hygiene rules that apply to every provider above** (from the caching guidance in the bundled skill, and they are provider-agnostic because they follow from prefix matching): the taxonomy must sit **first and byte-identical**; never interpolate `datetime.now()`, a UUID, or a run ID above it; serialise any JSON in the prefix with `sort_keys=True`; do not vary the tool list between annotation calls. If cache reads are zero across repeated calls, a silent invalidator is in the prefix.

# 19. Consolidated conflicts and unverified items

**Hard conflicts (two sources disagree):**
1. **OpenAI Terra/Luna** — $2.50/$15 & $1/$6 vs $2/$12 & $0.20/$1.20.
2. **DeepSeek V4-Pro** — $1.32/$3.96 peak (post-2026-08-16) vs a stale $0.435/$0.87.
3. **DeepSeek V4-Flash cache-hit** — $0.0028/M (2%) contradicts the documented 90% discount ($0.014/M).
4. **GLM-5.2 cached input** — $0.26/M vs "limited-time free."
5. **Qwen3-Max/3.7-Max** — $1.20–$1.25 in / $3.75–$6.00 out, plus length-tiered pricing up to $3.00 in.
6. **MiniMax M2** — $0.255/$1.02 vs $0.260/$1.00.

**Could not verify at all:**
- Every price against a **vendor pricing page or live API** (WebFetch blocked, no API key).
- **All endpoint hostnames** — from SDK docs/prior knowledge, not confirmed today.
- **Anthropic Sonnet 5 permanence** at $2/$10 (single source, contradicts the cached skill table).
- **OpenAI batch discount** for GPT-5.6; **OpenAI rate-limit tier numbers**.
- **Gemini Pro tier newer than 3.1 Pro**; Gemini model ID strings.
- **Moonshot per-tier RPM/TPM** (explicitly unpublished per source).
- **Structured-output support** for MiniMax, Doubao, and xAI.
- **Batch API existence** at Moonshot, Zhipu, MiniMax, Volcengine.
- **Everything about Doubao/Volcengine** beyond price — no caching, batch, schema, or rate-limit data found.
- **Bedrock / Azure / Vertex** — search returned demonstrably stale data (Claude Opus 4.7 at $15/$75, Gemini 1.5). Section is low-confidence throughout.
- **Any benchmark for Chinese search-query intent classification or Chinese short-text understanding** — none exists publicly that I could find. Build your own.
- **Current SuperCLUE 2026 leaderboard table** (site fetch blocked).
- **Llama 4 Maverick pricing.**
- **Mistral exact model IDs and context windows.**


---

## Recommendations

- Do NOT act on any price in this report without re-verifying against the vendor's own pricing page. WebFetch was blocked for every pricing domain (platform.claude.com, platform.openai.com, ai.google.dev, api-docs.deepseek.com) and there is no API key on this machine, so 100% of these numbers come from third-party search summaries dated 2026-08-18, not from a vendor page or a live API call.
- Optimise the DEEP tier, not the annotators. At QMine's shape (5,000 rows, 25/call, 2 annotators) the fast tier costs ~$5.76/run on Sonnet 5 while the deep tier costs ~$15 — 72% of spend from 11% of calls. Switching the fast tier to the cheapest Chinese model saves at most $5.60/run. The fast tier is not where the money is.
- Best single change to test: GLM-5.2 (`glm-5.2`, $1.40/$0.26-cached/$4.40, 1M ctx, open weights `zai-org/GLM-5.2`) in the fast tier — ~48% cheaper than Sonnet 5 on annotation AND the only Chinese lab with independent corroboration of reliable strict JSON/tool-calling (BFCL near GPT-5). For 13 all-Pydantic roles, schema reliability outranks price.
- Three model IDs are dead or dying and must be audited in config NOW: `deepseek-chat` and `deepseek-reasoner` were RETIRED 2026-07-24; `kimi-k2.5` and `moonshot-v1` retire 2026-08-31 (13 days out). Also DeepSeek V4-Pro's price tripled-to-quadrupled on 2026-08-16 (to $1.32/$3.96 peak) — any pre-August DeepSeek cost model is wrong.
- Do not put DeepSeek, Qwen, or Moonshot in a role needing strict schema without first measuring parse-failure rate. They provide JSON *mode* plus application-side validation, NOT constrained decoding like Anthropic/OpenAI/Google. Every one of QMine's 13 roles uses Pydantic structured output — this is the largest adoption risk, larger than price.
- Carry the repo's existing Haiku caching constraint (config.py:211-212 docstring) to every candidate: the question is not 'what does it cost' but 'does my ~4K-token taxonomy prefix actually cache'. Anthropic minimums are non-monotonic (512 on Opus 5 / 1024 on Sonnet 5 / 4096 on Opus 4.6 and Haiku 4.5), and on Qwen the batch discount and context-cache discount are MUTUALLY EXCLUSIVE per request.
- The Chinese-language advantage is real but probably smaller than headlines suggest for YOUR task. C-SimpleQA shows Chinese-native models beating GPT/o1 on Chinese-culture questions, but RAG collapses the gap from 42.4% to 9.3% — and QMine's annotators receive the taxonomy in-context, which is structurally the RAG condition. No public benchmark covers Chinese search-query intent classification. Run the existing blind-panel protocol with a Chinese model in the annotator seat and measure per-row Cohen's kappa against referee-adjudicated gold — not aggregate label distribution (cf. the prior 'cheap estimators lie' finding, Spearman 0.43 by luck).
- If routing via OpenRouter, pin a single model or set `provider: {require_parameters: true}` — with a `models` list or `route: "fallback"` active, structured-output enforcement silently degrades to function-calling because the backend model is unknown at request time.
- Anthropic remains the strongest operational fit for the deep tier regardless of price: constrained decoding + strict tools, 50% batch, cached input excluded from ITPM (80% hit rate on a 2M ITPM limit ≈ 10M input tok/min — bigger than a tier upgrade), and refusal handling already wired in registry.py. Note Bedrock and Vertex do NOT support automatic prompt caching or the Batches API for Claude, so a move there changes your caching strategy.
- Two provider quirks that favour a 5,000-call annotation job: DeepSeek publishes concurrency limits only (500 Pro / 2,500 Flash) and degrades to latency rather than 429s — budget wall-clock, not retries; and Qwen's Batch API is 50% off AND exempt from real-time rate limits, making it the best fit for the blind `namer` role (one call per cluster, latency-insensitive, no shared prefix to cache).

## Unverified

- METHOD-LEVEL, AFFECTS EVERYTHING: WebFetch was blocked by network/enterprise policy for every pricing domain attempted (platform.claude.com, platform.openai.com, ai.google.dev, api-docs.deepseek.com). No API key exists on this machine. Therefore ZERO prices in this report were verified against a vendor pricing page or a live API — all are from WebSearch summaries of third-party aggregator/blog sources run 2026-08-18.
- OpenAI GPT-5.6 Terra and Luna pricing: sources conflict between $2.50/$15 & $1/$6 versus $2/$12 & $0.20/$1.20. I favoured the former because it scales exactly to the independently-reported 2x-input/1.5x-output long-context rates ($5/$22.50, $2/$9), but this is inference, not verification.
- DeepSeek V4-Flash cache-hit input reported at $0.0028/M = 2% of the $0.14 cache-miss rate, which contradicts DeepSeek's documented 90% caching discount (which would be $0.014/M). One of the two figures is wrong.
- Anthropic Claude Sonnet 5 at $2/$10 reported as now-permanent with the 2026-09-01 increase to $3/$15 cancelled — single source, and it contradicts the bundled claude-api skill's cached table (2026-06-24) which lists $3/$15 with $2/$10 as introductory through 2026-08-31.
- Google Gemini lineup is the least clear in the survey: I found Flash models at generations 2.5, 3.5, 3.6, and 3.7 but could NOT confirm any Pro-tier model newer than Gemini 3.1 Pro. Exact API model ID strings for any Gemini model were not verified. Intro pricing on 3.6/3.7 Flash expires 2026-12-31.
- Bedrock / Azure / Vertex section is LOW CONFIDENCE — search returned demonstrably stale data (quoted 'Claude Opus 4.7 at $15/$75' when Anthropic's own rate is $5/$25, and referenced Gemini 1.5 Flash). Do not use any aggregator price from that section.
- All API endpoint hostnames (api.anthropic.com, api.deepseek.com, dashscope-intl.aliyuncs.com, api.z.ai, api.moonshot.ai, api.minimax.io, ark.cn-beijing.volces.com, etc.) come from SDK documentation and prior knowledge — none was re-verified on 2026-08-18.
- Structured-output support is UNKNOWN for MiniMax, Doubao/Volcengine, and only 'reported' (unverified) for xAI. Batch API existence could not be confirmed or refuted for Moonshot, Zhipu, MiniMax, or Volcengine — I found no evidence either way rather than evidence of absence.
- Rate-limit tiers: OpenAI per-tier RPM/TPM not verified; Moonshot explicitly does not publish per-tier RPM/TPM for K3 (one source warns to treat any cited number as unverified); Gemini, xAI, Zhipu, MiniMax, and Doubao rate limits all unknown. Anthropic's tier numbers are from third-party trackers, not platform.claude.com.
- Doubao/Volcengine is the least-documented provider from outside China — I found pricing only, with no data on caching, batch, structured output, or rate limits. Its USD prices are third-party conversions from CNY console pricing and carry FX risk.
- Chinese-language benchmark evidence is thin and generation-lagged: the only concrete C-Eval/CMMLU pair I could verify (92.1 / 91.5) is for Doubao Seed 1.6, not the current Seed 2.0/2.1. The SuperCLUE 68.90 figure is for Qwen2.5-72B. No C-Eval/CMMLU/C-SimpleQA numbers were found for GLM-5.2, Qwen3.8-Max, Kimi K3, DeepSeek V4, or MiniMax M3. The current SuperCLUE 2026 leaderboard could not be retrieved (fetch blocked).
- No public benchmark exists (that I could find) for Chinese search-query intent classification or Chinese short-text/colloquial-query understanding — the task closest to QMine's actual workload. A targeted search for this returned nothing on-point.
- MiniMax M3's $0.30/$1.20 is described as a 'Permanent 50% off' promotional rate. Promotional pricing described as permanent is exactly the category most likely to revert.
- Qwen3-Max/3.7-Max uses prompt-length-tiered pricing (reported input range $1.20 to $3.00 across its context window) — the exact tier boundaries were not verified, making cost modelling for long prompts unreliable.
- Mistral: no exact model ID strings, context windows, batch discount, or caching details verified — only headline Large 3 pricing at $0.50/$1.50. Llama 4 Maverick appears in leaderboards but no current price was found.
- My QMine cost model (~$5.76 fast tier / ~$15 deep tier per 5,000-row run) rests on assumed token shapes: ~4,000-token taxonomy prefix, ~800 fresh input and ~1,200 output tokens per 25-row batch, ~50 deep-tier calls at 1.5M input / 300K output. These are estimates from the stated pipeline shape, NOT measured from an actual run — the 72%/11% deep-tier conclusion should be re-derived from real ledger data before acting on it.

## Sources

- https://www.aipricing.guru/anthropic-pricing/ — Anthropic Claude API pricing (Fable/Opus/Sonnet), accessed 2026-08-18 via WebSearch summary
- https://benchlm.ai/anthropic/api-pricing — Claude API Pricing August 2026, accessed 2026-08-18
- https://platform.claude.com/docs/en/about-claude/pricing — Anthropic official pricing page (FETCH BLOCKED by network policy; appeared in search results only)
- https://www.cloudzero.com/blog/claude-pricing/ — Claude pricing 2026, accessed 2026-08-18
- https://en.wikipedia.org/wiki/GPT-5.6 — GPT-5.6 Sol/Terra/Luna model IDs, context, pricing, accessed 2026-08-18
- https://simonwillison.net/2026/Jul/9/gpt-5-6/ — GPT-5.6 family release notes 2026-07-09, accessed 2026-08-18
- https://developers.openai.com/api/docs/pricing — OpenAI official pricing (FETCH BLOCKED; search result only)
- https://developers.openai.com/api/docs/guides/prompt-caching — OpenAI prompt caching, prompt_cache_key requirement
- https://openrouter.ai/openai/gpt-5.6-terra — GPT-5.6 Terra pricing/benchmarks, accessed 2026-08-18
- https://aws.amazon.com/about-aws/whats-new/2026/08/gpt-sol-terra-luna-long-context-bedrock/ — GPT-5.6 1M context on Bedrock (Aug 2026)
- https://aws.amazon.com/blogs/machine-learning/introducing-explicit-prompt-caching-for-openai-gpt-5-6-models-on-amazon-bedrock/ — explicit prompt caching for GPT-5.6 on Bedrock
- https://ai.google.dev/gemini-api/docs/latest-model — What's new in Gemini 3.7 Flash (thinking_level, structured outputs)
- https://deepmind.google/models/model-cards/gemini-3-7-flash/ — Gemini 3.7 Flash model card, released 2026-08-13
- https://ai.google.dev/gemini-api/docs/pricing — Gemini Developer API pricing (FETCH BLOCKED; search result only)
- https://www.verdent.ai/guides/gemini-3-1-pro-pricing — Gemini 3.1 Pro pricing, 2M context, caching, thinking tokens
- https://benchlm.ai/google/api-pricing — Gemini API pricing August 2026
- https://www.metacto.com/blogs/the-true-cost-of-google-gemini-a-guide-to-api-pricing-and-integration — Gemini 3.5 Flash / 3.1 Pro / 2.5 Lite pricing
- https://www.aipricing.guru/xai-pricing/ — xAI Grok API pricing (Grok 4.6 released 2026-08-12)
- https://mem0.ai/blog/xai-grok-api-pricing — Grok API pricing per model
- https://costgoat.com/pricing/grok-api — Grok API pricing calculator, Aug 2026
- https://deepseek.ai/pricing — DeepSeek V4-Flash & V4-Pro per-token costs 2026
- https://www.androidheadlines.com/2026/08/deepseek-v4-pro-ai-model-quadruples-prices-remains-cheaper.html — DeepSeek V4 Pro price increase, August 2026
- https://www.aipricing.guru/deepseek-pricing/ — DeepSeek V4 peak & off-peak pricing (effective 2026-08-16 16:00 UTC)
- https://benchlm.ai/deepseek/api-pricing — DeepSeek API pricing August 2026
- https://www.morphllm.com/deepseek-v4 — DeepSeek V4 1.6T MoE, 1M context, architecture and pricing
- https://deepseek.ai/deepseek-v4 — DeepSeek V4-Pro (1.6T) / V4-Flash (284B) complete guide
- https://openrouter.ai/deepseek/deepseek-v4-flash-0731 — DeepSeek V4 Flash 0731 model page
- https://chat-deep.ai/docs/api-rate-limits/ — DeepSeek API rate limits: V4 concurrency (500 Pro / 2500 Flash)
- https://chat-deep.ai/docs/deepseek-context-caching/ — DeepSeek context caching cost guide
- https://www.nxcode.io/resources/news/deepseek-api-pricing-complete-guide-2026 — DeepSeek pricing, cache, rate limits
- https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek — LLM API rate limits by tier 2026 (Anthropic Start/Build/Scale RPM/ITPM/OTPM)
- https://standardcompute.com/rate-limits/anthropic — Anthropic API rate limits & usage tiers 2026
- https://ofox.ai/blog/qwen-3-8-max-price-context-window-api-access-open-weights-2026/ — Qwen 3.8 Max price, context, API access
- https://www.qwencloud.com/models/qwen3.8-max — Qwen3.8-Max model ID and specs (GA 2026-08-03)
- https://benchlm.ai/alibaba/api-pricing — Qwen API pricing August 2026 (Qwen3.5 Plus & Flash)
- https://felloai.com/qwen-pricing/ — Qwen pricing 2026, Qwen3.8-Max API costs
- https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output — Qwen structured output (response_format json_object)
- https://www.alibabacloud.com/help/en/model-studio/context-cache — Qwen Context Cache (implicit 20% / explicit 10%)
- https://www.alibabacloud.com/help/en/model-studio/model-pricing — Alibaba Cloud Model Studio model pricing + batch 50%
- https://benchlm.ai/moonshot/api-pricing — Kimi API pricing August 2026 (K3 at $3/$15)
- https://www.nxcode.io/resources/news/kimi-k2-5-pricing-plans-api-costs-2026 — Kimi K2.5 pricing, limits, K3 upgrade, retirement notice
- https://trilogyai.substack.com/p/kimi-k3-is-live-pricing-benchmarks — Kimi K3 pricing and benchmarks
- https://www.verdent.ai/guides/agents/kimi-k3-api-guide — Kimi K3 API guide: pricing, context, caching
- https://www.layer3labs.io/guides/kimi-k3-limits — Kimi K3 context window and rate limits
- https://docs.z.ai/guides/llm/glm-5.2 — GLM-5.2 official overview (Z.AI developer docs)
- https://www.siliconflow.com/blog/glm-5-2-api-guide — GLM-5.2 pricing, model ID, 1M context
- https://openrouter.ai/z-ai/glm-5.2 — GLM 5.2 API pricing & benchmarks
- https://venturebeat.com/technology/z-ais-open-weights-glm-5-2-beats-gpt-5-5-on-multiple-long-horizon-coding-benchmarks-for-1-6th-the-cost — GLM-5.2 open weights and benchmarks
- https://www.aipricing.guru/z-ai-pricing/ — Z.ai API pricing $1.40/$4.40 + GLM Coding Plan tiers
- https://openrouter.ai/minimax/minimax-m3 — MiniMax M3 API pricing & benchmarks
- https://codersera.com/blog/minimax-m3-developer-guide/ — MiniMax M3 developer guide: MSA, 1M context, pricing
- https://benchlm.ai/models/minimax-m3 — MiniMax M3 benchmarks, pricing, speed (August 2026)
- https://pricepertoken.com/pricing-page/provider/minimax — MiniMax all-model pricing 2026
- https://www.llmreference.com/model/seed-2.1-pro/volcengine — Doubao Seed 2.1 Pro on Volcengine: pricing (¥6/¥30 per MTok), API, specs
- https://cloudprice.net/models/bytedance-doubao-seed-2-pro — Doubao Seed 2 Pro pricing & specs (256K ctx, 128K out)
- https://dev.to/tokenmixai/doubao-api-setup-2026-19-bytedance-models-0022m-floor-python-in-5-min-2akn — Doubao API setup, 19 models, $0.022/M floor
- https://china-llm.com/provider/bytedance-doubao — ByteDance Doubao (Volcengine) pricing and products
- https://benchlm.ai/mistral/api-pricing — Mistral API pricing August 2026 (Large 3 & Medium 3.5)
- https://www.cloudzero.com/blog/mistral-api-pricing/ — Mistral API pricing 2026
- https://amnic.com/blogs/llama-api-pricing — Llama API per-token costs by provider 2026
- https://www.aipricing.guru/meta-pricing/ — Llama API pricing across 5 providers
- https://blog.alephant.io/openmodels-vs-openrouter-together-fireworks-deepinfra-2026/ — OpenRouter/Together/Fireworks/DeepInfra comparison with Chinese-model pricing
- https://openrouter.ai/blog/insights/the-open-weight-models-that-matter-june-2026/ — OpenRouter open-weight model landscape June 2026
- https://pricepertoken.com/endpoints/compare/deepinfra-vs-openrouter — DeepInfra vs OpenRouter pricing comparison 2026
- https://openrouter.ai/docs/guides/features/structured-outputs — OpenRouter structured outputs (json_schema, streaming)
- https://python.useinstructor.com/integrations/openrouter/ — Instructor + OpenRouter: json_schema vs functionCalling fallback under multi-model routing
- https://www.eesel.ai/blog/groq-pricing — Groq pricing 2026, every model and free tier
- https://verticalapi.com/vs/groq-vs-cerebras/ — Groq vs Cerebras speed and price comparison 2026
- https://artificialanalysis.ai/providers/groq — Groq intelligence, performance & price
- https://deepinfra.com/blog/deepseek-v4-pro-pricing-guide-2026-providers-cost-analysis — DeepSeek V4 Pro pricing across providers
- https://arxiv.org/html/2411.07140 — Chinese SimpleQA: A Chinese Factuality Evaluation for LLMs (CC subtopic advantage; RAG collapses gap 42.4%->9.3%)
- https://github.com/CLUEbenchmark/SuperCLUE — SuperCLUE Chinese LLM benchmark (site www.SuperClueAI.com updated 2026-02-06)
- https://explore.n1n.ai/blog/chinese-ai-model-benchmarks-2026-deepseek-glm-kimi-qwen-2026-06-28 — Chinese AI model benchmarks 2026 (Doubao Seed 1.6 CMMLU 91.5 / C-Eval 92.1)
- https://www.index.dev/blog/chinese-ai-models — Top 6 Chinese AI models 2026 with C-Eval/CMMLU data
- https://checkaimodels.com/en/articles/china-ai-models-landscape-2026/ — DeepSeek vs Qwen vs Kimi vs GLM vs MiniMax; GLM tool-calling/structured-output reliability
- https://nextfuture.io.vn/blog/2026-chinese-llm-stack-qwen-deepseek-minimax-kimi-glm-compared — 2026 Chinese LLM stack comparison, structured output guidance
- https://tech-insider.org/best-open-source-llm-2026/ — Best open-source LLM 2026 rankings (Elo: GLM-5 1451, Kimi K2.5 1447, DeepSeek-R1 1436)
- https://www.turingpost.com/p/chinesemodels — Kimi K2 vs DeepSeek-R1 vs Qwen3 vs GLM-4.5 2026 guide
- https://www.alphamatch.ai/blog/open-source-llm-comparison-blog-2026 — Open-source LLM revolution 2026, Chinese model benchmarks
- https://dl.acm.org/doi/10.1145/3769086 — ACM TMIS: comparison of US and China LLMs (GPT-4 Turbo leads English, Chinese LLMs lead Chinese contexts)
- https://k21academy.com/ai-ml/amazon-bedrock-vs-azure-openai-vs-google-vertex-ai — Bedrock vs Azure vs Vertex comparison 2026 (LOW CONFIDENCE - contains stale model data)
- https://www.swfte.com/blog/aws-bedrock-guide-2026 — AWS Bedrock 2026 models and pricing (LOW CONFIDENCE - stale figures)
- Local repo: /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/config.py:198-233 — LLMConfig deep_model/fast_model, Haiku caching-minimum rationale
- Local repo: /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/llm/registry.py:43-63,326-330 — ROLE_TIER deep/fast assignment and _NO_TEMPERATURE model list
- Bundled claude-api skill (cached 2026-06-24): shared/models.md, shared/prompt-caching.md, shared/platform-availability.md, shared/model-migration.md — Anthropic model table, per-model cache minimums, Bedrock/Vertex/Foundry feature availability