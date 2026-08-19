# Live Model Catalogues & Auto-Refresh

> Gathered 2026-08-18. Facts marked verified were fetched live; the model-landscape
> dossier in particular flags its prices as secondary-source and unconfirmed —
> which is why the running system fetches prices from a live catalogue rather
> than embedding any table from this document.

All network facts below were verified live from this machine on **2026-08-18, 06:45–07:15 UTC** (the `date:` response header on OpenRouter read `Tue, 18 Aug 2026 06:51:56 GMT`, confirming the clock). Nothing here is from memory. Where I could not reach a source, I say so rather than guessing.

Two caveats on my own environment, because they bound the evidence:
- `models.dev` is **unreachable from here** — `curl` fails with exit 35 (TLS connect error) on `https://models.dev/api.json` and every other path, and `WebFetch` refuses the domain ("unable to verify if domain is safe"). I therefore **cannot confirm or deny** its schema. I have not described it.
- `groq.com` and `mistral.ai` also returned HTTP 000 (network-blocked here), so their auth requirement is inferred from the pattern, not measured.

---

## 1. OpenRouter — `GET https://openrouter.ai/api/v1/models`

**Auth: none required to LIST.** Verified: HTTP 200, 678,422 bytes, no `Authorization` header sent. **413 models across 59 providers.** Every one of the 413 carries pricing; 19 are zero-priced free-tier variants.

Response headers are the most useful thing here for cache design:
```
cache-control: public, max-age=300, stale-while-revalidate=3600, stale-if-error=3600
cf-cache-status: HIT
age: 38
```
That is a 5-minute freshness window with a 1-hour stale-if-error grace — OpenRouter is explicitly telling you a 5-minute TTL and that serving stale on failure is sanctioned.

Real trimmed record (`anthropic/claude-opus-5-fast`, verbatim):
```json
{
  "id": "anthropic/claude-opus-5-fast",
  "canonical_slug": "anthropic/claude-opus-5-fast-20260723",
  "name": "Claude Opus 5 (Fast)",
  "created": 1784912546,
  "context_length": 1000000,
  "architecture": {
    "modality": "text+image+file->text",
    "input_modalities": ["text", "image", "file"],
    "output_modalities": ["text"],
    "tokenizer": "Claude"
  },
  "pricing": {
    "prompt": "0.00001", "completion": "0.00005", "web_search": "0.01",
    "input_cache_read": "0.000001", "input_cache_write": "0.0000125",
    "input_cache_write_1h": "0.00002"
  },
  "top_provider": { "context_length": 1000000, "max_completion_tokens": 128000, "is_moderated": true },
  "supported_parameters": ["include_reasoning","max_tokens","reasoning","reasoning_effort",
                           "response_format","stop","structured_outputs","tool_choice","tools","verbosity"],
  "reasoning": { "mandatory": false, "default_enabled": true,
                 "supported_efforts": ["max","xhigh","high","medium","low"], "default_effort": "high" }
}
```
Prices are **strings, per token** — `"0.00001"` = $10/Mtok. Parse with `Decimal`, not `float`.

`structured_outputs` appears in `supported_parameters` for **335 of 413** models (`response_format` for 356). Since every QMine role uses Pydantic structured output, that field is the natural eligibility filter.

**Sub-endpoints I probed:**
- `GET /api/v1/models/user` → **HTTP 401** `{"error":{"message":"Unauthorized","code":401}}`. Needs a key; it returns the key's permitted subset.
- `GET /api/v1/models/{id}/endpoints` → **200, no auth**, and it is richer than the list: per-provider pricing and live health. For `anthropic/claude-opus-5` it returns separate entries for `Amazon Bedrock` and `Claude Platform on AWS`, each with `pricing`, `max_completion_tokens: 128000`, `supports_implicit_caching`, and **`uptime_last_30m: 99.919`, `uptime_last_5m: 100`, `uptime_last_1d: 99.926`**, plus `latency_last_30m` / `throughput_last_30m` (null for this model). This is the only free source of provider reliability telemetry I found.

**Licence/terms:** I could not fetch OpenRouter's docs or ToS — `WebFetch` blocked the domain the same way it blocked models.dev. So I can state the *technical* access facts (measured) but **not** the contractual terms. Treat redistribution as unverified.

---

## 2. LiteLLM — `model_prices_and_context_window.json`

`https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`

**Auth: none.** 1,747,806 bytes. **3,040 top-level keys (3,039 real + a `sample_spec` documentation stub — skip that key).** **124 providers**, **145 distinct field names**. Serves an `ETag` (`"fbd03e5e…"`) and `max-age=300`, so conditional GETs work.

Coverage that matters:
- `mode` distribution: `chat` 2330, `image_generation` 209, **`embedding` 124**, `responses` 85, `rerank` 25. OpenRouter is a chat gateway and covers embeddings barely at all — QMine does clustering, so if it ever prices an embedding model, only LiteLLM has it.
- `input_cost_per_token` on 2,556 entries; `supports_response_schema` true on 910; `supports_prompt_caching` true on 660; **`deprecation_date` on 334**.

Real record (`claude-opus-4-5`, verbatim):
```json
{
  "input_cost_per_token": 5e-06, "output_cost_per_token": 2.5e-05,
  "cache_creation_input_token_cost": 6.25e-06, "cache_read_input_token_cost": 5e-07,
  "cache_creation_input_token_cost_above_1hr": 1e-05,
  "litellm_provider": "anthropic", "mode": "chat",
  "max_input_tokens": 200000, "max_output_tokens": 64000, "max_tokens": 64000,
  "supports_function_calling": true, "supports_tool_choice": true,
  "supports_response_schema": true, "supports_native_structured_output": true,
  "supports_prompt_caching": true, "supports_reasoning": true, "supports_vision": true,
  "supports_pdf_input": true, "supports_computer_use": true, "supports_assistant_prefill": true,
  "prompt_cache_min_tokens": 4096
}
```
Prices are **JSON numbers, per token**.

**Commit cadence — measured, not estimated.** `GET https://api.github.com/repos/BerriAI/litellm/commits?path=model_prices_and_context_window.json&per_page=30` returned 30 commits spanning **2026-08-13 → 2026-08-17** — roughly **six commits per day on this one file**. Sample messages: `fix(gemini): price gemini 3.6 flash at Google's introductory rates on every serv…` (2026-08-17T18:42:57Z), `fix(model_map): flag native structured outputs on Anthropic-direct claude-sonnet…`, `feat(xai): day-0 pricing for grok-4.6`. Day-0 pricing is a stated practice.

**Licence:** GitHub's API reports `NOASSERTION` (56,598 stars, `pushed_at: 2026-08-18T06:15:30Z`) because `LICENSE` opens with a dual-license preamble. The actual text: *"All content that resides under the `enterprise/` directory… is licensed under the license defined in `enterprise/LICENSE`. Content outside of the above mentioned directories… is available under the MIT license."* The price file is at repo root → **MIT**. This is the only source in this investigation with a clean, verified, redistribution-permitting licence.

---

## 3. Provider-native list endpoints — measured

| Provider | Endpoint | Result (2026-08-18) | Pricing? |
|---|---|---|---|
| OpenAI | `GET /v1/models` | **401** | No |
| Anthropic | `GET /v1/models` | **401** | No |
| DeepSeek | `GET /models` | **401** | No |
| Together | `GET /v1/models` | **401** | No |
| Fireworks | `GET /inference/v1/models` | **401** | No |
| Google | `GET /v1beta/models` | **403** | No |
| Groq / Mistral | — | **000** (blocked here) | Unverified |

**Confirmed: not one provider-native list endpoint returns pricing.** Pricing is a commercial fact that lives on marketing pages, not in the API. Any "stay current on price automatically" design must therefore depend on a third-party aggregator — there is no first-party alternative.

**But Anthropic's does return authoritative capability metadata**, and this turns out to be the crux. Per the bundled `claude-api` reference (`shared/models.md`), `GET /v1/models/{id}` returns `id`, `display_name`, `max_input_tokens`, `max_tokens`, and — since March 2026 — a nested `capabilities` dict with `supported` booleans at each leaf:
```python
caps = client.models.retrieve("claude-opus-5").capabilities
caps["structured_outputs"]["supported"]
caps["thinking"]["types"]["adaptive"]["supported"]
caps["effort"]["max"]["supported"]
```
Note there is **no `context_window` field** — it is `max_input_tokens`. `capabilities` is an untyped dict: bracket access, not attribute access.

---

## 4. Artificial Analysis, and the rest

- **Artificial Analysis** — `GET https://artificialanalysis.ai/api/v2/data/llms/models` → **401 `{"error":"API key is required"}`**. Free key via an Insights Platform account; publishes Intelligence/Coding/Agentic/Math/Multilingual indices, input/output/blended/cache price per Mtok by model *and provider*, plus median and percentile output speed, TTFT, and end-to-end latency. **Licence is the blocker: the free tier is "exploration and internal workflows" only; redistribution requires a commercial package.** Fine as an internal input, not shippable in a run manifest you publish.
- **simonw/llm-prices** — there is **no top-level `prices.json`** (that path 404s). The data is per-vendor under `data/`: `anthropic.json` (4,568 B), `openai.json` (12,767 B), `google.json`, `deepseek.json`, `mistral.json`, `qwen.json`, `moonshot-ai.json`, `minimax.json`, `amazon.json`, `meta-ai.json`. Schema is **price *history*** — genuinely distinctive:
  ```json
  { "vendor": "anthropic", "models": [
      { "id": "claude-3.5-haiku", "name": "Claude 3.5 Haiku",
        "price_history": [ { "input": 0.8, "output": 4, "from_date": null, "to_date": null, "input_cached": null } ] } ] }
  ```
  Units are **$/Mtok** (not per-token — do not mix them up with the other two). Cadence ~weekly (2026-08-13, 08-12, 08-10, 08-06, 07-30, 07-24). Pricing only, no capability flags. Its 2026-08-10 commit — *"Sonnet 5 price is no longer going up in September"* — is a good freshness signal.
- **Helicone** `https://www.helicone.ai/api/llm-costs` → HTTP 000 from here; unverified.
- **HuggingFace** `https://huggingface.co/api/models?inference_provider=all` → **200, no auth**, returns `id`, `downloads`, `likes`, `tags`, `pipeline_tag`, `createdAt`. **No pricing, no context length.** Useful for open-weights discovery, useless for cost.
- **Vercel ai-sdk registry** — guessed path 404'd; I did not locate a public JSON endpoint.

---

## 5. The finding that decides the architecture

I scored both machine-readable catalogs against **authoritative Anthropic behavior** (the bundled `claude-api` skill reference, which is the source of truth for Anthropic API semantics), on the two capability facts QMine actually hardcodes.

**Test A — does the model reject `temperature`?** (Anthropic-direct returns HTTP 400 on Fable 5, Opus 5, Sonnet 5, Opus 4.8, Opus 4.7.)

| Model | Truth: rejects? | LiteLLM `supports_sampling_params` | OpenRouter `supported_parameters` |
|---|---|---|---|
| claude-opus-5 | **yes** | ✅ False | ❌ says temperature supported |
| claude-opus-4-8 | **yes** | ✅ False | ❌ says supported |
| claude-mythos-5 | **yes** | ✅ False | ❌ absent from catalog |
| claude-sonnet-5 | **yes** | ✅ False | ✅ absent → correct |
| claude-fable-5, claude-opus-4-7 | yes | ✅ | ✅ |
| haiku-4-5, sonnet-4-6, opus-4-5, opus-4-6 | no | ✅ | ✅ |

**LiteLLM 10/10. OpenRouter 7/10.**

**Test B — `prompt_cache_min_tokens`** against the authoritative table (512 / 1024 / 2048 / 4096 tiers): **LiteLLM 12/12 exact.** OpenRouter **has no such field at all**.

The reason OpenRouter is wrong is structural, not a bug: `supported_parameters` describes **what the OpenRouter gateway accepts and normalizes**, not what the upstream provider's own API accepts. A gateway that silently drops `temperature` for you will honestly report it as "supported." **Gateway capability metadata is not a substitute for provider-direct capability metadata.** This is the single most important design constraint in this report.

### Two live bugs this surfaced in QMine

**(a) `_NO_TEMPERATURE` is missing the fast model.** In `/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/llm/registry.py`:
```python
_NO_TEMPERATURE = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-mythos-5")
```
`claude-sonnet-5` is absent — and `config.py` sets `fast_model: str = "claude-sonnet-5"`. Sonnet 5 rejects sampling params with a 400. Currently masked only because `LLMConfig.temperature` defaults to `None`; the moment anyone sets a temperature, **every fast-tier call — annotators (600–5000 rows), namers, adversary, researchers, l2_interpreter — 400s.** That is the overwhelming majority of the pipeline's calls. Both independent catalogs flagged this; the hand-written tuple did not. The docstring at `config.py:200` says "Opus 5 and Fable 5 removed the parameter" and omits Sonnet 5.

**(b) `estimated_cost_usd` is wrong in both directions.** In `src/qmine/llm/budget.py`:
```python
def estimated_cost_usd(self, in_rate: float = 3.0, out_rate: float = 15.0) -> float:
    return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000
```
Flat $3/$15 for a run whose deep tier is `claude-opus-5` (**$5/$25**, both catalogs agree) and fast tier is `claude-sonnet-5` (**$2/$10**, both agree). Deep-tier spend is **under**-reported ~40%; fast-tier **over**-reported ~50%. It also ignores prompt caching entirely, though `UsageLedger` tracks `cache_hits` and Anthropic cache reads bill at ~0.1×. The fix needs no new data: `by_role` already holds per-role `input_tokens`/`output_tokens`, and `ROLE_TIER` maps role → tier, so a correct per-tier cost is computable from state that already exists.

Worth noting on Sonnet 5's price: the `claude-api` skill's cached table (dated 2026-06-24) says **$3/$15 with a $2/$10 intro through 2026-08-31**; simonw's 2026-08-10 commit says the increase was cancelled; and **both live catalogs report $2/$10 today.** A hardcoded constant would have been wrong twice in three months. This is precisely the case for a live catalog.

---

## 6. Recommended refresh architecture

### The load-bearing decision: refresh is a *build* step, never a *run* step

A run must never touch the network for catalog data. Fetching at run time makes cost accounting depend on when you ran, silently couples reproducibility to a CDN, and breaks the offline stand-in this machine already relies on. Instead: a **vendored, SHA-pinned snapshot in git**, refreshed by an explicit command.

This is not theoretical. I caught drift **inside this session**. The `raw.githubusercontent.com/.../main/` copy I fetched at 06:51 differed from the copy at commit `94a29e07085dd9d8f2269ee93e899ef0e374cdc7` — same 3,040 keys, no models added or removed, but **`gemini-3.6-flash` had every price field halved** (`input_cost_per_token` 1.5e-06 → 7.5e-07; output 7.5e-06 → 3.75e-06; same for the `_flex`, `_priority`, `_batches`, and cache-read variants, across all three aliases `gemini-3.6-flash`, `gemini/gemini-3.6-flash`, `vertex_ai/gemini-3.6-flash`). That is the 2026-08-17 "introductory rates" commit; the CDN was serving a pre-fix copy under `max-age=300`. **A 2× cost error, invisible, purely from CDN staleness on the `main` ref.** Pinning by SHA is the only deterministic read.

### Source roles

| Role | Source | Why |
|---|---|---|
| **Primary — price + capability** | LiteLLM, pinned by commit SHA | MIT-licensed, 10/10 and 12/12 against authoritative Anthropic behavior, ~6 commits/day, 124 providers, covers embeddings, has `deprecation_date` |
| **Cross-check — price only** | OpenRouter `/api/v1/models` | Independent, no auth, all 413 priced. **Never** used for provider-direct capability |
| **Authority — capability, when a key exists** | Anthropic `GET /v1/models/{id}` | First-party truth for `capabilities`, `max_input_tokens`, `max_tokens`. No pricing |
| **Local override** | `configs/model_overrides.yaml` | Escape hatch, wins over everything |
| **Not depended on** | Artificial Analysis (licence), simonw (pricing-only, weekly), HF (no prices), models.dev (unverifiable from here) | |

Authority ladder for a capability lookup: `local override` → `Anthropic /v1/models` (only if a key is present) → `LiteLLM snapshot` → fail loudly. OpenRouter never enters this ladder.

### Layout

```
configs/
  catalog.lock.json          # pin: sha, sha256, fetched_at, source urls
  catalog.snapshot.json      # the pruned catalog, committed to git
  model_overrides.yaml       # hand-held facts that beat every source
```

Prune before committing — QMine needs a handful of models, not 3,040. That keeps the snapshot diffable in review, which is the point: **a price change should show up as a reviewable diff in a pull request, not as a silent behavior change at 3am.**

### `qmine catalog refresh`

```python
# src/qmine/llm/catalog.py
from __future__ import annotations
import hashlib, json, urllib.request
from decimal import Decimal
from pathlib import Path

LITELLM_REPO = "BerriAI/litellm"
LITELLM_PATH = "model_prices_and_context_window.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
UA = {"User-Agent": "qmine-catalog/1.0"}

def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()

def resolve_head_sha() -> str:
    """Resolve main->SHA, then fetch BY SHA. Never read /main/ for values:
    raw.githubusercontent serves it with max-age=300 and can lag a price fix."""
    api = f"https://api.github.com/repos/{LITELLM_REPO}/commits?path={LITELLM_PATH}&per_page=1"
    return json.loads(_get(api))[0]["sha"]

def fetch_litellm(sha: str) -> tuple[dict, str]:
    raw = _get(f"https://raw.githubusercontent.com/{LITELLM_REPO}/{sha}/{LITELLM_PATH}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def fetch_openrouter() -> dict[str, dict]:
    data = json.loads(_get(OPENROUTER_URL))["data"]
    # anthropic/claude-opus-4.5 -> claude-opus-4-5, to join against LiteLLM keys
    return {m["id"].split("/")[-1].replace(":batch", "").replace(".", "-"): m for m in data}

def refresh(models: list[str], out_dir: Path, tol: Decimal = Decimal("0.02")) -> dict:
    sha = resolve_head_sha()
    ll, digest = fetch_litellm(sha)
    try:
        orx = fetch_openrouter()
    except Exception as exc:            # cross-check is advisory, never fatal
        orx, warn = {}, [f"openrouter unreachable: {exc}"]
    else:
        warn = []

    snap = {}
    for m in models:
        if m not in ll:
            raise KeyError(f"{m!r} absent from LiteLLM@{sha[:8]} — check for a rename/retirement")
        e = ll[m]
        snap[m] = {
            "input_per_mtok":  float(Decimal(str(e["input_cost_per_token"])) * 1_000_000),
            "output_per_mtok": float(Decimal(str(e["output_cost_per_token"])) * 1_000_000),
            "cache_read_per_mtok": float(Decimal(str(e.get("cache_read_input_token_cost", 0))) * 1_000_000),
            "max_input_tokens": e.get("max_input_tokens"),
            "max_output_tokens": e.get("max_output_tokens"),
            # False means "rejects sampling params" (Opus 5, Sonnet 5, Fable 5...).
            # Absent means unconstrained -> default True.
            "accepts_sampling_params": e.get("supports_sampling_params", True) is not False,
            "supports_response_schema": bool(e.get("supports_response_schema")),
            "prompt_cache_min_tokens": e.get("prompt_cache_min_tokens"),
            "deprecation_date": e.get("deprecation_date"),
        }
        o = orx.get(m)
        if o:   # price-only cross-check; capability flags are deliberately ignored
            for fld, key in (("input_per_mtok", "prompt"), ("output_per_mtok", "completion")):
                a, b = Decimal(str(snap[m][fld])), Decimal(o["pricing"][key]) * 1_000_000
                if b and abs(a - b) / b > tol:
                    warn.append(f"PRICE DISAGREEMENT {m}.{fld}: litellm={a} openrouter={b}")

    lock = {"litellm_sha": sha, "litellm_sha256": digest,
            "fetched_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "models": sorted(models), "warnings": warn}
    (out_dir / "catalog.snapshot.json").write_text(json.dumps(snap, indent=2, sort_keys=True))
    (out_dir / "catalog.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True))
    return lock
```

Refresh weekly via the existing `Makefile`, or on a `catalog:` CI schedule that opens a PR. **Fail the refresh loudly on a missing model ID** — a model vanishing from LiteLLM is exactly the signal you want (rename, retirement) and must not be papered over with a default.

### Run-time load — offline by construction

```python
def load_catalog(cfg_dir: Path, overrides: dict | None = None) -> dict:
    """Pure local read. No network, ever. Works air-gapped and in CI unchanged."""
    snap = json.loads((cfg_dir / "catalog.snapshot.json").read_text())
    lock = json.loads((cfg_dir / "catalog.lock.json").read_text())
    for model, patch in (overrides or {}).items():
        snap.setdefault(model, {}).update(patch)   # local override wins over every source
    return {"models": snap, "provenance": lock}
```

Degradation is trivial because there is nothing to degrade: the file is in the repo. The offline heuristic provider needs no catalog at all — `model_name()` already returns `"offline-heuristic"`, so price lookup should short-circuit to zero cost, and `provenance_note()` continues to carry its existing disclaimer.

### Wiring into the three hardcoded tables

1. **Replace `_NO_TEMPERATURE`** with the catalog flag, closing bug (a):
   ```python
   def _accepts_temperature(self, model: str) -> bool:
       entry = self.catalog["models"].get(model)
       if entry is None:
           raise LLMUnavailable(f"{model!r} not in pinned catalog; run `qmine catalog refresh`")
       return entry["accepts_sampling_params"]
   ```
   Refusing to guess for an unknown model is deliberate — a silent `True` is what produces a 400 on 5,000 annotation calls.

2. **Replace the `$3/$15` default**, closing bug (b) — per-tier, using data the ledger already has:
   ```python
   def estimated_cost_usd(self, prices: dict[str, dict], role_tier, model_for_tier) -> float:
       total = 0.0
       for role, u in self.by_role.items():
           p = prices[model_for_tier(role_tier.get(role, "fast"))]
           total += (u["input_tokens"] * p["input_per_mtok"]
                     + u["output_tokens"] * p["output_per_mtok"]) / 1_000_000
       return round(total, 4)
   ```

3. **Make the fast-tier choice checkable instead of commented.** `config.py`'s docstring justifies Sonnet-over-Haiku by Haiku 4.5's 4096-token minimum cacheable prefix. The catalog carries that as data and I verified it 12/12: `claude-haiku-4-5` = 4096, `claude-sonnet-5` = 1024, `claude-opus-5` = 512. So assert it:
   ```python
   assert cat[cfg.fast_model]["prompt_cache_min_tokens"] <= taxonomy_prefix_tokens, (
       f"{cfg.fast_model} needs a {cat[cfg.fast_model]['prompt_cache_min_tokens']}-token "
       f"prefix to cache; taxonomy prefix is {taxonomy_prefix_tokens} — bulk labelling "
       f"would silently pay full price on every call")
   ```
   The reasoning that currently lives in a comment becomes a test that fires if anyone swaps the fast model.

### Principle 8 — reproducibility

Reproducibility here means a run's **model choice and the prices it was costed at** are both replayable even after the catalog moves. The pin does that:

- `catalog.lock.json` carries `litellm_sha` **and** `litellm_sha256` of the exact bytes. Anyone can re-fetch `https://raw.githubusercontent.com/BerriAI/litellm/<sha>/model_prices_and_context_window.json` and verify the digest. Git commit SHAs are immutable; `main` is not.
- Extend `ModelRegistry.usage()` — which already emits `provider`, `deep_model`, `fast_model`, `estimated_cost_usd` — with `catalog_sha`, `catalog_sha256`, and the resolved per-model price/capability block actually used. The manifest then answers "why did this run cost that, and why was temperature omitted?" without any network access.
- **Do not add the catalog to the LLM cache key.** `hash_params` in `complete()` keys on `role/provider/model/system/user/schema/temperature`. A price refresh must not invalidate cached judgments — prices affect accounting, not the judgment. Model *identity* is already in the key, which is the part that legitimately should invalidate.
- Add `catalog.lock.json` to the run's input-hash set so a changed catalog is visible as a run-level provenance difference while leaving per-call replay intact.

### One caution given this repo's history

The memory note *"Resume safety is not checkpointing — nodes reading process memory failed silently on resume"* applies directly. Do not stash the catalog in a module-level global or on a long-lived object that a resumed run reconstructs differently. Load it from `catalog.snapshot.json` on every node entry, or thread it through LangGraph state. A catalog that is present on the first pass and absent on resume reproduces exactly that bug class — and it would fail *silently*, because a missing price yields a plausible-looking cost of zero rather than an exception.


---

## Recommendations

- PRIMARY = LiteLLM `model_prices_and_context_window.json`, pinned by git commit SHA (not `main`). It is MIT-licensed (verified in LICENSE text; GitHub's API misreports NOASSERTION due to the dual-license preamble), commits ~6x/day (30 commits over 2026-08-13..17), covers 3,039 models / 124 providers including 124 embedding models, and carries `deprecation_date`. Decisively, it scored 10/10 on sampling-parameter support and 12/12 on `prompt_cache_min_tokens` against authoritative Anthropic behavior.
- NEVER use OpenRouter's `supported_parameters` for provider-direct capability decisions. It scored only 7/10 — it reports `temperature` as supported for claude-opus-5 and claude-opus-4-8, which return HTTP 400 on Anthropic-direct. This is structural, not a bug: the field describes what the OpenRouter gateway accepts and normalizes away, not what the upstream API accepts. Use OpenRouter as an independent PRICE cross-check only (413 models, all priced, no auth, `stale-if-error=3600`).
- Fetch by commit SHA, never by `main`. I caught live drift mid-session: the `/main/` CDN copy had gemini-3.6-flash priced at exactly 2x the pinned-SHA copy across every price field (input 1.5e-06 vs 7.5e-07), because raw.githubusercontent serves `max-age=300` and lagged the 2026-08-17 'introductory rates' fix. A 2x cost error, silent, from CDN staleness alone.
- FIX A LIVE BUG: `_NO_TEMPERATURE` in src/qmine/llm/registry.py omits `claude-sonnet-5`, which is the configured `fast_model`. Sonnet 5 rejects sampling params with a 400. Currently masked only because `LLMConfig.temperature` defaults to None — set any temperature and every fast-tier call (annotators over 600-5000 rows, namers, adversary) fails. Both independent catalogs flagged this; the hand-written tuple did not.
- FIX A SECOND LIVE BUG: `UsageLedger.estimated_cost_usd` hardcodes $3/$15 per Mtok, but the configured models are claude-opus-5 ($5/$25) and claude-sonnet-5 ($2/$10) — both catalogs agree. Deep-tier spend is under-reported ~40%, fast-tier over-reported ~50%, and prompt caching (~0.1x on reads) is ignored despite `cache_hits` being tracked. Per-tier cost is computable from data already in `by_role` + `ROLE_TIER`.
- Make refresh a BUILD step, never a RUN step: vendor `catalog.snapshot.json` + `catalog.lock.json` into git via `qmine catalog refresh`. Runs do a pure local file read, so offline capability is structural rather than a fallback path, and every price change arrives as a reviewable PR diff instead of a silent 3am behavior change.
- For Principle 8, record `litellm_sha` AND `litellm_sha256` plus the resolved per-model price/capability block in the run manifest, extending the existing `ModelRegistry.usage()`. Anyone can re-fetch that exact SHA and verify the digest. Critically, do NOT add the catalog to the LLM response cache key — a price refresh must not invalidate cached judgments; model identity is already in `hash_params`.
- Turn the fast-tier design rationale into an assertion. `config.py` justifies Sonnet-over-Haiku in a comment via Haiku 4.5's 4096-token minimum cacheable prefix; LiteLLM carries `prompt_cache_min_tokens` as data (haiku-4-5=4096, sonnet-5=1024, opus-5=512, verified 12/12). Assert it against the taxonomy prefix so the reasoning fires as a test if anyone swaps the fast model.
- Use Anthropic's `GET /v1/models/{id}` as the capability AUTHORITY when a key is present (it returns `max_input_tokens`, `max_tokens`, and a nested `capabilities` dict with `supported` leaves — note there is no `context_window` field). Confirmed no provider-native endpoint returns pricing, so a third-party aggregator is unavoidable for cost. Authority ladder: local override -> Anthropic /v1/models -> LiteLLM snapshot -> fail loudly.
- Load the catalog on every node entry or thread it through LangGraph state — never a module-level global. Per this repo's own `qmine-resume-safety-bug-class` memory, nodes reading process memory failed silently on resume; a missing price degrades to a plausible cost of zero rather than an exception, reproducing exactly that bug class.

## Unverified

- models.dev is UNVERIFIABLE from this machine — curl fails with exit 35 (TLS connect error) on every path, and WebFetch refuses the domain on policy grounds. I deliberately did not describe its schema or coverage rather than answer from memory. Assignment item 3 is genuinely unanswered; re-run from an unrestricted network before depending on it.
- OpenRouter's licence and terms for programmatic use are UNVERIFIED. WebFetch blocked openrouter.ai the same way it blocked models.dev, so I could confirm the technical access facts (measured directly via curl) but not the contractual ones. Do not assume redistribution rights for its pricing data.
- Helicone's cost API (https://www.helicone.ai/api/llm-costs) returned HTTP 000 from this environment — network-blocked, not confirmed absent. Groq and Mistral list endpoints likewise returned 000; their 401-auth requirement is inferred from the pattern of the six providers I did measure, not observed.
- No API key is present on this machine (consistent with the stored `qmine-no-api-key` memory), so all provider-native endpoints were confirmed only up to their auth challenge (401/403). I could not observe an actual Anthropic `/v1/models` response body; its schema is taken from the bundled claude-api skill reference, which is authoritative for Anthropic semantics but is not a live measurement.
- The claim that OpenRouter's `supported_parameters` reflects gateway normalization rather than upstream behavior is my inference from the 3/10 mismatch pattern (all three errors in the same direction: reporting a parameter as supported where Anthropic-direct 400s). It fits the evidence and the gateway architecture, but I did not confirm it against OpenRouter's documentation, which I could not fetch.
- Sonnet 5's price is genuinely in flux and sources disagree: the claude-api cached table (2026-06-24) says $3/$15 with a $2/$10 intro through 2026-08-31; simonw's 2026-08-10 commit says the increase was cancelled; both live catalogs report $2/$10 today. I used $2/$10 as current, but the post-2026-08-31 rate should be re-verified — this is precisely the volatility that motivates the whole design.
- Artificial Analysis's licensing terms come from search-result summaries of its data-api pages, not from the pages themselves (the API returned 401 without a key). The free-tier/commercial-redistribution split is directionally reliable but should be read in the actual terms before any dependency.
- I did not exercise the proposed refresh code end-to-end against the QMine codebase — no test run, no import into the real registry. The catalog field names and values it reads are all verified live, but integration (LangGraph state threading, CLI wiring, Makefile target) is designed, not validated.

## Sources

- https://openrouter.ai/api/v1/models — fetched 2026-08-18 06:51:56 GMT, HTTP 200 unauthenticated, 678,422 bytes, 413 models / 59 providers; cache-control: public, max-age=300, stale-while-revalidate=3600, stale-if-error=3600
- https://openrouter.ai/api/v1/models/user — HTTP 401 {"error":{"message":"Unauthorized","code":401}}, verified 2026-08-18
- https://openrouter.ai/api/v1/models/anthropic/claude-opus-5/endpoints — HTTP 200 unauthenticated; per-provider pricing plus uptime_last_30m/5m/1d, latency and throughput telemetry, verified 2026-08-18
- https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json — 1,747,806 bytes, 3,040 keys (3,039 real + sample_spec), 124 providers, 145 distinct fields; ETag + max-age=300, verified 2026-08-18
- https://raw.githubusercontent.com/BerriAI/litellm/94a29e07085dd9d8f2269ee93e899ef0e374cdc7/model_prices_and_context_window.json — SHA-pinned fetch; differed from the /main/ CDN copy by a 2x gemini-3.6-flash price correction, demonstrating CDN staleness, verified 2026-08-18
- https://api.github.com/repos/BerriAI/litellm/commits?path=model_prices_and_context_window.json&per_page=30 — 30 commits spanning 2026-08-13 to 2026-08-17T18:42:57Z (~6/day), verified 2026-08-18
- https://raw.githubusercontent.com/BerriAI/litellm/main/LICENSE — MIT for content outside enterprise/; GitHub API reports NOASSERTION due to the dual-license preamble; 56,598 stars, pushed_at 2026-08-18T06:15:30Z
- https://api.github.com/repos/simonw/llm-prices/contents/data — per-vendor files (anthropic.json 4,568 B, openai.json 12,767 B, google.json, deepseek.json, mistral.json, qwen.json, moonshot-ai.json, minimax.json, amazon.json, meta-ai.json); no top-level prices.json (404), verified 2026-08-18
- https://raw.githubusercontent.com/simonw/llm-prices/main/data/anthropic.json — price_history schema with from_date/to_date/input_cached, units $/Mtok; commit cadence ~weekly (2026-08-13, 08-12, 08-10, 08-06, 07-30, 07-24)
- https://artificialanalysis.ai/api/v2/data/llms/models — HTTP 401 {"error":"API key is required"}, verified 2026-08-18
- https://artificialanalysis.ai/data-api and /data-api/docs — free tier for internal use, commercial package required for redistribution (via search results; pages not directly fetched)
- https://api.openai.com/v1/models (401), https://api.anthropic.com/v1/models (401), https://api.deepseek.com/models (401), https://api.together.xyz/v1/models (401), https://api.fireworks.ai/inference/v1/models (401), https://generativelanguage.googleapis.com/v1beta/models (403) — all probed 2026-08-18; none return pricing
- https://huggingface.co/api/models?inference_provider=all — HTTP 200 unauthenticated; returns id/downloads/likes/tags/pipeline_tag, no pricing or context length, verified 2026-08-18
- https://models.dev/api.json — UNREACHABLE from this environment (curl exit 35 TLS connect error; WebFetch domain verification refused), 2026-08-18
- https://www.helicone.ai/api/llm-costs — HTTP 000 (network-blocked from this environment), 2026-08-18
- Bundled claude-api skill reference (shared/models.md, shared/model-migration.md, shared/prompt-caching.md, shared/tool-use-concepts.md) — authoritative Anthropic model IDs, pricing, /v1/models response shape, sampling-parameter removal on Fable 5 / Opus 5 / Sonnet 5 / Opus 4.8 / Opus 4.7, and the prompt-cache minimum table (512/1024/2048/4096) used to score both catalogs
- /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/llm/registry.py — ROLE_TIER table, _NO_TEMPERATURE tuple (missing claude-sonnet-5), hash_params cache key, provenance_note
- /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/llm/budget.py — UsageLedger.estimated_cost_usd hardcoded in_rate=3.0 / out_rate=15.0
- /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/config.py — LLMConfig with deep_model=claude-opus-5, fast_model=claude-sonnet-5, temperature=None, and the Haiku-4.5 4096-token cache-prefix rationale at lines 192-222