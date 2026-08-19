# Routing research — adversarial completeness pass

# Adversarial Completeness Review of 6 QMine Extension Dossiers

**Verification basis.** Everything below was checked by me from this machine on **2026-08-18, 07:07–07:20 UTC** (`date -u` confirmed). Unlike most of the dossiers, the network was **reachable** for me: OpenRouter returned HTTP 200 / 678,422 bytes, and `raw.githubusercontent.com` and `api.github.com` both served content. So where dossiers said "WebFetch blocked, trusting search snippets," I re-checked against live machine-readable catalogs and, in several cases, **the dossiers are wrong**.

Two catalogs used as ground truth:
- **LiteLLM** `model_prices_and_context_window.json` pinned at commit `94a29e07085dd9d8f2269ee93e899ef0e374cdc7` (latest commit touching that path, dated **2026-08-17T18:42:57Z**), sha256 `782e2ec4578c5bb74d5abff37ed507c666c2cddd7c654de20e709c336c558dc0`, 1,747,833 bytes, **3,040 keys**.
- **OpenRouter** `/api/v1/models` (413 models) and `/api/v1/models/{id}/endpoints`.

---

## (a) CONTRADICTIONS BETWEEN DOSSIERS — RESOLVED BY LIVE CHECK

### A1. Claude Sonnet 5 price — `routing` is wrong, and its entire cost model is inflated

Three dossiers, three answers, all dated 2026-08-18. Live result:

| Source | Input $/MTok | Output $/MTok |
|---|---|---|
| LiteLLM @ `94a29e07` | **2.00** | **10.00** |
| OpenRouter `anthropic/claude-sonnet-5` | **2.00** | **10.00** |
| `model-landscape` dossier | 2.00 | 10.00 ✅ |
| `live-catalogs` dossier | 2.00 | 10.00 ✅ |
| **`routing` dossier** | **3.00** | **15.00** ❌ |

LiteLLM also carries `cache_read_input_token_cost = 2e-07` (= $0.20/MTok = exactly 0.1× input), internally consistent with $2.00.

**Consequence:** the `routing` dossier states "Every Sonnet-5 figure I quote uses the $3/$15 list rate." Its fast tier is therefore **inflated ~50%**. That propagates into its headline table (`QMine today $16.03`, `+Batch $11.40`), its per-role attribution (`annotator_a $5.08`), and its central conclusion that "routing is the wrong lever." The *direction* of that conclusion probably survives — a cheaper fast tier makes routing even less attractive — but **no number in that dossier's §1 should be quoted**.

### A2. GPT-5.6 Terra/Luna — `model-landscape` picked the wrong side of its own conflict

`model-landscape` flagged a conflict and explicitly chose $2.50/$15 (Terra) and $1.00/$6.00 (Luna), reasoning that these "scale exactly to the independently-reported 2×/1.5× long-context rates." That inference was wrong. LiteLLM native keys:

```
gpt-5.6        in=  5.000  out= 30.000  cacheR=0.5000  ctx=1,050,000
gpt-5.6-sol    in=  5.000  out= 30.000  cacheR=0.5000  ctx=1,050,000
gpt-5.6-terra  in=  2.000  out= 12.000  cacheR=0.2000  ctx=1,050,000
gpt-5.6-luna   in=  0.200  out=  1.200  cacheR=0.0200  ctx=1,050,000
```

Corroborated by Azure rows (`azure/gpt-5.6-luna` $0.20/$1.20; `azure/eu/gpt-5.6-terra` $2.20/$13.20 = exactly 1.1× the $2/$12 base, the standard EU uplift). The `routing` dossier's $2/$12 for Terra was **right**.

**Luna is 5× cheaper than `model-landscape` claimed** ($0.20 vs $1.00 input; $1.20 vs $6.00 output). Any "cheapest viable fan-out model" analysis that excluded Luna on price excluded it on a 5× error — and Luna carries `supports_response_schema: True` plus a 1.05M context, making it a genuine annotator candidate that the dossiers dismissed.

### A3. DeepSeek — `model-landscape`'s two most alarming claims are unsupported

`model-landscape` asserted (i) `deepseek-chat` and `deepseek-reasoner` were **RETIRED 2026-07-24**, and (ii) V4-Pro pricing **tripled-to-quadrupled** to $1.32/$3.96 effective 2026-08-16. It escalated (i) to a recommendation: "Three model IDs are dead or dying and must be audited in config NOW."

LiteLLM at a commit dated **2026-08-17** — i.e. *after* both claimed events:

```
deepseek-chat               in=0.2800 out=0.4200  deprecation_date=None
deepseek-reasoner           in=0.2800 out=0.4200  deprecation_date=None
deepseek/deepseek-v4-pro    in=0.4350 out=0.8700  deprecation_date=None
deepseek/deepseek-v4-flash  in=0.1400 out=0.2800  deprecation_date=None
```

`deepseek/deepseek-v4-pro` at **$0.435/$0.87** matches the `routing` dossier exactly, not `model-landscape`. And this is not a catalog that ignores lifecycle: it carries **335 `deprecation_date` entries**, including 8 Moonshot/Kimi ones, and its recent commit log shows active lifecycle work — `litellm_model_registry_lifecycle_audit` and `litellm_model_map_deprecation_refresh` (both merged 2026-08-15), plus `fix(model_prices): revert unverified Gemini deprecation dates` and `fix(model_prices): drop xai/grok-4.6-latest, xAI does not serve that alias`.

This cannot *disprove* the retirement (a catalog can lag first-party), but a maintainer who audited deprecations on 08-15 and did not mark `deepseek-chat` is meaningful counter-evidence. **Same applies to `kimi-k2.5` "retires 2026-08-31"** — absent from LiteLLM's Moonshot deprecation set. Act on neither claim; this is precisely the class of fact a live catalog exists to settle.

### A4. Gemini 3.6 Flash — I reproduced the CDN drift live, and it is still present

The `live-catalogs` dossier's most important structural claim. I diffed `/main/` against the pinned SHA:

```
sha keys 3040   main keys 3040   differing keys: 3
gemini-3.6-flash          sha in=7.5e-07  main in=1.5e-06   (2x)
gemini/gemini-3.6-flash   sha out=3.75e-06 main out=7.5e-06 (2x)
vertex_ai/gemini-3.6-flash              (same 2x)
```

`94a29e07` is the newest commit for that path (`fix(gemini): price gemini 3.6 flash at Google's introductory rates`) yet `/main/` still served the pre-fix values ~13 hours later. **Confirmed independently. Exactly 3 keys differ, all the same model, all exactly 2×.** This is the single best-supported design argument in the entire dossier set: **fetch by commit SHA, never by `main`.**

### A5. OpenRouter capability data — right conclusion, wrong diagnosis

`live-catalogs` scored OpenRouter 7/10 and attributed the failure to "gateway normalization." I confirmed the symptom and found the actual mechanism.

Model-level `/api/v1/models` for `anthropic/claude-opus-5` **does** list `temperature`. But `/api/v1/models/anthropic/claude-opus-5/endpoints` shows per-provider:

```
Amazon Bedrock          temperature: False
Claude Platform on AWS  temperature: False
Anthropic               temperature: False
Google                  temperature: False
Azure                   temperature: True   <-- only these
Azure                   temperature: True
UNION has temperature: True
```

So model-level `supported_parameters` is a **union across heterogeneous endpoints**, and two Azure endpoints poison it. Practical upshot is *better* than the dossier concluded: **the `/endpoints` sub-resource is per-provider and correct** — the same free, unauthenticated resource it praised for uptime telemetry (I measured `uptime_last_30m` 99.97 / 99.95 / 100). Rule: never read model-level `supported_parameters` for capability; `/endpoints` is usable.

---

## (b) UNVERIFIED CLAIMS I VERIFIED — the ones that would break code

### B1. `_NO_TEMPERATURE` bug — CONFIRMED, and it is real

`src/qmine/llm/registry.py:327`:
```python
_NO_TEMPERATURE = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-mythos-5")
```
`claude-sonnet-5` absent; `config.py` sets `fast_model = "claude-sonnet-5"`. LiteLLM independently: `claude-sonnet-5 → supports_sampling_params: False`. Full verified table (LiteLLM, `False` = rejects):

| model | sampling | cache min | in/out $ |
|---|---|---|---|
| claude-opus-5 | False | **512** | 5.00 / 25.00 |
| claude-fable-5 | False | **512** | 10.00 / 50.00 |
| claude-sonnet-5 | **False** ← missing from tuple | 1024 | 2.00 / 10.00 |
| claude-opus-4-8 | False | 1024 | 5.00 / 25.00 |
| claude-opus-4-7 | False | 2048 | 5.00 / 25.00 |
| claude-opus-4-6 | *(absent → allowed)* | 4096 | 5.00 / 25.00 |
| claude-haiku-4-5 | *(absent → allowed)* | **4096** | 1.00 / 5.00 |
| claude-sonnet-4-6 | *(absent → allowed)* | 1024 | 3.00 / 15.00 |

Masked only because `LLMConfig.temperature` defaults to `None`. Set any temperature and every fast-tier call 400s. **Real, fix it.**

### B2. `estimated_cost_usd` bug — CONFIRMED, but the proposed fix would introduce a new bug

`src/qmine/llm/budget.py:88` hardcodes `in_rate=3.0, out_rate=15.0`. Wrong for both configured models.

**However**, `live-catalogs` also wrote: "prompt caching (~0.1× on reads) is ignored despite `cache_hits` being tracked." I traced `cache_hits` and this is **wrong**. In `registry.py`:

```python
if self.cfg.cache_llm_calls:
    hit = self._cache_get(cache_key)
    if hit is not None:
        self.ledger.record(role, cached=True)   # <-- QMine's LOCAL response cache
        return schema.model_validate(hit) if schema else hit
```

`cache_hits` counts QMine's own **local response-dedup cache**, which costs **$0**, not 0.1×. Pricing those hits at 0.1× would *invent* spend that never occurred. Anyone implementing the fix must not conflate the two.

### B3. There is NO prompt caching and NO batch API in QMine — this invalidates several headline recommendations

```
grep -rn "cache_control" src/qmine/   -> no matches
grep -rn "batches|batch_api" src/qmine/ -> only an unrelated prose comment
```

Messages are built plainly: `[SystemMessage(content=system), HumanMessage(content=user)]`.

Two consequences the dossiers missed:

1. **The Haiku cache-minimum rationale currently justifies nothing.** `config.py`'s docstring ("on Haiku the prompt cache silently never engages") is the empirical anchor that `model-landscape` and `routing` both elevated into a *general design law* — "carry this constraint to every candidate," "make cache-prefix feasibility a HARD eligibility gate." But **no prompt cache engages on any model today**, because `cache_control` is never sent. The constraint is real *conditional on implementing caching*, and the sequencing is backwards: implement caching, measure the real prefix, then gate.

2. **`routing`'s #1 recommendation — "Turn on the Batch API first, −29%/−39%" — is not a switch.** No batch code exists, and LangChain's `ChatAnthropic` does not expose Messages Batches. This is a build project with out-of-order results keyed by `custom_id`. `unknown-domain` likewise budgets "$0.46 batched" / "$1.10 batched" against capability that does not exist.

### B4. Other claims — all CONFIRMED

- **bge tokenizer bug** (`multilingual-embeddings`): from `.hf` cache — `bge-small-zh-v1.5` `do_lower_case = False`; `bge-base-zh-v1.5` and `bge-large-zh-v1.5` both `True`. Real inconsistency.
- **`ops/language.py` exists** (15,516 bytes) with `classify_row:85`, `alignment_probe:162`, `minority_dilution:201`, `char_ngram_for:242`. Confirms the dossier's "not greenfield" warning.
- **TradingAgents staleness** — fetched `model_catalog.py` live: `openai.deep` = `gpt-5.5`, `gpt-5.4`, `gpt-5.2`, `gpt-5.5-pro`; **no `gpt-5.6-*`**. `anthropic.deep` = fable-5, opus-4-8, sonnet-5, opus-4-7; **no `claude-opus-5`**. Custom-ID row present only for `deepseek` and `ollama` of the 8 providers I tested — `openai`/`anthropic`/`google`/`xai` are closed dropdowns. All confirmed.
- **`models.dev` unreachable** — `curl` exit 35 (TLS). Helicone: HTTP 000. `live-catalogs` was honest.
- **No API key** — `api.anthropic.com/v1/models` → 401; env has `ANTHROPIC_BASE_URL` set but no `ANTHROPIC_API_KEY`.

### B5. Things NO dossier found

**`src/qmine/llm/requirements.py` already exists (10,420 bytes) and is DEAD CODE.** It defines `RoleRequirement` with `reasoning: Literal["light","standard","strong","frontier"]`, `blast_radius: Literal["contained","phase","run"]`, `min_context_tokens`, `needs_structured_output`, `typical_calls`, `output_tokens_per_call`, `multilingual_critical`, and a `cost_sensitivity` property — covering **all 14 roles**. Its docstring states the exact thesis `routing` and `tradingagents-providers` spent pages arguing for. `grep` across `src/` and `tests/` for `RoleRequirement|ROLE_REQUIREMENTS|llm.requirements` returns **nothing outside the file itself**. Six dossiers proposed building infrastructure that is already written and merely unwired.

**There are 14 roles, not 13.** `ROLE_TIER` includes `domain_scout` (deep), also present in `agents/roles.py:521`. The task brief and `routing`'s "7 deep roles" both omit it — an off-by-one in every cost model.

**Only two providers are implemented.** `registry.get()` branches on `is_offline` and `elif self.provider == "anthropic"` — nothing else. `config.py` declares `provider: Literal["anthropic","openai","mock","auto"]`, so `"openai"` is *configurable but unimplemented*. Every multi-provider proposal presupposes a provider layer that does not exist.

**Offline-safety bug in credential detection:**
```python
return bool(os.environ.get("ANTHROPIC_API_KEY")
         or os.environ.get("ANTHROPIC_AUTH_TOKEN")
         or os.environ.get("LANGSMITH_API_KEY"))
```
A `LANGSMITH_API_KEY`-only environment resolves the provider to `"anthropic"` and then fails at call time with an auth error, instead of degrading to the offline stand-in. Tracing/observability creds are not inference creds.

**The tree moved during the research.** Nine source files carry 2026-08-18 mtimes: `config.py`, `llm/registry.py`, `llm/requirements.py`, `agents/roles.py`, `ops/templates.py`, `ops/language.py`, and three graph nodes. No git repo, so no history. Dossier line-number citations may already be stale, and `requirements.py`'s provenance is ambiguous — it may have been authored mid-session rather than pre-existing.

---

## (c) STALENESS RISK — what breaks in 3 months

**Rots fastest (weeks):** every price. The Sonnet-5 saga is self-demonstrating — three dossiers, three prices, all "verified 2026-08-18," one wrong by 50%. Also: Gemini 3.6/3.7 Flash introductory rates ($0.75/$3.75) expire **2026-12-31** and revert to $1.50/$7.50 — a 2× step change already visible as the `/main`-vs-SHA diff. MiniMax M3's "permanent 50% off" and Qwen's length-tiered pricing are in the same category.

**Rots medium (months):** model IDs. TradingAgents is the controlled experiment: GPT-5.6 went GA 2026-07-09, their last commit was 2026-07-18 — **nine days later** — and the catalog still shipped `gpt-5.5`. Hand-maintained catalogs rot on *every* axis simultaneously: stale IDs, stale labels ("Gemini 3.5 Flash — Latest"), stale docstrings, orphaned entries (`claude-mythos-preview`), and stale `-preview` suffixes.

**Rots slowest — and matters most:** capability facts. `supports_sampling_params`, `prompt_cache_min_tokens`, `supports_response_schema`. **These are the ones that break code rather than budgets.** A wrong price yields a wrong number in a report; a wrong sampling-param flag yields HTTP 400 across thousands of annotator calls. Note the cache minimums are **non-monotonic** across generations (512 → 1024 → 2048 → 4096 → 4096) — no heuristic reproduces them, so they must be data.

**Design implications, in priority order:**

1. **SHA-pinned LiteLLM snapshot as a build step**, never a run step. Justified not by the dossier's argument but by drift I reproduced live.
2. **Wire `deprecation_date` (335 entries) into the refresh as a hard failure.** It is the built-in early-warning for exactly the DeepSeek/Kimi question this review could not settle. Also fail on a *missing* key — a model vanishing is signal (rename/retirement), and defaulting past it is how you get a silent zero.
3. **Prefer capability over price in the snapshot.** Cache minimums and sampling flags change rarely and break loudly; prices change weekly and break quietly.
4. **Do not hand-maintain a dropdown.** Every provider gets a custom-ID escape hatch, and validation must be **advisory** (TradingAgents' `warn_if_unknown_model` with a `RuntimeWarning`), not the hard `LLMUnavailable` raise `live-catalogs` proposed — see (d).

---

## (d) OFFLINE / NO-KEY BEHAVIOUR — where the proposals break

Confirmed baseline: no API key present; `_resolve_provider("auto")` → `"offline"`; `model_name(tier)` → `"offline-heuristic"`; `OfflineHeuristicModel` serves every role.

**Breaks 1 — price lookup KeyErrors offline.** `live-catalogs` proposes:
```python
p = prices[model_for_tier(role_tier.get(role, "fast"))]
```
Offline, `model_name()` returns the literal string `"offline-heuristic"`, which is in no catalog → `KeyError`. Its own `load_catalog` has no offline branch. **Required:** short-circuit to zero cost when `is_offline`, and keep `provenance_note()` carrying the existing disclaimer. Its warning about the `qmine-resume-safety-bug-class` applies to its own design: a missing price degrades to a plausible-looking **$0.00**, not an exception.

**Breaks 2 — the strict `_accepts_temperature` is hostile.** Raising `LLMUnavailable` for any model not in the pinned snapshot converts "user set a brand-new model ID via env" into a hard stop, and makes the snapshot a gate on trying anything new. Advisory warning + explicit override is the right failure mode.

**Breaks 3 — offline template certification is the most dangerous interaction in the set.** `unknown-domain` proposes LLM pair-adjudication that stamps `trusted=True` on template groups (80 pairs, Wilson LB₉₅ ≥ 0.90). Offline, those calls hit `synthesize()`. The stand-in is seeded by role and prompt and **will return schema-valid agreement judgments** — so an offline run could emit certified-trusted template families whose certification is a hash function. Since those groups then judge the alpha sweep and `template_fragmentation`, this silently corrupts the metric that replaced silhouette. **Certification must be structurally impossible offline** — not "passed" by the stand-in. Same applies to `domain_scout` vertical inference and risk screening.

**Breaks 4 — batch/caching recommendations presuppose absent machinery** (B3). Additionally, per the dossiers' own reading, Batches and automatic prompt caching are unsupported for Claude on Bedrock and Vertex, so a provider move changes the strategy again.

**Breaks 5 — embeddings bake-off needs network.** The `.hf` cache holds only `bge-{small,base,large}-zh-v1.5` (plus e5-small per the dossier). The recommended `gte-multilingual-base`, `Qwen3-Embedding-0.6B`, and `bge-m3` are **not cached** → the P3 bake-off fails offline. `lingua` also adds a ~90 MB dependency.

**Breaks 6 — the `LANGSMITH_API_KEY` credential bug** (B5) is precisely an offline-degradation failure: it prevents the fallback to offline in an environment that has tracing but no inference.

---

## (e) REMAINING RISKS

1. **Sequencing inversion on caching.** Two dossiers make cache-prefix feasibility a *hard eligibility gate* while the pipeline sends no `cache_control`. Gating on a constraint that binds nothing is how you reject a good model for a reason that does not apply. Order: implement caching → measure the real taxonomy prefix (nobody measured the "~4,000 token" figure; it is assumed in both cost models) → then gate.
2. **Every cost model here is unmeasured.** `routing` used a 50%-wrong Sonnet price; `model-landscape` states its token shapes are "estimates from the stated pipeline shape, NOT measured"; both omit `domain_scout`. The `qmine-cheap-estimators-lie` precedent (Spearman 0.43 by luck) applies directly — re-derive from real ledger data before acting on the "72% of spend in the deep tier" conclusion.
3. **The κ-independence critique is correct but currently unactionable.** Two annotators on one model measure within-model sampling variance, not inter-rater reliability. But fixing it needs cross-provider routing, which does not exist (B5), and changing annotator models invalidates the κ ≥ 0.90 blocking gate, which was calibrated same-model. Do not change annotators without re-deriving the threshold.
4. **Synthetic-only calibration.** `unknown-domain`'s MCI ≥ 0.80 threshold and `multilingual`'s per-language-centring results both come from synthetic corpora the authors generated. `multilingual` is admirably explicit that its ARI-intent = 1.000 is a 48-query probe and that its 3,968-query corpus clusters by topic, not intent. Treat both as hypotheses.
5. **Only 2 of 6 recommended encoders were actually tested** (`bge-small/base-zh`, `e5-small`); the three recommended multilingual models were never run.
6. **Single-source dependency on LiteLLM.** MIT licence per the dossier's read of `LICENSE` (I did not re-read it); GitHub's API reports `NOASSERTION`. It is community-maintained and demonstrably fallible — it may well be the party that is wrong about DeepSeek. Keep OpenRouter `/endpoints` as an independent cross-check, and keep the local override file that beats both.
7. **OpenRouter ToS unverified** for redistributing pricing into a published run manifest.
8. **Dossiers cite a moving tree** (9 files changed 2026-08-18, no git history). Re-confirm every line-number citation before editing.

---

## Verification script (re-runnable)

```python
#!/usr/bin/env python3
"""Re-verify the load-bearing facts. Network required; exits non-zero on drift."""
import json, hashlib, urllib.request, sys

UA = {"User-Agent": "qmine-verify/1.0"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read()

sha = json.loads(get("https://api.github.com/repos/BerriAI/litellm/commits"
                     "?path=model_prices_and_context_window.json&per_page=1"))[0]["sha"]
raw = get(f"https://raw.githubusercontent.com/BerriAI/litellm/{sha}/"
          "model_prices_and_context_window.json")
print(f"litellm sha={sha}\nsha256={hashlib.sha256(raw).hexdigest()}")
cat = json.loads(raw)

# Facts this review established on 2026-08-18. Failure => the landscape moved.
EXPECT = {
    "claude-opus-5":   (5.0, 25.0, 512,  False),
    "claude-sonnet-5": (2.0, 10.0, 1024, False),   # NOT 3/15; NOT sampling-capable
    "claude-haiku-4-5":(1.0,  5.0, 4096, None),
    "gpt-5.6-terra":   (2.0, 12.0, None, None),    # NOT 2.50/15
    "gpt-5.6-luna":    (0.2,  1.2, None, None),    # NOT 1.00/6.00
    "deepseek/deepseek-v4-pro": (0.435, 0.87, None, None),  # NOT 1.32/3.96
}
bad = 0
for mid, (i, o, mp, sp) in EXPECT.items():
    e = cat.get(mid)
    if e is None:
        print(f"GONE   {mid}  <- rename or retirement"); bad += 1; continue
    got = (round(e.get("input_cost_per_token", 0) * 1e6, 4),
           round(e.get("output_cost_per_token", 0) * 1e6, 4),
           e.get("prompt_cache_min_tokens"), e.get("supports_sampling_params"))
    ok = got[0] == i and got[1] == o and got[2] == mp and got[3] == sp
    print(("OK     " if ok else "DRIFT  ") + f"{mid:28} {got}")
    bad += not ok
    if e.get("deprecation_date"):
        print(f"  !! deprecation_date={e['deprecation_date']}"); bad += 1

# /main/ is NOT safe to read: reproduce the CDN staleness check.
main = json.loads(get("https://raw.githubusercontent.com/BerriAI/litellm/main/"
                      "model_prices_and_context_window.json"))
drift = [k for k in set(cat) | set(main) if cat.get(k) != main.get(k)]
print(f"\n/main/ differs from sha on {len(drift)} keys: {sorted(drift)[:5]}")
sys.exit(1 if bad else 0)
```

When I ran the equivalent today: all six `EXPECT` rows matched, no deprecation dates fired, and `/main/` differed on exactly 3 keys (all `gemini-3.6-flash` aliases, all exactly 2×).


## Recommendations

- Do not act on the `routing` dossier's cost figures. Verified live on 2026-08-18 against both LiteLLM (SHA 94a29e07, dated 2026-08-17) and OpenRouter: claude-sonnet-5 is $2.00/$10.00 per MTok, not the $3.00/$15.00 that dossier used throughout. Its fast tier is inflated ~50%, which propagates into every number in its headline table ($16.03 today, $11.40 batched, annotator_a $5.08) and its per-role attribution.
- Drop the `model-landscape` GPT-5.6 and DeepSeek claims. LiteLLM native keys give gpt-5.6-terra $2.00/$12.00 and gpt-5.6-luna $0.20/$1.20 (that dossier explicitly chose $2.50/$15 and $1.00/$6.00 and reasoned its way to the wrong answer -- Luna is 5x cheaper than it claimed, and with supports_response_schema=True and 1.05M context it is a real annotator candidate it dismissed). Separately, deepseek-chat / deepseek-reasoner are still present with deprecation_date=None, and deepseek/deepseek-v4-pro is $0.435/$0.87 not $1.32/$3.96, in a catalog that carries 335 deprecation dates and ran a lifecycle audit on 2026-08-15. Its 'audit these dead model IDs NOW' recommendation is unsupported.
- Fetch the catalog by commit SHA, never by `main`. I reproduced the drift live: /main/ differs from the pinned SHA on exactly 3 keys, all gemini-3.6-flash aliases, all exactly 2x (input 1.5e-06 vs 7.5e-07), still stale ~13 hours after the fix commit landed. This is the best-supported design argument in the whole dossier set.
- Fix the two confirmed repo bugs, but not the way `live-catalogs` proposed. (1) `_NO_TEMPERATURE` at registry.py:327 omits claude-sonnet-5, the configured fast_model, and LiteLLM independently reports supports_sampling_params=False for it -- masked only because temperature defaults to None. (2) budget.py:88 hardcodes $3/$15. But `cache_hits` in the ledger counts QMine's LOCAL response-dedup cache (registry.py records cached=True on a `_cache_get` hit), which costs $0, not Anthropic prompt-cache reads at 0.1x. Pricing those at 0.1x would invent spend that never happened.
- Recognise that QMine has NO prompt caching (`cache_control` appears nowhere) and NO batch API. This invalidates two headline recommendations: `routing`'s '#1: turn on the Batch API first, -29%/-39%' is a build project, not a switch (LangChain's ChatAnthropic does not expose Messages Batches), and the cache-minimum eligibility gate that both `model-landscape` and `routing` elevated to a design law currently gates on a constraint that binds nothing. Implement caching, measure the actual taxonomy prefix (the ~4,000-token figure is assumed in both cost models, never measured), then gate.
- Read `src/qmine/llm/requirements.py` before building anything. It already defines RoleRequirement with reasoning tier, blast_radius, min_context_tokens, needs_structured_output, typical_calls, output_tokens_per_call, multilingual_critical and a cost_sensitivity property, covering all 14 roles -- exactly what `routing` and `tradingagents-providers` proposed building from scratch. It is imported nowhere in src/ or tests/. Also note there are 14 roles, not 13: `domain_scout` is missing from the task brief and from `routing`'s cost model.
- Refine the OpenRouter verdict rather than discarding it. Model-level `supported_parameters` is a UNION across heterogeneous endpoints -- for claude-opus-5 only the two Azure endpoints claim temperature, and that poisons the model-level field. The `/api/v1/models/{id}/endpoints` sub-resource is per-provider and correct, unauthenticated, and also carries uptime telemetry (I measured 99.97 / 100). Never read model-level for capability; /endpoints is usable.
- Make offline degradation explicit rather than incidental. `model_name()` returns the literal 'offline-heuristic', so `live-catalogs`' proposed `prices[model_for_tier(role)]` KeyErrors on every offline run and its `load_catalog` has no offline branch -- and a missing price degrades to a plausible $0.00, reproducing this repo's own recorded resume-safety bug class. Also make template certification structurally impossible offline: `unknown-domain`'s LLM pair-adjudication would be served by the heuristic stand-in and could stamp trusted=True on template groups that then judge the alpha sweep.
- Fix the credential-detection bug nobody flagged: `_has_anthropic_credentials()` counts LANGSMITH_API_KEY as an Anthropic credential, so a tracing-only environment resolves the provider to 'anthropic' and then fails at call time instead of degrading to the offline stand-in. Tracing creds are not inference creds.
- Prefer capability data over price data in whatever snapshot you build, and make validation advisory. Sampling-param flags and prompt_cache_min_tokens (non-monotonic: 512/1024/2048/4096, so no heuristic reproduces them) break code with HTTP 400s; prices break budgets quietly. Wire LiteLLM's 335 deprecation_date entries into the refresh as a hard failure, fail on a missing key too (a vanished model is signal), but keep unknown-model validation as a RuntimeWarning like TradingAgents does -- `live-catalogs`' proposed LLMUnavailable raise turns 'user tried a new model via env' into a hard stop.