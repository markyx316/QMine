# TradingAgents Multi-Provider Teardown

> Gathered 2026-08-18. Facts marked verified were fetched live; the model-landscape
> dossier in particular flags its prices as secondary-source and unconfirmed —
> which is why the running system fetches prices from a live catalogue rather
> than embedding any table from this document.

> **Verification basis.** All source quoted below was fetched from `raw.githubusercontent.com/TauricResearch/TradingAgents/main` on **2026-08-18**. Repo state: version `0.3.1` (`pyproject.toml`), CHANGELOG entry `[0.3.1] — 2026-07-05`, latest commit `a33fd4c0` dated **2026-07-18T15:55:04Z**. Model-ID/price currency checks were run against live sources on **2026-08-18** (cited at the end). Anthropic model/price facts come from the `claude-api` skill's catalog (cached 2026-06-24).

---

# 1. How a provider is selected and instantiated

## 1.1 The dispatch: 4 native SDK clients + one OpenAI-compatible registry

It is **not** a pure base_url swap, and it is **not** per-provider SDKs everywhere either. It is a deliberate two-lane split, and the comment says so explicitly. `tradingagents/llm_clients/factory.py` is the whole thing — 45 lines:

```python
def create_llm_client(provider: str, model: str, base_url: str | None = None, **kwargs) -> BaseLLMClient:
    provider_lower = provider.lower()

    # Native (non-OpenAI) APIs are matched first so their string check doesn't
    # import the OpenAI client. Everything else is OpenAI-compatible and routes
    # through the provider registry (single source of truth).
    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)
    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)
    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)
    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return BedrockClient(model, base_url, **kwargs)

    from .openai_client import OpenAIClient, is_openai_compatible
    if is_openai_compatible(provider_lower):
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
```

Lane A (native APIs, genuinely different wire formats): `anthropic` → `langchain_anthropic.ChatAnthropic`, `google` → `langchain_google_genai.ChatGoogleGenerativeAI`, `azure` → `langchain_openai.AzureChatOpenAI`, `bedrock` → `langchain_aws.ChatBedrockConverse` (lazily imported; optional `[bedrock]` extra).

Lane B is the interesting one — a **declarative frozen-dataclass registry** in `openai_client.py`:

```python
@dataclass(frozen=True)
class ProviderSpec:
    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API
```

with 16 rows:

```python
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama"),
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI
    ),
}
```

**20 provider keys total** (4 native + 16 compatible). Per-provider wire quirks are handled by `chat_class` subclasses of `NormalizedChatOpenAI`, not by if-ladders in the client: `DeepSeekChatOpenAI` round-trips `reasoning_content` (overrides `_get_request_payload` and `_create_chat_result`), `MinimaxChatOpenAI` injects `reasoning_split: True` via `extra_body`, `LocalCompatibleChatOpenAI` suppresses object-form `tool_choice` for arbitrary local servers.

One genuinely sharp detail — they discovered that "OpenAI-compatible" is not transitive with OpenAI's own Responses API:

```python
def _is_native_openai_base_url(base_url: str | None) -> bool:
    """The Responses API (/v1/responses) only exists on native OpenAI. A custom
    base_url on the ``openai`` provider (a proxy, gateway, or local server)
    speaks only Chat Completions, so the Responses API must stay off there even
    though the provider spec enables it (#1024)."""
    if not base_url: return True
    if "://" not in base_url: base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")
```

## 1.2 Auth

Canonical env-var map in `llm_clients/api_key_env.py` — one source of truth consulted by both the client and the CLI prompt. Notable: dual-region providers get *separate* keys because the accounts aren't interchangeable (`DASHSCOPE_API_KEY` vs `DASHSCOPE_CN_API_KEY`, `ZHIPU_API_KEY` vs `ZHIPU_CN_API_KEY`, `MINIMAX_API_KEY` vs `MINIMAX_CN_API_KEY` — issue #758). Bedrock maps to `None` (AWS credential chain). Missing keys raise at client construction with a fix-it message naming the env var and `.env` file.

## 1.3 SDK dependencies (`pyproject.toml`)

```
"langchain-core>=0.3.81", "langchain-anthropic>=0.3.15",
"langchain-google-genai>=4.0.0", "langchain-openai>=0.3.23",
"langgraph>=0.4.8", "langgraph-checkpoint-sqlite>=2.0.0",
"questionary>=2.1.0", "typer>=0.21.0", "rich>=14.0.0",
```
plus optional `bedrock = ["langchain-aws>=1.5.0"]`. **Everything goes through LangChain** — no raw `anthropic` or `openai` SDK usage anywhere. `requirements.txt` is literally one character: `.`

---

# 2. How models are chosen — and no, there is zero automatic selection

**The user's belief is correct. Verified: there is no capability-based or cost-based selection anywhere in the repo.** I grepped `cli/main.py`, `cli/stats_handler.py`, `default_config.py`, `model_catalog.py`, and `trading_graph.py` for `cost|price|budget|$` — the only hit in the entire codebase is a display string (see §3.4).

Three configuration surfaces, in precedence order:

**(a) Hardcoded defaults** (`default_config.py`):
```python
"llm_provider": "openai",
"deep_think_llm": "gpt-5.5",
"quick_think_llm": "gpt-5.4-mini",
"backend_url": None,
```

**(b) Env overrides** — a nice, genuinely borrowable pattern. A single `_ENV_OVERRIDES` table drives coercion typed off the *existing default's* type, and it **fails loudly** rather than silently defaulting:

```python
def _coerce(value: str, reference):
    """Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run."""
```
Keys exposed: `TRADINGAGENTS_LLM_PROVIDER`, `_DEEP_THINK_LLM`, `_QUICK_THINK_LLM`, `_LLM_BACKEND_URL`, `_TEMPERATURE`, `_LLM_MAX_RETRIES`, `_GOOGLE_THINKING_LEVEL`, `_OPENAI_REASONING_EFFORT`, `_ANTHROPIC_EFFORT`, `_MAX_DEBATE_ROUNDS`, `_MAX_RISK_ROUNDS`, `_CHECKPOINT_ENABLED`, `_OUTPUT_LANGUAGE`, `_BENCHMARK_TICKER`.

**(c) Interactive CLI menu** — 8 questionary steps in `cli/main.py`. Step 6 provider, step 7 the two models, step 8 the provider-specific reasoning knob. Every step is **individually skippable via env** — the pattern is uniform:

```python
if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
    selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
    selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
    console.print(f"[green]✓ Thinking agents from environment:[/green] ...")
else:
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)
```

The picker is a static dropdown over `MODEL_OPTIONS[provider][mode]` where `mode ∈ {"quick","deep"}`. **The "capability matching" is nothing more than which hardcoded list a model appears in**, plus a human-readable label:

```python
"anthropic": {
    "quick": [
        ("Claude Sonnet 5 - Best speed and intelligence balance", "claude-sonnet-5"),
        ("Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"),
    ],
    "deep": [
        ("Claude Fable 5 - Most capable, long-running agents", "claude-fable-5"),
        ("Claude Opus 4.8 - Frontier agentic coding and reasoning", "claude-opus-4-8"),
        ("Claude Sonnet 5 - Near-frontier intelligence at Sonnet cost", "claude-sonnet-5"),
        ("Claude Opus 4.7 - Previous frontier, long-running agents", "claude-opus-4-7"),
    ],
},
```

**The one exception — and the only live-catalog code in the repo** — is OpenRouter (`cli/utils.py`):

```python
def _fetch_openrouter_models() -> list[tuple[str, str]]:
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
    models = resp.json().get("data", [])
    # Newest first so the top-N shown really is the latest available
    models.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return [(m.get("name") or m["id"], m["id"]) for m in models]
```
filtered against a hardcoded namespace allowlist and truncated to 5:
```python
_OPENROUTER_MAINSTREAM = {"openai", "anthropic", "google", "deepseek", "qwen",
                          "mistralai", "meta-llama", "x-ai", "z-ai", "minimax", "moonshotai"}
top = (mainstream or models)[:5]
```
Note the reasoning in the comment: *"Provider names are stable (unlike model IDs), so this rarely needs touching."* They understood the staleness problem — and solved it for exactly one provider.

## 2.1 Role → tier assignment is positional and hardcoded

This is the closest structural parallel to QMine. `trading_graph.py` builds **exactly two** clients:

```python
deep_client = create_llm_client(provider=self.config["llm_provider"], model=self.config["deep_think_llm"], ...)
quick_client = create_llm_client(provider=self.config["llm_provider"], model=self.config["quick_think_llm"], ...)
self.deep_thinking_llm = deep_client.get_llm()
self.quick_thinking_llm = quick_client.get_llm()
self.graph_setup = GraphSetup(self.quick_thinking_llm, self.deep_thinking_llm, self.tool_nodes, self.conditional_logic)
self.reflector = Reflector(self.quick_thinking_llm)
self.signal_processor = SignalProcessor(self.quick_thinking_llm)
```

and `graph/setup.py` assigns them **literally, per node**:

| Role | Tier |
|---|---|
| market / social / news / fundamentals analysts | `quick_thinking_llm` |
| bull_researcher, bear_researcher | `quick_thinking_llm` |
| **research_manager** | **`deep_thinking_llm`** |
| trader | `quick_thinking_llm` |
| aggressive / neutral / conservative debators | `quick_thinking_llm` |
| **portfolio_manager** | **`deep_thinking_llm`** |
| Reflector, SignalProcessor (outside graph) | `quick_thinking_llm` |

**15 call sites, 13 graph agents, exactly 2 on the deep tier.** That is the same shape as QMine's 13 roles → 2 models — and it's assigned by whoever wrote `setup.py`, not by any policy the user can express or override without editing Python. Both providers must also be the *same* provider: `llm_provider` is a single scalar, so you cannot run Haiku for annotators and Opus for the referee across vendors, or even mix Anthropic + a local Ollama.

---

# 3. Which model IDs are hardcoded right now, and how stale they are

This is the strongest evidence for the maintenance argument. **The repo's last commit is 2026-07-18. I verified currency on 2026-08-18 — one month later — and the lists were already stale at the moment of that final commit for two of the three closed providers.**

## 3.1 OpenAI — one full generation behind, and unreachable from the menu

Hardcoded (`model_catalog.py` + `default_config.py`):
```python
"openai": {
    "quick": [("GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"),
              ("GPT-5.4 Nano - Cheapest, high-volume tasks", "gpt-5.4-nano"),
              ("GPT-5.5 - Latest frontier, 1M context", "gpt-5.5")],
    "deep":  [("GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"),
              ("GPT-5.4 - Previous-gen frontier, 1M context, cost-effective", "gpt-5.4"),
              ("GPT-5.2 - Strong reasoning, cost-effective", "gpt-5.2"),
              ("GPT-5.5 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.5-pro")],
},
```

**Verified 2026-08-18:** OpenAI's current family is **GPT-5.6**, GA **2026-07-09**, with model IDs `gpt-5.6-sol` (frontier), `gpt-5.6-terra` (balanced), `gpt-5.6-luna` (high-volume/cheap); the bare alias `gpt-5.6` routes to Sol. GPT-5.5 ($5/$30 per 1M) is **no longer published on OpenAI's current API pricing page and is treated as legacy**; GPT-5.4 ($2.50/$15) is two generations behind.

The damning bit: **GPT-5.6 GA was 2026-07-09; the repo's last commit is 2026-07-18 — nine days later — and still ships `gpt-5.5` as `deep_think_llm` and `gpt-5.4-mini` as `quick_think_llm`.**

Worse, this is a **dead end in the UI**. `openai` is not in `_ANY_MODEL_PROVIDERS`, and its `MODEL_OPTIONS` entry contains **no `("Custom model ID", "custom")` row**. So:

> A CLI user on the default provider **cannot select `gpt-5.6-sol` at all through the menu.** Their only escape is `TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.6-sol` or editing `default_config.py`.

The same closed-dropdown problem applies to **`anthropic`, `google`, and `xai`** — all four lack a custom-ID row. Only `deepseek`, `qwen*`, `glm*`, `minimax*`, and `ollama` offer `custom`.

## 3.2 Anthropic — missing the current Opus

Hardcoded deep list: `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-opus-4-7`. Quick: `claude-sonnet-5`, `claude-haiku-4-5`.

**Missing: `claude-opus-5`** ($5/$25 per 1M, 1M context) — which was already in the Anthropic catalog as of the 2026-06-24 snapshot, i.e. **~3 weeks before their last commit**. The CHANGELOG entry for 0.3.1 says *"Added Claude Sonnet 5 (`claude-sonnet-5`) and Fable 5 (`claude-fable-5`); effort control now covers the Claude 5 line"* — so they were tracking the Claude 5 line and still missed the Opus tier of it.

The IDs they *do* list are correct and current: `claude-fable-5` ($10/$50), `claude-opus-4-8` ($5/$25), `claude-opus-4-7` ($5/$25), `claude-sonnet-5` ($3/$15, intro $2/$10 through 2026-08-31), `claude-haiku-4-5` ($1/$5, 200K context).

Effort gating (`anthropic_client.py`) is the most careful model-version logic in the repo, and is *forward*-compatible by design:
```python
_EFFORT_EXACT = {"claude-mythos-preview", "claude-mythos-5"}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}
```
This correctly *would* admit `claude-opus-5` → `(5, 0) >= (4, 5)` → True. So the gate is fine; only the **catalog** is stale. Two residual defects: `claude-mythos-preview` is superseded by `claude-mythos-5` and should be dropped; and the CLI only offers three effort levels —
```python
def ask_anthropic_effort() -> str | None:
    """Controls token usage and response thoroughness on Claude 4.5 / 4.6 / 4.7
    models. The API also accepts "max"; we expose low/medium/high..."""
```
— a **stale docstring** (the Claude 5 line is unmentioned) and a menu that omits **`xhigh`**, which is the recommended setting for coding/agentic work on Opus 4.7/4.8 and Sonnet 5, and omits `max`.

## 3.3 Google — two releases behind, and a likely-wrong `-preview` suffix

Hardcoded:
```python
"google": {
    "quick": [("Gemini 3.5 Flash - Latest, frontier agentic + coding (GA)", "gemini-3.5-flash"),
              ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite")],
    "deep":  [("Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"),
              ("Gemini 3.5 Flash - Latest GA, strong agentic + coding", "gemini-3.5-flash")],
},
```
**Verified 2026-08-18:** `gemini-3.7-flash` launched **2026-08-13** ($0.75/$3.75); **Gemini 3.6 Flash** and **Gemini 3.5 Flash-Lite** released **2026-07-21** (3.6 Flash $1.50/$7.50; 3.5 Flash-Lite $0.30/$2.50); **Gemini 3.1 Pro** is $2.00/$12.00. So the label *"Gemini 3.5 Flash — Latest… (GA)"* is now **two releases wrong**, and `gemini-3.1-pro-preview` carries a `-preview` suffix for a model that is now referred to plainly as Gemini 3.1 Pro — a likely 404 or deprecated alias.

## 3.4 The only price in the entire repository

```python
("GPT-5.5 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.5-pro"),
```
That price was accurate. It is a **display string in a dropdown label**. No code reads it, no arithmetic uses it, and it is attached to a model that is now legacy. This is the cost-awareness story in its entirety.

## 3.5 In fairness: the parts that *are* well-maintained

- **DeepSeek** carries a documented, **future-dated** deprecation: *"the deepseek-chat / deepseek-reasoner aliases are deprecated (2026-07-24) and now map to V4 Flash; expose the V4 IDs directly."* That deprecation date is *after* their last commit — they wrote it ahead of time.
- **Qwen** shows real judgment about aliases: *"the version-less aliases (qwen-plus, qwen-flash) are documented by Alibaba as auto-upgrading pointers… which means their behavior shifts when Alibaba rotates the backing model. Users who want a specific generation pick it explicitly."* Only versioned IDs are exposed.
- **Volatile providers are deliberately not catalogued.** `_CUSTOM_ONLY` covers `mistral`, `kimi`, `groq`, `nvidia`, `bedrock`, `openai_compatible`, with the comment: *"Providers that serve many / frequently-changing models: offer only 'Custom model ID' rather than a list that goes stale."* **They diagnosed the exact problem and then applied the cure only to the providers they cared least about.**
- **Validation is advisory, not blocking** (`base_client.py`), which is the right call:
```python
def warn_if_unknown_model(self) -> None:
    if self.validate_model(): return
    warnings.warn(f"Model '{self.model}' is not in the known model list for "
                  f"provider '{self.get_provider_name()}'. Continuing anyway.", RuntimeWarning, stacklevel=2)
```
A stale catalog therefore never hard-blocks an env-set new model — it just emits a `RuntimeWarning`. Good failure mode; but it also means the stale menu quietly misinforms every interactive user.

---

# 4. Failures, rate limits, structured output

## 4.1 Failure handling: thin, and entirely delegated to the SDK

There is **no fallback chain, no cross-provider failover, no circuit breaker, no per-role degradation.** The whole story is one integer forwarded to every provider:

```python
# tradingagents/graph/trading_graph.py
max_retries = self.config.get("llm_max_retries")
if max_retries is not None and max_retries != "":
    kwargs["max_retries"] = _coerce_max_retries(max_retries)
```
```python
def _coerce_max_retries(value):
    if isinstance(value, bool):
        raise ValueError(f"llm_max_retries must be an integer, not a boolean: {value!r}")
    try: n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm_max_retries must be an integer, got {value!r}") from exc
    if n < 0: raise ValueError(f"llm_max_retries must be >= 0, got {n}")
    return n
```
`default_config.py`: *"SDK retry budget forwarded to every provider chat client. None leaves each provider/SDK at its own default (usually 2). Raise it to ride out bursty 429 throttling on rate-limited deployments instead of aborting a run (#1091)."*

So: **rate limits = "turn the SDK's retry count up."** A 429 storm that outlasts the budget kills the run. That is exactly the failure mode QMine cannot afford across 5,000 annotator rows.

## 4.2 The real resilience story is checkpointing, not retries

This is genuinely good and worth stealing. Opt-in `checkpoint_enabled` recompiles the graph with a per-ticker `SqliteSaver`, and — critically — the thread ID folds in everything that changes graph *shape*, so a resume under different settings starts fresh instead of silently continuing the wrong graph:

```python
def _run_signature(self, asset_type: str) -> str:
    """Graph-shape inputs that must invalidate a checkpoint if changed. (#1089)"""
    return "|".join([
        "analysts=" + ",".join(self.selected_analysts),
        f"debate={self.config['max_debate_rounds']}",
        f"risk={self.config['max_risk_discuss_rounds']}",
        f"asset={asset_type}",
    ])
```
Cleared on success (`clear_checkpoint`) so stale state can't leak into the next run.

## 4.3 Structured output — the best-engineered part of the codebase

Three layers, and this is where TradingAgents is genuinely ahead of most multi-provider wrappers.

**(a) A declarative per-model capability table** (`llm_clients/capabilities.py`) — explicitly built to avoid model-name if-ladders:

```python
@dataclass(frozen=True)
class ModelCapabilities:
    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: StructuredMethod  # function_calling | json_mode | json_schema | none
    requires_reasoning_content_roundtrip: bool = False
    requires_reasoning_split: bool = False
```
with exact-ID matches first, then **forward-compatible regex patterns**:
```python
_BY_PATTERN: list[tuple[re.Pattern[str], ModelCapabilities]] = [
    (re.compile(r"^deepseek-v\d"), _DEEPSEEK_THINKING),
    (re.compile(r"^deepseek-reasoner"), _DEEPSEEK_THINKING),
    (re.compile(r"^MiniMax-M\d"), _MINIMAX_THINKING),
]
```
*"New `deepseek-v5-*` / `deepseek-reasoner-*` or `MiniMax-M3*` variants inherit the thinking-mode quirks automatically."* The module docstring even credits its source: *"Pattern adapted from the per-model `compat:` flags DeepSeek themselves publish in their integration guides."*

**(b) Capability-aware dispatch at the binding site:**
```python
def with_structured_output(self, schema, *, method=None, **kwargs):
    caps = get_capabilities(self.model_name)
    if caps.preferred_structured_method == "none":
        raise NotImplementedError(...)
    method = method or caps.preferred_structured_method
    # When the model rejects tool_choice, suppress langchain's hardcoded value.
    if method == "function_calling" and not caps.supports_tool_choice:
        kwargs.setdefault("tool_choice", None)
    return super().with_structured_output(schema, method=method, **kwargs)
```
This exists because DeepSeek V4/reasoner and MiniMax M2.x **400 on the function-spec dict LangChain sends** for `tool_choice` — real bugs (#678, #826), fixed once declaratively rather than N times in agent code.

**(c) Two-stage graceful degradation** (`agents/utils/structured.py`) — bind-time and invoke-time:
```python
def bind_structured(llm, schema, agent_name):
    try: return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning("%s: provider does not support with_structured_output (%s); "
                       "falling back to free-text generation", agent_name, exc)
        return None

def invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name):
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling the tool
                raise ValueError("structured output returned no parsed result")
            return render(result)
        except Exception as exc:
            logger.warning("%s: structured-output invocation failed (%s); retrying once as free text", agent_name, exc)
    response = plain_llm.invoke(prompt)
    return response.content
```
Plus a prompt-level guard for a subtle failure: *"Schema-only structured output binds exactly one tool (the schema itself), so a model that reaches for a search tool emits an unknown tool call and the whole structured attempt is discarded for a free-text retry"* → they inject a `NO_EXTERNAL_TOOLS` instruction (#1130).

And a deterministic backstop so the pipeline never returns nothing usable — `agents/utils/rating.py` heuristically parses a 5-tier rating out of prose when structured output degraded.

**(d) Cross-provider content normalization**, because Responses-API and Gemini-3 return typed block lists where downstream code expects a string:
```python
def normalize_content(response):
    content = response.content
    if isinstance(content, list):
        texts = [item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                 else item if isinstance(item, str) else "" for item in content]
        response.content = "\n".join(t for t in texts if t)
    return response
```
applied by `NormalizedChatOpenAI` / `NormalizedChatAnthropic` / `NormalizedChatGoogleGenerativeAI` / `NormalizedAzureChatOpenAI` / `NormalizedChatBedrockConverse`.

**Test coverage backs this up**: `test_capabilities.py`, `test_structured_agents.py` (16KB), `test_structured_agent_prompts.py`, `test_deepseek_reasoning.py` (10KB), `test_minimax.py`, `test_provider_registry.py`, `test_model_validation.py`, `test_openai_compatible_provider.py`, `test_bedrock_provider.py`, `test_llm_max_retries.py`, plus `scripts/smoke_structured_output.py`.

---

# 5. What is genuinely good and worth borrowing

1. **`ProviderSpec` as a frozen dataclass registry.** One row per provider replaces a base-URL dict + auth branches + client-class branches. Adding NVIDIA NIM was one line. **Adopt this shape directly for QMine.**
2. **`ModelCapabilities` + `get_capabilities()` with exact-ID-then-regex resolution.** This is the single most valuable idea in the repo for QMine, because *every* QMine role uses Pydantic structured output. Their table already encodes the two known killers (DeepSeek/MiniMax rejecting `tool_choice`) that would otherwise silently break annotators mid-run.
3. **Two-stage structured-output degradation + a deterministic parser backstop.** `bind_structured` → `invoke_structured_or_freetext` → `parse_rating`. For QMine's annotators (600–5,000 rows), never blocking the pipeline on one malformed JSON is worth more than the structured guarantee itself.
4. **Env-override table with type-coerced, loud-failing parsing.** `_ENV_OVERRIDES` + `_coerce` raising on `treu`. Directly portable.
5. **Uniform "env set → skip the prompt" CLI pattern.** Same interactive binary is fully non-interactive in CI, per-step, with a `✓ … from environment:` echo. The UX is excellent — steal the pattern, including the echo.
6. **Regional endpoint sub-prompts.** `ask_qwen_region` / `ask_glm_region` / `ask_minimax_region` keep the main menu at 17 rows while handling non-interchangeable CN/intl credentials. **This matters directly for QMine — the corpus is mostly Chinese, and GLM/Qwen/MiniMax CN endpoints are live candidates.**
7. **`ensure_api_key()`** — prompts with `questionary.password`, persists via `dotenv.set_key`, exports into `os.environ`, and correctly *never* prompts for `key_optional` providers. Fails the run before the first API call rather than 40 minutes in.
8. **Checkpoint thread-ID folding graph shape** (`_run_signature`). Given the note in QMine's memory that *"resume safety is not checkpointing"* and nodes reading process memory failed silently on resume, this is a directly relevant hardening pattern.
9. **`normalize_content`** for typed-block responses across five client classes.
10. **The deep/quick two-tier split itself** is real and useful as a *floor* — it captures the single biggest cost lever (don't run 5,000 annotator calls on the frontier model) with almost no machinery.
11. **`_is_native_openai_base_url`** — the recognition that "OpenAI-compatible" ≠ "supports the Responses API."
12. **The `_CUSTOM_ONLY` philosophy**, stated in their own comment. Borrow the *principle* and apply it universally, which they did not.

---

# 6. What is missing or weak — the bar we must clear

1. **No live model catalog.** 100+ model IDs are hardcoded Python tuples, refreshed only by human PR. OpenRouter is the sole exception. **Measured consequence: on 2026-08-18 the OpenAI list is a full generation stale (no `gpt-5.6-*`), the Google "latest" label is two releases stale (`gemini-3.7-flash` shipped 2026-08-13), and the Anthropic list is missing `claude-opus-5`.**
2. **The stale list is a hard wall on four providers.** `openai`, `anthropic`, `google`, `xai` have no `("Custom model ID", "custom")` row. A menu user on the default provider **cannot pick the current frontier OpenAI model at all**. Minimum bar for QMine: *every* provider gets a custom-ID escape hatch.
3. **Zero cost awareness.** `StatsCallbackHandler` accumulates `llm_calls`, `tool_calls`, `tokens_in`, `tokens_out` behind a lock — and stops there. `get_stats()` returns those four integers. **No price table, no per-role attribution, no run-cost estimate, no post-run report.** The only `$` in the repo is a dropdown label. For QMine — 600–5,000 annotator rows batched 25/call, plus 20–200 namer calls, plus a long-context `tree_auditor` and `reporter` — a run's cost can differ 30× by routing, and this design makes that invisible.
4. **No capability matching whatsoever.** "Quick vs deep" is a *label on a hardcoded list*, not a property. Nothing in the system knows that `tree_auditor` needs a long context window, that `annotator_a` needs cheap high-throughput structured output, or that `namer` needs strong Chinese. QMine's 13 roles have at least four distinct requirement axes (volume, stakes, context length, language) — a two-bucket scheme cannot express them.
5. **No budget enforcement.** Nothing caps spend. No per-run ceiling, no per-role cap, no pre-flight estimate, no abort-on-overrun. A runaway debate loop (`max_recur_limit: 100`) burns until it finishes.
6. **No fallback chain.** One provider, two models, `max_retries`. No cross-model or cross-provider failover, no degradation ladder (deep → quick → cached), no partial-completion salvage. A sustained 429 on the deep model kills the run at the `portfolio_manager` node after every analyst has already been paid for.
7. **Single-provider lock per run.** `llm_provider` is one scalar shared by both tiers. You cannot mix Haiku-for-volume with Opus-for-judgment across vendors, or route Chinese-heavy roles to GLM while keeping the referee on Claude.
8. **No prompt caching anywhere.** Not in any client, not in any agent. For QMine's annotators this is the single largest unexploited saving: a stable taxonomy/system prefix re-sent across every 25-row batch is exactly the shape prompt caching is built for (~0.1× on cached reads). TradingAgents leaves it entirely on the table.
9. **Reasoning/effort exposure is incomplete and provider-siloed.** Anthropic gets `low/medium/high` (missing `xhigh`, the recommended agentic setting, and `max`); Google gets `high/minimal`; OpenAI gets `low/medium/high`. All three are **one global value for the whole run** — you cannot run the annotators at low effort and the referee at high, which is precisely the trade QMine wants.
10. **Stale metadata rides along with the stale IDs.** The `ask_anthropic_effort` docstring still says "Claude 4.5 / 4.6 / 4.7"; `claude-mythos-preview` lingers in `_EFFORT_EXACT` after `claude-mythos-5` superseded it; the `-preview` suffix on `gemini-3.1-pro-preview` is very likely wrong now. Human-maintained model metadata rots on every axis at once, not just the ID.
11. **No structured-output *verification* of the catalog.** `validate_model()` compares against `get_known_models()` — a list derived from the same stale hardcoded tuples. The validator can never detect that the catalog itself is out of date; it can only tell you a model isn't on a list that hasn't been updated in a month.
12. **Language is a report-formatting toggle, not a routing input.** `output_language` (English/Chinese/Japanese/… + custom) only controls the language of analyst reports — *"Internal agent debate stays in English for reasoning quality."* Nothing routes a model based on its competence in the corpus language. For a mostly-Chinese search-query log, model choice per language is a first-class capability question, and TradingAgents doesn't model it at all.


---

## Recommendations

- Steal the `ProviderSpec` frozen-dataclass registry verbatim as QMine's provider layer — one declarative row per provider (base_url, base_url_env, key_optional, placeholder_key, require_base_url, chat_class) replaces every per-provider if-branch, and their `_is_native_openai_base_url` guard proves the pattern survives contact with real gateway/proxy deployments.
- Steal and then extend `llm_clients/capabilities.py`. The exact-ID → regex-pattern → default resolution order with per-model flags (`supports_tool_choice`, `preferred_structured_method`, `requires_reasoning_content_roundtrip`, `requires_reasoning_split`) is exactly the right shape for a system where all 13 QMine roles use Pydantic structured output. Extend the dataclass with the fields TradingAgents lacks: `input_price_per_mtok`, `output_price_per_mtok`, `cached_input_price`, `context_window`, `supports_prompt_caching`, `effort_levels`, `strong_languages`.
- Steal the two-stage structured-output degradation (`bind_structured` at construction → `invoke_structured_or_freetext` at call time → deterministic heuristic parser as backstop). For 600–5,000 annotator rows, never blocking the pipeline on one malformed JSON is worth more than the structured guarantee itself. Add their `NO_EXTERNAL_TOOLS` prompt guard — schema-only binding means a model that reaches for a tool silently discards the whole structured attempt.
- Beat them on the maintenance problem, which is the explicit ask: build a live catalog with a cached fallback. They already proved the pattern works (`_fetch_openrouter_models` hits `https://openrouter.ai/api/v1/models`, sorts by `created` desc) and already diagnosed the disease in a code comment ('offer only Custom model ID rather than a list that goes stale') — they just never applied the cure to openai/anthropic/google/xai. Anthropic's `GET /v1/models` returns `max_input_tokens`, `max_tokens`, and a full `capabilities` tree; OpenAI and Google both expose model-list endpoints.
- Never build a closed dropdown. Every provider gets a `Custom model ID` row unconditionally. TradingAgents' worst single failure is that a CLI user on the default provider literally cannot select `gpt-5.6-sol` — the menu has no such entry and no escape hatch, six weeks after GA.
- Replace the two-tier `deep_think_llm` / `quick_think_llm` scalar with a per-role routing policy keyed on declared requirements, not labels. QMine's 13 roles vary on at least four axes TradingAgents cannot express: volume (annotators 600–5,000 rows batched 25/call vs referee's handful), stakes (architect/referee/tree_auditor errors propagate into every downstream artifact), context length (tree_auditor sees all cluster namings at once; reporter sees the whole evidence bundle), and language (mostly Chinese). Let each role declare `min_context`, `volume_class`, `stakes`, `needs_language`, and let the router match against the live catalog.
- Add real cost accounting, which TradingAgents has zero of. Their `StatsCallbackHandler` already collects `llm_calls`/`tool_calls`/`tokens_in`/`tokens_out` under a lock via LangChain callbacks — take that class and multiply by a price field on the capability record. Emit a pre-flight run estimate, live per-role attribution, and a post-run breakdown. The only dollar figure anywhere in their repo is `($30/$180 per 1M tokens)` inside a dropdown label string that no code reads.
- Add a budget ceiling and a fallback chain — the two things they have none of. Per-run and per-role spend caps with abort-or-degrade on overrun; an ordered fallback ladder per role (preferred model → cheaper same-provider → different provider → free-text degrade) rather than their single `llm_max_retries` integer forwarded to the SDK. A sustained 429 on their deep model kills the run at `portfolio_manager` after every analyst call has already been paid for.
- Allow per-role provider mixing. TradingAgents' `llm_provider` is one scalar shared by both tiers, so you cannot run Haiku for annotators and Opus for the referee across vendors, nor route Chinese-heavy roles (namer, l2_interpreter) to GLM/Qwen while keeping tree_auditor on Claude. Their regional sub-prompts (`ask_qwen_region`/`ask_glm_region`/`ask_minimax_region`, with separate CN vs international API keys) are directly relevant given QMine's mostly-Chinese corpus — borrow those, but make the provider a per-role field.
- Adopt prompt caching, which TradingAgents does not use anywhere. QMine's annotators re-send a stable taxonomy/system prefix across every 25-row batch — the textbook prompt-caching shape, at roughly 0.1x on cached reads. Make `supports_prompt_caching` a capability field and have the router prefer cache-capable models for high-volume roles.
- Borrow their env-configuration ergonomics wholesale: the `_ENV_OVERRIDES` table with type coercion driven by the existing default's type and a `ValueError` on a misspelled boolean rather than a silent fallback, plus the uniform per-step 'env set -> skip the prompt, echo a green checkmark' CLI pattern that makes the same interactive binary fully non-interactive in CI. Also borrow `ensure_api_key()`, which prompts with `questionary.password`, persists via `dotenv.set_key`, and fails before the first API call rather than 40 minutes into a run.
- Borrow their checkpoint thread-ID design, given QMine's known 'resume safety is not checkpointing' bug class. `_run_signature()` folds every graph-shape-affecting choice (selected analysts, debate depth, risk depth, asset mode) into the checkpoint thread ID so a resume under different settings starts fresh instead of silently continuing the wrong graph (#1089) — and the checkpoint is cleared on success so stale state cannot leak forward. For QMine the analogous signature must include the model routing policy itself: resuming a run under different model assignments must not reuse checkpoints.

## Unverified

- I could not fetch platform.openai.com, benchlm.ai, or devtk.ai directly (all blocked by domain-safety checks), so the OpenAI GPT-5.6 model IDs (`gpt-5.6-sol` / `-terra` / `-luna`, GA 2026-07-09) and prices (GPT-5.5 $5/$30, GPT-5.5 Pro $30/$180, GPT-5.4 $2.50/$15) come from aggregated web-search results rather than OpenAI's own pricing page. The IDs were corroborated across several independent sources and I consider them reliable; treat the exact price figures as second-hand.
- Whether `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, and `gpt-5.2` still resolve on the API is not fully established. Sources say GPT-5.5 is 'no longer published on OpenAI's current API pricing page, so it is shown as legacy' — legacy is not the same as retired. TradingAgents' defaults may still work today; the point stands that they are a generation behind and that the menu offers no path to the current family.
- Same caveat for Google: `gemini-3.7-flash` (launched 2026-08-13), `gemini-3.6-flash`, and `gemini-3.5-flash-lite` (2026-07-21) come from web-search aggregation, not from ai.google.dev directly. I did not verify whether `gemini-3.1-pro-preview` still resolves or has been replaced by a GA `gemini-3.1-pro` alias — I flagged it as 'likely wrong' rather than confirmed broken.
- The Anthropic model IDs and prices are from the `claude-api` skill's cached catalog dated 2026-06-24, not a live fetch on 2026-08-18. It is possible a newer Anthropic model shipped between 2026-06-24 and today that would make TradingAgents' Anthropic list even more stale than I report — my claim that it is missing `claude-opus-5` is a lower bound on the staleness, not an upper bound.
- I read `graph/setup.py` via grep rather than in full, so the role-to-tier table (13 graph agents, exactly 2 on `deep_thinking_llm`) reflects the `create_*` call sites visible on lines 76-92. I did not verify there is no other LLM assignment elsewhere in that file, though `trading_graph.py` constructs only two clients so no third tier can exist.
- I did not read every agent factory (`bull_researcher.py`, `market_analyst.py`, etc.), so my statement that structured output flows through `bind_structured` / `invoke_structured_or_freetext` is based on that module's docstring naming the Portfolio Manager, Trader, and Research Manager as its three callers, plus the `test_structured_agents.py` coverage. Other agents may use structured output through a different path or not at all.
- My 'no prompt caching anywhere' claim is based on grepping the five LLM client files plus `trading_graph.py` and finding no `cache_control` / caching parameter. I did not grep every one of the ~60 files in the repo, so a caching call in an agent factory would have been missed — though the client layer is where it would have to be plumbed, and it is not there.
- The claim that the OpenAI CLI dropdown has no custom-ID escape rests on `MODEL_OPTIONS['openai']` lacking a `('Custom model ID', 'custom')` tuple and `openai` being absent from `_ANY_MODEL_PROVIDERS`. I traced `_select_model` and confirmed the only paths to a free-text model entry are `openrouter`, `azure`, and the `choice == 'custom'` branch — but I did not run the CLI to observe the behavior empirically.

## Sources

- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/default_config.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/graph/trading_graph.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/graph/setup.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/factory.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/base_client.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/openai_client.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/anthropic_client.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/google_client.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/azure_client.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/bedrock_client.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/model_catalog.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/capabilities.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/validators.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/llm_clients/api_key_env.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/agents/utils/structured.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/agents/utils/rating.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/cli/utils.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/cli/main.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/cli/stats_handler.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/cli/models.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/cli/config.py (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/pyproject.toml (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/requirements.txt (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/README.md (fetched 2026-08-18)
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/CHANGELOG.md (fetched 2026-08-18)
- https://api.github.com/repos/TauricResearch/TradingAgents/git/trees/main?recursive=1 (file inventory, 2026-08-18)
- https://api.github.com/repos/TauricResearch/TradingAgents/commits?per_page=5 (latest commit a33fd4c0, 2026-07-18T15:55:04Z)
- claude-api skill, shared/models.md + Current Models table (Anthropic model IDs and pricing, cached 2026-06-24)
- https://apidog.com/blog/how-to-use-gpt-5-6-api/ (GPT-5.6 Sol/Terra/Luna model IDs, verified via search 2026-08-18)
- https://www.vellum.ai/blog/gpt-5-6-sol-terra-luna-explained (GPT-5.6 tier naming and GA date 2026-07-09)
- https://developers.openai.com/api/docs/guides/latest-model (OpenAI current-model guidance, via search result 2026-08-18)
- https://benchlm.ai/openai/api-pricing (OpenAI API pricing August 2026 — GPT-5.5 legacy status, GPT-5.4 pricing; via search snippet, direct fetch blocked)
- https://devtk.ai/en/blog/openai-api-pricing-guide-2026/ (GPT-5.6/5.5/5.4 pricing; via search snippet, direct fetch blocked)
- https://www.cometapi.com/how-much-is-gpt-5-5/ (GPT-5.5 and GPT-5.5 Pro $30/$180 pricing)
- https://ai.google.dev/gemini-api/docs/pricing (Gemini Developer API pricing, via search result 2026-08-18)
- https://devtk.ai/en/blog/gemini-api-pricing-guide-2026/ (Gemini 3.7 Flash launch 2026-08-13; 3.6 Flash and 3.5 Flash-Lite released 2026-07-21; per-model pricing)
- https://benchlm.ai/google/api-pricing (Gemini API pricing August 2026)
- https://openrouter.ai/api/v1/models (the endpoint TradingAgents itself calls in cli/utils._fetch_openrouter_models)