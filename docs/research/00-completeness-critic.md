# Completeness critic — contradictions, corrections, residual risk

> An adversarial pass over the other eight dossiers: what was asserted but not
> verified, where two dossiers disagreed, and what had to be re-checked.

> **Verification method.** I did not take any dossier claim on trust. I used the pre-existing venvs in the shared scratchpad (`langgraph 1.2.11`, `langchain 1.3.15`, `langchain-core 1.5.5`, `anthropic 0.122.0`, `langsmith 0.11.0`), built a new venv with `scikit-learn==1.9.0` + `numpy 2.5.2`, queried PyPI JSON for every version claim, pulled sentence-transformers `model.py` from GitHub master, and loaded the Anthropic-authored `claude-api` skill for model/pricing/batch facts. Everything below marked ✅ was executed, not read. Scratchpad: `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/8075e8db-1d8b-4b54-b04d-7d995fbae90d/scratchpad/` (venvs: `venv`, `lgvenv`, `sk`; my test scripts are `/tmp/v1.py` … `/tmp/v8.py`).

---

# (a) CONTRADICTIONS

## C1. Default recursion limit: 10007 vs 1000 — **langgraph-core is right**

`langgraph-core` says `DEFAULT_RECURSION_LIMIT = 10007`; `langgraph-multiagent` says "default **1000** super-steps since v1.0.6".

✅ **Verified: 10007.**
```
langgraph._internal._config.DEFAULT_RECURSION_LIMIT == 10007
```
The multiagent dossier quoted the docs; the docs are stale. Source reads `int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))`, so it is also environment-overridable — meaning the effective default is deployment-dependent either way. **Set `recursion_limit` explicitly on every `invoke`.** An unbounded P6 refinement loop will spin ~10k super-steps before erroring, which at LLM prices is a five-figure accident.

## C2. "There is no official SqliteStore" — **FALSE, and this is the most consequential error in the dossiers**

`langgraph-memory` states flatly: *"there is **no official `SqliteStore`.** `langgraph-checkpoint-sqlite` ships `SqliteSaver`/`AsyncSqliteSaver` (checkpointers) only"* and recommends writing a custom `BaseStore` subclass over SQLite, tested against `InMemoryStore` as an oracle.

✅ **`langgraph.store.sqlite` exists, ships in `langgraph-checkpoint-sqlite` 3.1.1 (already a dependency), and works with cross-connection persistence.**

```python
import inspect, langgraph.store.sqlite as S
S.__all__  # -> ['AsyncSqliteStore', 'SqliteStore', 'aio', 'base']
# distribution: importlib.metadata.files('langgraph-checkpoint-sqlite')
#   -> langgraph/store/sqlite/{__init__,aio,base}.py
```

Verified signatures:
```python
SqliteStore(conn: sqlite3.Connection, *,
            deserializer: Callable[[bytes|str|orjson.Fragment], dict] | None = None,
            index: SqliteIndexConfig | None = None,
            ttl: TTLConfig | None = None)
SqliteStore.from_conn_string(conn_string: str, *, index=None, ttl=None) -> Iterator[SqliteStore]   # context manager
AsyncSqliteStore(conn: aiosqlite.Connection, *, deserializer=None, index=None, ttl=None)
AsyncSqliteStore.from_conn_string(...) -> AsyncIterator[AsyncSqliteStore]
# both have .setup()
```

✅ End-to-end run (persistence across two separate connections):
```python
with SqliteStore.from_conn_string(p) as st:
    st.setup()
    st.put(("qm","decisions","k12"), "d1", {"summary":"K=42 chosen","seed":7})
    st.search(("qm",))            # -> both items, prefix match confirmed
    st.list_namespaces(prefix=("qm",))  # -> [('qm','decisions','k12')]
with SqliteStore.from_conn_string(p) as st2:
    st2.get(("qm","decisions","k12"),"d1").value  # -> {'summary': 'K=42 chosen', 'seed': 7}  ✅ PERSISTED
```

**Impact:** the whole "Tier 2 = custom BaseStore over SQLite" workstream in the memory dossier is unnecessary. Use `SqliteStore` for local/offline and `PostgresStore` for deployed. Note the store's index config type is `SqliteIndexConfig`, not the generic `IndexConfig` — check its fields before enabling semantic search.

## C3. `BaseStore` custom-backend contract: 5 async methods vs `batch`/`abatch`

`langgraph-memory` says a custom store must implement `aput/aget/adelete/asearch/alist_namespaces` ("all five async methods are required").

✅ **Verified: `BaseStore.__abstractmethods__ == frozenset({'batch', 'abatch'})`.** The five CRUD methods are *concrete* convenience wrappers that funnel into `batch`/`abatch`. Anyone following the dossier writes five methods and still gets `TypeError: Can't instantiate abstract class` for the two that actually matter. Moot in practice given C2, but a real trap if we ever back the store with our own DB.

## C4. LangGraph `RetryPolicy` and Anthropic 429 — the dossier's fix is aimed at the wrong exception

`production-ops` says: *"429 is NOT retried by the default policy (it's 4xx). For Anthropic 429/overloaded you must either extend `retry_on` or rely on the provider SDK's own retries"* and supplies a custom `retry_on` predicate.

✅ **Half right, and the half that's wrong matters.** `default_retry_on` uses a *denylist* of non-retryable exception types, plus special-casing for `httpx`/`requests` HTTP errors:

| Exception | `default_retry_on` |
|---|---|
| `httpx.HTTPStatusError` 429 | **False** ✅ |
| `httpx.HTTPStatusError` 503 | True ✅ |
| `ConnectionError` | True ✅ |
| `ValueError` | False ✅ |
| **`anthropic.RateLimitError`** | **True** ✅ |

`anthropic.RateLimitError`'s MRO is `RateLimitError → APIStatusError → APIError → AnthropicError → Exception` — it is **not** an `httpx.HTTPStatusError`, so it never hits the 4xx special case and falls through to "retry". If we call Anthropic through `langchain-anthropic` or the raw SDK (we will), 429s **are** retried by the stock policy. Writing the custom predicate is harmless but the stated justification is false; the real risk is the opposite one the dossier flags later — **stacked retries** (LangGraph 3 × SDK `max_retries=2` × middleware = up to 18 calls).

✅ `RetryPolicy()` defaults confirmed: `initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True`.

## C5. `langchain.agents.middleware` export list

`langgraph-multiagent` lists exports including `LLMToolEmulator`, `PIIMiddleware`, etc., and separately claims `deepagents` ships `RubricMiddleware`.

✅ Actual `dir(langchain.agents.middleware)` on 1.3.15 adds several the dossier missed — `CodexSandboxExecutionPolicy`, `DockerExecutionPolicy`, `HostExecutionPolicy`, `TracePolicy`, `ModelCallResult`, `InputAgentState`, `OutputAgentState`, `PIIDetectionError`, `PIIMatch`, `RedactionRule`, `omit_payload`, `configure_trace_policy`, `internal_call_transformer`, `model_fallback`, `tool_error` — and contains **no `RubricMiddleware`** (that's a `deepagents` symbol, unverified, and `deepagents` is not installed). Both `ExtendedModelResponse` and `ModelCallResult` exist. Minor, but don't code against the dossier's list verbatim.

## C6. Two dossiers disagree on `temperature` on Sonnet 5

`agent-best-practices` flags this as an open uncertainty; `production-ops` says flatly "rejected (400)". ✅ The Anthropic skill resolves it: on **Opus 5 / Opus 4.8 / 4.7 / Fable 5**, `temperature`/`top_p`/`top_k` are **removed → 400 on any value**. On **Sonnet 5**, a **non-default value** returns 400 but passing the default (or omitting) is accepted. Practical guidance is identical — omit them — but the error surface differs if legacy code passes `temperature=1.0`.

---

# (b) VERIFIED CORRECTIONS

## LangGraph 1.2.11 — exact signatures (all `inspect.signature`, not docs)

```python
StateGraph.__init__(self, state_schema, context_schema=None, *,
                    input_schema=None, output_schema=None, **kwargs)

StateGraph.compile(self, checkpointer=None, *, cache=None, store=None,
                   interrupt_before=None, interrupt_after=None, debug=False,
                   name=None, transformers=None) -> CompiledStateGraph

StateGraph.add_node(self, node, action=None, *, defer=False, metadata=None,
                    input_schema=None, retry_policy=None, cache_policy=None,
                    error_handler=None, destinations=None, timeout=None,
                    trace_policy=None, **kwargs) -> Self

StateGraph.set_node_defaults(self, *, retry_policy=None, cache_policy=None,
                             error_handler=None, timeout=None) -> Self

CompiledStateGraph.invoke(input, config=None, *, context=None,
    stream_mode='values', print_mode=(), output_keys=None,
    interrupt_before=None, interrupt_after=None, durability=None,
    control=None, version: Literal['v1','v2'] = 'v1')

CompiledStateGraph.stream(... same ..., subgraphs=False, debug=None, version='v1')
CompiledStateGraph.stream_events(input, config=None, *,
    version: Literal['v1','v2','v3'] = 'v2', interrupt_before=None,
    interrupt_after=None, control=None, transformers=None)

get_state(config, *, subgraphs=False) -> StateSnapshot
update_state(config, values, as_node=None, task_id=None) -> RunnableConfig

interrupt(value: Any) -> Any
Command(*, graph=None, update=None, resume=None, goto: Send|Sequence[Send|N]|N = ())
Command.PARENT == "__parent__"
Send(node: str, arg: Any, *, timeout: float|timedelta|TimeoutPolicy|None = None)
CachePolicy(*, key_func=default_cache_key, ttl: int|None = None)
TimeoutPolicy(*, run_timeout=None, idle_timeout=None, refresh_on: Literal['auto','heartbeat']='auto')
SqliteCache(*, path: str, serde=None)
Durability = Literal['sync','async','exit']
StreamMode = Literal['values','updates','checkpoints','tasks','debug','messages','custom']
NodeError fields: ('node', 'error')
```

✅ **`NodeTimeoutError.__mro__ == (NodeTimeoutError, Exception, BaseException, object)`; `issubclass(NodeTimeoutError, TimeoutError) is False`.** The docs are wrong. `except TimeoutError:` will not catch it. Import `from langgraph.errors import NodeTimeoutError`.

✅ **`langgraph.config.get_store` and `get_stream_writer()` both import cleanly**; `get_stream_writer() -> Callable[[Any], None]`.

✅ **Store API** (`BaseStore`): `get(ns, key, *, refresh_ttl=None)`, `search(ns_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None)`, `put(ns, key, value, index=False|list[str]|None, *, ttl=NOT_GIVEN)`, `delete(ns, key)`, `list_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)`. `InMemoryStore(*, index: IndexConfig|None = None)`. `IndexConfig` keys: `dims, embed, fields`. `TTLConfig` keys: `refresh_on_read, omit_expired, default_ttl, **sweep_interval_minutes**` (the memory dossier omitted the last one).

## `create_agent` vs `create_react_agent` — both confirmed

✅ `create_react_agent` still imports from `langgraph.prebuilt` and emits, verbatim:
```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`.
Please update your import to `from langchain.agents import create_agent`.
Deprecated in LangGraph V1.0 to be removed in V2.0.
```
✅ `create_agent` signature matches the multiagent dossier **exactly** (14 params, `transformers` last). `langchain.tools.ToolRuntime` resolves to `langgraph.prebuilt.tool_node.ToolRuntime`. `ToolStrategy` fields: `schema, schema_specs, tool_message_content, handle_errors`. `ProviderStrategy` fields: `schema, schema_spec` (dossier said `schema, strict` — **wrong**, there is no `strict` field on `ProviderStrategy` in 1.3.15).

## Send-payload isolation (the P7 anti-anchoring guarantee) — re-verified independently

✅ Ran a fresh graph with `secret_labels: ["LEAK"]` in parent state and `Send("w", {"cid": c})` fan-out. Worker observed:
```
[('c0', ['cid']), ('c1', ['cid'])]
```
Worker `state.keys()` is exactly `['cid']`. Parent state is structurally unreachable. This is the one architectural claim in the dossiers that is both load-bearing and fully confirmed — build P7 on it.

Also ✅ in the same graph: `defer=True` fan-in, `interrupt()` → `__interrupt__` payload with a stable `Interrupt(value=..., id=...)`, and `Command(resume=True)` resumption, all under `SqliteSaver` with `durability="sync"`, **with zero network access**.

## scikit-learn 1.9.0 — all deprecation claims verified

```python
LogisticRegression(penalty='deprecated', *, C=1.0, l1_ratio=0.0, dual=False, tol=1e-4,
                   fit_intercept=True, intercept_scaling=1, class_weight=None,
                   random_state=None, solver='lbfgs', max_iter=100, verbose=0,
                   warm_start=False, n_jobs=None)
```
- ✅ `penalty='l2'` → `FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. … Use l1_ratio=0 instead of penalty…`
- ✅ `solver='liblinear'` + 3 classes → `ValueError: The 'liblinear' solver does not support multiclass classification (n_classes >= 3)`.
- ✅ `AgglomerativeClustering` has **no** `affinity` param; signature is `(n_clusters=2, *, metric='euclidean', memory, connectivity, compute_full_tree, linkage='ward', distance_threshold, compute_distances)`. **`linkage="ward"` + `metric="l2"` is ACCEPTED** (confirms the 1.9 change).
- ✅ `cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None, replace_undefined_by=nan)`. Degenerate case `cohen_kappa_score([0,0,0,0],[0,0,0,0], replace_undefined_by=0.0)` → `0.0` with `UndefinedMetricWarning`. **This is the correct P2 κ-gate implementation** — pass `replace_undefined_by=0.0` so a degenerate annotator fails the gate instead of NaN-poisoning a mean.
- ✅ `KMeans(n_init='auto')`, `MiniBatchKMeans(batch_size=1024, n_init='auto', reassignment_ratio=0.01)`, `BisectingKMeans(init='random', n_init=1, bisecting_strategy='biggest_inertia')` — defaults exactly as the ml-stack dossier states.
- ✅ `sklearn.cluster.HDBSCAN(..., store_centers=None, copy='warn')` — `copy='warn'` confirmed.
- ✅ `CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble='auto')`; `"temperature"` present in source ✅. `sklearn.frozen.FrozenEstimator` ✅. `sklearn.metrics.metric_at_thresholds` ✅. `sklearn.set_config(sparse_interface="sparray")` ✅.
- ⚠️ **Correction:** `SVC(probability=True)` warns *"deprecated in 1.9 and will be removed in version 1.11"* — the ml-stack dossier said deprecated in 1.8.

## sentence-transformers 5.7.0 — `inputs=` rename confirmed, plus two methods the dossier missed

✅ Fetched `sentence_transformers/sentence_transformer/model.py` from master (1427 lines). `def encode(self, inputs: Sequence[SingleInput] | SingleInput, prompt_name=..., prompt=..., batch_size=..., show_progress_bar=..., output_value=..., precision=..., convert_to_numpy=..., convert_to_tensor=..., device=..., normalize_embeddings=..., truncate_dim=..., pool=..., chunk_size=..., **kwargs)` — the `sentences=` → `inputs=` rename is real.

⚠️ **Not in any dossier:** the same module defines **`encode_query()`** and **`encode_document()`** as first-class overloaded methods (6 overloads each, lines 228–470). For asymmetric models (bge-zh, Qwen3-Embedding) these are the correct API and handle the instruction prefix automatically — cleaner than hand-managing `prompt=` and far less error-prone for our P3 bake-off. Investigate before writing manual prompt plumbing.

## Anthropic — model IDs, pricing, batches, refusal (from the Anthropic-authored `claude-api` skill + SDK introspection)

| Model | ID | Ctx | $/1M in | $/1M out |
|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | 1M | 10.00 | 50.00 |
| **Claude Mythos 5** | `claude-mythos-5` | 1M | 10.00 | 50.00 |
| Claude Opus 5 | `claude-opus-5` | 1M | 5.00 | 25.00 |
| Claude Opus 4.8 | `claude-opus-4-8` | 1M | 5.00 | 25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | 3.00 (intro 2.00 → 2026-08-31) | 15.00 (intro 10.00) |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | 1.00 | 5.00 |

`claude-mythos-5` (Project Glasswing only) appears in none of the dossiers — irrelevant to us, but the tables were presented as complete. All other IDs/prices ✅ match `production-ops`. IDs are complete as-is — **never append date suffixes**.

**Thinking/effort — corrections that matter for our cost model:**
- On **Opus 5, thinking is ON by default**; omitting `thinking` runs adaptive. ✅ Confirms `agent-best-practices`. **`max_tokens` caps thinking + response together** — any P7 naming call sized at `max_tokens=1024` from an Opus-4.8-era assumption will now truncate.
- `thinking: {"type":"disabled"}` on Opus 5 is legal **only at effort `high` or below**; pairing with `xhigh`/`max` is a 400, validated **per request**.
- `budget_tokens` → 400 on Opus 5/4.8/4.7/Fable 5/Sonnet 5.
- `thinking.display` defaults to `"omitted"` — set `"summarized"` explicitly or your audit log captures empty strings.
- ⚠️ **New, unflagged risk:** with `thinking: {"type":"disabled"}` on Opus 5, the model can emit a **tool call as plain text** — the turn succeeds, the call never runs, no error. In an agentic loop the bogus text pollutes history. Also leaks `<thinking>` tags. **Do not disable thinking on Opus 5; use `effort: "low"/"medium"` instead.**

**Batches ✅ all confirmed:** ≤100,000 requests or 256 MB; most <1h, max 24h; results retained 29 days; 50% off; prompt caching works inside batches; results in arbitrary order — **key by `custom_id`**. SDK surface verified on `anthropic 0.122.0`: `client.messages.batches` → `create, retrieve, list, results, cancel, delete`. `batches.create(*, requests: Iterable[Request], user_profile_id=..., ...)`. `results()` returns `JSONLDecoder[MessageBatchIndividualResponse]`. **`fallbacks` is rejected on the Batches API** — so the batch path needs client-side re-submission of refused `custom_id`s.

**Refusal ✅:** HTTP **200** with `stop_reason == "refusal"`. `stop_details` is populated **only** when `stop_reason == "refusal"` and is `null` for every other stop reason — **branch on `stop_reason`, never on `stop_details`**. Categories are an open set: `"cyber"`, `"bio"`, `"reasoning_extraction"`, `"frontier_llm"`, or `null`. Server-side fallback: `fallbacks="default"` requires beta `server-side-fallback-2026-07-01`; the array form `fallbacks=[{"model": "claude-opus-4-8"}]` requires `server-side-fallback-2026-06-01`; **pairing either header with the other form is a 400**.

**SDK shape ✅:** `client.messages.parse` exists and accepts `output_format` **and** `output_config` and `thinking`. It does **not** accept `betas` or `fallbacks` — those live only on `client.beta.messages.*`. Error classes present include `OverloadedError` and `RetryableError`, which no dossier mentions and which are useful for our retry predicate.

**JSON-Schema limits ✅ (confirms `agent-best-practices`):** recursive schemas unsupported; `minimum`/`maximum`/`minLength`/`maxLength` unsupported (SDKs strip them and validate client-side). Our two-level taxonomy must be modeled as flat `{l1, l2}` or a bounded `L1(children: list[L2])` — **never** a self-referential `Node(children: list[Node])`.

## Prompt-cache minimums ✅ (non-monotonic — worth restating)

512 (Opus 5, Fable 5, Mythos 5) · **1024** (Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5, Opus 4.1/4) · 2048 (Opus 4.7, Haiku 3.5) · **4096** (Opus 4.6, Opus 4.5, **Haiku 4.5**). Below the minimum it silently doesn't cache. **This bites our design directly:** the plan routes bulk P2 labeling to Haiku 4.5, whose minimum is 4096 — a 2K-token taxonomy prefix will silently never cache on Haiku while caching fine on Opus 5. Either pad the cached prefix past 4096 tokens or route bulk labeling to Sonnet 5.

---

# THE BIG GAP: offline / no-API-key deterministic mode

**No dossier addressed this, and the obvious approach does not work.**

✅ **`GenericFakeChatModel.with_structured_output(...)` raises `NotImplementedError: with_structured_output is not implemented for this model.`** — for both the default tool-calling method and `method="json_mode"`. The same is true of every `langchain_core.language_models.fake_chat_models` class (`FakeChatModel`, `FakeListChatModel`, `FakeMessagesListChatModel`, `ParrotFakeChatModel`). Since P2 labels and P7 naming both depend on structured output, **the stock fakes cannot run our pipeline offline.** `production-ops` recommends `GenericFakeChatModel` as the testing path without noticing this.

✅ What *does* work on the fakes: `GenericFakeChatModel(messages=iter([...]))` yields scripted `AIMessage`s including `tool_calls` (verified: tool call then text). Fields confirmed: `messages`; `FakeListChatModel` has `responses, sleep, i, error_on_chunk_number`.

✅ **Working replacement, verified end-to-end.** A ~25-line `BaseChatModel` subclass that implements `bind_tools` restores `with_structured_output` for free (it routes through the tool-calling path), and works inside `create_agent(response_format=...)`:

```python
class DeterministicFakeChat(BaseChatModel):
    responder: Any = None       # fn(messages, tools) -> AIMessage
    bound_tools: list = []
    @property
    def _llm_type(self): return "deterministic-fake"
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=self.responder(messages, self.bound_tools))])
    def bind_tools(self, tools, **kw):
        specs = [convert_to_openai_tool(t) for t in tools]
        return self.__class__(responder=self.responder, bound_tools=specs)
```
With a `responder` that hashes the last `HumanMessage` and fills the bound schema:
```
plain:      echo:hello
structured: name='cluster_87' confidence=0.787   | deterministic across calls: True   ✅
create_agent structured_response: name='cluster_87' confidence=0.787                  ✅
```
Same input → identical `ClusterName` every run, no network, no key.

✅ **LangSmith is genuinely opt-in.** With all `LANGSMITH_*`/`LANGCHAIN_*` unset, `langsmith.utils.tracing_is_enabled()` is `False`; with `LANGSMITH_TRACING=false`, also `False`. `Client.evaluate` does expose `upload_results` ✅. Combined with `SqliteSaver` + `SqliteStore` + `SqliteCache`, the **entire 12-phase pipeline can run with zero network egress** — which is the CI story the dossiers gestured at but never demonstrated.

---

# (c) REMAINING RISKS

1. **`SqliteIndexConfig` is unexplored.** `SqliteStore` takes `index: SqliteIndexConfig | None`, a *different* type from the generic `IndexConfig(dims, embed, fields)`. Whether it supports the same `fields=["summary","rationale"]` / `index=False` semantics, and whether it needs `sqlite-vec` (present in the venv), is unverified. If P12 semantic search over the decision ledger matters, verify before designing around it.
2. **`langmem` is 10 months stale and I did not execute it.** `langmem 0.0.30` (2025-10-27) declares `langchain-core>=0.3.46`; installed is 1.5.5. It *should* import, but its docs target `create_react_agent` (now deprecation-warning). `trustcall` (its dedup engine) last shipped 2025-04-14. Given C2, the primary reason to adopt langmem has evaporated — but if we want `create_prompt_optimizer` for the P2 rubric loop, smoke-test it first.
3. **`agentevals` 0.0.9 last shipped 2025-07-24** and depends on `openevals>=0.0.20` (current 0.2.0, 2026-04-07). `production-ops` recommends agentevals for trajectory evals; it may be effectively unmaintained. `openevals` pulls `langchain-openai` — an unwanted dependency for an Anthropic-only stack.
4. **Determinism of the real LLM path is still unsolved and nobody should pretend otherwise.** `temperature=0` is a 400 on our target models. The response-cache design in `agent-best-practices` is the right answer, but it means P0 "reproducibility" for LLM steps = *replay*, not *regeneration*. Record `model_served` separately from `model` (they differ under fallbacks).
5. **Node timeouts require async nodes.** LangGraph rejects `timeout=` on sync nodes at compile time. Our P3/P4 nodes are sync sklearn/pandas. Either `async def` + `asyncio.to_thread`, or accept no per-node timeouts. I verified the parameter exists but did not trigger the compile-time rejection.
6. **Unverified by me** (asserted in dossiers, plausible, but untested here): `Overwrite`, `DeltaChannel`, `TracePolicy` runtime behavior, `PostgresSaver`/`PostgresStore` (never installed), `stream_events(version="v3")` projections, `defer=True` interaction with error handlers, `Command.PARENT` + return-annotation compile error, node cache cross-thread bleed, and every UMAP/HDBSCAN/faiss claim (I installed sklearn only — not umap-learn, hdbscan, faiss, torch, or sentence-transformers).
7. **The multi-agent literature numbers are unverifiable and dated.** Anthropic's "90.2% improvement" and "15× tokens" are from a June 2025 post using Opus 4 / Sonnet 4. Do not put them in a budget model for 2026-era models.
8. **`create_agent` + fake model works, but the LLM-in-the-loop determinism of P7 still depends on the provider.** My Send-isolation proof used plain Python workers and a scripted fake; I did not prove isolation holds when the worker internally invokes a real `create_agent` (it should — the agent only sees what you pass to `.invoke()` — but the model call itself is nondeterministic).


## Recommendations

- Use the official `langgraph.store.sqlite.SqliteStore` (ships in the already-required `langgraph-checkpoint-sqlite` 3.1.1) instead of writing a custom BaseStore — I verified it exists, has `.setup()`, does prefix search, and persists across separate connections, which deletes an entire workstream the memory dossier proposed.
- Write a ~25-line `DeterministicFakeChat(BaseChatModel)` that implements `bind_tools`, because `GenericFakeChatModel.with_structured_output()` raises NotImplementedError and would otherwise block every offline test of P2 labeling and P7 naming; I verified the replacement works with `create_agent(response_format=...)` and is byte-identical across runs.
- Set `recursion_limit` explicitly on every invoke — the verified default is 10007 (not 1000), so an unbounded P6 refinement loop would burn ~10k super-steps of LLM calls before erroring.
- Import `NodeTimeoutError` from `langgraph.errors` and never write `except TimeoutError` — verified MRO is (NodeTimeoutError, Exception, BaseException, object), contradicting the docs.
- Do not disable thinking on Opus 5: it can emit tool calls as plain text that silently never execute (no error, poisoning agentic history) and leak `<thinking>` tags; use `effort: 'low'/'medium'` to control cost instead.
- Re-size `max_tokens` on every Opus 5 call — thinking is ON by default and `max_tokens` caps thinking plus response together, so limits carried over from a non-thinking model will truncate mid-answer.
- Route bulk P2 labeling to Sonnet 5 rather than Haiku 4.5, or pad the cached prefix past 4096 tokens — Haiku 4.5's minimum cacheable prefix is 4096 (vs 512 on Opus 5), so a 2K taxonomy prefix silently never caches and the 50%-batch + cache saving evaporates.
- Model the two-level taxonomy as flat `{l1, l2}` or a bounded `L1(children: list[L2])` — recursive JSON schemas are unsupported by Anthropic structured outputs, and Pydantic `max_length`/`ge` constraints are stripped before transmission and enforced client-side only.
- Branch on `stop_reason == 'refusal'` before ever reading `response.content`, and note `stop_details` is null for every non-refusal stop reason; add client-side re-submission of refused `custom_id`s on the batch path since `fallbacks` is rejected by the Batches API.
- Pass `replace_undefined_by=0.0` to `sklearn.metrics.cohen_kappa_score` in the P2 gate (new in 1.9, verified) so a degenerate single-label annotator scores 0 and fails the gate rather than returning NaN and poisoning downstream means.
- Drop the custom `retry_on` predicate written to 'fix' Anthropic 429s — `anthropic.RateLimitError` is not an httpx.HTTPStatusError and is already retried by `default_retry_on`; spend the effort instead on preventing stacked retries (LangGraph 3 x SDK 2 x middleware = up to 18 calls).
- Investigate `SentenceTransformer.encode_query()` / `encode_document()` (present in 5.7.0 master, absent from every dossier) before hand-rolling instruction-prefix plumbing for asymmetric models like Qwen3-Embedding.
- Build the P7 blind-naming fan-out on `Send(node, {only_shard_fields})` — I independently re-verified that a worker's `state.keys()` contains exactly the Send payload keys and parent `secret_labels` is unreachable, making anti-anchoring structural rather than prompt-dependent.
- Treat P0 'reproducibility' for LLM steps as replay, not regeneration: `temperature=0` is a 400 on the target models, so build the sha256-keyed record/replay cache and record `model_served` separately from `model` since they diverge under server-side fallbacks.
- Smoke-test `langmem` and `agentevals` before adopting either — langmem last shipped 2025-10-27 against langchain-core 0.3.x (installed: 1.5.5) and agentevals last shipped 2025-07-24; with SqliteStore available, the main reason to depend on langmem is gone.