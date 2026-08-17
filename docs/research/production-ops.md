# Production Observability, Evaluation & Cost Control

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

# Production concerns for a LangGraph multi-agent batch data-science system (verified 2026-08-17)

## 0. Method / caveat on sources

`docs.langchain.com`, `pypi.org` and `github.com` are **blocked to WebFetch** in this environment. I obtained everything two ways instead:

1. `curl https://pypi.org/pypi/<pkg>/json` for exact version + release dates.
2. Sparse `git clone` of **`langchain-ai/docs`** (the source repo behind `docs.langchain.com`), reading `src/langsmith/*.mdx` and `src/oss/langgraph/*.mdx`, `src/oss/langchain/*.mdx` directly. These are the same files that render as the doc URLs in the assignment.
3. **I installed the packages into a venv and introspected/ran them.** Several claims below are empirically verified, not just read.

## 1. Version matrix (PyPI, fetched 2026-08-17)

| Package | Latest | Uploaded |
|---|---|---|
| `langgraph` | **1.2.11** | 2026-08-11 |
| `langgraph-checkpoint` | 4.2.0 | 2026-08-07 |
| `langgraph-checkpoint-sqlite` | 3.1.1 | 2026-07-30 |
| `langgraph-cli` | 0.4.31 | 2026-07-10 |
| `langgraph-sdk` | 0.4.2 | 2026-06-01 |
| `langsmith` | **0.11.0** | 2026-08-14 |
| `langchain` | 1.3.15 | 2026-08-11 |
| `langchain-core` | 1.5.5 | 2026-08-14 |
| `langchain-anthropic` | 1.5.6 | 2026-08-13 |
| `anthropic` | 0.122.0 | 2026-08-13 |

Pin these. Note the LangGraph 1.x line: **per-node `timeout=`, `error_handler=`, and `set_node_defaults()` require `langgraph>=1.2`** (docs state this explicitly). JS equivalents need `@langchain/langgraph>=1.4.0`.

---

## 2. LangSmith tracing — env vars, offline, instrumentation

### 2.1 Exact env var names (verified against docs + SDK)

| Var | Meaning |
|---|---|
| `LANGSMITH_TRACING` | `"true"` enables tracing. **Unset/false ⇒ fully offline, zero network.** |
| `LANGSMITH_API_KEY` | API key |
| `LANGSMITH_ENDPOINT` | API base URL. **Required for non-US regions and self-hosting.** EU: `https://eu.api.smith.langchain.com`. *No trailing slash* — a trailing slash causes auth errors. |
| `LANGSMITH_PROJECT` | Project name (default project is literally `default`) |
| `LANGSMITH_WORKSPACE_ID` | Required when the API key spans multiple workspaces |
| `LANGSMITH_TRACING_SAMPLING_RATE` | float `0.0`–`1.0`, global probabilistic sampling |
| `LANGSMITH_TRACING_MODE` | `langsmith` (default) \| `otel` \| `hybrid` |
| `LANGSMITH_HIDE_INPUTS` / `LANGSMITH_HIDE_OUTPUTS` | blanket redaction |
| `LANGSMITH_TEST_SUITE`, `LANGSMITH_EXPERIMENT`, `LANGSMITH_EXPERIMENT_METADATA` (JSON), `LANGSMITH_TEST_CACHE`, `LANGSMITH_TEST_TRACKING=false` | pytest plugin |
| `LANGCHAIN_*` legacy aliases | `LANGCHAIN_API_KEY`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT` still read; `LANGSMITH_PROJECT` needs JS SDK ≥ 0.2.16 |
| `LANGCHAIN_CALLBACKS_BACKGROUND` | JS only; `true` off-serverless for latency, `false` in serverless so traces flush |

### 2.2 Offline story (critical for your laptop requirement — empirically verified)

I ran a full LangGraph graph (SQLite checkpointer, node cache, retries, fake model) with all `LANGSMITH_*`/`LANGCHAIN_*` env vars deleted and `LANGSMITH_TRACING=false`. It ran with no network access and no errors. **Tracing is strictly opt-in; there is no phone-home.** So the design rule is: *never* hard-require LangSmith. Make it a config flag.

Three additional offline levers:

- `ls.tracing_context(enabled=False)` — highest-priority runtime override (beats `ls.configure(enabled=...)`, which beats the env var). Also accepts `enabled="local"`.
- `client.evaluate(..., upload_results=False)` — runs the app *and* evaluators exactly as normal, returns the same `ExperimentResults`, records **nothing** to LangSmith (including app/evaluator traces). `ExperimentResults.to_pandas()` works offline. This is your CI/laptop eval path.
- `LANGSMITH_TEST_TRACKING=false` — dry-run mode for the pytest plugin.

Self-hosting: `LANGSMITH_ENDPOINT=https://langsmith.yourdomain.com/api`, plus the Terraform/Helm/K8s stacks documented in `self-host-*.mdx` (AWS/Azure/GCP), `LANGSMITH_LICENSE_KEY`. There is also a **`langsmith-collector`** (OTLP→LangSmith proxy) and `tracing_mode="otel"` if you'd rather ship OTLP to your own collector and keep everything on-prem.

### 2.3 `@traceable` and friends (Python)

```python
from langsmith import traceable, trace, tracing_context, get_current_run_tree, uuid7, Client

@traceable(run_type="tool", name="Regex family miner", tags=["p1"], metadata={"phase": "P1"})
def mine_templates(queries: list[str]) -> dict: ...

# invocation-time extras
mine_templates(qs, langsmith_extra={"run_id": uuid7(), "tags": ["rerun"], "metadata": {"k": "v"}})

# dynamic metadata inside a run
rt = get_current_run_tree(); rt.metadata["kappa"] = 0.93; rt.tags.append("gate-passed")

# context manager (Python only) — good for wrapping a whole phase
with trace("P4 clustering battery", "chain", project_name="qim", inputs={"k_grid": ks}) as rt:
    out = run_battery(...)
    rt.end(outputs={"best_k": out.k})
```

- `run_type` values that matter: `"llm"` (gets token/cost rendering), `"tool"`, `"chain"`, `"retriever"`.
- **Custom run IDs must be UUIDv7** — `from langsmith import uuid7` (Python ≥0.4.43). UUIDv7 embeds a timestamp so run ordering inside a trace stays correct. Use this to correlate a LangSmith run with your own `run_id` in the manifest.
- `RunTree` API is the low-level escape hatch; note it **ignores `LANGSMITH_TRACING`** and always posts once you call `.post()`. Avoid it.
- **Flush before exit** (background thread uploads): `client.flush()` in a `finally:`. Essential for a batch job that `sys.exit()`s.

### 2.4 Per-node tracing control in LangGraph — `TracePolicy` (new, under-documented)

`StateGraph.add_node(..., trace_policy=TracePolicy(process_inputs=fn, process_outputs=fn))`. Frozen dataclass with exactly two fields. Scope: transforms only *that node's own* run record — child runs and the root graph run are unaffected. Docstring explicitly says it is **not** for secret redaction (use `Client(hide_inputs=..., hide_outputs=..., anonymizer=...)` for that). This is the right tool for a node whose state contains a 5000-row dataframe you don't want in the trace payload:

```python
from langgraph.types import TracePolicy
builder.add_node("embed_bakeoff", embed_bakeoff,
    trace_policy=TracePolicy(process_inputs=lambda s: {"n_rows": len(s["queries"])},
                             process_outputs=lambda o: {"model": o["best_model"], "dim": o["dim"]}))
```

### 2.5 Multi-tenant / redaction / sampling

- `tracing_context(replicas=[{"project_name": "...", "updates": {"inputs": {}, "outputs": {}}}])` — keeps the trace shape/timing/errors but blanks payloads per-request. **Always set `project_name` on the replica**, or the `updates` may be dropped and unredacted data sent.
- `Client(tracing_sampling_rate=0.25)` per-client + `tracing_context(client=...)` for per-operation sampling.
- Sampling = probabilistic; `tracing_context(enabled=...)` = deterministic. Combine.

### 2.6 Per-node latency/cost in the UI

For LangChain/LangGraph calls, token counts and cost are automatic. For hand-rolled LLM calls, set `run_type="llm"` + metadata `{"ls_provider": ..., "ls_model_name": ...}` and attach usage via `get_current_run_tree().set(usage_metadata={...})` — the `usage_metadata` dict supports `input_tokens`, `output_tokens`, `total_tokens`, `input_token_details: {"cache_read": N}`. **Gotcha for threaded/aggregated cost:** child runs must carry the thread metadata (`session_id`/`thread_id`) or their tokens are excluded from thread-level aggregation.

---

## 3. Evaluation

### 3.1 Datasets & experiments (langsmith 0.11 API — note the shape change)

The canonical entry point is now the **client method**, not only the module function:

```python
from langsmith import Client
client = Client()
ds = client.create_dataset("qim-gold-5k", description="P2 gold labels")
client.create_examples(dataset_id=ds.id, examples=[
    {"inputs": {"query": "how to solve quadratic equations"},
     "outputs": {"l1": "concept_learning", "l2": "algebra_method"}},
])
res = client.evaluate(target, data="qim-gold-5k", evaluators=[...], summary_evaluators=[...],
                      experiment_prefix="p2-classifier", max_concurrency=8,
                      num_repetitions=1, blocking=False, upload_results=True,
                      error_handling="log")
```

Verified `Client.evaluate` signature: `(target, /, data=None, evaluators=None, summary_evaluators=None, metadata=None, experiment_prefix=None, description=None, max_concurrency=0, num_repetitions=1, blocking=True, experiment=None, upload_results=True, error_handling='log', **kwargs)`. Module-level `langsmith.evaluate` / `langsmith.aevaluate` still exist (lazy `__getattr__`); `evaluate_comparative` is **not** exported top-level in 0.11 — use `client.evaluate((exp_a, exp_b), evaluators=[...])`.

### 3.2 Evaluator signatures (argument names are load-bearing — matched by name)

Single-experiment: any subset of `inputs: dict`, `outputs: dict`, `reference_outputs: dict`, `run: Run`, `example: Example`.
Pairwise: `outputs: list[dict]` (two items), `runs: list[Run]`, plus `inputs`/`reference_outputs`/`example`.

Returns: a `bool`/`float`, a `dict` with `{"key", "score", "comment"}`, or a list of such dicts for multiple metrics. Pairwise returns `{"key", "scores": {run_id: score}, "comment"}` or (Python only) a 2-item list of scores. Docs recommend prefixing pairwise feedback keys `pairwise_`/`ranked_`.

### 3.3 LLM-as-judge

Use `openevals` (`pip install openevals`):
```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, model="anthropic:claude-opus-5", feedback_key="correctness")
def correctness(inputs, outputs, reference_outputs): return judge(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)
```
`CORRECTNESS_PROMPT` is just an f-string with `{inputs}/{outputs}/{reference_outputs}` — trivially customizable for taxonomy-fit / naming-quality rubrics.

`langchain.agents.middleware` also ships **`RubricGradingMiddleware`** (built-in middleware §"Rubric grading") if you want grading inside the agent loop.

### 3.4 Pairwise experiments

`client.evaluate(("experiment-1","experiment-2"), evaluators=[ranked_preference], randomize_order=True, max_concurrency=5, load_nested=False)`. `randomize_order=True` mitigates positional bias. Perfect for P7 blind-naming A/B (naming agent v1 vs v2) and P5 K-selection comparisons.

### 3.5 CI

Two supported shapes:

**(a) pytest plugin** (`pip install -U "langsmith[pytest]"`, needs `langsmith>=0.3.4`):
```python
import pytest
from langsmith import testing as t

@pytest.mark.langsmith(output_keys=["expected_label"])
@pytest.mark.parametrize("query,expected_label", CASES)
def test_l1_classifier(query, expected_label):
    t.log_inputs({"query": query}); t.log_reference_outputs({"label": expected_label})
    got = classify(query); t.log_outputs({"label": got})
    with t.trace_feedback():                       # judge traced separately
        t.log_feedback(key="exact", score=float(got == expected_label))
    assert got == expected_label
```
Extras: `LANGSMITH_TEST_SUITE="QIM classifier"`, `LANGSMITH_EXPERIMENT="baseline"`, `pytest --langsmith-output` (rich table; **incompatible with pytest-xdist**), `pytest -n auto` for xdist, session-scoped `langsmith_experiment_metadata` fixture in `conftest.py` (needs `langsmith>=0.7.13`), `langsmith.expect(...)` with `.to_contain`, `expect.embedding_distance(...).to_be_less_than(0.5)`, `expect.edit_distance(...)`.
**HTTP cassette caching**: `LANGSMITH_TEST_CACHE=tests/cassettes pytest tests/` — commit the cassettes so CI never pays for LLM calls. `langsmith>=0.4.10` allows selective `@pytest.mark.langsmith(cached_hosts=["https://api.anthropic.com"])`.

**(b) `evaluate()` + quality gate** (the doc's own pattern): run with `blocking=False`, iterate, compute an aggregate, `sys.exit(1)` below threshold. `result["run"].inputs/.outputs/.id`, `result["evaluation_results"]["results"][i].{key,score,comment,source_run_id}`, `result["example"].inputs/.outputs`.

**Spend control on evaluators** (new): org-wide weekly evaluator spend cap, resets Monday 00:00 UTC, per-project/dataset override; only OpenAI/Anthropic/Gemini with configured pricing. When the cap is hit LangSmith *pauses that evaluator* — traces and the agent keep running.

---

## 4. Testing multi-agent graphs

### 4.1 Fake LLMs (Python — exact names)

`langchain_core.language_models.fake_chat_models` exports: `FakeChatModel`, `FakeListChatModel`, `FakeListChatModelError`, `FakeMessagesListChatModel`, `GenericFakeChatModel`, `ParrotFakeChatModel`.

- **`GenericFakeChatModel(messages=iter([...]))`** — the one the docs recommend. Accepts `AIMessage` objects *or* plain strings; yields one per `invoke()`; supports streaming. **Verified working**: I queued an `AIMessage` with `tool_calls=[{"name","args","id","type":"tool_call"}]` then a string, and got the tool call then `"second"`.
- **`FakeListChatModel(responses=[...], sleep=..., error_on_chunk_number=...)`** — string list; `error_on_chunk_number` injects a mid-stream failure, which is how you unit-test streaming error handling.
- `FakeMessagesListChatModel(responses=[AIMessage,...])`.
- All of them inherit `rate_limiter`, `cache`, `callbacks`, `disable_streaming` fields — so you can attach your budget-guard callback in tests.
- (JS has a richer builder `fakeModel().respond(...).respondWithTools(...).alwaysThrow(...).structuredResponse(...)` with `.callCount`/`.calls` spy assertions. Python has no direct equivalent — write a tiny `BaseChatModel` subclass if you want call recording.)

### 4.2 Node isolation

Compiled graphs expose `graph.nodes` — `compiled.nodes["p3_embed"].invoke({...})` runs one node and **bypasses the checkpointer**. This is the cheapest unit test for a 12-phase pipeline.

### 4.3 Partial execution ("run only P4→P6")

Documented three-step recipe:
1. compile with `InMemorySaver()`,
2. `compiled.update_state(config={"configurable":{"thread_id":"t"}}, values={...}, as_node="p3")` — inject the state *as if* P3 had just finished,
3. `compiled.invoke(None, config, interrupt_after="p6")`.

This is how you regression-test a middle phase without re-running the expensive upstream phases.

### 4.4 Checkpoint snapshot tests

`st = graph.get_state(config)` returns a `StateSnapshot` with `.values`, `.next`, `.tasks` (each task has `.name`, `.error`), `.config`, `.metadata`. Snapshot-assert `st.values` after each superstep; assert `st.next == ("p5_k_selection",)` for routing tests. `graph.get_state_history(config)` gives the full time-travel list.

### 4.5 Determinism

Seed everything at P0 and assert reproducibility by hashing artifacts, not by comparing floats. Set `PYTHONHASHSEED`, numpy/sklearn `random_state`, `torch.manual_seed`. **Important gotcha:** the default LangGraph cache key is `pickle.dumps(...)` of node inputs — pickle of a `dict` is stable but pickle of a `set` is *not* order-stable across processes. If you cache a node whose input contains sets, pass a custom `key_func`.

---

## 5. Reliability (LangGraph 1.2 — this is the strongest part of the stack for you)

### 5.1 `RetryPolicy` — verified constructor

`RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True, retry_on=default_retry_on)` from `langgraph.types`.

`default_retry_on` source (read from the installed package) — retries **everything except**: `ValueError, TypeError, ArithmeticError, ImportError, LookupError, NameError, SyntaxError, RuntimeError, ReferenceError, StopIteration, StopAsyncIteration, OSError`. It retries `ConnectionError`; for `httpx.HTTPStatusError` and `requests.HTTPError` it retries **only 5xx**. `NodeTimeoutError` is retryable by default.

**⚠️ 429 is NOT retried by the default policy** (it's 4xx). For Anthropic 429/overloaded you must either extend `retry_on` or rely on the provider SDK's own retries (see 5.6):

```python
from langgraph.types import RetryPolicy, default_retry_on
import anthropic

def retry_on_llm(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIStatusError)):
        return getattr(exc, "status_code", 0) in (408, 409, 429, 500, 502, 503, 504, 529)
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return default_retry_on(exc)

builder.add_node("label", label_node,
    retry_policy=RetryPolicy(max_attempts=6, initial_interval=1.0,
                             backoff_factor=2.0, max_interval=120.0,
                             jitter=True, retry_on=retry_on_llm))
```
`retry_policy` also accepts a **`Sequence[RetryPolicy]`** — multiple policies with different `retry_on` predicates on one node.

**Verified**: a node raising `ConnectionError` twice succeeded on attempt 3 under `RetryPolicy(max_attempts=3)`.

### 5.2 `TimeoutPolicy` (langgraph ≥ 1.2)

`TimeoutPolicy(run_timeout=None, idle_timeout=None, refresh_on="auto"|"heartbeat")`; `add_node(..., timeout=60 | timedelta(...) | TimeoutPolicy(...))`.

- `run_timeout` = hard wall-clock cap per attempt, never refreshed.
- `idle_timeout` = progress-resetting. Under `refresh_on="auto"` it resets on state writes, stream output, child-task scheduling, `runtime.stream_writer` calls, and **any LangChain callback event from the node or its descendants** (LLM tokens, tool calls, chain start/end). Under `"heartbeat"`, only explicit `runtime.heartbeat()` resets it. `runtime.heartbeat()` is a no-op outside an idle-timed attempt, so call it unconditionally inside long loops (e.g. per-batch in a 5000-row labeling loop).
- **Node timeouts apply to async nodes only.** A sync node with `timeout=` is **rejected at compile time**. Wrap blocking sklearn/pandas work in `asyncio.to_thread`.
- On fire: raises `NodeTimeoutError(node, elapsed, kind: "idle"|"run", idle_timeout, run_timeout)`, **clears the failed attempt's writes**, then the retry policy decides.
- `Send("process_item", {...}, timeout=TimeoutPolicy(idle_timeout=15))` overrides the target node's static timeout per fan-out item — ideal for map-reduce over 5000 rows.

### 5.3 Error handlers / Saga compensation (langgraph ≥ 1.2)

`add_node(..., error_handler=fn)`. Handler signature `(state)`, `(state, error: NodeError)`, or `(state, error: NodeError, config: RunnableConfig)` — the `error: NodeError` param is injected **by type annotation** (same mechanism as `runtime: Runtime`). `NodeError` is a frozen dataclass with `.node: str` and `.error: BaseException`. Handler can return a state update or a `Command(update=..., goto="...")`.

Ordering is fixed: exception (incl. `NodeTimeoutError`) → `retry_policy` → (exhausted) → `error_handler`. **`interrupt()` bypasses both** (uses `GraphBubbleUp`). Subgraph exceptions surface to the parent node and hit the parent's handler. **Failure provenance is checkpointed** — if the process crashes after the node fails but before the handler completes, the handler sees the same `NodeError` on resume.

### 5.4 `set_node_defaults` (langgraph ≥ 1.2)

```python
graph = (StateGraph(State)
    .set_node_defaults(retry_policy=RetryPolicy(max_attempts=3),
                       timeout=TimeoutPolicy(run_timeout=1800, idle_timeout=120),
                       error_handler=mark_phase_failed,
                       cache_policy=CachePolicy(ttl=86400))
    .add_node("p1_audit", p1)          # inherits all four
    .add_node("p8_governance", p8, error_handler=rollback_merges)  # per-node wins
    .compile())
```
Resolved at `compile()` time, so order doesn't matter. **Applicability matrix**: error-handler nodes inherit `retry_policy` and `timeout` but are excluded from `error_handler` (can't catch themselves) and `cache_policy` (unsafe). **Defaults are not inherited by subgraphs** — set them per graph.

### 5.5 Durability + partial failure recovery (empirically verified)

`durability: Literal["sync","async","exit"]` on `invoke`/`stream`/`ainvoke`/`astream`:
- `"exit"` — persist only when execution exits (fastest; **no mid-run crash recovery**).
- `"async"` — persist while the next step runs (good default; small crash window).
- `"sync"` — persist before the next step starts (**use this for a 12-phase job**).

**Verified end-to-end**: with `SqliteSaver` + `durability="sync"`, a graph `a→b→c` where `b` raised: `get_state()` afterwards showed `values={'a':'A'}`, `next=('b',)`, `tasks=[('b', "RuntimeError('phase B crashed')")]`. Calling `graph.invoke(None, config, durability="sync")` after fixing the fault **resumed and re-ran only `b` and `c`** (`a` ran exactly once). That is your "fix P6 and rerun without redoing P0–P5" story, for free.

Checkpointers: `langgraph.checkpoint.memory.InMemorySaver` (bundled), `langgraph-checkpoint-sqlite` → `SqliteSaver.from_conn_string("./run.sqlite")` (context manager) / `AsyncSqliteSaver`, `langgraph-checkpoint-postgres` → `PostgresSaver`/`AsyncPostgresSaver` (call `.setup()` once). Serializer: `JsonPlusSerializer(pickle_fallback=True)` — **you need this for pandas DataFrames / numpy arrays in state**. Encryption: `EncryptedSerializer.from_pycryptodome_aes()` reading `LANGGRAPH_AES_KEY`. `DeltaChannel` (langgraph ≥1.2, beta) stores incremental deltas instead of full channel values — relevant if you keep a growing list of 5000 labeled rows in state.

### 5.6 Graceful shutdown

```python
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained
control = RunControl()
signal.signal(signal.SIGTERM, lambda *_: control.request_drain("sigterm"))
try:
    graph.invoke(inputs, config, control=control, durability="sync")
except GraphDrained as e:
    log.info("drained: %s; resume with same thread_id", e.reason)
```
Stops after the current superstep and saves a resumable checkpoint. `request_drain()` is thread-safe.

### 5.7 Provider-level retries (complementary layer)

- `ChatAnthropic(max_retries=2)` (default 2; SDK auto-retries 408/409/429/5xx with backoff), `default_request_timeout` (alias `timeout`).
- Runnable-level: `model.with_retry(stop_after_attempt=6)`.
- Agent-level middleware: `ModelRetryMiddleware(max_retries=3, retry_on=(Exception,), on_failure='continue'|'error'|callable, backoff_factor=2.0, initial_delay=1.0, max_delay=60.0, jitter=True)` — `on_failure='continue'` returns an `AIMessage` with the error so the agent can recover instead of aborting.
- `ModelFallbackMiddleware("anthropic:claude-sonnet-5", "anthropic:claude-haiku-4-5")` — provider/model redundancy.

**Don't stack all four blindly** — 3 retries × 3 SDK retries × 3 middleware retries = 27 calls. Pick one primary layer per failure class: SDK retries for transport/429, LangGraph `RetryPolicy` for node-level idempotent work, middleware fallback for outages.

### 5.8 Idempotent artifact writes (design guidance, no doc)

Nodes get retried and resumed, so **every node must be safe to run twice**. Pattern:
- write to `artifacts/<run_id>/<phase>/<name>.<ext>.tmp`, `fsync`, then `os.replace()` (atomic on POSIX);
- name outputs by a content hash of `(input_hash, config_hash, seed)` — a re-run with identical inputs produces an identical path, so re-running is a no-op;
- keep node state **references** (paths + hashes), not payloads — this keeps checkpoints small, keeps traces readable, and makes the pickle cache key stable;
- record `runtime.execution_info.node_attempt` in the artifact sidecar so you can tell a retried write apart from a first write.

`Runtime.execution_info` fields: `node_attempt` (1-indexed), `node_first_attempt_time`, `thread_id`, `run_id`, `checkpoint_id`, `task_id`. Available even without a retry policy. Use it for fallback-on-retry logic:
```python
def label_node(state: State, runtime: Runtime):
    if runtime.execution_info.node_attempt > 2:
        return {"labels": call_cheaper_model(state)}   # degrade after repeated failures
    return {"labels": call_opus(state)}
```

---

## 6. Cost control

### 6.1 Node-level caching — `CachePolicy` (verified)

`CachePolicy(key_func=default_cache_key, ttl: int | None = None)` from `langgraph.types`; enable with `builder.compile(cache=...)`.

Caches available: `langgraph.cache.memory.InMemoryCache`, `langgraph.cache.sqlite.SqliteCache(path="...")` (ships with `langgraph-checkpoint-sqlite`), `langgraph.cache.redis`. `ttl` is seconds; `None` = never expires.

**Verified behavior**: an `expensive_node` with `CachePolicy(ttl=60)` ran once across **two different `thread_id`s** — the cache is graph-scoped and keyed on node input, not thread. Second run's update carried `'__metadata__': {'cached': True}`. `default_cache_key` = `pickle.dumps((frozen_args, frozen_kwargs), protocol=5)`.

For your pipeline, `SqliteCache` on P3 (embedding bake-off) and P4 (clustering battery) is a large win: re-running the graph after tweaking P5 skips recomputation entirely. **Do set a `ttl`** or a stale embedding cache will silently survive a model change — better, put the model name/version into the node input so the key changes.

### 6.2 Two-tier model routing

```python
from langchain.chat_models import init_chat_model
DEEP  = init_chat_model("anthropic:claude-opus-5",  max_tokens=16000, thinking={"type": "adaptive"},
                        output_config={"effort": "high"})
CHEAP = init_chat_model("anthropic:claude-haiku-4-5", max_tokens=1024)
```
`init_chat_model(model, *, model_provider=None, configurable_fields=None, config_prefix=None, **kwargs)` — the `"provider:model"` prefix form works and passes kwargs to `ChatAnthropic`. `configurable_fields="any"` returns a `_ConfigurableModel` so you can flip model per-invocation via `config={"configurable": {"model": ...}}` — handy for a `--cheap` CI flag.

Route: Haiku 4.5 for the 5000-row bulk labeling and regex-family triage; Opus 5 for taxonomy design, referee adjudication, and the tree-audit agent.

### 6.3 Anthropic prompt caching (the biggest lever for 5000-row labeling)

Two mechanisms, both current in `langchain-anthropic` 1.5.6:

**(a) Automatic (requires `langchain-anthropic>=1.4.0`)** — invocation-level, mirrors the API:
```python
resp = model.invoke(messages, cache_control={"type": "ephemeral"})            # 5m default
resp = model.invoke(messages, cache_control={"type": "ephemeral", "ttl": "1h"})
```
Applies the breakpoint to the last cacheable block and moves it forward as the conversation grows.

**(b) Explicit breakpoints** — `{"type":"text","text":TAXONOMY_SPEC,"cache_control":{"type":"ephemeral","ttl":"1h"}}` inside a system/user content block. Also on tool definitions: `@tool(description=..., extras={"cache_control": {"type": "ephemeral"}})`.

**(c) Agent middleware** — `AnthropicPromptCachingMiddleware(ttl="5m"|"1h", type="ephemeral", min_messages_to_cache=0, unsupported_model_behavior="warn"|"ignore"|"raise")` from `langchain_anthropic.middleware`. Marks the system prompt and tool definitions, passes `cache_control` through `model_settings`. Use with `create_agent`.

Verify hits via `response.usage_metadata["input_token_details"]` → `{'cache_read': N, 'cache_creation': N, 'ephemeral_5m_input_tokens': N, 'ephemeral_1h_input_tokens': N}`. 1-hour writes cost 2× base input (vs 1.25× for 5-min); reads ~0.1×.

**For your gold-labeling job**: put the entire taxonomy spec + annotation guidelines + few-shot examples in a cached 1h prefix and vary only the query at the tail. With a ~10k-token guideline over 5000 rows that's the difference between ~50M full-price input tokens and ~5M cache-read-priced ones.

Prompt-caching design rules that bite (from `shared/prompt-caching.md` in the Claude API skill): render order is `tools → system → messages`; **any byte change anywhere in the prefix invalidates everything after it** — so no `datetime.now()`, no UUIDs, no unsorted `json.dumps` in the prefix; max 4 breakpoints; minimum cacheable prefix is model-dependent (512 tokens on Opus 5, 1024 on Opus 4.8/Sonnet 5, 4096 on Opus 4.6/Haiku 4.5); changing the tool list or the model invalidates everything.

### 6.4 Rate limiting

`from langchain.rate_limiters import InMemoryRateLimiter` — verified signature `(*, requests_per_second: float = 1, check_every_n_seconds: float = 0.1, max_bucket_size: float = 1)`. Thread-safe, shareable across threads **in the same process** (it's a token bucket, not distributed). Pass as `init_chat_model(..., rate_limiter=rate_limiter)` or `ChatAnthropic(rate_limiter=...)`.

Caveat the docs state explicitly: it limits **request count only, not token volume** — so it won't protect you from ITPM limits on 5000 long prompts. Combine with `max_concurrency` on `evaluate()` / a semaphore around the fan-out, and set `RetryPolicy` to retry 429 with jitter.

### 6.5 Token accounting per node — `UsageMetadataCallbackHandler` (this is the answer to "cost tracking per node")

`langchain_core.callbacks` exports `UsageMetadataCallbackHandler` and `get_usage_metadata_callback` (added in `langchain-core` 0.3.49). It aggregates `AIMessage.usage_metadata` keyed by model name:

```python
from langchain_core.callbacks import get_usage_metadata_callback

def p2_labeling_node(state, config):
    with get_usage_metadata_callback() as cb:
        out = do_labeling(state)
    # cb.usage_metadata == {"claude-haiku-4-5": {"input_tokens":..., "output_tokens":..., "total_tokens":..., "input_token_details": {...}}}
    return {"labels": out, "usage": {"p2_labeling": cb.usage_metadata}}
```
Accumulate a `usage` channel with a merging reducer, and you get a per-node/per-phase token+cost ledger inside the checkpoint, available offline, with no LangSmith dependency.

### 6.6 Budget guard that aborts a run (verified pattern)

**Verified gotcha**: a `BaseCallbackHandler` that raises is *swallowed* by LangChain unless you set the class attribute `raise_error = True`. With it set, the exception propagates out of `invoke()`.

```python
from langchain_core.callbacks import BaseCallbackHandler

class BudgetExceeded(RuntimeError): ...

class BudgetGuard(BaseCallbackHandler):
    raise_error = True                     # <-- REQUIRED, else silently logged & ignored
    def __init__(self, max_usd, price_in, price_out):
        self.max_usd, self.pi, self.po, self.usd = max_usd, price_in, price_out, 0.0
    def on_llm_end(self, response, **kw):
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.usd += um.get("input_tokens",0)/1e6*self.pi + um.get("output_tokens",0)/1e6*self.po
        if self.usd > self.max_usd:
            raise BudgetExceeded(f"budget ${self.max_usd} exceeded: ${self.usd:.4f}")

guard = BudgetGuard(max_usd=50.0, price_in=5.0, price_out=25.0)   # Opus 5 pricing
graph.invoke(inputs, {"configurable": {"thread_id": rid}, "callbacks": [guard]}, durability="sync")
```
I ran this against a fake model emitting 700k-token usage per call and it aborted at the right iteration with the correct running total. Because `durability="sync"` checkpointed each superstep, the aborted run is resumable after you raise the cap.

Complementary, agent-loop-level: `ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")` (thread limiting requires a checkpointer) and `ToolCallLimitMiddleware`.

### 6.7 Anthropic Message Batches API — 50% discount, and the integration gap

**Finding: `langchain-anthropic` 1.5.6 has NO Message Batches integration.** I grepped the installed package: zero references to `batches`. `Runnable.batch()` / `.abatch()` is *client-side concurrency* (thread pool / asyncio), **not** the discounted Batches API. Anyone assuming otherwise will pay full price.

For the 5000-row gold-labeling job, drop to the raw `anthropic` SDK (0.122.0, verified `client.messages.batches` has `create/retrieve/list/results/cancel/delete`) inside a LangGraph node:

```python
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

client = anthropic.Anthropic()
batch = client.messages.batches.create(requests=[
    Request(custom_id=f"row-{i}",
            params=MessageCreateParamsNonStreaming(
                model="claude-haiku-4-5", max_tokens=256,
                system=[{"type":"text","text":GUIDELINES,"cache_control":{"type":"ephemeral"}}],
                messages=[{"role":"user","content":q}]))
    for i, q in enumerate(queries)])
# poll batch.processing_status until "ended"; then stream results
for r in client.messages.batches.results(batch.id):
    ...  # r.custom_id, r.result.type in {"succeeded","errored","canceled","expired"}
```
Facts that matter for pipeline design: up to **100,000 requests or 256 MB per batch**; most complete within 1 hour, **max 24 h**; results retained 29 days; **50% off all token usage**; prompt caching works inside batches; **results come back in arbitrary order — key by `custom_id`, never by position**. `fallbacks` is rejected on the Batches API.

Model the batch as three LangGraph nodes — `submit_batch` (idempotent: store `batch.id` in state), `poll_batch` (`TimeoutPolicy(idle_timeout=...)` + `runtime.heartbeat()`, retry policy on transient), `collect_results` — so a crash mid-poll resumes against the *existing* batch id rather than re-submitting and double-paying.

---

## 7. Deployment

### 7.1 `langgraph.json` schema (key fields, Python)

Schema URL: `https://langgra.ph/schema.json` (canonical JSON schema at `github.com/langchain-ai/langgraph/blob/main/libs/cli/schemas/schema.json`).

| Key | Notes |
|---|---|
| `dependencies` | **required**. `"."`, a dir containing `pyproject.toml`/`requirements.txt`/`setup.py`, or package names. Do **not** write `"requirements.txt"` — write `"./"`. |
| `graphs` | **required**. `{"id": "./pkg/file.py:graph"}` — a `CompiledStateGraph` **or** a factory function taking config and returning a `StateGraph`/`CompiledStateGraph` (used for `tracing_context` per-run customization). |
| `env` | path to `.env` or an inline `{"VAR": "value"}` map |
| `python_version` | `3.11` (default) / `3.12` / `3.13` |
| `base_image` | pin e.g. `"langchain/langgraph-server:0.2"` (cli ≥0.2.8) |
| `image_distro` | `debian`(default) / `wolfi`(recommended, smaller+more secure) / `bookworm` / `bullseye` (cli ≥0.2.11) |
| `pip_installer` | `auto`(default, uses `uv pip`) / `pip` / `uv` (v0.3+) |
| `dockerfile_lines` | extra Dockerfile lines — how you install system deps (e.g. `libgomp1` for sklearn/hdbscan) |
| `checkpointer` | `{backend: "default"|"mongo"|"custom", path, ttl: {strategy, sweep_interval_minutes, default_ttl, sweep_limit}, serde: {allowed_json_modules, pickle_fallback}}` |
| `store` | `{index: {embed, dims, fields}, ttl: {...}}` — semantic search over the BaseStore |
| `http` | `{app, cors, configurable_headers, logging_headers, middleware_order, mount_prefix, disable_meta/assistants/runs/threads/store/ui/mcp/a2a/webhooks}` |
| `auth` | `./pkg/auth.py:auth` (`langgraph_sdk.Auth`) |
| `api_version` | pin the server semver, e.g. `"0.3"` |

### 7.2 CLI

`pip install langgraph-cli` (add `[inmem]` for `dev`).
- `langgraph dev` — no Docker, hot reload, **port 2024**, state pickled to a local directory, DAP debugging (`--debug-port`, `--wait-for-client`), `--allow-blocking` (suppresses the sync-I/O-in-async errors — **you will need this** for pandas/sklearn nodes), `--no-browser`, `--tunnel`, `--n-jobs-per-worker` (default 10). Python ≥3.11 only.
- `langgraph up` — Docker: server + Postgres + Redis, **port 8123**, `--watch`, `--recreate`; production use needs a license key.
- `langgraph build -t img [--platform linux/amd64,linux/arm64]`, `langgraph dockerfile` (emit a Dockerfile for custom builds), `langgraph deploy [--deployment-type serverless|dedicated|dev|prod] [--remote/--no-remote] [--deployment-id]` (beta; Cloud only; reads `LANGGRAPH_HOST_API_KEY`/`LANGSMITH_API_KEY`/`LANGCHAIN_API_KEY`), `langgraph new --template ...`.
- CI health check: poll the server's `/ok` endpoint (the reference CI repo polls for 30 s then fails).

### 7.3 The purely-local, offline deployment you actually want

**Recommendation: do not use Agent Server for the pipeline itself.** `langgraph dev` is an interactive-development affordance (and its quickstart assumes a LangSmith key); a 12-phase batch job doesn't need an HTTP server, assistants, threads, or Postgres. The offline-on-a-laptop shape is:

```
qim/
├── pyproject.toml            # pinned versions
├── langgraph.json            # OPTIONAL — only so Studio/`langgraph dev` can inspect the graph
├── qim/graph.py              # builds + compiles the graph
├── qim/cli.py                # `python -m qim.cli run --phases p0..p12 --config cfg.yaml`
├── runs/<run_id>/
│   ├── manifest.json
│   ├── checkpoints.sqlite    # SqliteSaver
│   ├── cache.sqlite          # SqliteCache
│   └── artifacts/…
```
`graph.compile(checkpointer=SqliteSaver(...), cache=SqliteCache(path=...))`, `durability="sync"`, `LANGSMITH_TRACING` unset by default and flipped on by `--trace`. Keep `langgraph.json` around anyway — it costs nothing and gives you Studio + a one-command Docker path (`langgraph build`) if you ever need to hand the pipeline to someone else.

---

## 8. Structured logging + run manifest / provenance (design guidance — no official doc)

There is no LangSmith/LangGraph feature for this; build it at P0. Recommended manifest, written atomically at run start and updated per phase:

```jsonc
{
  "run_id": "0198f2c1-...",            // UUIDv7 — also used as LangSmith run_id and thread_id
  "started_at": "2026-08-17T09:12:03Z",
  "git": {"sha": "…", "dirty": false, "branch": "main"},
  "env": {"python": "3.12.4", "platform": "darwin-arm64",
          "packages": {"langgraph": "1.2.11", "langsmith": "0.11.0",
                       "langchain-anthropic": "1.5.6", "scikit-learn": "…"}},
  "seeds": {"global": 42, "numpy": 42, "sklearn": 42, "PYTHONHASHSEED": "42"},
  "config_hash": "sha256:…",           // canonicalised (sorted-keys) JSON of the full config
  "input_hash": "sha256:…",            // hash of the query-log file(s) + row count
  "models": {"deep": "claude-opus-5", "cheap": "claude-haiku-4-5"},
  "langsmith": {"enabled": true, "project": "qim", "trace_url": "…"},
  "phases": [
    {"id": "p3_representation", "status": "ok", "attempt": 1,
     "started_at": "…", "ended_at": "…",
     "inputs": {"queries": "artifacts/p1/queries.parquet@sha256:…"},
     "outputs": {"embeddings": "artifacts/p3/emb.npy@sha256:…"},
     "usage": {"claude-haiku-4-5": {"input_tokens": 1200000, "output_tokens": 90000}},
     "cost_usd": 1.65, "cached": false}
  ]
}
```

Implementation notes:
- **Logs**: `structlog` (or stdlib `logging` + a JSON formatter) with a `contextvars`-bound `run_id`/`phase`/`node_attempt`, emitted as JSONL to `runs/<run_id>/log.jsonl`. Bind `runtime.execution_info.{task_id, checkpoint_id, node_attempt}` on every record so a retried attempt is distinguishable in the log.
- **Correlate with LangSmith**: use the *same* UUIDv7 as `thread_id`, LangSmith `run_id` (`langsmith_extra={"run_id": rid}`), and manifest `run_id`. Then `metadata={"run_id": rid, "git_sha": sha, "config_hash": ch}` on `tracing_context` so LangSmith is filterable by commit.
- **Config hash**: canonicalise with `json.dumps(cfg, sort_keys=True, separators=(",",":"))` before hashing, or hashes drift across runs and destroy both your cache keys and your prompt cache.
- Manifest writes must be atomic (`tmp` + `os.replace`) for the same reason node artifacts must be.

---

## 9. Anthropic API specifics current in 2026

**Model IDs** (no date suffixes — these strings are complete as-is):

| Model | ID | Context | $/1M in | $/1M out |
|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | 1M | 10.00 | 50.00 |
| Claude Opus 5 | `claude-opus-5` | 1M | 5.00 | 25.00 |
| Claude Opus 4.8 | `claude-opus-4-8` | 1M | 5.00 | 25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | 3.00 (intro 2.00 thru 2026-08-31) | 15.00 (intro 10.00) |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1M | 3.00 | 15.00 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | 1.00 | 5.00 |

Default to `claude-opus-5` for deep work, `claude-haiku-4-5` for bulk. Batch API halves all of these.

**Via LangChain**: `init_chat_model("anthropic:claude-opus-5", ...)` or `ChatAnthropic(model="claude-opus-5", ...)`. Verified `ChatAnthropic` fields and defaults: `max_retries=2`, `default_request_timeout=None` (alias `timeout`), `max_tokens=None` (alias `max_tokens_to_sample`), `stream_usage=True`, plus `thinking`, `output_config`, `reasoning_effort`, `betas`, `context_management`, `mcp_servers`, `inference_geo`, `rate_limiter`, `model_kwargs`, `default_headers`.

**Thinking / effort (breaking, current)**: on Opus 5 / Opus 4.8 / 4.7 / Sonnet 5 / Fable 5, `thinking={"type":"enabled","budget_tokens":N}` is **removed → 400**. Use `thinking={"type":"adaptive"}` and `output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}`. `temperature`/`top_p`/`top_k` are **rejected (400)** on these models — LangChain's `ChatAnthropic` still exposes the fields, so don't set them. On **Opus 5 thinking is ON by default** (omitting `thinking` runs adaptive) and `max_tokens` caps thinking + text together — an old `max_tokens` sized for non-thinking output will truncate. `thinking.display` defaults to `"omitted"`; set `{"type":"adaptive","display":"summarized"}` if you surface reasoning. `thinking={"type":"disabled"}` is only legal at effort `high` or below on Opus 5.

**Assistant prefill returns 400** on Opus 5 / Sonnet 5 / Fable 5 / the 4.6–4.8 family. Use structured outputs (`output_config.format` / `with_structured_output`) instead — relevant to your P2 labeling and P7 naming schemas.

**Rate limits**: separate buckets per model tier; Claude Opus 5 does **not** draw from the combined Opus 4.x pool. 429 responses carry `retry-after`; the Anthropic SDK auto-retries 408/409/429/5xx (`max_retries`, default 2). 529 = `overloaded_error`, retryable. Batch API requests draw from a separate batch quota.

**Refusals**: Opus 5 / Fable 5 ship elevated cybersecurity/bio safeguards; a declined request returns **HTTP 200** with `stop_reason: "refusal"` and a `stop_details.category`. Code that reads `response.content[0]` unconditionally breaks. Politics/sports query-mining corpora are the realistic trigger for you. Handle it, and consider server-side fallbacks (`fallbacks: "default"` + beta header `server-side-fallback-2026-07-01`) — **but note `fallbacks` is rejected on the Batches API**, so your batch path needs a client-side re-submit of refused `custom_id`s on a fallback model.

---

## 10. Concrete gaps / risks for this project

1. **No batch-API integration in LangChain** — you must write the Anthropic Batches node by hand. This is the single largest cost lever (50% × prompt caching) for the 5000-row gold-labeling phase and it is not on the paved path.
2. **Node timeouts require async nodes** — a 12-phase data-science pipeline is mostly sync sklearn/pandas. Either make phase nodes `async def` + `asyncio.to_thread`, or accept no timeouts on those nodes. `langgraph dev --allow-blocking` masks the warning but doesn't give you timeouts.
3. **429 is not in the default retry set** — silently, a rate-limited node fails immediately unless you either write a custom `retry_on` or rely on the SDK's own retries.
4. **Callback exceptions are swallowed** unless `raise_error = True` — a budget guard without it looks like it works and doesn't.
5. **Node cache is shared across threads and its default key is `pickle`** — cross-run bleed is a feature until it isn't. Always version the cache key on model name + config hash, and set a `ttl`.
6. **`langgraph dev` state is pickled to a local dir, not Postgres** — fine for dev, not a durability story. For the real run use `SqliteSaver`/`PostgresSaver` + `durability="sync"`.

## Sources actually read
(Files read from the `langchain-ai/docs` repo, which is the source of the `docs.langchain.com` pages; plus PyPI JSON and live package introspection.)


---

## Recommendations carried into the design

- Treat LangSmith as strictly optional: leave LANGSMITH_TRACING unset by default and gate it behind a --trace CLI flag, since a graph with the vars unset runs fully offline (verified) and client.evaluate(upload_results=False) gives you the identical ExperimentResults object with zero network.
- Run the 12 phases as LangGraph nodes over SqliteSaver with durability="sync" — verified that a crashed node leaves state.next pointing at it and graph.invoke(None, config) resumes re-running only the failed node, which is exactly the 'fix P6, don't redo P0-P5' requirement.
- Set graph-wide policy once with StateGraph.set_node_defaults(retry_policy=..., timeout=TimeoutPolicy(run_timeout=..., idle_timeout=...), error_handler=mark_phase_failed, cache_policy=CachePolicy(ttl=...)) (requires langgraph>=1.2) and override per-node only where a phase genuinely differs.
- Write a custom retry_on predicate wrapping langgraph.types.default_retry_on, because the default explicitly does NOT retry 4xx — so Anthropic 429 / rate_limit_error is currently a hard node failure.
- Make every node idempotent: write artifacts to a content-hash-named tmp path then os.replace(), keep only path+hash references in graph state, and stamp runtime.execution_info.node_attempt into each artifact sidecar so retried writes are auditable.
- Use langgraph.cache.sqlite.SqliteCache with CachePolicy(ttl=...) on P3/P4 (embedding bake-off, clustering battery) but bake the model name and config hash into the node input, because the cache is graph-scoped, shared across thread_ids, and keyed by pickle of the input.
- For P2's 5000-row gold labeling, drop to the raw anthropic SDK Message Batches API (50% discount) inside three nodes — submit/poll/collect — because langchain-anthropic 1.5.6 has no batch integration and Runnable.batch() is only client-side concurrency at full price.
- Layer Anthropic prompt caching under the batch job: put the taxonomy spec + annotation guidelines in a 1h ephemeral cache_control block and vary only the query at the tail, verifying hits via response.usage_metadata['input_token_details']['cache_read'].
- Route two tiers via init_chat_model('anthropic:claude-haiku-4-5') for bulk labeling/triage and init_chat_model('anthropic:claude-opus-5', thinking={'type':'adaptive'}, output_config={'effort':'high'}) for taxonomy design, referee adjudication, and the P7 tree audit.
- Implement the budget guard as a BaseCallbackHandler subclass with the class attribute raise_error = True (verified: without it LangChain logs and swallows the exception), and pair it with ModelCallLimitMiddleware for agent-loop call caps.
- Build the per-phase cost ledger with langchain_core.callbacks.get_usage_metadata_callback() wrapped around each LLM-calling node and accumulate it into a 'usage' state channel, giving you offline per-node token/cost accounting with no LangSmith dependency.
- Add TracePolicy(process_inputs=..., process_outputs=...) to any node whose state carries dataframes or embedding matrices, so traces record shapes and hashes instead of megabyte payloads.
- Test with GenericFakeChatModel(messages=iter([...])) for scripted tool-call/text sequences, unit-test single nodes via compiled.nodes['p4'].invoke(...), and test mid-pipeline phases with update_state(as_node='p3') + invoke(None, interrupt_after='p6').
- Gate CI on client.evaluate(..., blocking=False) with a sys.exit(1) threshold, and use the pytest plugin with LANGSMITH_TEST_CACHE=tests/cassettes committed so CI never pays for LLM calls; add LANGSMITH_TEST_TRACKING=false for fully offline test runs.
- Use pairwise evaluation (client.evaluate((exp_a, exp_b), evaluators=[...], randomize_order=True)) for P7 blind-naming variants and P5 K-selection comparisons, since it is the only eval mode designed for 'which of these two is better' without a gold label.
- Ship the pipeline as a CLI over an in-process compiled graph rather than Agent Server; keep langgraph.json anyway (dependencies, graphs, python_version, image_distro:'wolfi', dockerfile_lines for sklearn system deps) so Studio and langgraph build remain available.
- Make phase nodes async def with asyncio.to_thread around pandas/sklearn work, because LangGraph rejects sync nodes that declare a timeout at compile time — otherwise you get no per-node timeout protection at all.
- Wire RunControl + SIGTERM to control.request_drain() and catch GraphDrained so an interrupted overnight run stops at a superstep boundary with a resumable checkpoint.
- Emit a runs/<run_id>/manifest.json written atomically at P0 containing run_id (UUIDv7, reused as thread_id and LangSmith run_id), git sha+dirty flag, seeds, canonicalised config hash, input hash, package versions, and a per-phase record of inputs/outputs/hashes/usage/cost.
- Handle stop_reason == 'refusal' explicitly before reading response.content — Opus 5 and Fable 5 return HTTP 200 on a safety decline, and politics/security query corpora are a realistic trigger; note server-side fallbacks are rejected on the Batches API so the batch path needs client-side re-submission of refused custom_ids.

## Unverified or version-dependent

- docs.langchain.com and github.com are blocked to WebFetch in this environment; I read the docs from a git clone of langchain-ai/docs main branch, which may be slightly ahead of what is published at docs.langchain.com.
- Model IDs and pricing for Claude (opus-5, sonnet-5, haiku-4-5, fable-5) come from the bundled claude-api skill's cached table (cached 2026-06-24), not from a live fetch of platform.claude.com, which was also unreachable. Verify pricing before building cost models on it.
- The Message Batches API limits I cite (100k requests / 256 MB, 24h max, 29-day result retention, 50% discount) come from the claude-api skill's cached batches doc, not a live docs fetch. The SDK method surface (create/retrieve/list/results/cancel/delete) I did verify directly against anthropic 0.122.0.
- I could not verify whether `langgraph dev` hard-requires a LANGSMITH_API_KEY or merely warns — the docs' quickstart lists it as a prerequisite but I did not install langgraph-cli[inmem] and run it. My recommendation routes around this by not using Agent Server for the batch pipeline.
- `tracing_context(enabled="local")` appears in the langsmith 0.11 type signature but I found no doc explaining its exact semantics; do not rely on it as the offline switch — use enabled=False or leave LANGSMITH_TRACING unset.
- langgraph.cache.redis exists as a module but I did not inspect its constructor; if you want a shared cache across machines, check its signature before use.
- Whether TracePolicy applies cleanly to nodes that are themselves Runnables (rather than plain functions) is documented ambiguously — the docstring says plain function nodes are traced with trace=False so child runs don't exist, implying different behavior for Runnable-bound nodes.
- The interaction between LangGraph's RetryPolicy and the Anthropic SDK's own max_retries is not documented; my guidance to pick one primary layer per failure class is a design judgment, not a documented recommendation.
- Exact Anthropic per-model rate limits by tier were not fetchable; the claim that Opus 5 has a separate bucket from the Opus 4.x pool comes from the cached migration guide.
- I did not test the langsmith pytest plugin, the cassette cache, or evaluate() end-to-end, since all require a LangSmith API key and network access to smith.langchain.com.

## Sources

- https://pypi.org/pypi/langgraph/json
- https://pypi.org/pypi/langsmith/json
- https://pypi.org/pypi/langchain/json
- https://pypi.org/pypi/langchain-core/json
- https://pypi.org/pypi/langchain-anthropic/json
- https://pypi.org/pypi/langgraph-cli/json
- https://pypi.org/pypi/langgraph-checkpoint/json
- https://pypi.org/pypi/langgraph-checkpoint-sqlite/json
- https://pypi.org/pypi/langgraph-sdk/json
- https://pypi.org/pypi/anthropic/json
- https://github.com/langchain-ai/docs (sparse clone of src/langsmith and src/oss, read directly)
- https://docs.langchain.com/langsmith/observability-quickstart
- https://docs.langchain.com/langsmith/evaluation-quickstart
- https://docs.langchain.com/langsmith/annotate-code
- https://docs.langchain.com/langsmith/conditional-tracing
- https://docs.langchain.com/langsmith/trace-without-env-vars
- https://docs.langchain.com/langsmith/add-metadata-tags
- https://docs.langchain.com/langsmith/sample-traces
- https://docs.langchain.com/langsmith/cost-tracking
- https://docs.langchain.com/langsmith/evaluator-spend
- https://docs.langchain.com/langsmith/handle-model-rate-limiting
- https://docs.langchain.com/langsmith/pytest
- https://docs.langchain.com/langsmith/local
- https://docs.langchain.com/langsmith/read-local-experiment-results
- https://docs.langchain.com/langsmith/evaluate-pairwise
- https://docs.langchain.com/langsmith/evaluate-graph
- https://docs.langchain.com/langsmith/cicd-pipeline-example
- https://docs.langchain.com/langsmith/cli
- https://docs.langchain.com/langsmith/application-structure
- https://docs.langchain.com/langsmith/local-dev-testing
- https://docs.langchain.com/langsmith/trace-with-langgraph
- https://docs.langchain.com/langsmith/log-traces-to-project
- https://docs.langchain.com/oss/python/langgraph/fault-tolerance
- https://docs.langchain.com/oss/python/langgraph/graph-api
- https://docs.langchain.com/oss/python/langgraph/use-graph-api
- https://docs.langchain.com/oss/python/langgraph/checkpointers
- https://docs.langchain.com/oss/python/langgraph/test
- https://docs.langchain.com/oss/python/langgraph/observability
- https://docs.langchain.com/oss/python/langchain/test/unit-testing
- https://docs.langchain.com/oss/python/langchain/models
- https://docs.langchain.com/oss/python/langchain/middleware/built-in
- https://docs.langchain.com/oss/python/integrations/chat/anthropic
- https://docs.langchain.com/oss/python/integrations/middleware/anthropic
- https://raw.githubusercontent.com/langchain-ai/langgraph/refs/heads/main/libs/cli/schemas/schema.json (referenced as the langgraph.json schema)