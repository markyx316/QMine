# LangGraph Core API (v1.2.11)

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

> **Method note:** `docs.langchain.com` is blocked for WebFetch in this environment, so I pulled the raw `.md` sources via `curl` (`https://docs.langchain.com/oss/python/langgraph/<slug>.md`) and — more importantly — **installed langgraph 1.2.11 into a venv and introspected/executed every API below**. Every signature in this report is `inspect.signature()` output from the real package, not from docs or memory. Where docs and runtime disagree, I flag it explicitly. Scratchpad with venv + test scripts: `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/8075e8db-1d8b-4b54-b04d-7d995fbae90d/scratchpad/`

---

# 0. Pinned versions (PyPI, verified 2026-08-17)

```
langgraph                    == 1.2.11    # requires-python >=3.10
langgraph-checkpoint         == 4.2.0     # bundled with langgraph
langgraph-checkpoint-sqlite  == 3.1.1     # separate install
langgraph-checkpoint-postgres== 3.1.2     # separate install
langgraph-prebuilt           == 1.1.0
langgraph-sdk                == 0.4.2
langgraph-cli                == 0.4.31
langchain-core               == 1.5.5
langchain                    == 1.3.15
```

Dependency constraints resolved from langgraph 1.2.11 metadata:
`langchain-core>=1.4.7,<2`, `langgraph-checkpoint>=4.1.0,<5.0.0`, `langgraph-prebuilt>=1.1.0,<1.2.0`, `langgraph-sdk>=0.4.2,<0.5.0`, `pydantic>=2.7.4`, `xxhash>=3.5.0`.

Suggested pin for our `requirements.txt`:
```
langgraph>=1.2.11,<1.3
langgraph-checkpoint-sqlite>=3.1.1,<4
langgraph-checkpoint-postgres>=3.1.2,<4
langchain-core>=1.5.5,<2
```

**Doc structure changed.** The URLs in the assignment are stale. Current slugs under `/oss/python/langgraph/`: `overview`, `graph-api`, `use-graph-api`, `persistence` (now a thin hub page), `checkpointers`, `stores`, `interrupts` (replaces `add-human-in-the-loop`), `fault-tolerance` (replaces `durable-execution`), `streaming`, `event-streaming` (new), `use-subgraphs` (replaces `subgraphs`), `use-time-travel`, `functional-api`, `use-functional-api`, `choosing-apis`, `backward-compatibility`, `pregel`, `changelog-py`, `test`, `errors/*`.

---

# 1. StateGraph construction

## Verified constructor

```python
StateGraph.__init__(
    self,
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
    **kwargs,  # DeprecatedKwargs
) -> None
```

### ⚠️ Renames — CONFIRMED by triggering the deprecation warnings

I passed the old kwargs and captured the exact warning text:

| Old kwarg | New kwarg | Warning text (verbatim) |
|---|---|---|
| `config_schema=` | `context_schema=` | ``` `config_schema` is deprecated and will be removed. Please use `context_schema` instead. Deprecated in LangGraph V1.0``` |
| `input=` | `input_schema=` | ``` `input` is deprecated and will be removed. Please use `input_schema` instead. Deprecated in LangGraph V0.5 to be removed in V2.0.``` |
| `output=` | `output_schema=` | ``` `output` is deprecated and will be removed. Please use `output_schema` instead. Deprecated in LangGraph V0.5 to be removed in V2.0``` |

All three still **work** (warn only). Use the new names in all new code.

## State schemas — three flavors

```python
from typing import Annotated, NotRequired
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from pydantic import BaseModel
import operator

# (a) TypedDict — recommended default, fastest
class PipelineState(TypedDict):
    run_id: str
    queries: list[str]
    metrics: NotRequired[dict]

# (b) dataclass — use when you want DEFAULT VALUES
@dataclass
class PipelineState:
    run_id: str
    queries: list[str] = field(default_factory=list)

# (c) Pydantic BaseModel — runtime validation of INPUTS ONLY
class PipelineState(BaseModel):
    run_id: str
    queries: list[str] = []
```

**Pydantic limitations (documented, important for us):**
- Graph output is **NOT** a pydantic instance (comes back as dict).
- Validation runs **only on input to the first node** — not on subsequent nodes or outputs.
- Validation error traces do not identify which node failed.
- Recursive validation is slow — prefer `dataclass` for hot paths.
- `create_agent` (langchain) does **not** support Pydantic state schemas.

## Reducers via `Annotated`

Reducer contract, exactly:
```python
new_value = reducer(left=current_state[key], right=node_update[key])
```
Left = accumulated state, right = latest node update. Default reducer discards `left` (last-write-wins).

```python
from typing import Annotated
import operator
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage

def merge_cluster_labels(left: dict[int, str], right: dict[int, str]) -> dict[int, str]:
    """Custom reducer: merge naming-agent outputs keyed by cluster id."""
    return {**left, **right}

class State(TypedDict):
    step_log:  Annotated[list[str], operator.add]          # append
    messages:  Annotated[list[AnyMessage], add_messages]   # id-aware upsert
    labels:    Annotated[dict[int, str], merge_cluster_labels]
    k_final:   int                                          # last-write-wins
```

`add_messages` import paths (both valid): `from langgraph.graph import add_messages` or `from langgraph.graph.message import add_messages`. It appends new messages, **overwrites by message `id`**, and deserializes `{"type":"human","content":...}` dicts into LangChain message objects. Prebuilt `MessagesState` = single `messages` key with `add_messages`:
```python
from langgraph.graph import MessagesState
class State(MessagesState):
    documents: list[str]
```

### `Overwrite` — bypass a reducer (v1.x)
```python
from langgraph.types import Overwrite

def replace_all(state: State):
    return {"step_log": Overwrite(["reset"])}     # bypasses operator.add
    # JSON-equivalent: {"step_log": {"__overwrite__": ["reset"]}}
```
Verified dataclass: `Overwrite(value: Any, type: Literal['__overwrite__'] = '__overwrite__')`.
**Constraint:** only one node may `Overwrite` a given key per super-step; two → `InvalidUpdateError`.

### ⚠️ Concurrent-write error — VERIFIED empirically
Two parallel nodes writing the same non-reduced key raises:
```
langgraph.errors.InvalidUpdateError:
At key 'x': Can receive only one value per step. Use an Annotated key to handle multiple values.
```
**This is the #1 thing that will bite our fan-out phases (P2 research agents, P4 algorithm battery, P7 naming agents).** Every channel written by parallel branches MUST have a reducer.

## Input / output / private schemas

```python
class InputState(TypedDict):     user_input: str
class OutputState(TypedDict):    graph_output: str
class OverallState(TypedDict):   foo: str; user_input: str; graph_output: str
class PrivateState(TypedDict):   bar: str

def node_1(state: InputState) -> OverallState:   return {"foo": state["user_input"] + " name"}
def node_2(state: OverallState) -> PrivateState: return {"bar": state["foo"] + " is"}
def node_3(state: PrivateState) -> OutputState:  return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
```
Two non-obvious rules:
1. A node can **write to any channel in the graph**, not just those in its declared input schema. The graph state is the union of all declared schemas.
2. Nodes can declare **additional private channels** (like `PrivateState.bar`) just by annotating; the channel is created automatically.

Per-node input narrowing is also available via `add_node(..., input_schema=Node2Input)`.

### ⚠️ Private channels are NOT redacted when streaming
Direct from docs — a trap for anything sensitive:
> Input, output, and private schemas constrain what each node *reads* and what `invoke` *returns*. They do **not** hide channels from `stream`.

`stream_mode="values"` emits **all** channels including private ones. To restrict: pass `output_keys=["graph_output"]` to `stream`/`invoke`, or use `stream_mode="updates"`.

## Runtime context (replaces `config["configurable"]` for DI)

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Ctx:
    llm_provider: str = "anthropic"
    embedding_model: str = "..."
    db_uri: str = "..."

def node_a(state: State, runtime: Runtime[Ctx]):
    llm = get_llm(runtime.context.llm_provider)   # dataclass → attribute access
    ...

graph = StateGraph(State, context_schema=Ctx).…compile()
graph.invoke(inputs, context={"llm_provider": "openai"})   # or context=Ctx()
```
If `context_schema` is a `TypedDict`, access is `runtime.context["llm_provider"]`; if a dataclass, `runtime.context.llm_provider`. **Use a dataclass** — it gives defaults and attribute access.

Verified `Runtime` attributes: `context`, `store`, `stream_writer`, `execution_info`, `server_info`, `control`, `drain_requested`, `drain_reason`, `heartbeat`, `previous`, `merge`, `override`, `patch_execution_info`.

---

# 2. Nodes and edges

## `add_node` — full verified signature

```python
StateGraph.add_node(
    self,
    node: str | StateNode,
    action: StateNode | None = None,
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: type[NodeInputT] | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: StateNode | None = None,          # NEW in 1.2
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,   # NEW in 1.2
    trace_policy: TracePolicy | None = None,          # NEW
    **kwargs,
) -> Self                                             # chainable
```

Node function signatures — LangGraph injects by **parameter name + type annotation**:
```python
def n(state): ...
def n(state, config: RunnableConfig): ...
def n(state, runtime: Runtime[Ctx]): ...
def handler(state, error: NodeError): ...   # error-handler nodes only
```

Other builder methods (all verified, all return `Self` so they chain):
```python
add_edge(start_key: str | list[str], end_key: str) -> Self
add_conditional_edges(source: str,
                      path: Callable[..., Hashable | Sequence[Hashable]] | Runnable,
                      path_map: dict[Hashable, str] | list[str] | None = None) -> Self
add_sequence(nodes: Sequence[StateNode | tuple[str, StateNode]]) -> Self
set_entry_point(key: str) -> Self        # == add_edge(START, key)
set_finish_point(key: str) -> Self       # == add_edge(key, END)
set_node_defaults(*, retry_policy=None, cache_policy=None,
                  error_handler=None, timeout=None) -> Self    # NEW in 1.2
```
`START`/`END` from `langgraph.graph`. Docs state `add_edge(START, ...)` / `add_edge(..., END)` are the **recommended modern syntax** over `set_entry_point`/`set_finish_point`.

`add_edge` accepts a **list** as `start_key` — that's a built-in "wait for all of these" fan-in.

## ⚠️ Never mix static edges with dynamic routing on the same node
Direct from docs:
> `Command` only adds dynamic edges — static edges defined with `add_edge` still execute. If `node_a` returns `Command(goto="my_other_node")` and you also have `graph.add_edge("node_a", "node_b")`, **both** `node_b` and `my_other_node` will run.

Same applies to tools returning `Command`.

## RetryPolicy — verified defaults

```python
from langgraph.types import RetryPolicy, default_retry_on

RetryPolicy(
    initial_interval: float = 0.5,
    backoff_factor:   float = 2.0,
    max_interval:     float = 128.0,
    max_attempts:     int   = 3,          # INCLUDING the first attempt
    jitter:           bool  = True,
    retry_on = default_retry_on,
)
```
`default_retry_on` retries on any exception **except**: `ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`, `SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`, `StopAsyncIteration`, `OSError`. For `requests`/`httpx` it retries only on **5xx**.

Extend rather than replace:
```python
def custom_retry_on(exc: BaseException) -> bool:
    if isinstance(exc, MyNonRetryable): return False
    return default_retry_on(exc)

builder.add_node("call_api", call_api,
                 retry_policy=RetryPolicy(max_attempts=3, retry_on=custom_retry_on))
```
`retry_policy` also accepts a **Sequence** of policies (different backoffs per exception class).

**Verified:** `RetryPolicy(max_attempts=3)` → the node function ran exactly **3** times.

## CachePolicy + caches

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache      # InMemoryCache(*, serde=None)
from langgraph.cache.sqlite import SqliteCache        # SqliteCache(*, path: str, serde=None)

builder.add_node("embed_bakeoff", embed_bakeoff, cache_policy=CachePolicy(ttl=3600))
graph = builder.compile(cache=SqliteCache(path="artifacts/node_cache.sqlite"))
```
`CachePolicy(*, key_func=default_cache_key, ttl: int|None = None)`. `ttl` in seconds; `None` = never expires. Default `key_func` hashes the node input (pickle-based). Cache hits surface in `stream_mode="updates"` as `{'__metadata__': {'cached': True}}`.

**`SqliteCache` is the right choice for our pipeline** — it survives process restarts, so re-running P3/P4 skips completed embedding/clustering work.

## Node timeouts (NEW in 1.2) — async only

```python
from langgraph.types import TimeoutPolicy
TimeoutPolicy(*, run_timeout: float|timedelta|None = None,
                 idle_timeout: float|timedelta|None = None,
                 refresh_on: Literal['auto','heartbeat'] = 'auto')

builder.add_node("call_model", call_model, timeout=30)                       # seconds
builder.add_node("call_model", call_model, timeout=timedelta(minutes=2))
builder.add_node("call_model", call_model,
                 timeout=TimeoutPolicy(run_timeout=120, idle_timeout=30))
```
- **Async nodes only.** Setting `timeout` on a sync node raises **at compile time** ("sync Python execution cannot be safely canceled in-process").
- Timeout applies **per attempt** — the timer resets on each retry.
- Timed-out attempts **do not commit their buffered writes** (no state leakage past the boundary).
- `Send(..., timeout=...)` allows per-fan-out-item timeouts.

### 🚨 DOC BUG — `NodeTimeoutError` does NOT subclass `TimeoutError`
Docs claim: *"LangGraph raises `NodeTimeoutError`, which subclasses Python's built-in `TimeoutError`."*

**Verified MRO in 1.2.11:**
```python
NodeTimeoutError.__mro__ == (NodeTimeoutError, Exception, BaseException, object)
isinstance(err, TimeoutError)  # -> False
```
Consequences: `except TimeoutError:` will **not** catch it, and `RetryPolicy(retry_on=TimeoutError)` will **not** match it. Always import and use the real class:
```python
from langgraph.errors import NodeTimeoutError
```

## Node error handlers (NEW in 1.2) — Saga/compensation

```python
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy

def payment_error_handler(state: State, error: NodeError) -> Command:
    # NodeError is a frozen dataclass: .node (str), .error (BaseException)
    return Command(update={"status": f"compensated: {error.error}"}, goto="finalize")

builder.add_node("charge_payment", charge_payment,
                 retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
                 error_handler=payment_error_handler)
```
Composition order (fixed): **attempt → retry_policy decides → only after retries exhausted does error_handler run**. The `error: NodeError` param is opt-in; `(state)` and `(state, runtime)` handler signatures also work.

**Verified end-to-end:** node raised `ConnectionError` 3×, handler fired, returned `{'status': 'handled ConnectionError from flaky'}` and routed to `fin`.

- ⚠️ `interrupt()` is **NOT** routed to the error handler — it uses the `GraphBubbleUp` mechanism and bypasses both retries and handlers. Good: HITL gates can't be swallowed by a handler.
- Failure provenance is **checkpointed** — if the process crashes after failure but before the handler completes, the handler sees the same `NodeError` on resume.
- Subgraph exceptions surface to the parent node; the parent's handler fires with the subgraph exception in `error.error`.

## `set_node_defaults` (NEW in 1.2)

```python
graph = (StateGraph(State)
    .set_node_defaults(retry_policy=RetryPolicy(max_attempts=3),
                       timeout=TimeoutPolicy(run_timeout=30),
                       error_handler=fallback_handler)
    .add_node("a", node_a)
    .add_node("b", node_b, retry_policy=RetryPolicy(max_attempts=5))  # per-node wins
    .add_edge(START, "a")
    .compile())
```
**Verified:** default `max_attempts=2` produced exactly 2 attempts; per-node override to 5 wins. Defaults resolve at `compile()` time (call order vs `add_node` is irrelevant).

Applicability matrix:
| default | regular nodes | error-handler nodes | reason |
|---|---|---|---|
| `retry_policy` | ✅ | ✅ | handlers retried on transient failures |
| `timeout` | ✅ | ✅ | stuck handlers cancelled too |
| `error_handler` | ✅ | ❌ | handlers must never catch themselves |
| `cache_policy` | ✅ | ❌ | caching a handler result is unsafe |

**Defaults are NOT inherited by subgraphs.** Each graph carries its own — call `set_node_defaults` on every subgraph builder.

## `defer=True` — fan-in that waits for uneven branches

```python
builder.add_node("aggregate_metrics", aggregate, defer=True)
```
A deferred node waits until **all pending tasks** are finished, not just its immediate predecessors. Essential when branches have different lengths (documented output confirms `d` runs after the entire multi-step `b` branch, not after the first super-step). **This is exactly what our P9 uniform-metrics-panel node needs.**

---

# 3. `Command` — combined update + routing

```python
from langgraph.types import Command
Command(*, graph: str | None = None,
           update: Any | None = None,
           resume: dict[str, Any] | Any | None = None,
           goto: Send | Sequence[Send | str] | str = ())
Command.PARENT == "__parent__"        # verified class constant
```

Four uses: `update` (state), `goto` (routing), `graph` (target parent), `resume` (post-interrupt input).

## Return from a node

```python
def route_after_kappa(state: State) -> Command[Literal["referee", "build_classifier"]]:
    if state["kappa"] < 0.9:
        return Command(update={"needs_referee": True}, goto="referee")
    return Command(update={"gold_frozen": True}, goto="build_classifier")
```
The `Command[Literal[...]]` **return annotation is required** for graph rendering and for LangGraph to know the possible destinations. Alternative: `add_node(..., destinations=("referee", "build_classifier"))`.

**Verified:** `destinations=("tail",)` produced edge `('router','tail', conditional=True)` in `get_graph().edges`, with no return annotation needed.

## 🚨 GOTCHA I HIT: `Command.PARENT` + return annotation = compile error

The docs' `Command.PARENT` example silently omits the return annotation. If you keep it, the **subgraph** tries to validate the parent's node name as one of its own:
```python
# ❌ FAILS at subgraph.compile():
#    ValueError: Found edge ending at unknown node `node_b`
def inner(s: S) -> Command[Literal["node_b"]]:
    return Command(update={"foo":"X"}, goto="node_b", graph=Command.PARENT)

# ✅ WORKS — omit the annotation when goto targets the PARENT graph
def inner(s: S):
    return Command(update={"foo":"X"}, goto="node_b", graph=Command.PARENT)
```
**Verified working:** parent result `{'foo': 'XB'}`.

Also required: when a subgraph node updates a key **shared** with the parent via `Command.PARENT`, that key **must have a reducer in the parent state** (I used `Annotated[str, operator.add]`).

## When `Command` vs conditional edges
- **`Command`** — you need to update state **and** route in one node (our phase-gate nodes: kappa check, K-triangulation, refinement loop).
- **`add_conditional_edges`** — routing only, no state change.
- Never both from the same node.

## ⚠️ `Command` as INPUT to `invoke`/`stream`
Only `Command(resume=...)` (optionally `+ update=`) is legitimate as graph input. Passing any `Command` resumes **from the latest checkpoint**, not `__start__`:
```python
# WRONG — graph appears stuck if it already finished
graph.invoke(Command(update={"messages":[...]}), config)
# CORRECT — plain dict restarts from __start__
graph.invoke({"messages":[...]}, config)
```

---

# 4. `Send` — map-reduce / dynamic fan-out

```python
from langgraph.types import Send
Send(node: str, arg: Any, *, timeout: float|timedelta|TimeoutPolicy|None = None)
```
`arg` is the **entire state** the target node receives — a *different* state shape per fan-out item.

**Fully verified working example (this is our P7 blind-naming pattern):**

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class S(TypedDict):
    clusters: list[str]
    named:    Annotated[list[str], operator.add]   # REDUCER REQUIRED for fan-in
    final:    str

def plan(s: S):     return {"clusters": [f"c{i}" for i in range(4)]}
def fanout(s: S):   return [Send("name_one", {"cluster": c}) for c in s["clusters"]]
def name_one(s):    return {"named": [f"named:{s['cluster']}"]}   # s is the Send arg
def reduce_(s: S):  return {"final": ",".join(sorted(s["named"]))}

b = StateGraph(S)
b.add_node("plan", plan)
b.add_node("name_one", name_one)
b.add_node("reduce_", reduce_, defer=True)          # wait for ALL fanned-out tasks
b.add_edge(START, "plan")
b.add_conditional_edges("plan", fanout, ["name_one"])   # 3rd arg = path_map for rendering
b.add_edge("name_one", "reduce_")
b.add_edge("reduce_", END)
graph = b.compile()

graph.invoke({"clusters": [], "named": [], "final": ""})
# -> final == 'named:c0,named:c1,named:c2,named:c3'   ✅ VERIFIED
```

Key mechanics:
1. `Send` objects are returned from a **conditional-edge function** (or from `Command(goto=[Send(...), ...])`).
2. Pass the destination list as `path_map` (`["name_one"]`) so the graph renders and validates.
3. **Aggregation is purely by reducer** — every target node writes into an `Annotated[..., operator.add]` channel. There is no separate "join" primitive.
4. Add `defer=True` to the reduce node so it waits for the whole fan-out even if branches differ in length.
5. `Send(..., timeout=...)` gives per-item timeouts — useful for capping a slow naming agent without failing the batch.

For our P4 algorithm battery: `Send("run_clusterer", {"algo": a, "params": p, "X_ref": ...})` over the 6 algorithms; results accumulate into `Annotated[list[ClusterResult], operator.add]`.

---

# 5. Compilation and execution

## `compile` — verified signature

```python
StateGraph.compile(
    self,
    checkpointer: Checkpointer = None,          # BaseCheckpointSaver | bool | None
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after:  All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,  # NEW: stream transformers
) -> CompiledStateGraph
```
Note: `interrupt_before`/`interrupt_after` are **not deprecated** at compile time — but docs explicitly say *"Static interrupts are **not** recommended for human-in-the-loop workflows. Use the `interrupt()` function instead."* They are positioned as **debugging breakpoints**.

## `invoke` / `ainvoke` — verified

```python
CompiledStateGraph.invoke(
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    context: ContextT | None = None,
    stream_mode: StreamMode = 'values',
    print_mode: StreamMode | Sequence[StreamMode] = (),
    output_keys: str | Sequence[str] | None = None,
    interrupt_before: All | Sequence[str] | None = None,   # runtime override
    interrupt_after:  All | Sequence[str] | None = None,
    durability: Durability | None = None,
    control: RunControl | None = None,
    version: Literal['v1','v2'] = 'v1',
) -> dict[str, Any] | Any
```

## `stream` / `astream` — verified
Same as above **plus** `subgraphs: bool = False`, `debug: bool | None = None`, and `stream_mode: StreamMode | Sequence[StreamMode] | None = None`.

## `stream_events` / `astream_events` (NEW, v1.2) — verified

```python
CompiledStateGraph.stream_events(
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    version: Literal['v1','v2','v3'] = 'v2',     # NOTE: default is v2, docs use v3
    interrupt_before=None, interrupt_after=None,
    control: RunControl | None = None,
    transformers: Sequence[...] | None = None,
) -> Any
```

### ⚠️ `version="v3"` emits a beta warning
Runtime warning captured verbatim:
```
LangChainBetaWarning: The v3 streaming protocol on Pregel is experimental.
  (langgraph/pregel/main.py:3708)
```
Docs recommend v3 for new apps, but the runtime marks it **experimental** and the parameter **defaults to v2**. For a production-grade system I'd use `stream(..., version="v2")` for the core orchestrator and treat `stream_events(version="v3")` as opt-in for the UI layer.

## Config keys

```python
config = {
    "configurable": {
        "thread_id":     "run-2026-08-17-k12",   # REQUIRED with a checkpointer
        "checkpoint_ns": "",                      # "" = root graph
        "checkpoint_id": "1f19a18b-...",          # target a specific checkpoint
    },
    "recursion_limit": 100,      # TOP-LEVEL, NOT inside "configurable"
    "tags": [...], "metadata": {...},
}
```

### 🚨 Default recursion limit is **10007**, not 25 and not 1000
Docs say *"Starting in version 1.0.6, the default recursion limit is set to 1000 steps."* **Source in 1.2.11 says otherwise:**
```python
# langgraph/_internal/_config.py:32
DEFAULT_RECURSION_LIMIT = int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))
```
So: default **10007**, and it's overridable process-wide via env var `LANGGRAPH_DEFAULT_RECURSION_LIMIT`. **Implication for us:** the old "recursion limit as an accidental infinite-loop guard" no longer applies — an unbounded P6 refinement loop will spin ~10k super-steps before erroring. **Set `recursion_limit` explicitly on every invoke, and/or use `RemainingSteps`.**

### Recursion counter access + `RemainingSteps`

```python
from langgraph.managed import RemainingSteps        # also: langgraph.managed.is_last_step
# verified: RemainingSteps == Annotated[int, RemainingStepsManager]

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    remaining_steps: RemainingSteps      # auto-populated managed channel

def route(state: State) -> Literal["refine", END]:
    if state["remaining_steps"] <= 2:
        return END                      # graceful degradation, state preserved
    return "refine"
```
Reactive alternative: `except GraphRecursionError` from `langgraph.errors` (terminates the run, no partial-result checkpoint).

Available `config["metadata"]` keys inside a node: `langgraph_step`, `langgraph_node`, `langgraph_triggers`, `langgraph_path`, `langgraph_checkpoint_ns`.

## Durability modes — verified `Durability = Literal['sync','async','exit']`

```python
graph.invoke(inputs, config, durability="sync")
```
| Mode | Behavior | Use |
|---|---|---|
| `"exit"` | Persist only when execution exits (success, error, or interrupt). Fastest; **no crash recovery mid-run**. | Short/cheap graphs |
| `"async"` | Persist asynchronously while the next step runs. Good perf; small crash-window risk. | Default-ish |
| `"sync"` | Persist synchronously before the next step starts. Highest durability. | **Our long-running P3/P4/P7 phases** |

---

# 6. Persistence

## Checkpointer libraries and exact import paths

```python
from langgraph.checkpoint.memory     import InMemorySaver     # bundled (MemorySaver = alias)
from langgraph.checkpoint.sqlite     import SqliteSaver       # pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres   import PostgresSaver     # pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
```
Verified constructors:
```python
InMemorySaver(*, serde: SerializerProtocol|None = None, factory: type[defaultdict] = defaultdict)
SqliteSaver(conn: sqlite3.Connection, *, serde: SerializerProtocol|None = None)
AsyncSqliteSaver(conn: aiosqlite.Connection, *, serde=None)
SqliteSaver.from_conn_string(conn_string: str) -> Iterator[SqliteSaver]        # CONTEXT MANAGER
AsyncSqliteSaver.from_conn_string(conn_string: str) -> AsyncIterator[...]      # CONTEXT MANAGER
```

### ⚠️ `from_conn_string` is a context manager, not a constructor
It returns an `Iterator`/`AsyncIterator`, so `SqliteSaver.from_conn_string("x.db")` alone gives you a generator, not a saver. For a **long-lived orchestrator** build the connection yourself:
```python
import sqlite3
conn = sqlite3.connect("artifacts/checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()          # creates tables/indexes; idempotent
```
`check_same_thread=False` is required because LangGraph executes nodes on a thread pool.

## `BaseCheckpointSaver` contract (custom backends)

```python
from langgraph.checkpoint.base import (
    BaseCheckpointSaver, ChannelVersions, Checkpoint, CheckpointMetadata, CheckpointTuple,
)

class MyCheckpointer(BaseCheckpointSaver):
    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint,
                   metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig: ...
    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]],
                          task_id: str, task_path: str = "") -> None: ...
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: ...
    async def alist(self, config: RunnableConfig | None, *, filter: dict|None = None,
                    before: RunnableConfig|None = None, limit: int|None = None
                    ) -> AsyncIterator[CheckpointTuple]: ...
    async def adelete_thread(self, thread_id: str) -> None: ...
```
Five methods, **all required** (missing → `NotImplementedError` at runtime). Sync counterparts: `put`, `put_writes`, `get_tuple`, `list`, `delete_thread`. Sync graph execution (`invoke`/`stream`) uses the sync set; async execution (`ainvoke`/`astream`) uses the `a*` set. A **conformance test suite** ships for validating custom checkpointers.

Storage model: two tables — **checkpoints** (one row per super-step: `channel_values`, `channel_versions`, `versions_seen`, parent link) and **writes** (one row per node output within a super-step: `(task_id, channel, value)`).

## Serializers

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# pickle fallback for numpy/pandas/sklearn objects — RELEVANT TO US
InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))

# AES encryption; reads LANGGRAPH_AES_KEY env var
serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = SqliteSaver(sqlite3.connect("cp.db"), serde=serde)
```
Default `JsonPlusSerializer` uses **ormsgpack + JSON** — it will not serialize DataFrames, numpy arrays, or sklearn models. **For our pipeline: do NOT put embeddings/models in graph state.** Put file paths in state and write artifacts to disk. If you must, use `pickle_fallback=True`.

## `StateSnapshot` fields (verified via `get_state`)

| Field | Type | Notes |
|---|---|---|
| `values` | `dict` | channel values at this checkpoint |
| `next` | `tuple[str, ...]` | nodes to execute next; `()` = complete |
| `config` | `dict` | `thread_id`, `checkpoint_ns`, `checkpoint_id` |
| `metadata` | `dict` | `source` ∈ `{"input","loop","update"}`, `writes`, `step` |
| `created_at` | `str` | ISO 8601 |
| `parent_config` | `dict \| None` | previous checkpoint; `None` for first |
| `tasks` | `tuple[PregelTask, ...]` | each has `id`, `name`, `error`, `interrupts`, and `state` when `subgraphs=True` |

## State access API — verified signatures

```python
get_state(config, *, subgraphs: bool = False) -> StateSnapshot
aget_state(config, *, subgraphs: bool = False) -> StateSnapshot
get_state_history(config, *, filter=None, before=None, limit=None) -> Iterator[StateSnapshot]
update_state(config, values, as_node: str|None = None, task_id: str|None = None) -> RunnableConfig
aupdate_state(...)  # same
bulk_update_state(config, supersteps: Sequence[Sequence[StateUpdate]]) -> RunnableConfig
```
`StateUpdate(values, as_node=None, task_id=None)` from `langgraph.types`.

History is **reverse-chronological** (newest first). Filtering idioms:
```python
history = list(graph.get_state_history(config))
before_node_b = next(s for s in history if s.next == ("node_b",))
step_2        = next(s for s in history if s.metadata["step"] == 2)
forks         = [s for s in history if s.metadata["source"] == "update"]
interrupted   = next(s for s in history if s.tasks and any(t.interrupts for t in s.tasks))
```

## Time travel — replay vs fork (BOTH VERIFIED END-TO-END)

```python
# Full run
graph.invoke({"k":0,"log":[]}, cfg)     # -> {'k':12,'log':['pick_k','cluster(k=12)']}

hist   = list(graph.get_state_history(cfg))
before = next(h for h in hist if h.next == ("cluster",))   # checkpoint before `cluster` ran

# (a) PURE REPLAY — pass the historical config, input=None
graph.invoke(None, before.config)
# -> {'k':12,'log':['pick_k','cluster(k=12)']}   (cluster re-executed)

# (b) FORK — update_state at that checkpoint creates a NEW checkpoint, returns its config
fork_cfg = graph.update_state(before.config, {"k": 40})
graph.invoke(None, fork_cfg)
# -> {'k':40,'log':['pick_k','cluster(k=40)']}   ✅ alternative trajectory
```
- Nodes **before** the checkpoint are skipped (results already saved); nodes **after** re-execute, including LLM calls and `interrupt()`s.
- `update_state` **never mutates** the original checkpoint — it appends a new one (`metadata.source == "update"`).
- `update_state` values pass **through reducers**, so a channel with `operator.add` **accumulates** rather than replaces. Use `Overwrite(...)` to force replacement.
- `as_node="pick_k"` makes the update look like it came from that node, so execution resumes at that node's **successor**. **Verified:** after `update_state(cfg, {...}, as_node="pick_k")`, `get_state(cfg).next == ('cluster',)`.

**This is exactly our P5/P6 K-retuning capability:** fork at the "K chosen" checkpoint, override `k`, replay downstream clustering — without re-running P0–P4.

## Cross-process durability — VERIFIED

I wrote to a SQLite file, then built a **brand-new graph object with a brand-new connection** to the same file:
```python
conn2 = sqlite3.connect("tt.sqlite", check_same_thread=False)
g2 = build(SqliteSaver(conn2))
g2.get_state(cfg).values      # -> {'k':99, 'log':[...,'manual']}   ✅ full state recovered
len(list(g2.get_state_history(cfg)))   # -> 9 checkpoints            ✅ full history recovered
```
Interrupts, forks, and history all survive process restarts.

## Pending writes (fault tolerance)
When one node fails mid-super-step, LangGraph persists the **successful** sibling nodes' writes to the writes table. On resume those nodes are **not re-run**. This is what makes our parallel research-agent / naming-agent phases cheap to retry.

## Checkpoint namespaces (subgraphs) — VERIFIED
- `checkpoint_ns == ""` → root graph.
- `"node_name:<uuid>"` → subgraph invoked as that node.
- Nested: joined with `|`, e.g. `"outer:uuid|inner:uuid"`.
- A `checkpoint_map` links root ↔ subgraph checkpoint ids.

Observed live:
```python
snap.tasks[0].state.config["configurable"]
# {'thread_id':'p3',
#  'checkpoint_ns':'child:f40d30c0-ea04-a05b-ab18-a53ab529faaf',
#  'checkpoint_id':'1f19a18c-d40f-6204-8000-f6f10078c00d',
#  'checkpoint_map': {'': '1f19a18c-d40e-...', 'child:f40d...': '1f19a18c-d40f-...'}}
```

## Stores (long-term, cross-thread memory)

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

store = InMemoryStore()
store.put(("k12", "taxonomy_versions"), memory_id, {"label": "homework_help", "v": 3})
items = store.search(("k12", "taxonomy_versions"), limit=100)
items = store.search(ns, query="natural language question", limit=3)   # semantic
namespaces = store.list_namespaces(prefix=("k12",), max_depth=2)

# semantic search config
store = InMemoryStore(index={"embed": init_embeddings("openai:text-embedding-3-small"),
                             "dims": 1536, "fields": ["label", "$"]})
store.put(ns, key, value, index=["label"])   # embed only this field
store.put(ns, key, value, index=False)       # store but don't embed
```
`Item` attributes: `value`, `key`, `namespace`, `created_at`, `updated_at` (+ `score` when `query` given).

`BaseStore` contract (5 required async + 5 optional sync, all verified present): `aput/aget/adelete/asearch/alist_namespaces` and `put/get/delete/search/list_namespaces`.
```python
aput(namespace, key, value, index=None)
aget(namespace, key)
adelete(namespace, key)
asearch(namespace_prefix, *, query=None, filter=None, limit=10, offset=0)
alist_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)
```
Gotchas: `namespace_prefix` matches **by prefix, not exactly** (`("alice",)` also returns `("alice","memories")`); results past `limit` are **silently truncated** (no overflow signal — paginate with `offset`); **default ordering is backend-dependent** (Postgres → `updated_at` DESC; InMemory → insertion order) — sort client-side.

Inject into nodes via `runtime.store` or `graph.compile(store=store)`.

## Checkpointer vs Store
| | Checkpointer | Store |
|---|---|---|
| Persists | graph state snapshots | app-defined KV data |
| Scope | one thread | across threads |
| Access | `thread_id` in config | read/write from nodes |
| For | continuity, HITL, time travel, fault tolerance | preferences, facts, shared knowledge |

## `DeltaChannel` (beta, ≥1.2)
`from langgraph.channels.delta import DeltaChannel` — stores incremental deltas instead of the full accumulated value at each super-step. Cuts checkpoint size for append-heavy channels. Requires a **bulk reducer**; `snapshot_frequency` bounds read latency. **Beta, API may change** — worth it for our long `step_log` / `messages` channels only after the core pipeline is stable.

## Troubleshooting notes from docs
- `PostgresSaver`: keep `thread_id` **under 255 chars** (column limit) — use a UUID/hash.
- `InMemorySaver`/`MemorySaver` lose everything on restart — never in production.
- Checkpoints grow unboundedly on long threads: `checkpointer.setup()` creates indexes; add a cron to delete checkpoints older than N days.

---

# 7. Human-in-the-loop

## `interrupt()` — verified signature: `interrupt(value: Any) -> Any`

```python
from langgraph.types import interrupt, Command

def taxonomy_gate(state: State):
    decision = interrupt({"question": "Approve taxonomy?", "tree": state["tree"]})
    return {"approved": bool(decision)}
```
Requires: (1) a checkpointer, (2) a `thread_id` in config, (3) JSON-serializable payload.

## Resume — VERIFIED live

```python
cfg = {"configurable": {"thread_id": "t1"}}
r = graph.invoke({"log": [], "approved": False}, cfg)
# -> {'log': [], 'approved': False,
#     '__interrupt__': [Interrupt(value={'question':'approve taxonomy?','n':3},
#                                 id='ebe98ac2edf872694b705d81c5827900')]}

st = graph.get_state(cfg)
st.next                                  # ('gate',)
[t.interrupts for t in st.tasks]         # [(Interrupt(value=..., id=...),)]

r2 = graph.invoke(Command(resume=True), cfg)
# -> {'log': ['gate:True', 'after'], 'approved': True}      ✅
```
With `stream_events(version="v3")`:
```python
stream = graph.stream_events(inputs, cfg, version="v3")
out = stream.output
stream.interrupted    # True
stream.interrupts     # [Interrupt(value={...}, id='3739...')]
graph.stream_events(Command(resume=False), cfg, version="v3").output
```
Verified `Interrupt` dataclass: `Interrupt(value: Any, id: str)`.

## Multiple simultaneous interrupts (parallel branches)
Resume **all at once** by mapping interrupt id → value:
```python
resume_map = {i.id: f"answer for {i.value}" for i in stream.interrupts}
graph.stream_events(Command(resume=resume_map), config, version="v3")
```
**Directly applicable to P2** (2 annotators + 1 referee interrupting in parallel) and **P7** (5 naming agents).

## Approve / edit / review patterns

```python
# Approve or reject — combine interrupt with Command routing
def approval_node(state) -> Command[Literal["proceed", "cancel"]]:
    ok = interrupt({"question": "Proceed?", "details": state["action_details"]})
    return Command(goto="proceed" if ok else "cancel")

# Review and edit — return the edited payload straight into state
def review_node(state):
    edited = interrupt({"instruction": "Edit these labels", "current": state["labels"]})
    return {"labels": edited}
```

## 🔴 Rules of interrupts (these WILL bite us)

`interrupt()` pauses by **raising a special exception**, and on resume **the entire node re-runs from the top**.

1. **Never wrap `interrupt` in a bare `try/except Exception`** — you'll swallow the pause signal. Catch specific exception types, or put `interrupt` outside the try.
2. **Never reorder / conditionally skip `interrupt` calls within a node.** Resume matching is **strictly index-based** against a per-task resume list. No `if`-guarded interrupts, no loops over data that can change between runs, **no `while True` validation loops** — use a conditional edge for validation retries instead.
3. **Only JSON-serializable payloads.** No functions, no class instances.
4. **Side effects before `interrupt` must be idempotent** — they re-run on every resume. Use upserts/idempotency keys, put side effects *after* the interrupt, or isolate them in a separate node.

## Static interrupts (breakpoints — debugging only)

```python
graph = builder.compile(checkpointer=cp,
                        interrupt_before=["node_a"],
                        interrupt_after=["node_b", "node_c"])
graph.invoke(inputs, cfg)
graph.invoke(None, cfg)     # resume with None input → run to next breakpoint
```
Also settable **per-invocation**: `graph.invoke(inputs, cfg, interrupt_before=[...], interrupt_after=[...])`. Not deprecated, but docs say **not recommended for HITL** — use `interrupt()`.

## Persisting HITL across process restarts
Confirmed by my SQLite restart test: with a durable checkpointer, a paused thread is fully recoverable in a new process — `get_state` shows `next` and the pending `Interrupt`, and `invoke(Command(resume=...), cfg)` continues correctly. Store the `thread_id` externally (DB row / file) — it is the only pointer needed.

## Subgraph interrupts propagate to the top
**Verified:** a subgraph node calling `interrupt()` surfaced at the *parent* `invoke` as `__interrupt__`, and `graph.invoke(Command(resume="Z"), cfg)` on the **parent** resumed the **subgraph** correctly. When resuming, the parent node re-runs from its start **and** the subgraph node re-runs from its start.

---

# 8. Streaming

## Two APIs

### (a) `stream()` / `astream()` with `version="v2"` (≥1.1, stable) — RECOMMENDED for our orchestrator

Every chunk is a uniform `StreamPart`:
```python
{"type": "values"|"updates"|"messages"|"custom"|"checkpoints"|"tasks"|"debug",
 "ns":   (),        # namespace tuple, populated for subgraph events
 "data": ...}       # payload varies by mode
```
Without `version="v2"` (i.e. v1 default), the shape **changes** based on options — single mode → raw data, multiple modes → `(mode, data)` tuples, `subgraphs=True` → `(ns, data)` tuples. **Always pass `version="v2"`.**

Verified `StreamMode = Literal['values','updates','checkpoints','tasks','debug','messages','custom']`.

| Mode | TypedDict | Payload |
|---|---|---|
| `values` | `ValuesStreamPart` | full state after each step |
| `updates` | `UpdatesStreamPart` | `{node_name: update}`; multiple updates in a step stream separately |
| `messages` | `MessagesStreamPart` | `(message_chunk, metadata)` LLM tokens |
| `custom` | `CustomStreamPart` | whatever `get_stream_writer()` emits |
| `checkpoints` | `CheckpointStreamPart` | same format as `get_state()`; **needs a checkpointer** |
| `tasks` | `TasksStreamPart` | task start/finish with results and errors; **needs a checkpointer** |
| `debug` | `DebugStreamPart` | `checkpoints` + `tasks` + extra metadata |

All importable from `langgraph.types`; `StreamPart` is a disjoint union on `part["type"]` giving full type narrowing.

**Verified multi-mode + subgraphs + custom:**
```python
for part in graph.stream(inputs, cfg, stream_mode=["updates","custom"],
                         subgraphs=True, version="v2"):
    print(part["type"], part["ns"], part["data"])
# updates ()  {'sub': {'foo': 'X'}}
# custom  ()  {'phase': 'node_b', 'pct': 50}
# updates ()  {'node_b': {'foo': 'B'}}
```

### (b) `stream_events(..., version="v3")` — typed projections (NEW 1.2, **beta**)

```python
stream = graph.stream_events(inputs, config, version="v3")

for message in stream.messages:          # chat model output
    for token in message.text:           # token deltas
        print(token, end="", flush=True)

for snapshot in stream.values: ...       # state snapshots
for subgraph in stream.subgraphs:        # nested runs, no ns-string parsing
    print(subgraph.graph_name, subgraph.path)
    for m in subgraph.messages: ...

final = stream.output                    # final output (drives the stream)
stream.interrupted                       # bool
stream.interrupts                        # tuple[Interrupt, ...]
stream.extensions                        # custom transformer projections
```
Multiple consumers read concurrently without starving each other. Sync interleaving in arrival order:
```python
for name, item in stream.interleave("values", "messages", "subgraphs"): ...
```
Async concurrent consumption:
```python
stream = await graph.astream_events(inputs, version="v3")
await asyncio.gather(consume_messages(), consume_subgraphs())
```
Also: `message.reasoning` (reasoning deltas), `message.tool_calls` (tool-call arg chunks), `str(message.text)` for the full text. Built-in `ToolCallTransformer`; register custom transformers at call time (`transformers=[...]`) or compile time (`compile(transformers=[...])`).

## Custom progress events — `get_stream_writer`

```python
from langgraph.config import get_stream_writer      # verified: () -> Callable[[Any], None]

def cluster_battery(state: State):
    writer = get_stream_writer()
    for i, algo in enumerate(ALGOS):
        writer({"phase": "P4", "algo": algo, "pct": 100*i//len(ALGOS)})
        ...
    return {...}
```
Works in nodes **and tools**. Consume with `stream_mode="custom"` (may be combined: `["updates","custom"]`, at least one must be `"custom"`).

### ⚠️ Python < 3.11 async caveats
- `get_stream_writer()` **does not work** in async code on Python < 3.11. Add an explicit `writer: StreamWriter` parameter to the node/tool instead (`from langgraph.types import StreamWriter`).
- For async LLM token streaming on Python < 3.11 you must explicitly thread `RunnableConfig` through to `ainvoke()`.
**→ Target Python 3.11+ (ideally 3.12) for our stack and this whole class of bug disappears.**

## ⚠️ Subgraph token streaming requires `subgraphs=True`
`create_agent(...)` returns a **compiled graph**, so adding it as a node makes it a subgraph. Without `subgraphs=True`, `stream_mode="messages"` on the parent emits **no tokens from the inner agent**. This surprises people because `agent.stream(...)` directly does work.

---

# 9. Subgraphs

## Two composition patterns

| Pattern | When | How |
|---|---|---|
| **Add as a node** | parent and subgraph **share state keys** | `builder.add_node("child", compiled_subgraph)` |
| **Call inside a node** | **different schemas** / need transformation | wrapper fn calls `subgraph.invoke(...)` and maps state both ways |

```python
# Shared schema — pass the compiled subgraph directly
subgraph = subgraph_builder.compile()
builder.add_node("node_1", subgraph)

# Different schema — wrapper transforms in and out
def call_subgraph(state: ParentState):
    out = subgraph.invoke({"bar": state["foo"]})
    return {"foo": out["bar"]}
builder.add_node("node_1", call_subgraph)
```

## Subgraph persistence — the `checkpointer=` argument on the **subgraph's** `.compile()`

| Mode | `checkpointer=` | Behavior |
|---|---|---|
| **Per-invocation** (default) | `None` | Fresh each call; **inherits parent's checkpointer** so interrupts + durable execution work within a call |
| **Per-thread** | `True` | State accumulates across calls on the same thread |
| **Stateless** | `False` | No checkpointing; plain function call; **no interrupts, no durable execution** |

| Feature | Per-invocation | Per-thread | Stateless |
|---|---|---|---|
| Interrupts (HITL) | ✅ | ✅ | ❌ |
| Multi-turn memory | ❌ | ✅ | ❌ |
| Multiple calls, different subgraphs | ✅ | ⚠️ ns conflicts | ✅ |
| Multiple calls, same subgraph | ✅ | ❌ | ✅ |
| State inspection | ⚠️ current invocation only | ✅ | ❌ |

**Per-invocation (the default) is right for our phase subgraphs.** The parent must be compiled with a real checkpointer for any of this to work.

## Viewing subgraph state
```python
snap = graph.get_state(config, subgraphs=True)
sub_state = snap.tasks[0].state          # StateSnapshot of the subgraph
```
⚠️ Requires LangGraph to **statically discover** the subgraph — i.e. added as a node or called inside a node. It does **not** work when a subgraph is called from inside a *tool* function or other indirection. (Interrupts still propagate regardless of nesting.)

## Streaming from subgraphs
`stream(..., subgraphs=True, version="v2")` → `part["ns"]` is `()` for root, `("node_name:<task_id>",)` for subgraphs. Or use `stream_events(version="v3").subgraphs` for the typed projection.

## Known rough edge (from docs)
> When a subgraph updates state, the parent graph may not see the changes immediately... each subgraph manages its own checkpoint namespace. **Fix:** use a **Store** for data that must cross graph boundaries.

---

# 10. Errors, recovery, graceful shutdown

`from langgraph.errors import ...` — verified exports:
`GraphRecursionError`, `InvalidUpdateError`, `NodeError`, `NodeTimeoutError`, `NodeCancelledError`, `NodeInterrupt`, `GraphInterrupt`, `GraphBubbleUp`, `GraphDrained`, `EmptyChannelError`, `EmptyInputError`, `ParentCommand`, `TaskNotFound`, `ErrorCode`.

Documented error codes: `GRAPH_RECURSION_LIMIT`, `INVALID_CHAT_HISTORY`, `INVALID_CONCURRENT_GRAPH_UPDATE`, `INVALID_GRAPH_NODE_RETURN_VALUE`, `MISSING_CHECKPOINTER`, `MULTIPLE_SUBGRAPHS`.

## Failure ↔ checkpoint interaction
- Checkpoints are written at **super-step boundaries**, never mid-node. A resumed run **re-runs the affected node from the start** — so node bodies must be idempotent.
- **Pending writes:** siblings that succeeded in the failed super-step are already durable and are not re-run.
- **Graph structure can change freely** between runs for completed threads. For **interrupted** threads, all topology changes are supported **except renaming/removing nodes**.
- **State migrations:** adding/removing keys is fully forward+backward compatible. **Renamed keys lose their saved state.** Incompatible type changes can break old threads.
- Tasks/interrupts inside a node impose stricter determinism on resume (cached task results are matched positionally).

## Graceful shutdown (≥1.2)
```python
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

control = RunControl()
graph.invoke(inputs, config, control=control)   # control= also on stream/ainvoke/astream
# from a SIGTERM handler or supervisor thread:
control.request_drain("sigterm")
```
Stops the run **after the current super-step** and saves a resumable checkpoint. Inside nodes:
```python
def my_node(state, runtime: Runtime):
    if runtime.drain_requested:
        return {"status": "skipped", "reason": runtime.drain_reason}
    return {"status": do_work()}
```
**This is the clean answer for killing a long P3/P4 run without losing hours of work.**

## Idle timeouts / heartbeats
`TimeoutPolicy(idle_timeout=..., refresh_on='auto'|'heartbeat')`. With `refresh_on='heartbeat'`, call `runtime.heartbeat()` from long-running loops to refresh the idle timer — good for a node chewing through 100k queries.

---

# 11. Functional API (`@entrypoint` / `@task`)

```python
from langgraph.func import entrypoint, task
```
Verified signatures:
```python
entrypoint(checkpointer: BaseCheckpointSaver|None = None,
           store: BaseStore|None = None,
           cache: BaseCache|None = None,
           context_schema: type|None = None,
           cache_policy: CachePolicy|None = None,
           retry_policy: RetryPolicy|Sequence[RetryPolicy]|None = None,
           timeout: float|timedelta|TimeoutPolicy|None = None)

task(func=None, *, name: str|None = None,
     retry_policy=None, cache_policy=None, timeout=None)
```

```python
@task(retry_policy=RetryPolicy(max_attempts=3), timeout=TimeoutPolicy(idle_timeout=30))
def embed_batch(rows: list[str]) -> list[list[float]]: ...

@entrypoint(checkpointer=checkpointer, store=store)
def pipeline(inp: dict, *, previous: Any = None,
             store: BaseStore, writer: StreamWriter, config: RunnableConfig) -> dict:
    futures = [embed_batch(b) for b in inp["batches"]]   # parallel, checkpointed
    return {"vecs": [f.result() for f in futures]}
```
- Entrypoint takes **exactly one positional arg** (use a dict for multiple values).
- Injectable params by name+annotation: `previous`, `store`, `writer`, `config`.
- Returns a `Pregel` object → `invoke`/`ainvoke`/`stream`/`astream`/`stream_events`.
- Tasks return **futures**; `.result()` (sync) or `await` (async).
- Tasks may **only** be called from an entrypoint, another task, or a **graph node** (nice — you can use `@task` inside `StateGraph` nodes to get sub-node checkpointing).
- Entrypoint I/O and task outputs must be **JSON-serializable**.
- `entrypoint.final[return_type, save_type]` decouples the returned value from the checkpointed value:
  ```python
  return entrypoint.final(value=previous, save=2 * number)
  ```
- `previous` = the previous invocation's saved value on the same `thread_id`.

## When Functional beats Graph
| Graph API | Functional API |
|---|---|
| need visualization of decision paths | wrapping existing imperative code with minimal change |
| shared state across many nodes | linear/simple control flow |
| explicit parallel fan-out/fan-in with reducers | ad-hoc parallelism via futures |
| team splits work per node | single-author script |
| time travel / state inspection per step | just want checkpointing + retries |

They **compose** — call an entrypoint from inside a graph node, or use `@task` inside nodes.

**For our 12-phase playbook: Graph API for the top-level orchestrator** (we need visualization, HITL gates, per-phase time travel, and typed shared state). **`@task` inside heavy nodes** (embedding bake-off, clustering battery) to get free sub-node checkpointing and parallelism without exploding the node count.

## Determinism rules (Functional API + tasks-in-nodes)
Resume replays saved task results **positionally**. Changing task call order, or making control flow depend on non-deterministic data before the resume point, will mismatch cached values. Encapsulate all randomness/API calls in `@task`.

---

# 12. Visualization

```python
graph.get_graph().draw_mermaid_png()      # bytes
graph.get_graph().draw_mermaid()          # mermaid source
graph.get_graph(xray=True)                # expand subgraphs
graph.get_graph().nodes / .edges          # edges have .source, .target, .conditional
```
Useful for our P11 report generation — emit the phase graph as mermaid straight into the report.

---

# 13. Consolidated gotcha list for our build

1. **`InvalidUpdateError` on parallel writes** — every channel touched by fan-out needs a reducer. Biggest practical constraint on the whole design.
2. **Default recursion limit is 10007**, not 25/1000 — set it explicitly per phase; use `RemainingSteps` for the P6 refinement loop.
3. **`NodeTimeoutError` is not a `TimeoutError`** — import it explicitly.
4. **`Command.PARENT` + `Command[Literal[...]]` annotation = compile error** — omit the annotation, or use `destinations=`.
5. **`Command.PARENT` on a shared key requires a reducer in the parent state.**
6. **`from_conn_string` is a context manager** — build `sqlite3.connect(..., check_same_thread=False)` yourself for long-lived orchestrators.
7. **Default serializer can't handle numpy/pandas/sklearn** — keep artifacts on disk, paths in state; or `JsonPlusSerializer(pickle_fallback=True)`.
8. **Private state channels leak into `stream_mode="values"`** — use `output_keys` or `updates` mode.
9. **Never bare-`try/except` around `interrupt()`**; never conditionally skip/reorder interrupts; no `while True` validation loops.
10. **Node bodies re-run on resume** — everything before an `interrupt()` must be idempotent.
11. **`stream_events(version="v3")` is beta** (emits `LangChainBetaWarning`) and the param **defaults to v2**.
12. **`subgraphs=True` is mandatory** to see tokens/updates from `create_agent`-style subgraph nodes.
13. **`set_node_defaults` is not inherited by subgraphs** — repeat per builder.
14. **Renamed state keys lose saved state** across migrations — plan schema versioning (see `backward-compatibility` doc's "pin a behavioral version in state" pattern).
15. **Python < 3.11 breaks `get_stream_writer()` in async** — target 3.11+/3.12.
16. **Static edges + `Command`/tool `goto` both fire** — pick one routing mechanism per node.


---

## Recommendations carried into the design

- Pin langgraph>=1.2.11,<1.3 with langchain-core>=1.5.5,<2 and target Python 3.12, since Python <3.11 breaks get_stream_writer() in async code and per-node timeouts/error handlers/set_node_defaults all require langgraph>=1.2.

## Unverified or version-dependent

- The docs state the default recursion limit is 1000 (as of v1.0.6), but the installed langgraph 1.2.11 source reads DEFAULT_RECURSION_LIMIT = int(getenv('LANGGRAPH_DEFAULT_RECURSION_LIMIT', '10007')) in langgraph/_internal/_config.py:32. I trust the source, but the env-var override means the effective default is deployment-dependent — always set recursion_limit explicitly.
- The docs claim NodeTimeoutError 'subclasses Python's built-in TimeoutError', but the verified MRO in 1.2.11 is (NodeTimeoutError, Exception, BaseException, object). This may be an intended fix that regressed, or a docs error; re-verify after any minor upgrade before relying on except TimeoutError.
- stream_events(version='v3') emits LangChainBetaWarning ('The v3 streaming protocol on Pregel is experimental') and the parameter still defaults to 'v2', yet the docs recommend v3 for new applications. The v3 projection API (stream.messages/.values/.subgraphs/.output/.interrupts/.extensions/.interleave) may change shape before it stabilizes.
- DeltaChannel is explicitly marked beta and requiring langgraph>=1.2, with the API subject to change; I read its docs but did not execute it.
- I did not install or execute langgraph-checkpoint-postgres, so PostgresSaver/AsyncPostgresSaver constructor and setup() signatures are taken from docs rather than introspection. The documented 255-char thread_id limit was not empirically confirmed.
- Node-level `timeout=` on sync nodes is documented to raise at compile time; I only verified the async timeout path (NodeTimeoutError raised correctly at 0.2s against a 2s sleep).
- The exact set of langchain-core message/streaming classes referenced by stream_mode='messages' (message.text, message.reasoning, message.tool_calls) was not exercised, since doing so requires live LLM credentials.
- Semantic search in InMemoryStore (the index={'embed':..., 'dims':..., 'fields':...} config) was not executed because it needs an embeddings provider; signatures come from docs.
- The `transformers=` parameter on compile() and stream_events() for custom stream projections is present in the verified signatures, but I did not build or run a custom transformer.
- Backward-compatibility guidance around renamed state keys losing saved state, and the 'pin a behavioral version in state' pattern, was read from docs but not tested against a real migrated thread.

## Sources

- https://docs.langchain.com/oss/python/langgraph/graph-api.md
- https://docs.langchain.com/oss/python/langgraph/use-graph-api.md
- https://docs.langchain.com/oss/python/langgraph/overview.md
- https://docs.langchain.com/oss/python/langgraph/persistence.md
- https://docs.langchain.com/oss/python/langgraph/checkpointers.md
- https://docs.langchain.com/oss/python/langgraph/stores.md
- https://docs.langchain.com/oss/python/langgraph/interrupts.md
- https://docs.langchain.com/oss/python/langgraph/fault-tolerance.md
- https://docs.langchain.com/oss/python/langgraph/streaming.md
- https://docs.langchain.com/oss/python/langgraph/event-streaming.md
- https://docs.langchain.com/oss/python/langgraph/use-subgraphs.md
- https://docs.langchain.com/oss/python/langgraph/use-time-travel.md
- https://docs.langchain.com/oss/python/langgraph/functional-api.md
- https://docs.langchain.com/oss/python/langgraph/use-functional-api.md
- https://docs.langchain.com/oss/python/langgraph/choosing-apis.md
- https://docs.langchain.com/oss/python/langgraph/backward-compatibility.md
- https://docs.langchain.com/oss/python/langgraph/pregel.md
- https://docs.langchain.com/oss/python/langgraph/changelog-py.md
- https://docs.langchain.com/oss/python/langgraph/test.md
- https://docs.langchain.com/oss/python/langgraph/add-memory.md
- https://docs.langchain.com/oss/python/langgraph/llms.txt
- https://pypi.org/pypi/langgraph/json
- https://pypi.org/pypi/langchain-core/json
- https://pypi.org/pypi/langchain/json
- https://pypi.org/pypi/langgraph-checkpoint/json
- https://pypi.org/pypi/langgraph-checkpoint-sqlite/json
- https://pypi.org/pypi/langgraph-checkpoint-postgres/json
- https://pypi.org/pypi/langgraph-prebuilt/json
- https://pypi.org/pypi/langgraph-sdk/json
- https://pypi.org/pypi/langgraph-cli/json
- https://github.com/langchain-ai/langgraph/issues/5023
- https://reference.langchain.com/python/langgraph/graph/state/StateGraph