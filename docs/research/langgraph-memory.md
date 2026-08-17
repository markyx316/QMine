# Memory Management & Context Engineering

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

> **Scope note:** every API below was verified against live docs/source on 2026-08-17, not from memory. Version pins verified from PyPI the same day: `langgraph==1.2.11` (2026-08-11), `langchain==1.3.15` (2026-08-11), `langchain-core==1.5.5` (2026-08-14), `langgraph-checkpoint-postgres==3.1.2`, `langgraph-checkpoint-sqlite==3.1.1`, `deepagents==0.7.6` (2026-08-13), **`langmem==0.0.30` (last release 2025-10-27 — ~10 months stale)**, `trustcall==0.0.39` (2025-04-14, stale).

---

## 0. The single most important finding: LangMem is no longer the recommended path

The task brief assumes LangMem is the memory SDK. **It is effectively in maintenance mode.** `langmem` 0.0.30 has not shipped since 2025-10-27 while `langgraph` went 0.6 → 1.2 and `langchain` went to 1.3. Its `pyproject.toml` declares `langgraph>=0.6.0,<2`, so it still *installs* and imports, but:

- Its docs are all written against `langgraph.prebuilt.create_react_agent`, which has been superseded by `langchain.agents.create_agent`.
- Its `SummarizationNode` / `summarize_messages` are **superseded by `langchain.agents.middleware.SummarizationMiddleware`**, which is actively developed and is what `deepagents` itself uses internally.
- Its `trustcall` dependency (the dedup/JSON-patch engine) is also stale (2025-04).

**Recommendation for our 12-phase system:** use `langgraph` core (checkpointer + `BaseStore`) + `langchain` middleware as the load-bearing stack. Treat LangMem as optional and pinned, used *only* for `create_memory_manager` / `create_prompt_optimizer` / `ReflectionExecutor` if you want them; do not build the critical path on it. The 2026-current LangChain-sanctioned pattern for long-horizon agent memory is the **`deepagents` filesystem-backed memory model** (`CompositeBackend` + `StoreBackend`), which is exactly the "files + store" architecture we want anyway.

**Second correction:** there is **no official `SqliteStore`.** `langgraph-checkpoint-sqlite` ships `SqliteSaver`/`AsyncSqliteSaver` (checkpointers) only. The official store backends are exactly: `InMemoryStore` (in `langgraph-checkpoint`), `PostgresStore` (`langgraph-checkpoint-postgres`), Redis (`langgraph-checkpoint-redis`), MongoDB (`langgraph-store-mongodb`), Upstash (`langgraph-store-upstash`). For local dev of our pipeline: SqliteSaver for checkpoints + InMemoryStore or a custom `BaseStore` subclass over SQLite for the store.

---

## 1. Short-term memory: checkpointer-backed thread state

### Checkpointer wiring

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Production:

```python
# pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()          # run ONCE, ideally as a deploy step, not at import
    graph = builder.compile(checkpointer=checkpointer)
```

Async variants: `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver`, `from langgraph.store.postgres.aio import AsyncPostgresStore`. Both are async context managers with `await .setup()`.

### Durability modes (LangGraph 1.x) — directly relevant to a 12-phase pipeline

```python
graph.stream({"input": "test"}, durability="sync")
```

- `"exit"` — persists only when the run exits (success/error/interrupt). Fastest; **no mid-run crash recovery**.
- `"async"` — persists asynchronously while the next step runs. Good default.
- `"sync"` — persists before the next step starts. Highest durability, some overhead.

For our pipeline: **use `"sync"` for phases that cost real money/time (P2 gold labelling, P3 embedding bake-off, P4 clustering battery) and `"async"` elsewhere.** A crash in the middle of a 40-minute HDBSCAN sweep should not replay the sweep.

### Checkpoint storage growth

`DeltaChannel` (requires `langgraph>=1.2`, **beta**) stores incremental deltas instead of the full accumulated channel value at every super-step. Relevant because our append-heavy channels (metric history, per-phase logs) otherwise get rewritten in full at each super-step.

Also: `checkpointer.delete_thread(thread_id)` deletes all checkpoints for a thread.

### Trimming

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}
```

`start_on`/`end_on` exist to keep the trimmed history *provider-valid* — most providers require an assistant message with tool calls to be followed by its tool results.

In `create_agent`, do it as middleware:

```python
from typing import Any
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime

@before_model
def trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    first_msg = messages[0]
    recent = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first_msg, *recent]}

agent = create_agent("gpt-5.5", tools=[...], middleware=[trim], checkpointer=InMemorySaver())
```

### Deleting messages

```python
from langchain.messages import RemoveMessage            # NOTE: new location in langchain 1.x
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:2]]}

def wipe(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

Requires a state key with the `add_messages` reducer (`MessagesState` / `AgentState` both provide it).

### Summarization — the current API

**Use this, not `langmem.short_term.SummarizationNode`:**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[SummarizationMiddleware(
        model="gpt-5.4-mini",
        trigger=("tokens", 4000),
        keep=("messages", 20),
    )],
    checkpointer=checkpointer,
)
```

Verified signature (from `langchain/agents/middleware/summarization.py`):

```python
def __init__(
    self,
    model: str | BaseChatModel,
    *,
    trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None = None,
    keep: ContextSize = ("messages", 20),
    token_counter: TokenCounter = count_tokens_approximately,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    trim_tokens_to_summarize: int | None = ...,
) -> None
```

`trigger` semantics are richer than the docs page suggests:
- Tuple = one threshold: `("messages", 50)`, `("tokens", 3000)`, `("fraction", 0.8)` (fraction of the model's max input tokens).
- Dict = **AND**: `{"tokens": 4000, "messages": 10}`.
- List = **OR** across items: `[("fraction", 0.8), ("messages", 100)]`, or `[{"tokens": 5000, "messages": 3}, {"tokens": 3000, "messages": 6}]`.

`keep` takes a single `ContextSize` only (no list).

**Legacy LangMem equivalent** (works, but prefer the above). Exact verified signatures:

```python
from langmem.short_term import SummarizationNode, RunningSummary, summarize_messages

def summarize_messages(
    messages: list[AnyMessage], *,
    running_summary: RunningSummary | None,
    model: LanguageModelLike,
    max_tokens: int,
    max_tokens_before_summary: int | None = None,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> SummarizationResult
```

`RunningSummary` fields: `summary: str`, `summarized_message_ids: set[str]`, `last_summarized_message_id: str | None`. `SummarizationNode.__init__` adds `input_messages_key="messages"`, `output_messages_key="summarized_messages"`, `name="summarization"`. The node writes its state under `context["running_summary"]`, and the trick in the docs is a **private input state** so the summarized list never leaks into the main channel:

```python
class State(MessagesState):
    context: dict[str, RunningSummary]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

def call_model(state: LLMInputState):   # <- private schema isolates the summarized list
    return {"messages": [model.invoke(state["summarized_messages"])]}
```

That private-input-schema idiom is worth stealing regardless of which summarizer we use — it is "isolate" applied inside a single graph.

---

## 2. Long-term memory: the `BaseStore` API

### Verified method signatures (from `langgraph/store/base/__init__.py`)

```python
store.get(namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None) -> Item | None

store.search(namespace_prefix: tuple[str, ...], /, *,
             query: str | None = None,
             filter: dict[str, Any] | None = None,
             limit: int = 10, offset: int = 0,
             refresh_ttl: bool | None = None) -> list[SearchItem]

store.put(namespace: tuple[str, ...], key: str, value: dict[str, Any],
          index: Literal[False] | list[str] | None = None, *,
          ttl: float | None | NotProvided = NOT_PROVIDED) -> None

store.delete(namespace: tuple[str, ...], key: str) -> None

store.list_namespaces(*, prefix=None, suffix=None, max_depth=None,
                      limit=100, offset=0) -> list[tuple[str, ...]]
```

Async: `aget/asearch/aput/adelete/alist_namespaces` — **all five async methods are required** for a custom store; sync are optional but recommended.

`Item` attributes: `value` (dict), `key`, `namespace` (tuple, serializes to list in JSON), `created_at`, `updated_at`. `SearchItem` adds `score` when `query` is given.

### Three behaviors that will bite us

1. **`namespace_prefix` matches by prefix, not exactly.** `search(("run_2026Q3",))` returns items under `("run_2026Q3","decisions")`, `("run_2026Q3","metrics")`, everything. To restrict to one level, pass the full namespace or filter client-side on `item.namespace`.
2. **Results past `limit` are silently truncated — no overflow signal.** Set `limit` above expected max, or paginate with `offset`.
3. **Default ordering is backend-dependent.** `PostgresStore`/`AsyncPostgresStore` order by `updated_at` DESC. `InMemoryStore` returns insertion order (most recent *last*). **Do not rely on order; sort client-side on `item.updated_at`.** This matters for our metric-history ledger.

Pagination idiom:

```python
page_size, offset = 50, 0
while True:
    page = store.search(("run_2026Q3", "metrics"), limit=page_size, offset=offset)
    if not page:
        break
    for item in page:
        ...
    offset += page_size
```

### Semantic search config

```python
from langchain.embeddings import init_embeddings
from langgraph.store.base import IndexConfig      # typed, preferred over a raw dict
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index=IndexConfig(
    embed=init_embeddings("openai:text-embedding-3-small"),
    dims=1536,
    fields=["food_preference", "$"],   # "$" = embed the whole JSON doc
))
```

`IndexConfig.embed` accepts **four** forms: a LangChain `Embeddings` instance, a sync `EmbeddingsFunc`, an async `AEmbeddingsFunc`, or a provider string like `"openai:text-embedding-3-small"`. Dims cheat-sheet from the source: `text-embedding-3-large`=3072, `-3-small`/`ada-002`=1536, `cohere:embed-english-v3.0`=1024, `-light-v3.0`=384.

Per-item override at write time:

```python
store.put(ns, key, {"food_preference": "I love Italian", "context": "dinner"},
          index=["food_preference"])   # embed only this field
store.put(ns, key2, {"system_info": "..."}, index=False)  # retrievable but NOT searchable
```

`index=False` is the right choice for our large machine-generated blobs (cluster centroids, confusion matrices) — store them, don't pay to embed them.

Postgres with index:

```python
with PostgresStore.from_conn_string(DB_URI, index=IndexConfig(embed=embed, dims=1536)) as store:
    store.setup()
```

Deployed on LangSmith/Agent Server, the store is provided automatically but **semantic search must be declared in `langgraph.json`**:

```json
{"store": {"index": {"embed": "openai:text-embeddings-3-small", "dims": 1536, "fields": ["$"]}}}
```

### TTL — underused, very relevant for us

`TTLConfig` (TypedDict): `refresh_on_read: bool` (default True), `omit_expired: bool` (default False), `default_ttl: float | None`. Combined with per-`put` `ttl=`, this gives automatic expiry for **ephemeral scratch memories** (e.g., P4's per-algorithm trial results) while decision/metric ledgers get `ttl=None` (permanent).

### Accessing the store inside nodes — three mechanisms, ranked

**(a) `Runtime` injection — the modern, recommended way.** Add a `Runtime` parameter and LangGraph injects it:

```python
from dataclasses import dataclass
import uuid
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, MessagesState, START

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    ns = (user_id, "memories")
    memories = await runtime.store.asearch(ns, query=state["messages"][-1].content, limit=3)
    info = "\n".join(d.value["data"] for d in memories)
    await runtime.store.aput(ns, str(uuid.uuid4()), {"data": "User prefers dark mode"})

builder = StateGraph(MessagesState, context_schema=Context)   # context_schema, not config_schema
graph = builder.compile(store=store)

graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    {"configurable": {"thread_id": "1"}},
    context=Context(user_id="1"),          # runtime context is a separate kwarg now
)
```

`Runtime` fields (verified from `langgraph/runtime.py`): `context`, `store`, `stream_writer`, `heartbeat`, `previous`, `execution_info`, `server_info`, `control`.

**(b) `get_runtime()` contextvar** — `from langgraph.runtime import get_runtime; rt = get_runtime(Context)`. Use inside helpers that can't take a parameter.

**(c) `get_store()` contextvar** — canonical import is **`from langgraph.config import get_store`**. LangMem's docs use `from langgraph.utils.config import get_store`, which the source now labels *"Backwards compat imports for config utilities, to be removed in v1"* — a re-export shim. **Use `langgraph.config`.** Handy inside `prompt` functions and tools:

```python
from langgraph.config import get_store

def prompt(state):
    store = get_store()
    memories = store.search(("memories",), query=state["messages"][-1].content)
    return [{"role": "system", "content": f"<memories>\n{memories}\n</memories>"}, *state["messages"]]
```

**(d) In `create_agent` tools:** `ToolRuntime`.

```python
from langchain.tools import tool, ToolRuntime

@tool
def lookup(query: str, runtime: ToolRuntime) -> str:
    """Look up a stored decision."""
    return str(runtime.store.search(("decisions",), query=query, limit=5))
```

### Custom store contract (if we back it with our own DB)

Suggested SQL schema straight from the docs:

```sql
CREATE TABLE store_items (
    namespace   TEXT[] NOT NULL,
    key         TEXT NOT NULL,
    value       JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (namespace, key)
);
CREATE INDEX ON store_items USING gin(namespace);
```

Requirements: prefix matching on `asearch`, ~O(1) exact `aget`, plain JSON-serializable dict values (no pickled Python objects), and `raise NotImplementedError` if `query` is passed to a backend without vector search. **There is no conformance suite** — test against `InMemoryStore` as the reference oracle:

```python
import pytest
from langgraph.store.memory import InMemoryStore
from your_module import YourStore

@pytest.fixture
async def store():
    async with YourStore.create() as s:
        yield s

@pytest.fixture
def reference():
    return InMemoryStore()

async def test_put_and_get(store, reference):
    ns = ("test", "ns")
    for s in (store, reference):
        await s.aput(ns, "k1", {"val": 1})
        item = await s.aget(ns, "k1")
        assert item is not None and item.value == {"val": 1}
```

Inspect the live contract with `import inspect; from langgraph.store.base import BaseStore; print(inspect.getsource(BaseStore))`.

---

## 3. Memory types mapped to our 12-phase data-science team

LangGraph/CoALA taxonomy: **semantic** = facts, **episodic** = experiences, **procedural** = instructions.

| Type | Store pattern | Our use |
|---|---|---|
| Semantic — **profile** | one JSON doc, updated in place | **Dataset profile** per corpus (K12/finance/sports): row count, language mix, dedup rate, template-family regex inventory from P1, PII flags. Small, strict schema, always loaded. |
| Semantic — **collection** | many docs, searched | **Taxonomy priors & label glossary** (P2), **template-family catalogue** (P1), **naming glossary** (P7). Higher recall; needs dedup/invalidations. |
| Episodic | collection of experience records | **Run experiences**: "on the finance corpus, HDBSCAN `min_cluster_size=25` collapsed to 3 clusters; Bisecting-KMeans at K=48 survived". Retrieved as few-shot exemplars when P4/P5 run on a *new* corpus. |
| Procedural | prompt rules in store, or files | **Evolving rubrics**: the annotator guideline that got kappa from 0.82 → 0.91 in P2; the P7 naming style rules; the P8 governance merge criteria. |

The canonical procedural-memory loop (pseudo-code from the LangGraph memory concept page):

```python
def call_model(state: State, store: BaseStore):
    instructions = store.get(("agent_instructions",), key="agent_a")[0]
    prompt = prompt_template.format(instructions=instructions.value["instructions"])
    ...

def update_instructions(state: State, store: BaseStore):
    instructions = store.search(("instructions",))[0]
    prompt = prompt_template.format(
        instructions=instructions.value["instructions"], conversation=state["messages"])
    new_instructions = llm.invoke(prompt)["new_instructions"]
    store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
```

This is exactly how our P2 annotation-guideline refinement loop and P7 naming-style loop should work: rubric lives in the store, referee feedback rewrites it, next run reads the improved version.

**Profile vs collection tradeoff (from the docs, worth internalizing):** profiles risk lossy overwrite and get error-prone as they grow (mitigate with strict decoding or JSON-patch via trustcall); collections give higher downstream recall but shift complexity to *update/delete* and to *search*, and lose the cross-memory relational context a unified profile has. For us: **profile for the dataset card, collection for everything episodic.**

---

## 4. Hot-path vs background memory formation

| | Latency | Availability | Load | Best for |
|---|---|---|---|---|
| Hot path (active/"conscious") | Higher | Immediate | During response | Critical context updates |
| Background ("subconscious"/sleep-time compute) | None | Delayed to next run | Between/after calls | Pattern analysis, summaries, consolidation |

Hot-path tradeoffs per the docs: real-time availability and transparency, but it adds a tool the agent must reason about, adds latency, and forces the agent to multitask between memory-writing and its actual job — which degrades both. Background tradeoffs: no user-facing latency and can synthesize *across* runs, but requires a second agent and a trigger policy (fixed delay with rescheduling, cron, or manual).

**LangMem hot-path tools** (verified signatures):

```python
def create_manage_memory_tool(
    namespace: tuple[str, ...] | str, *,
    instructions: str = "...",
    schema: typing.Type = str,
    actions_permitted: tuple[Literal["create","update","delete"], ...] | None = ("create","update","delete"),
    store: BaseStore | None = None,
    name: str = "manage_memory",
)

def create_search_memory_tool(
    namespace: tuple[str, ...] | str, *,
    instructions: str = _MEMORY_SEARCH_INSTRUCTIONS,
    store: BaseStore | None = None,
    response_format: Literal["content", "content_and_artifact"] = "content",
    name: str = "search_memory",
)
```

`actions_permitted` is the important one for us: **`actions_permitted=("create",)` makes a memory namespace append-only**, which is how we enforce an immutable decision ledger.

Namespaces support `{template}` variables resolved from `configurable` at runtime, and **read/write namespaces can differ** — this is the anti-anchoring primitive:

```python
# Agent A writes only to its own space but reads the whole team space
agent_a_tools = [
    create_manage_memory_tool(namespace=("memories", "team_a", "agent_a")),
    create_search_memory_tool(namespace=("memories", "team_a")),
]
```

**Background formation:**

```python
from langmem import ReflectionExecutor, create_memory_store_manager
from langgraph.func import entrypoint

memory_manager = create_memory_store_manager("anthropic:claude-3-5-sonnet-latest",
                                             namespace=("memories",))
executor = ReflectionExecutor(memory_manager)     # debounces + cancels superseded work

@entrypoint(store=store)
def chat(message: str):
    response = llm.invoke(message)
    to_process = {"messages": [{"role": "user", "content": message}, response]}
    executor.submit(to_process, after_seconds=1800)   # reschedules if new msgs arrive
    return response.content
```

Serverless caveat from the docs: local threads die between invocations — use the remote executor `ReflectionExecutor("my_memory_manager", ("memories",), url="http://localhost:2024")`.

**Schema-driven vs freeform.** `create_memory_manager(model, schemas=[...], instructions=..., enable_inserts=bool)`. `enable_inserts=True` → collection; `enable_inserts=False` + a single Pydantic schema → profile. Episodic schema example verbatim from the conceptual guide:

```python
from pydantic import BaseModel, Field
from langmem import create_memory_manager

class Episode(BaseModel):
    """How to handle a specific situation, including reasoning and what made it work."""
    observation: str = Field(..., description="The situation and relevant context")
    thoughts: str   = Field(..., description="Key considerations and reasoning process")
    action: str     = Field(..., description="What was done in response")
    result: str     = Field(..., description="What happened and why it worked")

manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Episode],
    instructions="Extract examples of successful interactions. Include the context, thought process, and why the approach worked.",
    enable_inserts=True,
)
episodes = manager.invoke({"messages": conversation})   # -> [ExtractedMemory(id=..., content=Episode(...))]
```

Prompt optimizer (procedural memory):

```python
from langmem import create_prompt_optimizer
optimizer = create_prompt_optimizer("anthropic:claude-3-5-sonnet-latest",
                                    kind="metaprompt", config={"max_reflection_steps": 3})
optimized = optimizer.invoke({"trajectories": [(trajectory, {"user_score": 0})], "prompt": prompt})
```

**Dedup / conflict resolution.** The docs are candid: with collections, the model must *delete or update* existing items, and models skew either toward over-inserting or over-updating. Mitigations named: strict decoding, `trustcall` (JSON-patch instead of regenerate), and **evaluation to tune the insert/update balance**. Relevance should not be pure cosine similarity — the docs explicitly say recall should combine similarity with *importance* and *strength* (recency/frequency). **We should implement that scoring ourselves**; the store gives us `updated_at` and `filter=` to build it.

**2026-current background consolidation (deepagents, the pattern I'd actually copy):** deploy a *separate* consolidation agent registered in `langgraph.json`, triggered by cron:

```python
from datetime import datetime, timedelta, timezone
from deepagents import create_deep_agent
from langchain.tools import tool, ToolRuntime
from langgraph_sdk import get_client

sdk_client = get_client(url="<DEPLOYMENT_URL>")

@tool
async def search_recent_conversations(query: str, runtime: ToolRuntime) -> str:
    """Search this user's conversations updated in the last 6 hours."""
    user_id = runtime.server_info.user.identity
    since = datetime.now(timezone.utc) - timedelta(hours=6)
    threads = await sdk_client.threads.search(
        metadata={"user_id": user_id}, updated_after=since.isoformat(), limit=20)
    out = []
    for t in threads:
        h = await sdk_client.threads.get_history(thread_id=t["thread_id"])
        out.append(h["values"]["messages"])
    return str(out)

agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    system_prompt="Review recent conversations and update the memory file. "
                  "Merge new facts, remove outdated information, keep it concise.",
    tools=[search_recent_conversations],
)
```

```python
cron_job = await client.crons.create(
    assistant_id="consolidation_agent",
    schedule="0 */6 * * *",     # UTC always
    input={"messages": [{"role": "user", "content": "Consolidate recent memories."}]},
)
```

**Explicit warning from the docs:** the cron interval must equal the lookback window inside the tool (`0 */6 * * *` ↔ `timedelta(hours=6)`). Run more often → reprocess duplicates; less often → silently drop memories. This is an easy bug to ship.

---

## 5. Context engineering tactics

### The write / select / compress / isolate framework (LangChain, verified)

- **Write** — save outside the window. *Scratchpads* (a tool that writes a file, or a field in the runtime state object) for within-session; *memories* for across-session. Anthropic's multi-agent researcher saves the LeadResearcher's plan to memory precisely because the window truncates at 200K.
- **Select** — pull the right thing back in. From a scratchpad (tool read, or developer chooses which state fields to expose per step); from memories (episodic few-shots / procedural instructions / semantic facts); **tools via RAG over tool descriptions** — the post cites papers showing ~3× improvement in tool-selection accuracy, and `langgraph-bigtool` for semantic search over tool descriptions; knowledge via RAG.
- **Compress** — summarization (recursive/hierarchical; Claude Code auto-compacts past ~95%; Cognition uses a *fine-tuned model* at agent-agent handoff boundaries) and trimming/pruning (heuristics, or trained pruners like Provence).
- **Isolate** — sub-agents with their own windows; sandboxed code execution (HuggingFace CodeAgent keeps token-heavy objects as sandbox variables and returns only selected values); and **the state object itself** — a schema whose non-`messages` fields are invisible to the LLM until needed.

The post's own selection warning is worth heeding: Simon Willison's ChatGPT injected his location from memory into an image request — "unexpected or undesired memory retrieval can make some users feel like the context window no longer belongs to them."

### Anthropic's context engineering guidance

- **Context rot**: as tokens increase, recall accuracy decreases — a performance *gradient*, not a cliff, driven by n² pairwise attention and by training distributions skewed to short sequences. Context is a finite resource with an **attention budget**.
- Goal: *"the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome."* Minimal ≠ short.
- **System prompt altitude**: the Goldilocks zone between brittle hardcoded if-else logic and vague guidance that assumes shared context. Organize into sections (`<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`). Start minimal on the best model, add instructions/examples only against observed failure modes.
- **Tools**: the #1 failure is bloated, overlapping tool sets. *"If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better."*
- **Examples**: curate diverse canonical examples; do **not** stuff a laundry list of edge cases.
- **Just-in-time context**: keep lightweight identifiers (file paths, stored queries, links) and load at runtime via tools. Metadata itself is signal — `test_utils.py` in `tests/` vs `src/core_logic/` implies different purpose; file sizes suggest complexity, timestamps proxy relevance. This enables **progressive disclosure**. Tradeoff: runtime exploration is slower and needs good tools/heuristics or the agent wastes context chasing dead ends.
- **Hybrid is often best**: Claude Code drops `CLAUDE.md` in up front, then uses glob/grep for JIT retrieval — avoiding stale indexes entirely.
- **Three long-horizon techniques**: **compaction** (maximize recall in the compaction prompt first, then tune for precision; "tool result clearing" is the safest lightest-touch form), **structured note-taking** (NOTES.md / to-do lists; Claude playing Pokémon maintains tallies across thousands of steps and resumes from its own notes after context resets), and **sub-agent architectures** (a subagent may burn tens of thousands of tokens and return only a **1,000–2,000 token** distilled summary).
- Selection rule: compaction for conversational back-and-forth; note-taking for iterative work with clear milestones; multi-agent for parallel exploration.

### The 2026-current mechanics (deepagents built-ins)

`create_deep_agent` ships context compression by default:

- **Offloading** — tool call *inputs* or *results* exceeding **20,000 tokens** are written to the filesystem and replaced with a path reference plus a 10-line preview; older large tool calls are truncated to pointers once the session crosses **85% of the model's window**.
- **Summarization** — `SummarizationMiddleware` is in the bare stack. Triggers at **85% of the model's `max_input_tokens`** (from the model profile), keeps **10% of tokens** as recent context, falls back to a 170,000-token trigger / 6 messages kept if no model profile. On any `ContextOverflowError` it immediately falls back to summarization and retries. Dual output: an **in-context structured summary** (session intent, artifacts created, next steps) *and* a **filesystem-preserved text rendering of the original messages** as the canonical record — so detail is recoverable by search. Filter summarization tokens out of streams via `metadata.get("lc_source") == "summarization"`.
- **On-demand compaction tool** — `create_summarization_tool_middleware(model, backend)` gives the agent a `compact_conversation` tool so it can compact *between tasks* instead of waiting for 85%. Both share the same engine/state.

Subagent best practices, verbatim in spirit:

```python
research_subagent = {
    "name": "researcher",
    "description": "Conducts research on a topic",
    "system_prompt": """You are a research assistant.
    IMPORTANT: Return only the essential summary (under 500 words).
    Do NOT include raw search results or detailed tool outputs.""",
    "tools": [web_search],
}
```

Plus: "use the filesystem for large data — subagents write results to files; the main agent reads what it needs."

**Filesystem-as-memory via `CompositeBackend`** — the concrete mechanism:

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    memory=["/memories/AGENTS.md"],
    skills=["/skills/"],
    backend=CompositeBackend(
        default=StateBackend(),                       # thread-scoped scratch
        routes={
            "/memories/": StoreBackend(namespace=lambda rt: (rt.server_info.assistant_id,)),
            "/skills/":   StoreBackend(namespace=lambda rt: (rt.server_info.assistant_id,)),
        },
    ),
)
```

Seeding, verbatim pattern:

```python
from deepagents.backends.utils import create_file_data
store.put(("my-agent",), "/memories/AGENTS.md", create_file_data("""## Response style
- Keep responses concise
- Use code examples where possible
"""))
```

Scoping choices: `(assistant_id,)` = agent-scoped (shared across users), `(user_id,)` = user-scoped, `(org_id,)` = org policy, `(assistant_id, user_id)` = both. **Skills are on-demand procedural memory**: only name+description are read at startup; the full `SKILL.md` is read only when a task matches — this is progressive disclosure implemented as a file convention.

---

## 6. Failure modes and how to avoid them

Breunig's four (adopted verbatim by LangChain and consistent with Anthropic's "context rot"):

| Failure | What happens | Mitigation for our pipeline |
|---|---|---|
| **Context poisoning** | A hallucination or bad tool output enters context and is treated as ground truth on every subsequent turn; compounds because agents build on their own working set. | Never let an LLM-authored number into the metric ledger. **Only the metrics node writes `metrics/*`, from actual computed values.** Make the ledger append-only (`actions_permitted=("create",)`) with provenance (`run_id`, `code_sha`, `seed`) so a poisoned entry is identifiable and revocable rather than silently persistent. |
| **Context distraction** | Context grows so long the model over-focuses on it and neglects training knowledge. | Hard per-phase token budgets; `SummarizationMiddleware(trigger=[("fraction", 0.8), ("messages", 100)])`; fresh thread per phase. |
| **Context confusion** | Superfluous info / too many overlapping tools degrade the response. | Per-phase tool allowlists — the P7 naming agent gets `read_cluster_samples` and nothing else. Anthropic's rule: if a human can't say which tool applies, neither can the agent. |
| **Context clash** | New info conflicts with earlier context. | Conflict-resolution pass in background consolidation; supersede-don't-append semantics for decisions (`{"status": "superseded_by": <id>}`); TTL on ephemeral trial results. |

**Long-horizon degradation** is the umbrella problem: attention budget depletion + compaction loss. Anthropic's warning on compaction is the operational one — *"overly aggressive compaction can result in the loss of subtle but critical context whose importance only becomes apparent later."* Tune the compaction prompt on real traces: **maximize recall first, then improve precision.**

**Security failure mode** (deepagents docs, and directly relevant to a multi-agent team writing to shared memory): if one agent can write memory another agent reads, that is a prompt-injection channel. Mitigations: default to the narrowest scope; make shared/policy memory **read-only** and populate it from application code, not the agent; put an `interrupt()` human-approval gate before writes to sensitive paths. Also: **concurrent writes to the same file cause last-write-wins conflicts** — mitigate by structuring memory as **separate files per topic** to reduce contention, or serialize writes through background consolidation.

---

## 7. Recommended architecture for the 12-phase pipeline

Three tiers, with a strict rule about what may enter context.

### Tier 1 — Artifact filesystem (the real memory; never in context)

```
runs/<run_id>/
  MANIFEST.json            # run_id, seed, code_sha, corpus_hash, phase statuses
  PHASE_LOG.md             # append-only one-line-per-phase ledger  (ALWAYS loaded)
  DECISIONS.md             # decision + rejected alternatives + rationale (ALWAYS loaded)
  p01_audit/{profile.json, template_families.json}
  p03_repr/{embeddings.npy, svd.pkl, alpha_sweep.parquet}
  p04_cluster/{trials.parquet, labels_*.npy}
  p05_k/{stability.parquet, k_triangulation.json}
  p07_naming/{blind_reports/*.md, tree_audit.md}
  p09_metrics/metrics_panel.parquet
```

Rationale: this is Anthropic's just-in-time model. Agents hold **paths**, not payloads. Numeric artifacts are parquet/npy and are *never* serialized into messages — a node reads them with pandas and emits a ≤500-token digest. `PHASE_LOG.md` + `DECISIONS.md` are the `CLAUDE.md`-equivalent: small, always-loaded, hand-curated-by-agent. Everything else is glob/grep/read on demand.

### Tier 2 — `BaseStore` (cross-run long-term memory)

Namespace design (prefix-searchable, so ordering matters):

```python
("qm", "decisions",  corpus, phase)     # semantic collection: decisions + REJECTED alternatives
("qm", "metrics",    corpus)            # metric history across quarterly reruns (P12 drift)
("qm", "episodes",   domain)            # episodic: "what worked on finance-like corpora"
("qm", "procedures", phase)             # procedural: annotation rubric, naming style, merge criteria
("qm", "profile",    corpus)            # semantic profile: dataset card (single doc)
("qm", "glossary",   corpus)            # naming glossary — WRITE-ONLY for P7 agents
```

Index config: `IndexConfig(embed="openai:text-embedding-3-small", dims=1536, fields=["summary","rationale"])`. Deliberately **not** `["$"]` — we do not want raw metric floats embedded. Store big blobs with `index=False`.

TTL: `ttl=None` for decisions/metrics/procedures; `ttl=90*24*3600` for `("qm","trials",...)` scratch.

### Tier 3 — Checkpointer (per-phase threads)

**One thread per phase**, not one per run: `thread_id = f"{run_id}:p{n:02d}"`. Consequences:

- Each phase starts with a clean window (isolate), so P7 physically cannot see P2's label chatter.
- `interrupt()` HITL gates resume cleanly at phase granularity.
- `durability="sync"` on expensive phases; `"async"` elsewhere.
- Phase→phase handoff is a **≤2,000-token structured digest** written to `PHASE_LOG.md` + a `("qm","decisions",...)` store entry — the Cognition "summarize at agent-agent boundaries" pattern.

### Enforcing P7 blind naming (anti-anchoring) — this is a namespace problem, and it's solvable declaratively

The playbook says naming agents must not see existing labels. Implement it as **capability isolation, not prompt instruction** (a prompt instruction is not a control):

```python
naming_agent_tools = [
    create_search_memory_tool(namespace=("qm", "glossary", corpus)),   # read: style only
    create_manage_memory_tool(namespace=("qm", "naming", corpus, agent_id),
                              actions_permitted=("create",)),          # write: own space, append-only
]
# NOT granted: any tool whose namespace touches ("qm","decisions",corpus,"p02") or label artifacts
```

Each of the 5 naming agents is a **subagent with its own context window**, receives only deterministic cluster samples (seeded sample, fixed order), writes to `("qm","naming",corpus,agent_id)`, and the tree-audit agent is the only one with read access across all five. This is the same read/write-namespace-split shown in the LangMem team pattern, plus deepagents' read-only-memory guidance.

### Where each context tactic lands

| Tactic | Where |
|---|---|
| Write | `PHASE_LOG.md` / `DECISIONS.md` scratchpad; store decision ledger |
| Select | JIT artifact reads by path; semantic search over `("qm","episodes",domain)` when starting a new corpus |
| Compress | `SummarizationMiddleware(trigger=[("fraction",0.8),("messages",100)], keep=("messages",20))` per phase; phase-boundary digests |
| Isolate | Thread-per-phase; subagents for P2's 5–9 researchers and P7's 5 namers; private input state schemas; non-`messages` state fields for artifact handles |

### P12 maintenance loop

Quarterly rerun = new `run_id`, same store. Drift sentinel = `store.search(("qm","metrics",corpus), limit=200)`, **sorted client-side on `item.updated_at`** (do not trust backend order), compared against the current panel. Novelty detection writes new template families into `("qm","decisions",corpus,"p01")` with supersede pointers. Consolidation agent on cron, lookback window == cron interval.

---

## 8. Evaluating memory: does it actually help?

**Framework first (LangChain):** before any context engineering, (1) have token-usage tracing per agent step — LangSmith — to find *where* to spend effort, and (2) have a way to test whether a change helps or hurts. Context engineering without an eval harness is guesswork.

**Memory ≠ long context.** The 2026 consensus: long context measures *capacity*, memory measures *continuity*. A model acing needle-in-a-haystack at 1M tokens is not thereby a system that handles cross-session memory. NIAH/RULER/BABILong/InfiniteBench/LongBench test attention over a single fixed input — they are the *substrate*, not the memory eval. Notably RULER shows many "128K-capable" models have effective working length nearer **32K–64K**, and BABILong finds LLMs effectively use only **10–20%** of nominal context. Budget accordingly: do not plan on 200K working context.

**The three real memory benchmarks:**
- **LoCoMo** (2024) — ~300 turns / ~9,000 tokens across up to 35 sessions; QA + event summarization. Headline: even with RAG or long-context models, systems "still substantially lag behind human performance." Limits: modest length by 2026 standards, and **does not score knowledge updates**.
- **LongMemEval** (2024) — 500 curated questions across five abilities: information extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, and **abstention**. `_S` ≈ 115K tokens / ~40 sessions; `_M` ≈ 500 sessions. The `_abs` questions grade whether the system correctly *declines* rather than fabricating — a failure mode most retrieval evals ignore. Paper reports a **~30% accuracy drop** for commercial assistants and long-context LLMs on sustained interaction. GPT-4o as judge. `LongMemEval-V2` (2026) extends to web-agent histories beyond 100M-token context.
- **BEAM** (ICLR 2026) — 100 conversations up to **10M tokens**, 2,000 probing questions, ten capabilities (fact/entity tracking, updating over time, contradiction resolution, temporal order, instructions-vs-preferences, multi-hop, summarization). Its LIGHT framework reports **+3.5% to +12.7%** over the strongest long-context baselines, **with the gap widening as token scale grows**. The practical answer to "can a 10M window let us skip building memory?" is **no**.

**What no public benchmark measures (and what we should measure ourselves)** — this list is the actually actionable part:

1. **Write quality.** Almost every benchmark grades retrieval only. A system that stores everything scores identically to one that stores only what matters — until token budgets and latency are involved.
2. **Forgetting / eviction / consolidation.** No widely adopted public benchmark scores these dynamics directly.
3. **Isolation under concurrent multi-user load.** Essentially untested academically; every production system must handle it. For us the analogue is **namespace leakage** — the P7 anti-anchoring guarantee.
4. **Token economy.** "A system that scores 92 with 7,000 tokens/query is not the same product as one that scores 92 with 70,000 tokens/query. Most leaderboard tables ignore this."
5. **Cross-session continuity at production scale** — the field has no stable definition of "long enough."

**Concrete eval plan for us** — build a small internal harness rather than importing a chat benchmark, since our memory is decisions/metrics, not user preferences:

- **Recall probes**: ~50 questions of the form "what K did we pick for the K12 corpus in Q1 and why was K=32 rejected?" with gold answers from `DECISIONS.md`. Score exact-decision retrieval.
- **Abstention probes** (steal LongMemEval's best idea): questions about decisions that were never made. Grade whether the agent declines. This directly measures **context poisoning resistance**.
- **Update probes**: change a decision in run N+1; verify the agent returns the new one and can name the superseded one. This measures conflict resolution, which no public benchmark covers well.
- **Isolation probes**: assert that a P7 naming agent's transcript contains zero P2 label strings. Automatable, and a hard gate.
- **Token economy**: log tokens-per-phase and p50 latency alongside accuracy; a memory change that improves recall 2% while tripling tokens is a regression.
- **A/B the whole thing**: run the pipeline with memory on/off on a held-out corpus and compare downstream quality (kappa in P2, cluster stability in P5). LangSmith evaluation is the intended harness; LangSmith Datasets are also an explicitly supported alternative to the store for holding few-shot/episodic examples when you want them tied to the eval harness.
- **Audit writes**: LangSmith tracing shows every memory file write as a tool call — use it as the provenance audit log.


---

## Recommendations carried into the design

- Do NOT build the critical path on LangMem — it last shipped 2025-10-27 (v0.0.30) against langgraph 0.6 while langgraph is now 1.2.11; use langgraph core (checkpointer + BaseStore) plus langchain.agents.middleware.SummarizationMiddleware instead, and pin langmem only if you want create_memory_manager/create_prompt_optimizer/ReflectionExecutor.
- Replace langmem.short_term.SummarizationNode with SummarizationMiddleware(model=..., trigger=[('fraction',0.8),('messages',100)], keep=('messages',20)), whose trigger accepts tuple (single), dict (AND), or list (OR) thresholds.
- Use one checkpointer thread per phase (thread_id = f'{run_id}:p{n:02d}') rather than one per run, so each phase starts with a clean context window and HITL interrupts resume at phase granularity.
- Set durability='sync' for expensive/irreversible phases (P2 gold labelling, P3 embedding bake-off, P4 clustering battery) and durability='async' elsewhere, so a crash never replays a 40-minute sweep.
- Keep all numeric artifacts (embeddings, trial parquets, label arrays) on a run-scoped filesystem and pass only paths through context, with small always-loaded PHASE_LOG.md and DECISIONS.md files acting as the CLAUDE.md-equivalent index.
- Enforce P7 blind naming as capability isolation, not prompt instruction: give naming subagents create_search_memory_tool only on ('qm','glossary',corpus) and create_manage_memory_tool with actions_permitted=('create',) on their own ('qm','naming',corpus,agent_id) namespace, and never grant a tool whose namespace reaches P2 labels.
- Make the decision and metric ledgers append-only with provenance (run_id, code_sha, seed) and let only the metrics node write computed numbers, so an LLM hallucination can never become persistent ground truth (context poisoning).
- Never rely on store.search ordering — PostgresStore orders by updated_at DESC while InMemoryStore returns insertion order — always sort client-side on item.updated_at, and set limit above expected max since results past limit are silently truncated with no overflow signal.
- Configure the store with IndexConfig(embed=..., dims=1536, fields=['summary','rationale']) rather than fields=['$'], and write large blobs with index=False so raw metric floats are stored but never embedded.
- Use TTL to separate permanence tiers: ttl=None for decisions/metrics/procedures, and a finite ttl for ephemeral per-trial scratch memories, via TTLConfig(default_ttl=..., refresh_on_read=..., omit_expired=True).
- Adopt the deepagents CompositeBackend pattern (StateBackend default + StoreBackend routed at /memories/ and /skills/) as the reference architecture for filesystem-as-memory, since it is the actively-maintained 2026 LangChain answer and mirrors the files+store design we want.
- If you add background consolidation, deploy it as a separate cron-triggered agent and keep the cron interval exactly equal to the tool's lookback window (e.g. '0 */6 * * *' with timedelta(hours=6)), or you will reprocess duplicates or silently drop memories.
- Structure long-term memory as separate files/keys per topic rather than one large file, because concurrent writes to the same key are last-write-wins and agent-scoped memory has real contention.
- Import get_store from langgraph.config (not langgraph.utils.config, which the source marks as a to-be-removed backcompat shim) and prefer Runtime injection with context_schema=Context plus graph.invoke(..., context=Context(...)) over config['configurable'] plumbing.
- Instruct every subagent to return a distilled report under ~500 words / 1,000-2,000 tokens and to write bulk output to files, since the whole point of subagent isolation is that the parent never sees the exploration.
- Budget for an effective working context of roughly 32K-64K tokens rather than the nominal window, per RULER and BABILong findings that models effectively use only 10-20% of nominal context.
- Build an internal memory eval harness with four probe types — recall, abstention (questions about decisions never made), update/supersede, and namespace-isolation assertions — since no public benchmark measures write quality, forgetting, or per-tenant isolation.
- Track tokens-per-phase and p50 latency alongside accuracy in every memory experiment, and treat a 2% recall gain that triples token cost as a regression.
- Plan on no official SqliteStore existing: langgraph-checkpoint-sqlite provides SqliteSaver/AsyncSqliteSaver (checkpointers) only, so for local dev pair SqliteSaver with InMemoryStore or a custom BaseStore subclass tested against InMemoryStore as the reference oracle.
- Tune any compaction prompt on real pipeline traces by maximizing recall first and only then improving precision, and preserve a full text rendering of the pre-compaction messages to the filesystem so detail stays recoverable.

## Unverified or version-dependent

- I could not fetch docs.langchain.com or blog.langchain.com via WebFetch (domain-safety block) and worked around it with curl against the Mintlify .md endpoints; content is authentic but I did not see rendered tabs/accordions, so a few code variants may exist that I did not surface.
- langmem 0.0.30 declares langgraph>=0.6.0,<2 so it should import under langgraph 1.2.11, but I did not actually install and run it — runtime compatibility with langchain-core 1.5.5 (especially message-class relocations) is unverified and should be smoke-tested before adopting.
- LangMem's docs are written against langgraph.prebuilt.create_react_agent; I did not verify whether that symbol still exists in langgraph 1.2.11 or whether langmem's tools work unchanged with langchain.agents.create_agent.
- trustcall (langmem's dedup/JSON-patch dependency) last released 2025-04-14; I did not verify its behavior under current langchain-core.
- SummarizationMiddleware's signature was read from langchain master on GitHub, not from the 1.3.15 release tag — the released signature could differ slightly. Also, the docs example passes trigger=('tokens', 4000) while the source default is trigger=None; the exact default-trigger behavior when omitted is unverified.
- DeltaChannel is documented as requiring langgraph>=1.2 and is explicitly in beta with a possibly-changing API; I did not read its source or verify the storage-vs-latency tradeoff numbers.
- Deepagents' compression constants (20,000-token offload threshold, 85% summarization trigger, 10% kept, 170,000-token fallback, 6-message fallback) come from the docs page, not from reading deepagents source, and may drift across its fast release cadence (0.7.6 shipped 2026-08-13).
- Several deepagents APIs shown (CompositeBackend/StoreBackend constructor forms, rt.server_info.assistant_id, rt.server_info.user.identity) appear in two slightly different shapes across the docs — with and without a leading rt positional arg — and the docs note rt.server_info requires deepagents>=0.5.0; the exact current constructor signature should be confirmed against installed source.
- Benchmark figures (LongMemEval ~30% accuracy drop, BEAM +3.5-12.7%, RULER effective 32K-64K, BABILong 10-20% context utilization) are quoted from a vendor blog post (mem0, Aug 2026) rather than read from the primary papers, and mem0 has a commercial interest in the memory-beats-long-context conclusion.
- I did not fetch the LongMemEval-V2, BEAM, or LoCoMo papers directly, so category counts and dataset sizes are second-hand.
- Whether an unofficial/community SqliteStore exists outside the five listed backends (InMemory, Postgres, Redis, MongoDB, Upstash) was not investigated beyond confirming langgraph-checkpoint-sqlite ships only checkpointers.
- The recommended thread-per-phase, three-tier architecture is my synthesis for this pipeline, not a pattern documented by LangChain; the individual primitives are verified but the composition is untested.
- Model identifiers appearing in fetched docs (gpt-5.5, gpt-5.4-mini, gemini-3.6-flash, claude-sonnet-4-6, claude-haiku-4-5-20251001) are reproduced as-written from the docs and were not independently verified as currently available.

## Sources

- https://docs.langchain.com/oss/python/langgraph/memory.md
- https://docs.langchain.com/oss/python/langgraph/add-memory.md
- https://docs.langchain.com/oss/python/langgraph/persistence.md
- https://docs.langchain.com/oss/python/langgraph/stores.md
- https://docs.langchain.com/oss/python/langgraph/checkpointers.md
- https://docs.langchain.com/oss/python/langchain/short-term-memory.md
- https://docs.langchain.com/oss/python/langchain/long-term-memory.md
- https://docs.langchain.com/oss/python/deepagents/memory.md
- https://docs.langchain.com/oss/python/deepagents/context-engineering.md
- https://docs.langchain.com/oss/deepagents/code/memory-and-skills.md
- https://docs.langchain.com/oss/python/integrations/long-term-memory/index.md
- https://docs.langchain.com/llms.txt
- https://langchain-ai.github.io/langmem/
- https://langchain-ai.github.io/langmem/concepts/conceptual_guide/
- https://langchain-ai.github.io/langmem/hot_path_quickstart/
- https://langchain-ai.github.io/langmem/background_quickstart/
- https://github.com/langchain-ai/langmem
- https://raw.githubusercontent.com/langchain-ai/langmem/main/README.md
- https://raw.githubusercontent.com/langchain-ai/langmem/main/pyproject.toml
- https://raw.githubusercontent.com/langchain-ai/langmem/main/src/langmem/__init__.py
- https://raw.githubusercontent.com/langchain-ai/langmem/main/src/langmem/short_term/summarization.py
- https://raw.githubusercontent.com/langchain-ai/langmem/main/src/langmem/knowledge/tools.py
- https://raw.githubusercontent.com/langchain-ai/langmem/main/docs/docs/concepts/conceptual_guide.md
- https://raw.githubusercontent.com/langchain-ai/langmem/main/docs/docs/guides/memory_tools.md
- https://raw.githubusercontent.com/langchain-ai/langmem/main/docs/docs/guides/delayed_processing.md
- https://raw.githubusercontent.com/langchain-ai/langmem/main/docs/docs/guides/dynamically_configure_namespaces.md
- https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain_v1/langchain/agents/middleware/summarization.py
- https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint/langgraph/store/base/__init__.py
- https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/config.py
- https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/utils/config.py
- https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/runtime.py
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://www.langchain.com/blog/context-engineering-for-agents
- https://mem0.ai/blog/ai-memory-benchmarks-in-2026
- https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html
- https://pypi.org/pypi/langgraph/json
- https://pypi.org/pypi/langmem/json
- https://pypi.org/pypi/langchain/json
- https://pypi.org/pypi/deepagents/json
- https://pypi.org/pypi/langgraph-checkpoint-sqlite/json