# Multi-Agent Orchestration Patterns

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

# Multi-Agent Orchestration in LangGraph + Literature — Verified Findings (as of 2026-08-17)

## 0. CRITICAL VERSION/API REALITY CHECK (verified live, not from memory)

I installed the packages in a scratch venv and introspected them, and pulled the live docs (`docs.langchain.com` serves `.md` for every page — append `.md` to any doc URL; `https://docs.langchain.com/oss/python/<section>/llms.txt` is the per-section index).

**PyPI versions (queried 2026-08-17):**

| Package | Latest | Released | requires_python |
|---|---|---|---|
| `langchain` | **1.3.15** | 2026-08-11 | >=3.10,<4 |
| `langgraph` | **1.2.11** | 2026-08-11 | >=3.10 |
| `deepagents` | **0.7.6** | 2026-08-13 | >=3.11,<4 |
| `langchain-anthropic` | **1.5.6** | 2026-08-13 | >=3.10 |
| `langgraph-supervisor` | 0.0.31 | **2025-11-19** (stale) | >=3.10 |
| `langgraph-swarm` | 0.1.0 | **2025-12-04** (stale) | >=3.10 |

**Things that changed and will break code written from memory:**

1. **The URL `docs.langchain.com/oss/python/langgraph/multi-agent` is 404.** Multi-agent docs moved to `/oss/python/langchain/multi-agent/{index,subagents,handoffs,router,skills,custom-workflow}`.
2. **The taxonomy changed.** The old "network / supervisor / supervisor-as-tool-calling / hierarchical / custom / swarm" list is gone. The current official taxonomy is: **Subagents (supervisor), Handoffs, Skills, Router, Custom workflow**. "Swarm" and "network" are no longer named patterns in the docs.
3. **`langgraph-supervisor` is officially deprecated.** Its README header states: *"We now recommend using the supervisor pattern directly via tools rather than this library for most use cases."* There is a formal migration guide at `/oss/python/migrate/langgraph-supervisor.md`: *"The `langgraph-supervisor` package is no longer actively maintained."* **Do not build on `create_supervisor`.**
4. **`create_react_agent` is deprecated.** Verified empirically — importing and calling it emits `LangGraphDeprecatedSinceV10: create_react_agent has been moved to 'langchain.agents'. Please update your import to 'from langchain.agents import create_agent'. Deprecated in LangGraph V1.0 to be removed in V2.0.` It still exists in langgraph 1.2.11 and still accepts `pre_model_hook`/`post_model_hook`/`prompt`/`version`, but those are the *old* API.
5. **`pre_model_hook` / `post_model_hook` do NOT exist on `create_agent`.** They were replaced wholesale by **middleware**. `state_modifier` is long gone (it was already replaced by `prompt` in v0, then `system_prompt` in v1).
6. **`InjectedState` / `InjectedToolCallId` / `InjectedStore` / `get_runtime()` are legacy.** The current single interface is **`ToolRuntime`**. Docs: *"Older examples used `InjectedState`, `InjectedStore`, `get_runtime()`, or `InjectedToolCallId`. Use `ToolRuntime` instead for one explicit interface to state, context, store, and execution metadata."* `InjectedToolCallId` still appears in some current doc examples (subagent outputs), so both work, but `ToolRuntime` is what to write.
7. **Deprecated symbols (LangGraph v1 migration table):** `create_react_agent`→`langchain.agents.create_agent`; `AgentState`/`AgentStatePydantic`/`AgentStateWithStructuredResponse`→`langchain.agents.AgentState` (no more Pydantic state); `HumanInterruptConfig`/`ActionRequest`→`InterruptOnConfig`; `HumanInterrupt`→`HITLRequest`; `ValidationNode`→ built into `create_agent`; `MessageGraph`→`StateGraph`.

---

## 1. `create_agent` — EXACT current signature (langchain 1.3.15, verified by `inspect.signature`)

```python
from langchain.agents import create_agent

create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]
```

Source: `libs/langchain_v1/langchain/agents/factory.py#L836`.

Notes that matter for us:
- `name=` is *"automatically used when adding the agent graph to another graph as a subgraph node — particularly useful for building multi-agent systems."*
- `state_schema` must extend `langchain.agents.AgentState` (a TypedDict). Docs recommend adding state via middleware `state_schema` instead, to keep extensions scoped.
- `cache: BaseCache` exists — useful for our deterministic re-runs of P7 naming across quarters.

### Structured output (`response_format`)

```python
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None          # `strict` requires langchain>=1.2

class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
```

Passing a bare Pydantic model auto-selects `ProviderStrategy` when the provider supports native structured output (OpenAI, Anthropic, Gemini, xAI), else falls back to `ToolStrategy`. **Raw JSON-Schema dicts are NOT auto-detected — they must be wrapped explicitly.** `ToolStrategy` also supports `Union[A, B]` schemas (model picks). This is exactly what we want for P7 naming agents returning a fixed `ClusterName` schema and for P2 annotators returning `Label`.

### Middleware — the replacement for pre/post hooks

Hooks (from `/oss/python/langchain/middleware/custom.md`):

| Hook | When |
|---|---|
| `before_agent` | Before agent starts (once per invocation) |
| `before_model` | Before each model call |
| `after_model` | After each model response |
| `after_agent` | After agent completes (once per invocation) |
| `wrap_model_call` | Around each model call (0/1/N handler calls → short-circuit, normal, retry) |
| `wrap_tool_call` | Around each tool call |

Execution order (documented): `before_*` first→last; `after_*` last→first; `wrap_*` nested (first middleware is outermost). Jump targets from a hook: return `{"jump_to": "end" | "tools" | "model"}`, and declare it with `@hook_config(can_jump_to=["end"])`.

Exported names verified from `langchain.agents.middleware` in 1.3.15:
`AgentMiddleware, AgentState, ClearToolUsesEdit, ContextEditingMiddleware, ExtendedModelResponse, FilesystemFileSearchMiddleware, HumanInTheLoopMiddleware, InterruptOnConfig, LLMToolEmulator, LLMToolSelectorMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware, ModelRequest, ModelResponse, ModelRetryMiddleware, PIIMiddleware, ProviderToolSearchMiddleware, ShellToolMiddleware, SummarizationMiddleware, TodoListMiddleware, ToolCallLimitMiddleware, ToolErrorMiddleware, ToolRetryMiddleware` + decorators `before_agent, before_model, after_model, after_agent, wrap_model_call, wrap_tool_call, dynamic_prompt, hook_config, model_call_limit, tool_call_limit, model_retry, tool_retry, summarization, context_editing, human_in_the_loop, pii, todo, shell_tool, file_search, tool_emulator, tool_selection, provider_tool_search`.

Verified `__init__` signatures:
```python
ModelCallLimitMiddleware(*, thread_limit: int|None = None, run_limit: int|None = None,
                         exit_behavior: Literal['end','error'] = 'end')
ToolCallLimitMiddleware(tool_name=None, thread_limit=None, run_limit=None,
                        exit_behavior: Literal['continue','error','end'] = 'continue')
ContextEditingMiddleware(*, edits: Iterable[ContextEdit]|None = None,
                         token_count_method: Literal['approximate','model'] = 'approximate')
ClearToolUsesEdit(trigger=100000, clear_at_least=0, keep=3, clear_tool_inputs=False,
                  exclude_tools=(), placeholder='[cleared]')
SummarizationMiddleware(model, *, trigger=None, keep=('messages', 20),
                        token_counter=count_tokens_approximately,
                        summary_prompt=<long default>, trim_tokens_to_summarize=4000)
```
`SummarizationMiddleware.trigger` accepts `('fraction', 0.8) | ('tokens', N) | ('messages', N)` or a list of such clauses.

`ModelCallLimitMiddleware`/`ToolCallLimitMiddleware` are our hard budget guards for P2 (annotator loops) and P7 (naming agents) — set `run_limit` so a wedged agent cannot burn the phase budget.

State updates from wrap hooks use `ExtendedModelResponse(model_response=..., command=Command(update={...}))`. Composition rule: commands apply inner→outer, **outer wins** on non-reducer keys; message updates are additive; commands from discarded retry attempts are dropped.

---

## 2. The five current architectures, with trade-offs (from `/multi-agent/index.md`)

| Pattern | Mechanism | Distributed dev | Parallel | Multi-hop | Direct user interaction |
|---|---|:--:|:--:|:--:|:--:|
| **Subagents** (supervisor) | main agent calls subagents *as tools*; subagents stateless, isolated context | ★★★★★ | ★★★★★ | ★★★★★ | ★ |
| **Handoffs** | tool updates a state var (`active_agent`/`current_step`) → routing/config change | – | – | ★★★★★ | ★★★★★ |
| **Skills** | one agent loads specialized prompts on demand (progressive disclosure) | ★★★★★ | ★★★ | ★★★★★ | ★★★★★ |
| **Router** | classification step fans out to specialists, then synthesizes | ★★★ | ★★★★★ | – | ★★★ |
| **Custom workflow** | bespoke `StateGraph`; mix deterministic + agentic nodes; embed other patterns as nodes | — | — | — | — |

**Published cost benchmarks** (from the same page — useful for budgeting our 12 phases):

- *One-shot* ("buy coffee"): Subagents 4 model calls; Handoffs/Skills/Router 3 each.
- *Repeat request* (same conv.): Subagents 4+4=8; Handoffs 3+2=5; Skills 3+2=5; Router 3+3=6. "Stateful patterns (Handoffs, Skills) save 40–50% of calls on repeat requests."
- *Multi-domain* (3 domains, ~2K tokens of docs each): Subagents 5 calls/~9K tokens; **Router 5 calls/~9K**; Skills 3 calls/**~15K** tokens; Handoffs **7+ calls/~14K+** (sequential, cannot parallelize).
- Verbatim conclusion: *"Subagents processes 67% fewer tokens overall due to context isolation."* And: *"Handoffs is inefficient here—it must execute sequentially and can't leverage parallel tool calling."*

**Mapping to our 12-phase playbook:**
- P0/P1/P3/P4/P5/P9/P10 are deterministic compute → **custom workflow** nodes (plain Python, no LLM), with LLM only where judgment is needed.
- P2 taxonomy research (5–9 agents) and P7 blind naming (5 agents + 1 auditor) → **Router/Send fan-out**, NOT subagents-as-tools, because we need *guaranteed* context isolation and deterministic sharding, not LLM-chosen delegation.
- P2 annotation (2 annotators + referee) and P6 refinement → **debate/judge** custom workflow.
- Overall phase driver → **custom workflow** `StateGraph` with `interrupt()` HITL gates, not a supervisor agent. Deterministic phase ordering is a hard requirement; don't let an LLM decide which phase runs.

---

## 3. Handoffs — exact implementation

### 3a. `Command` primitive (`/oss/python/langgraph/graph-api.md`)

`Command` takes four params: `update` (state updates), `goto` (navigate), `graph` (target parent graph), `resume` (post-interrupt value).

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(update={"foo": "bar"}, goto="my_other_node")
```

**Gotchas documented:**
- You *must* annotate the return type with the destinations (`Command[Literal[...]]`) or graph rendering/routing metadata breaks.
- *"`Command` only adds dynamic edges—static edges defined with `add_edge` still execute."* If `node_a` returns `Command(goto="x")` **and** you have `add_edge("node_a","node_b")`, **both** run. Use one or the other.
- `Command(resume=...)` is the **only** `Command` valid as input to `invoke()/stream()`. `Command(update=...)` as input resumes from the latest checkpoint and will look "stuck". To continue a thread, pass a plain dict.

### 3b. `Command.PARENT` from a subgraph / tool

```python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(update={"foo": "bar"}, goto="other_subgraph", graph=Command.PARENT)
```

**Hard requirement:** *"When you send updates from a subgraph node to a parent graph node for a key that's shared by both parent and subgraph state schemas, you MUST define a reducer for the key you're updating in the parent graph state."*

### 3c. Handoff tool — current (ToolRuntime) form

```python
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """Transfer to the sales agent."""
    last_ai_message = next(
        m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferred to sales agent",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai_message, transfer_message],   # pair, not full history
        },
        graph=Command.PARENT,
    )
```

**Why the AIMessage+ToolMessage pair is mandatory:** *"LLMs expect tool calls to be paired with their responses... Without this pairing, the receiving agent will see an incomplete conversation and may produce errors."* And: *"Why not pass all subagent messages? The receiving agent may become confused by irrelevant internal reasoning, and token costs increase unnecessarily... consider summarizing the subagent's work in the ToolMessage content instead."*

### 3d. Legacy handoff tool (still in `langgraph-swarm` / `langgraph-supervisor` READMEs) — for reference only

```python
from typing import Annotated
from langchain.tools import tool, BaseTool, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

def create_custom_handoff_tool(*, agent_name: str, name: str|None, description: str|None) -> BaseTool:
    @tool(name, description=description)
    def handoff_to_agent(
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all relevant context."],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(content=f"Successfully transferred to {agent_name}",
                                   name=name, tool_call_id=tool_call_id)
        return Command(goto=agent_name, graph=Command.PARENT,
                       update={"messages": state["messages"] + [tool_message],
                               "active_agent": agent_name,
                               "task_description": task_description})
    return handoff_to_agent
```
Two documented requirements when returning `Command` from tools: (1) a tool node that handles `Command`-returning tools (`ToolNode`/`create_agent`'s tool node), (2) both the parent graph and the destination agent must have the updated keys in their state schemas.

### 3e. Single-agent-with-middleware handoffs (recommended over subgraphs)

Docs: *"Use single agent with middleware for most handoffs use cases—it's simpler. Only use multiple agent subgraphs when you need bespoke agent implementations."*

```python
@wrap_model_call
def apply_step_config(request: ModelRequest, handler) -> ModelResponse:
    step = request.state.get("current_step", "triage")
    config = CONFIGS[step]
    request = request.override(
        system_prompt=config["prompt"].format(**request.state),
        tools=config["tools"],
    )
    return handler(request)
```
`ModelRequest.override(system_prompt=..., tools=...)` is the API for per-call prompt/tool swapping. This is a very clean fit for our P6 iterative refinement (merge → split → reassign phases) inside one agent.

---

## 4. Subagents (supervisor) pattern — the replacement for `create_supervisor`

### Minimal tool-wrapped subagent
```python
from langchain.agents import create_agent
from langchain.tools import tool

research_agent = create_agent(model=model, tools=[web_search],
                              system_prompt="You are a research expert.")

@tool("research_expert", description="Research expert for current events and web lookups.")
def call_research_agent(query: str) -> str:
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

supervisor = create_agent(model=model, tools=[call_research_agent, call_math_agent],
                          system_prompt="Route research questions to research_expert and math to math_expert.",
                          checkpointer=InMemorySaver())
```

### Single dispatch `task` tool (Claude-Code-style; better for N teams / dynamic registry)
```python
SUBAGENTS = {"research": research_agent, "writer": writer_agent}

@tool
def task(agent_name: str, description: str) -> str:
    """Launch an ephemeral subagent for a task.
    Available agents:
    - research: Research and fact-finding
    - writer: Content creation and editing
    """
    agent = SUBAGENTS[agent_name]
    result = agent.invoke({"messages": [{"role": "user", "content": description}]})
    return result["messages"][-1].content
```
Agent-discovery options and when to use each (documented): **system-prompt enumeration** (<10 static agents), **enum constraint** on `agent_name` (<10, type-safe), **tool-based discovery** (`list_agents`) for >10 or dynamic registries.

### Subagent input/output context engineering
```python
# INPUT: pull extra state into the subagent
@tool("subagent1_name", description="...")
def call_subagent1(query: str, runtime: ToolRuntime[None, CustomState]):
    subagent_input = some_logic(query, runtime.state["messages"])
    result = subagent1.invoke({"messages": subagent_input,
                               "example_state_key": runtime.state["example_state_key"]})
    return result["messages"][-1].content

# OUTPUT: pass state back to the supervisor alongside the text
@tool("subagent1_name", description="...")
def call_subagent1(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    result = subagent1.invoke({"messages": [{"role": "user", "content": query}]})
    return Command(update={
        "example_state_key": result["example_state_key"],
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)],
    })
```
Documented failure mode: *"A common failure mode is that the sub-agent performs tool calls or reasoning but doesn't include results in its final message—remind it that the supervisor only sees the final output."*

### Checkpointing rules for nested agents (critical for our HITL gates)
- Default: subagents use the **inherited checkpointer** (per-invocation persistence) — fresh state each call, interrupt-capable, parallel-safe. `checkpointer=True` on the subagent switches to "continuations" mode (its own persistent history).
- **Rule for `interrupt()` propagation through nested `create_agent` layers:** (1) *compile only the outermost graph with a checkpointer*; (2) always pass `thread_id` in `configurable`. Then `interrupt()` in a deep tool bubbles up as `__interrupt__` on the outer graph and `Command(resume=...)` works.
- **Known limitation:** *"Because subagents are called inside tool functions, LangGraph cannot statically discover them. `get_state` with `subgraphs` will not return subagent state."* If you need to inspect nested state during an interrupt, invoke the subagent from a **node function in a custom graph** instead. → For our HITL gates (P2 gold labels, P5 K choice, P7 naming approval, P8 governance merges) we should use **explicit graph nodes**, not tool-wrapped subagents.

### Built-in SubAgentMiddleware (deepagents ≥0.6.5)
```python
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

backend = StateBackend()
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(backend=backend, subagents=[{
            "name": "researcher",
            "description": "Searches and returns a structured summary.",
            "system_prompt": "Use the search tool to research the question and summarize key points.",
            "tools": [search],
            "model": "anthropic:claude-sonnet-4-6",
            "middleware": [],
        }]),
    ],
)
```
`SubAgent` dict fields: `name, description, system_prompt` (required); `tools, model, middleware, interrupt_on, skills, response_format, permissions` (optional). Inheritance semantics: `tools`/`model`/`interrupt_on`/`permissions` inherit from the main agent unless overridden; `system_prompt`, `middleware`, `skills` do **not** inherit. `response_format` makes the parent receive JSON instead of free text — directly useful for our naming agents. `CompiledSubAgent(name=, description=, runnable=<compiled graph>)` lets you plug an arbitrary LangGraph graph in as a subagent. A `general-purpose` subagent is auto-added *"primarily for context isolation."*

`RubricMiddleware(model=..., max_iterations=3)` (deepagents ≥0.6.5, **beta**) is a packaged evaluator-optimizer: declare what "done" looks like as a rubric; the agent self-evaluates and iterates until satisfied or the cap is hit. Good candidate for P7 tree-audit and P11 report quality.

---

## 5. Parallel fan-out of N identical workers — the `Send` API (and **empirically verified isolation**)

This is the single most important section for our anti-anchoring blind-review protocol, and I verified the semantics by running real code against `langgraph==1.2.11`.

### Canonical Send / map-reduce
```python
from langgraph.types import Send

def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
```
`Send(node_name, state)` — "the number of objects may be unknown ahead of time... the input State to the downstream Node should be different (one for each generated object)."

### ✅ VERIFIED: Send payload is the worker's ENTIRE input — parent state does NOT leak

I ran this:
```python
class Overall(TypedDict):
    secret_labels: list[str]           # must NOT reach workers
    shards: list[str]
    results: Annotated[list, operator.add]

def worker(state):
    return {"results": [(state.get("shard"), sorted(state.keys()), state.get("secret_labels"))]}

def fanout(state): return [Send("worker", {"shard": s}) for s in state["shards"]]
```
Output:
```
WORKER SAW: ('A', ['shard'], None)
WORKER SAW: ('B', ['shard'], None)
WORKER SAW: ('C', ['shard'], None)
```
**The worker's `state.keys()` is exactly `['shard']`.** `secret_labels` is `None`. This is a *hard* isolation guarantee at the framework level — it does not depend on prompt discipline. For P7 blind naming this means: put existing labels, prior names, and any taxonomy strings in the parent state ONLY, and pass each naming agent a `Send` payload containing only `{"cluster_id", "sample_queries", "centroid_terms", "seed"}`. There is no path by which the agent can see the anchor.

### ✅ VERIFIED: aggregation order is deterministic (Send order, not completion order)
With randomized per-worker sleeps (0–50ms) over 6 workers, 5 consecutive runs all produced `['s0','s1','s2','s3','s4','s5']`. Reducer writes are applied in task-creation order, not wall-clock finish order. **This is essential for our P0 reproducibility requirement** — an `operator.add` accumulator over a Send fan-out is deterministic given deterministic worker outputs.

### Fan-in: use `defer=True`
```python
builder.add_node("aggregate", aggregate_fn, defer=True)
```
*"Deferring node execution is useful when you want to delay the execution of a node until all other pending tasks are completed. This is particularly relevant when branches have different lengths, which is common in workflows like map-reduce flows."* Verified working with Send fan-out in my test.

### Concurrency + retry controls
```python
graph.invoke(inputs, {"configurable": {"max_concurrency": 10}})
```
Plus `RetryPolicy` per node: *"Only failing branches are retried, so you needn't worry about performing redundant work."*
Recursion limit: default **1000 super-steps** since v1.0.6; set via top-level `config={"recursion_limit": N}` (NOT inside `configurable`). `RemainingSteps` managed value + `config["metadata"]["langgraph_step"]` let you degrade gracefully before hitting the wall.

### Ready-to-adapt: 5 BLIND naming agents + 1 tree auditor (P7)

```python
import operator, json
from typing import Annotated, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.agents import create_agent

# ---------- schemas ----------
class ClusterName(BaseModel):
    """A blind-generated name for one cluster."""
    name: str = Field(description="2-6 word intent label, no vendor/brand jargon")
    definition: str = Field(description="One sentence: what belongs in this cluster")
    boundary_note: str = Field(description="What is explicitly EXCLUDED")
    confidence: float = Field(ge=0.0, le=1.0)

class NamingState(TypedDict):
    clusters: list[dict]                 # {"cluster_id", "samples", "top_terms"}
    existing_labels: dict                # ANCHOR — never enters a Send payload
    proposals: Annotated[list, operator.add]
    consensus: dict

# ---------- one blind namer, N replicas ----------
NAMER_PROMPT = (
    "You are naming a cluster of user search queries. You have NEVER seen any existing "
    "taxonomy, label, or name for this data. Do not guess at or reproduce any external "
    "labeling scheme. Base the name ONLY on the queries and terms shown.\n"
    "Return a name that is: (a) 2-6 words, (b) an INTENT, not a topic, "
    "(c) mutually exclusive from a sibling that differs only in surface wording."
)

def _make_namer(agent_id: int):
    return create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[],                              # blind: no retrieval, no lookup
        system_prompt=NAMER_PROMPT,
        response_format=ClusterName,           # ProviderStrategy auto-selected
        name=f"namer_{agent_id}",
    )

NAMERS = {i: _make_namer(i) for i in range(5)}

def name_worker(state: dict) -> dict:
    """Worker sees ONLY: agent_id, cluster_id, samples, top_terms, seed."""
    agent = NAMERS[state["agent_id"]]
    payload = json.dumps(
        {"queries": state["samples"], "distinctive_terms": state["top_terms"]},
        ensure_ascii=False, sort_keys=True,
    )
    out = agent.invoke({"messages": [{"role": "user", "content": payload}]})
    proposal = out["structured_response"]
    return {"proposals": [{
        "cluster_id": state["cluster_id"],
        "agent_id": state["agent_id"],
        "name": proposal.name,
        "definition": proposal.definition,
        "boundary_note": proposal.boundary_note,
        "confidence": proposal.confidence,
    }]}

def fan_out_naming(state: NamingState):
    # deterministic ordering => deterministic aggregation
    return [
        Send("name_worker", {
            "agent_id": a,
            "cluster_id": c["cluster_id"],
            "samples": c["samples"],          # already deterministically sampled w/ seed
            "top_terms": c["top_terms"],
        })
        for c in sorted(state["clusters"], key=lambda x: x["cluster_id"])
        for a in range(5)
    ]

def consensus(state: NamingState) -> dict:
    by_cluster: dict[Any, list] = {}
    for p in state["proposals"]:
        by_cluster.setdefault(p["cluster_id"], []).append(p)
    return {"consensus": {cid: pick_consensus_name(props)   # your voting / embedding-agreement fn
                          for cid, props in sorted(by_cluster.items())}}

g = (
    StateGraph(NamingState)
    .add_node("name_worker", name_worker)
    .add_node("consensus", consensus, defer=True)     # waits for ALL 5*K workers
    .add_conditional_edges(START, fan_out_naming, ["name_worker"])
    .add_edge("name_worker", "consensus")
    .add_edge("consensus", END)
    .compile()
)
result = g.invoke(
    {"clusters": clusters, "existing_labels": legacy_labels, "proposals": [], "consensus": {}},
    {"configurable": {"max_concurrency": 8}},
)
```

Three isolation properties this buys us, all framework-enforced:
1. `existing_labels` lives in `NamingState` and is *structurally unreachable* from `name_worker`.
2. Each of the 5 agents is a separately-constructed `create_agent` with no checkpointer → no shared thread, no shared message history. Agent *i* cannot see agent *j*'s proposal.
3. Aggregation into `proposals` is order-deterministic, so consensus is reproducible given a fixed seed.

The **tree-audit agent** runs *after* `consensus` as a separate node (it is *allowed* to see everything — sibling names, hierarchy, and the existing taxonomy), so it is a distinct node downstream, never a Send worker.

---

## 6. Debate / adversarial / judge-panel patterns in graph form

### The literature pattern (Anthropic's `building-effective-agents` taxonomy)
- **Parallelization** has two variants: **Sectioning** (independent subtasks in parallel) and **Voting** (*"Running the same task multiple times to get diverse outputs"*). Our 5 blind namers are **Voting**; our 5–9 taxonomy researchers are **Sectioning**. Documented example: *"Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives."* → directly applicable to our P2 annotator/referee design.
- **Evaluator-optimizer**: *"particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. Two signs of good fit: LLM responses can be demonstrably improved when a human articulates their feedback; and the LLM can provide such feedback."*

### LangGraph evaluator-optimizer (verbatim shape from `workflows-agents.md`)
```python
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Decide if the joke is funny or not.")
    feedback: str = Field(description="If not funny, provide feedback on how to improve it.")

evaluator = llm.with_structured_output(Feedback)

def route_joke(state: State):
    return "Accepted" if state["funny_or_not"] == "funny" else "Rejected + Feedback"

optimizer_builder.add_conditional_edges(
    "llm_call_evaluator", route_joke,
    {"Accepted": END, "Rejected + Feedback": "llm_call_generator"},
)
```
**This loop as written has no round cap.** Add one — either a counter in state, or `ModelCallLimitMiddleware(run_limit=...)`, or rely on `recursion_limit` + `RemainingSteps`.

### TradingAgents-style bull-vs-bear debate — actual production source (`TauricResearch/TradingAgents`, fetched from `main`)

**Debate state** (`tradingagents/agents/utils/agent_states.py`):
```python
class InvestDebateState(TypedDict):
    bull_history: Annotated[str, "Bullish Conversation history"]
    bear_history: Annotated[str, "Bearish Conversation history"]
    history: Annotated[str, "Conversation history"]
    current_response: Annotated[str, "Latest response"]
    judge_decision: Annotated[str, "Final judge decision"]
    count: Annotated[int, "Length of the current conversation"]

class RiskDebateState(TypedDict):
    aggressive_history: str; conservative_history: str; neutral_history: str
    history: str; latest_speaker: str
    current_aggressive_response: str; current_conservative_response: str; current_neutral_response: str
    judge_decision: str; count: int
```

**Convergence / round limits** (`tradingagents/graph/conditional_logic.py`) — this is the whole mechanism, and it's dead simple:
```python
class ConditionalLogic:
    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_debate(self, state) -> str:
        if state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds:
            return "Research Manager"                      # -> judge
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state) -> str:
        if state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds:
            return "Portfolio Manager"                     # -> judge
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
```
Key design notes: **the cap is `n_speakers * max_rounds` on a monotonically-incremented `count`**; turn-taking is a string-prefix check on the last speaker; there is **no semantic convergence test** — it's a pure hard cap, and the *judge node* (Research Manager / Portfolio Manager) is the convergence mechanism. Each debater node appends to a shared `history` string AND its own `bull_history`/`bear_history`, then bumps `count`.

**Bull node** (abridged, real code) — note it receives *all* analyst reports plus `history` and `current_response` (the last bear argument), and explicitly instructs rebuttal:
```python
def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        ds = state["investment_debate_state"]
        prompt = f"""You are a Bull Analyst advocating for investing...
- Bear Counterpoints: Critically analyze the bear argument with specific data...
- Engagement: Present your argument in a conversational style, engaging directly with
  the bear analyst's points and debating effectively rather than just listing data.
Market research report: {state['market_report']}
...
Conversation history of the debate: {ds.get('history','')}
Last bear argument: {ds.get('current_response','')}
"""
        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"
        return {"investment_debate_state": {
            "history": ds.get("history","") + "\n" + argument,
            "bull_history": ds.get("bull_history","") + "\n" + argument,
            "bear_history": ds.get("bear_history",""),
            "current_response": argument,
            "count": ds["count"] + 1,
        }}
    return bull_node
```
Note this is the **opposite** of our blind protocol: debaters get FULL shared context deliberately. That's correct for debate (you must rebut) and wrong for blind naming (you must not anchor). Our system needs both, in different phases.

### Adapting this to P2 (2 annotators + 1 referee, κ ≥ 0.9) and P6 (merge/split adjudication)

```python
class AdjudicationState(TypedDict):
    item: dict
    a_history: Annotated[list, operator.add]
    b_history: Annotated[list, operator.add]
    transcript: Annotated[list, operator.add]
    last_speaker: str
    round_count: int
    verdict: dict | None

MAX_ROUNDS = 2          # => hard cap of 2*MAX_ROUNDS turns

def route_debate(state) -> str:
    if state["round_count"] >= 2 * MAX_ROUNDS:
        return "referee"
    # early convergence: both annotators agree on the label AND both confidence >= .8
    if len(state["transcript"]) >= 2:
        last_two = state["transcript"][-2:]
        if last_two[0]["label"] == last_two[1]["label"] and min(t["confidence"] for t in last_two) >= 0.8:
            return "referee"
    return "annotator_b" if state["last_speaker"] == "annotator_a" else "annotator_a"

builder.add_conditional_edges("annotator_a", route_debate, ["annotator_b", "referee"])
builder.add_conditional_edges("annotator_b", route_debate, ["annotator_a", "referee"])
```
This adds the semantic convergence test that TradingAgents lacks — necessary because we have a measurable target (κ ≥ 0.9), so we can stop early on agreement rather than always burning the full round budget.

**Judge-panel note:** Anthropic's evaluation section explicitly warns against over-engineering judge panels: *"We experimented with multiple judges to evaluate each component, but found that a single LLM call with a single prompt outputting scores from 0.0-1.0 and a pass-fail grade was the most consistent and aligned with human judgements."* → For P9/P11 quality scoring, use **one** judge with a rubric, not a panel. Panels are for *generation* diversity (P7 naming), not for *scoring*.

---

## 7. Context isolation, token economics, and when multi-agent is NOT worth it

**Anthropic's numbers (from `anthropic.com/engineering/multi-agent-research-system`, verbatim):**
- *"a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by **90.2%** on our internal research eval."*
- *"three factors explained **95%** of the performance variance in the BrowseComp evaluation... **token usage by itself explains 80%** of the variance, with the number of tool calls and the model choice as the two other explanatory factors."*
- *"agents typically use about **4× more tokens** than chat interactions, and multi-agent systems use about **15× more tokens** than chats."*
- *"For economic viability, multi-agent systems require tasks where the value of the task is high enough to pay for the increased performance."*
- **When NOT to use it:** *"some domains that require all agents to share the same context or involve many dependencies between agents are not a good fit... most coding tasks involve fewer truly parallelizable tasks than research, and LLM agents are not yet great at coordinating and delegating to other agents in real time."*
- **When it DOES fit:** *"valuable tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools."*
- *"The essence of search is compression: distilling insights from a vast corpus. Subagents facilitate compression by operating in parallel with their own context windows... Each subagent also provides separation of concerns—distinct tools, prompts, and exploration trajectories—which reduces path dependency."*

**Cognition's counterpoint (`cognition.ai/blog/dont-build-multi-agents`, Walden Yan, 2025-06-12):**
- **Principle 1:** *"Share context, and share full agent traces, not just individual messages."*
- **Principle 2:** *"Actions carry implicit decisions, and conflicting decisions carry bad results."*
- *"I would argue that Principles 1 & 2 are so critical, and so rarely worth violating, that you should by default rule out any agent architectures that don't abide by them."*
- Flappy Bird example: subagent 1 builds a Super Mario background, subagent 2 builds a non-Flappy bird; the merger is left combining two miscommunications. Even with the original task copied to subagents, the two produce *"a bird and background with completely different visual styles"* because they can't see each other's implicit decisions.
- On Claude Code (June 2025): *"it never does work in parallel with the subtask agent, and the subtask agent is usually only tasked with answering a question, not writing any code... The benefit of having a subagent in this case is that all the subagent's investigative work does not need to remain in the history of the main agent."*
- Recommended fallback: single-threaded linear agent + a dedicated **compression model** that *"compress[es] a history of actions & conversation into key details, events, and decisions. This is hard to get right."*
- Explicitly names OpenAI `swarm` and Microsoft `autogen` as *"actively push[ing] concepts which I believe to be the wrong way of building agents."*

**Harrison Chase's synthesis (`blog.langchain.com/how-and-when-to-build-multi-agent-systems/`, 2025-06-16)** — this is the reconciliation and it is the key decision rule for us:
- *"Multi-agent systems designed primarily for 'reading' tasks tend to be more manageable than those focused on 'writing' tasks... **read actions are inherently more parallelizable than write actions.** When you attempt to parallelize writing, you face the dual challenge of effectively communicating context between agents and then merging their outputs coherently."*
- *"conflicting write actions typically produce far worse outcomes than conflicting read actions."*
- *"Anthropic's Claude Research... the multi-agent architecture primarily handles the research (reading) component. The actual writing—synthesizing findings into a coherent report—is deliberately handled by a **single main agent in one unified call**."*
- On framework choice: *"you need to have full control what gets passed into the LLM, and full control over what steps are run and in what order... LangGraph... is a low-level orchestration framework with **no hidden prompts, no enforced 'cognitive architectures'**."*

**Applying the read/write rule to our 12 phases:**

| Phase | Read or Write? | Verdict |
|---|---|---|
| P2 taxonomy research (5–9 agents) | READ | ✅ parallelize (Send fan-out over research questions) |
| P2 gold labeling | WRITE, but per-item independent | ✅ parallelize over items; ❌ do not parallelize the *schema* decision |
| P3/P4/P5 | deterministic compute | ✅ parallelize (no LLM) |
| P6 hierarchy build/refine | **WRITE, globally coupled** | ❌ single agent; merge/split decisions carry implicit decisions that conflict |
| P7 blind naming | READ+propose, per-cluster independent | ✅ parallelize; consensus is a *programmatic* merge, not an agent merge |
| P7 tree audit | WRITE, global | ❌ single auditor with full context |
| P8 governance merges | WRITE, global | ❌ single agent + deterministic lookup-table execution |
| P11 report synthesis | WRITE | ❌ **single agent, one unified call** — this is exactly Anthropic's CitationAgent/report-writer choice |

**LangChain's own context-engineering framing** (`/oss/python/langchain/context-engineering.md`): three controllable context types — **Model context** (transient: instructions, message history, tools, response format), **Tool context** (persistent: reads/writes to state, store, runtime context), **Life-cycle context** (persistent: summarization, guardrails, logging). Three data sources: **Runtime Context** (static config, conversation-scoped), **State** (short-term memory, conversation-scoped), **Store** (long-term memory, cross-conversation). *"When agents fail, it's usually because... the 'right' context was not passed to the LLM."*

Anthropic's appendix tip that maps perfectly to our artifact/provenance requirement: **"Subagent output to a filesystem to minimize the 'game of telephone.'"** — *"Rather than requiring subagents to communicate everything through the lead agent, implement artifact systems where specialized agents can create outputs that persist independently. Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator. This prevents information loss during multi-stage processing and reduces token overhead."* Our phases should write parquet/JSON artifacts and pass **paths + hashes**, never dataframes-as-text.

---

## 8. Orchestrator prompt design

Anthropic's rules, verbatim and each with the failure it prevents:

1. **"Think like your agents."** Build simulations with the exact prompts and tools, watch step-by-step. *"This immediately revealed failure modes: agents continuing when they already had sufficient results, using overly verbose search queries, or selecting incorrect tools."*
2. **"Teach the orchestrator how to delegate."** *"Each subagent needs **an objective, an output format, guidance on the tools and sources to use, and clear task boundaries**. Without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information."* Concrete failure: *"one subagent explored the 2021 automotive chip crisis while 2 others duplicated work investigating current 2025 supply chains, without an effective division of labor."*
3. **"Scale effort to query complexity."** The explicit ladder they embed in prompts: *"Simple fact-finding requires just **1 agent with 3-10 tool calls**, direct comparisons might need **2-4 subagents with 10-15 calls each**, and complex research might use **more than 10 subagents** with clearly divided responsibilities."* Prevents the observed failure of *"spawning 50 subagents for simple queries."*
4. **"Tool design and selection are critical."** Heuristics they inject: *"examine all available tools first, match tool usage to user intent, search the web for broad external exploration, or prefer specialized tools over generic ones."* *"Bad tool descriptions can send agents down completely wrong paths."*
5. **"Let agents improve themselves."** A tool-testing agent that rewrites flawed tool descriptions produced a **40% decrease in task completion time**.
6. **"Start wide, then narrow down."** *"Agents often default to overly long, specific queries that return few results."*
7. **"Guide the thinking process."** Extended thinking as a controllable scratchpad for the lead agent to plan (*"determining query complexity and subagent count, and defining each subagent's role"*); interleaved thinking for subagents after tool results.
8. **"Parallel tool calling transforms speed."** *"(1) the lead agent spins up **3-5 subagents in parallel** rather than serially; (2) the subagents use **3+ tools in parallel**. These changes cut research time by up to **90%** for complex queries."*
9. Meta-rule: *"the best prompts for these agents are not just strict instructions, but **frameworks for collaboration that define the division of labor, problem-solving approaches, and effort budgets**."*

LangChain's subagent-spec guidance (deepagents best practices):
- Descriptions: ✅ `"Analyzes financial data and generates investment insights with confidence scores"` / ❌ `"Does finance stuff"`.
- System prompts: numbered procedure + explicit **Output format** section + *"Keep your response under 500 words to maintain clean context."*
- **Minimize tool sets** — *"Only give subagents the tools they need. This improves focus and security."*
- **Choose models by task** (large-context model for long docs, stronger reasoning model for numerics).
- **Return concise results** — explicitly list what NOT to include: *"Raw data / Intermediate calculations / Detailed tool outputs."*

**A concrete orchestrator prompt skeleton for our phase driver** (synthesizing all of the above):
```
You coordinate a 12-phase query-intent-mining pipeline. You do NOT do analysis yourself.

EFFORT LADDER (choose before delegating):
- Dataset < 10k queries, single vertical  -> 3 research agents, 1 pass, K-sweep 4..12
- 10k-500k, single vertical               -> 5 research agents, 2 passes, K-sweep 6..40
- >500k or multi-vertical                 -> 9 research agents, 3 passes, K-sweep 10..120
Never exceed the ladder. Overinvestment in simple datasets is the most common failure.

FOR EACH DELEGATION you MUST specify:
  1. OBJECTIVE      - one sentence, non-overlapping with other agents' objectives
  2. OUTPUT FORMAT  - the exact schema name the agent must return
  3. TOOLS/SOURCES  - which tools, which artifact paths, which shards
  4. BOUNDARIES     - what this agent must NOT do, and which shard IDs belong to others

ANTI-DUPLICATION: before dispatching N agents, write the N objectives out and verify
no two share a primary noun phrase. If two overlap, merge them.

ARTIFACTS: agents return artifact PATHS + sha256, never inline data.
```

---

## 9. Documented failure modes and mitigations

| Failure mode | Source | Mitigation |
|---|---|---|
| Spawning 50 subagents for a simple query | Anthropic | Effort-scaling ladder in orchestrator prompt + `ModelCallLimitMiddleware(run_limit=...)` |
| Subagents duplicating each other's work (2021 chip crisis vs 2025 supply chain) | Anthropic | Objective + output format + tools + **explicit boundaries** per subagent; pre-dispatch overlap check |
| Subagents "distracting each other with excessive updates" | Anthropic | Return only final summary; artifact-passing by reference |
| Subagent does the work but omits results from its final message | LangChain subagents doc | Prompt: *"the supervisor only sees the final output"*; or use `response_format` so the schema forces it |
| Conflicting implicit decisions across parallel writers | Cognition | Don't parallelize writes; single writer for P6/P8/P11 |
| Malformed history after handoff (unpaired tool call) | LangChain handoffs doc | Always emit `[last_ai_message, ToolMessage(tool_call_id=...)]` in the `Command.update` |
| Context bloat from passing full subagent history on handoff | LangChain handoffs doc | Pass only the pair; summarize in ToolMessage content |
| Missing parent reducer on `Command.PARENT` shared key | LangGraph graph-api | Define reducer on every parent key a subgraph writes |
| `Command(goto=...)` + static `add_edge` from same node → both fire | LangGraph graph-api | One routing mechanism per node |
| `Command(update=...)` passed to `invoke()` → graph appears stuck | LangGraph graph-api | Only `Command(resume=...)` is valid as input; use plain dict to continue |
| Evaluator-optimizer infinite loop | Anthropic BEA / observed in docs example | Round counter in state + `recursion_limit` + `RemainingSteps` fallback node |
| Subagent state invisible to `get_state(subgraphs=True)` | LangChain subagents doc | Invoke subagent from a **node function**, not a tool, when you need to inspect state at an interrupt |
| Interrupt doesn't propagate from nested agent | supervisor migration guide | Compile ONLY outermost graph with checkpointer; always pass `thread_id` |
| Errors compound; restart from scratch is expensive | Anthropic | Durable checkpointing + resume-from-failure; *"letting the agent know when a tool is failing and letting it adapt works surprisingly well"*; combine with deterministic retry + regular checkpoints |
| Non-determinism makes debugging impossible | Anthropic / LangChain | Full production tracing (LangSmith); *"monitor agent decision patterns and interaction structures"* |
| Deploying mid-run breaks in-flight agents | Anthropic | **Rainbow deployments** — gradually shift traffic, keep both versions running |
| Synchronous lead-agent bottleneck | Anthropic (acknowledged, unsolved) | Accept it; or async job pattern (start/status/get_result three-tool pattern from LangChain subagents doc) |
| Agents preferring SEO content farms over authoritative sources | Anthropic (found by human testers) | Source-quality heuristics in prompt; human eval catches what automation misses |
| Evals delayed because "we need 100s of cases" | Anthropic | *"start with a set of about 20 queries representing real usage patterns"* |

---

## 10. Two remaining API details worth having on hand

**Router with parallel fan-out (current doc form):**
```python
from langgraph.types import Send

class ClassificationResult(TypedDict):
    query: str
    agent: str

def route_query(state: State):
    classifications = classify_query(state["query"])
    return [Send(c["agent"], {"query": c["query"]}) for c in classifications]
```
Stateful-router warning from the docs: *"Stateful routers require custom history management... Consider the handoffs pattern or subagents pattern instead—both provide clearer semantics for multi-turn conversations."* The recommended cheap fix is the **tool wrapper**: wrap the stateless router graph as a `@tool` on a conversational agent.

**Agent-as-node inside a bigger StateGraph** (this is the shape our phase driver should take):
```python
email_agent = create_agent(model="claude-sonnet-4-6", tools=[read_email, send_email],
                           middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})])
graph = (StateGraph(AgentState)
         .add_node("classify", classify_node)
         .add_node("email_agent", email_agent)      # compiled agent used directly as a node
         .add_edge(START, "classify")
         .add_conditional_edges("classify", route)
         .compile())
```
*"Middleware is not a separate runtime: hooks run inside the compiled LangGraph that `create_agent` returns... every middleware hook continues to run."* Reach for this *"when the surrounding topology is more than a standard 'loop until done': classifying input before routing to one of several agents, fanning out work in parallel, or stitching agent calls together with deterministic steps."* — which is exactly our 12-phase pipeline.


---

## Recommendations carried into the design

- Do NOT use `langgraph-supervisor` or `create_react_agent` — both are officially deprecated (supervisor's last release was 2025-11, and `create_react_agent` raises `LangGraphDeprecatedSinceV10`, slated for removal in LangGraph v2); build on `langchain.agents.create_agent` (langchain 1.3.15) + `langgraph` 1.2.11 with tool-wrapped subagents.
- Make the 12-phase driver a hand-written `StateGraph` (custom workflow), not a supervisor agent — phase ordering, HITL gates, and reproducibility are hard requirements that must not be delegated to an LLM's routing decision.
- Implement the P7 blind naming protocol with `Send("name_worker", {only_the_shard_fields})`: I verified empirically on langgraph 1.2.11 that a Send worker's `state.keys()` contains ONLY the Send payload keys, so putting `existing_labels` in the parent state makes anchoring structurally impossible rather than prompt-dependent.
- Rely on the verified determinism of Send fan-in: with randomized worker latency, `Annotated[list, operator.add]` aggregation returned results in Send-dispatch order (not completion order) on 5/5 runs — pair it with `defer=True` on the aggregator node so it waits for all N workers.
- Apply Harrison Chase's read/write rule as the parallelization gate: parallelize READ-heavy phases (P2 research, P7 naming proposals) and per-item-independent labeling, but keep globally-coupled WRITE phases (P6 hierarchy refinement, P8 governance merges, P11 report synthesis) in a single agent making one unified call.
- Replace all `pre_model_hook`/`post_model_hook`/`state_modifier` thinking with middleware: `wrap_model_call` + `request.override(system_prompt=..., tools=...)` for per-phase dynamic configuration, and `ModelCallLimitMiddleware(run_limit=N)` / `ToolCallLimitMiddleware` as hard per-phase budget guards.
- Use `ToolRuntime` (not `InjectedState`/`InjectedToolCallId`) for all state-aware tools, and when a tool returns `Command` for a handoff, always emit the `[last_ai_message, ToolMessage(tool_call_id=runtime.tool_call_id)]` pair — an unpaired tool call corrupts the receiving agent's history.
- Copy TradingAgents' debate cap mechanism verbatim for P2 annotator adjudication (`count >= n_speakers * max_rounds` on a monotonic counter, with a judge node as the terminal), but add the semantic early-exit TradingAgents lacks: stop as soon as both annotators agree with confidence >= threshold, since we have a measurable κ >= 0.9 target.
- Use ONE LLM judge with a 0.0–1.0 rubric for P9/P11 scoring, not a judge panel — Anthropic explicitly reported that a single judge call was more consistent and more human-aligned than multiple judges; reserve multi-agent panels for generation diversity (P7 naming votes), not scoring.
- Budget for ~15x token cost versus a single-agent baseline (Anthropic's measured multiplier) and justify it only for the genuinely parallel READ phases; have subagents write artifacts to disk and return paths + sha256 rather than inline data (Anthropic's 'avoid the game of telephone' pattern), which also gives us P11 provenance for free.
- Compile ONLY the outermost graph with a checkpointer and always pass `thread_id` so `interrupt()` propagates up through nested `create_agent` layers; and invoke subagents from explicit node functions (not tool wrappers) at every HITL gate, because `get_state(subgraphs=True)` cannot see subagents called inside tools.
- Embed an explicit effort-scaling ladder in the orchestrator prompt (dataset-size tiers → number of research agents, passes, K-sweep range) plus a mandatory 4-part delegation template (objective / output format / tools+shards / boundaries), which are Anthropic's two documented fixes for the 'spawned 50 subagents' and 'three subagents duplicated the same search' failure modes.

## Unverified or version-dependent

- `docs.langchain.com` is blocked for WebFetch in this environment; I retrieved everything via curl of the Mintlify `.md` endpoints. Content is authoritative but I could not render the diagrams/images, so a few flow diagrams are described only by their alt text and surrounding prose.
- Model IDs appearing in the current docs (`anthropic:claude-sonnet-4-6`, `openai:gpt-5.5`, `google_genai:gemini-3.6-flash`, `ollama:north-mini-code-1.0`) are taken verbatim from the docs pages; I did not independently verify these model strings resolve, and `create_agent` model strings should be re-checked against `init_chat_model` at implementation time.
- `RubricMiddleware` is explicitly marked BETA and requires `deepagents>=0.6.5`; the docs warn the API may change. I did not install deepagents or verify its signature empirically.
- I verified Send-payload isolation and fan-in determinism on langgraph 1.2.11 with plain Python worker nodes. I did NOT verify the same isolation holds when the worker node internally invokes a compiled `create_agent` — it should (the agent gets only what you pass to `.invoke()`), but the LLM call itself is nondeterministic, so end-to-end reproducibility of P7 still depends on temperature/seed control at the provider level, which LangChain does not guarantee.
- `langgraph-supervisor-py` and `langgraph-swarm-py` are NOT archived on GitHub (both pushed 2026-07-15) despite stale PyPI releases and the README deprecation notice. Their maintenance trajectory is ambiguous; the docs' recommendation to migrate off `create_supervisor` is unambiguous, but `langgraph-swarm` has no equivalent migration guide and no explicit deprecation notice — its future is the least certain item here.
- The supervisor README links to a tutorial at `docs.langchain.com/oss/python/langchain/supervisor` which is not in the current section index; that URL appears stale. The live equivalent is `/oss/python/langchain/multi-agent/subagents-personal-assistant`, which I did not fetch.
- Anthropic's 15x token multiplier and 90.2% eval improvement are from the June 2025 post using Claude Opus 4 / Sonnet 4; these ratios may differ substantially with 2026-era models and the post has not been updated.
- I did not fetch the `subagents-personal-assistant`, `router-knowledge-base`, `skills-sql-assistant`, or `handoffs-customer-support` tutorials, which likely contain additional end-to-end runnable code for each pattern.
- The `transformers: Sequence[TransformerFactory]` parameter on `create_agent` is new and sparsely documented (stream transformers); I captured only the docstring and did not explore its use.

## Sources

- https://docs.langchain.com/llms.txt
- https://docs.langchain.com/oss/python/langgraph/llms.txt
- https://docs.langchain.com/oss/python/langchain/llms.txt
- https://docs.langchain.com/oss/python/migrate/llms.txt
- https://docs.langchain.com/oss/python/langchain/multi-agent/index.md
- https://docs.langchain.com/oss/python/langchain/multi-agent/subagents.md
- https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs.md
- https://docs.langchain.com/oss/python/langchain/multi-agent/router.md
- https://docs.langchain.com/oss/python/langchain/multi-agent/skills.md
- https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow.md
- https://docs.langchain.com/oss/python/langchain/agents.md
- https://docs.langchain.com/oss/python/langchain/tools.md
- https://docs.langchain.com/oss/python/langchain/context-engineering.md
- https://docs.langchain.com/oss/python/langchain/structured-output.md
- https://docs.langchain.com/oss/python/langchain/middleware/overview.md
- https://docs.langchain.com/oss/python/langchain/middleware/custom.md
- https://docs.langchain.com/oss/python/langchain/middleware/built-in.md
- https://docs.langchain.com/oss/python/langgraph/workflows-agents.md
- https://docs.langchain.com/oss/python/langgraph/graph-api.md
- https://docs.langchain.com/oss/python/langgraph/use-graph-api.md
- https://docs.langchain.com/oss/python/migrate/langgraph-supervisor.md
- https://docs.langchain.com/oss/python/migrate/langgraph-v1.md
- https://docs.langchain.com/oss/python/deepagents/subagents.md
- https://reference.langchain.com/python/langchain/agents/factory/create_agent.md
- https://www.anthropic.com/engineering/multi-agent-research-system
- https://www.anthropic.com/engineering/building-effective-agents
- https://cognition.ai/blog/dont-build-multi-agents
- https://blog.langchain.com/how-and-when-to-build-multi-agent-systems/
- https://raw.githubusercontent.com/langchain-ai/langgraph-supervisor-py/main/README.md
- https://raw.githubusercontent.com/langchain-ai/langgraph-swarm-py/main/README.md
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/graph/conditional_logic.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/agents/utils/agent_states.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/main/tradingagents/agents/researchers/bull_researcher.py
- https://pypi.org/pypi/langchain/json
- https://pypi.org/pypi/langgraph/json
- https://pypi.org/pypi/deepagents/json
- https://pypi.org/pypi/langgraph-supervisor/json
- https://pypi.org/pypi/langgraph-swarm/json
- https://api.github.com/repos/langchain-ai/langgraph-supervisor-py
- https://api.github.com/repos/langchain-ai/langgraph-swarm-py