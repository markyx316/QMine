# TradingAgents Architecture Teardown

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

> **Method note.** I did not rely on memory. I downloaded the repo at HEAD (`https://github.com/TauricResearch/TradingAgents/archive/refs/heads/main.zip`, version `0.3.1` per `pyproject.toml`, files dated 2026-07-18) into `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/8075e8db-1d8b-4b54-b04d-7d995fbae90d/scratchpad/TradingAgents-main/`, and separately pulled the **v0.1.0** tag from `raw.githubusercontent.com` because HEAD has *removed* several things the assignment asked about (notably `FinancialSituationMemory` / ChromaDB and the `toolkit` argument). I also extracted the full arXiv PDF (27 pages, 101k chars) to `.../scratchpad/paper.txt`.
>
> **Headline correction to the brief:** `FinancialSituationMemory` + ChromaDB + `get_memories(situation, n_matches)` **no longer exists**. It was ChromaDB in v0.1.x, then BM25 in v0.2.x, then deleted in v0.2.4 and replaced by an append-only markdown decision log. The *reasons they deleted it* are the single most useful thing in this whole repo for our design, so I document both generations below.

---

## 1. Overall graph topology — exact node names and edges

Built in `tradingagents/graph/setup.py`, class `GraphSetup.setup_graph(selected_analysts=("market","social","news","fundamentals"))`, returning an uncompiled `StateGraph(AgentState)`.

**Nodes.** For each selected analyst, three nodes are registered from a declarative spec table (`tradingagents/graph/analyst_execution.py`, `ANALYST_NODE_SPECS`):

| key | agent_node | clear_node | tool_node | report_key |
|---|---|---|---|---|
| `market` | `Market Analyst` | `Msg Clear Market` | `tools_market` | `market_report` |
| `social` | `Sentiment Analyst` | `Msg Clear Sentiment` | `tools_social` | `sentiment_report` |
| `news` | `News Analyst` | `Msg Clear News` | `tools_news` | `news_report` |
| `fundamentals` | `Fundamentals Analyst` | `Msg Clear Fundamentals` | `tools_fundamentals` | `fundamentals_report` |

Plus the fixed downstream nodes: `Bull Researcher`, `Bear Researcher`, `Research Manager`, `Trader`, `Aggressive Analyst`, `Neutral Analyst`, `Conservative Analyst`, `Portfolio Manager`. (In v0.1.0 the risk trio was named `Risky Analyst` / `Safe Analyst` / `Neutral Analyst` and the final judge was `Risk Judge`; renamed since.)

**Edges — the analyst chain is SEQUENTIAL, not parallel.** This is the most important structural fact and it contradicts the paper's Figure 1 ("Four analysts *concurrently* gather relevant market information"):

```python
workflow.add_edge(START, plan.specs[0].agent_node)
for i, spec in enumerate(plan.specs):
    workflow.add_conditional_edges(
        spec.agent_node,
        getattr(self.conditional_logic, f"should_continue_{spec.key}"),
        [spec.tool_node, spec.clear_node],
    )
    workflow.add_edge(spec.tool_node, spec.agent_node)      # ReAct tool loop
    if i < len(plan.specs) - 1:
        workflow.add_edge(spec.clear_node, plan.specs[i + 1].agent_node)
    else:
        workflow.add_edge(spec.clear_node, "Bull Researcher")
```

So the real topology is:

```
START → Market Analyst ⇄ tools_market → Msg Clear Market
      → Sentiment Analyst ⇄ tools_social → Msg Clear Sentiment
      → News Analyst ⇄ tools_news → Msg Clear News
      → Fundamentals Analyst ⇄ tools_fundamentals → Msg Clear Fundamentals
      → Bull Researcher ⇄ Bear Researcher   (ping-pong, count-limited)
      → Research Manager → Trader → Aggressive Analyst
      → Conservative Analyst → Neutral Analyst  (3-way round-robin, count-limited)
      → Portfolio Manager → END
```

**Debate edges use a *shared complete* path map** (a bug fix, issue #1088, worth stealing):

```python
DEBATE_PATH_MAP = {"Bull Researcher": "Bull Researcher",
                   "Bear Researcher": "Bear Researcher",
                   "Research Manager": "Research Manager"}
RISK_ANALYSIS_PATH_MAP = {"Aggressive Analyst": ..., "Conservative Analyst": ...,
                          "Neutral Analyst": ..., "Portfolio Manager": ...}

for debate_node in ("Bull Researcher", "Bear Researcher"):
    workflow.add_conditional_edges(debate_node,
        self.conditional_logic.should_continue_debate, DEBATE_PATH_MAP)
workflow.add_edge("Research Manager", "Trader")
workflow.add_edge("Trader", "Aggressive Analyst")
for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
    workflow.add_conditional_edges(risk_node,
        self.conditional_logic.should_continue_risk_analysis, RISK_ANALYSIS_PATH_MAP)
workflow.add_edge("Portfolio Manager", END)
```

In v0.1.0 each edge got only its *partial* map (`{"Bear Researcher": ..., "Research Manager": ...}`), so any router fall-through — a renamed node, a translated speaker label — crashed LangGraph mid-run with a missing path_map key. The fix: **every edge driven by a shared router maps the router's entire return range.** There is a dedicated regression test at `tests/test_risk_router_path_map.py` that parametrizes drift cases (`""`, `"Aggressive Risk Analyst"`, `"Agresivo"`) and asserts `target in RISK_ANALYSIS_PATH_MAP`.

---

## 2. State design

`tradingagents/agents/utils/agent_states.py`. Three TypedDicts; `AgentState` extends LangGraph's `MessagesState` (so `messages` carries the built-in add-messages reducer; **everything else is last-write-wins**).

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
    history: str
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: str
    current_conservative_response: str
    current_neutral_response: str
    judge_decision: str
    count: int

class AgentState(MessagesState):
    company_of_interest: str; asset_type: str; instrument_context: str; trade_date: str
    sender: str
    market_report: str; sentiment_report: str; news_report: str; fundamentals_report: str
    investment_debate_state: InvestDebateState
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: RiskDebateState
    final_trade_decision: str
    past_context: Annotated[str, "Memory log context injected at run start"]
```

**How debate history accumulates.** Pure string concatenation, done by hand inside each node. From `researchers/bull_researcher.py`:

```python
argument = f"Bull Analyst: {response.content}"
new_investment_debate_state = {
    "history": history + "\n" + argument,
    "bull_history": bull_history + "\n" + argument,
    "bear_history": investment_debate_state.get("bear_history", ""),   # copied forward by hand
    "current_response": argument,
    "count": investment_debate_state["count"] + 1,
}
return {"investment_debate_state": new_investment_debate_state}
```

Three things to note: (a) the speaker-prefixed string (`"Bull Analyst: ..."`) *is* the routing signal — the router does `current_response.startswith("Bull")`; (b) because the sub-dict is replaced wholesale, **every node must hand-copy every sibling field** or silently wipe it (this caused real bug #503, "portfolio manager state fix"); (c) `count` is incremented manually by each participant, so the counter is only correct if every node remembers to bump it.

`Propagator.create_initial_state()` (`graph/propagation.py`) seeds both debate dicts with empty strings and `count: 0`, and seeds `messages: [("human", company_name)]`.

---

## 3. Conditional logic — termination and tool loops

`tradingagents/graph/conditional_logic.py`, class `ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)`.

**Tool-call loop** — one method per analyst, all identical:

```python
def should_continue_market(self, state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_market"
    return "Msg Clear Market"
```

i.e. "if the model emitted tool calls, go execute them and come back; otherwise the analyst is done — wipe the message channel and hand off." The only guard against an infinite ReAct loop is `recursion_limit: 100` from `Propagator.get_graph_args()`.

**Debate termination — pure round counting, no convergence detection:**

```python
def should_continue_debate(self, state: AgentState) -> str:
    if state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds:
        return "Research Manager"
    if state["investment_debate_state"]["current_response"].startswith("Bull"):
        return "Bear Researcher"
    return "Bull Researcher"

def should_continue_risk_analysis(self, state: AgentState) -> str:
    if state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds:
        return "Portfolio Manager"
    if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
        return "Conservative Analyst"
    if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
        return "Neutral Analyst"
    return "Aggressive Analyst"
```

The multipliers encode the number of participants: `2 *` for the two-agent research debate, `3 *` for the three-agent risk debate. With the default `max_debate_rounds=1`, the "debate" is literally **one bull turn plus one bear turn**, then the judge. (The inline comment says "3 rounds of back-and-forth between 2 agents" while the code says `2 *` — a stale comment that has survived since v0.1.0.)

The final `return` in each router is the fall-through that the shared path map exists to make safe.

---

## 4. Agent construction pattern

**Factory closure over the LLM, returning a plain node function.** Signature changed between versions: v0.1.0 was `create_market_analyst(llm, toolkit)` and `create_bull_researcher(llm, memory)`; HEAD is `create_market_analyst(llm)` — tools are now module-level `@tool` functions imported directly, and memory was removed from the researcher path entirely.

Two distinct sub-patterns:

**(a) Tool-using analysts** (`analysts/market_analyst.py`) — ChatPromptTemplate + `MessagesPlaceholder` + `bind_tools`:

```python
def create_market_analyst(llm):
    def market_analyst_node(state):
        tools = [get_stock_data, get_indicators, get_verified_market_snapshot]
        system_message = ("""You are a trading assistant tasked with analyzing financial
markets. Your role is to select the **most relevant indicators** ... up to **8 indicators**
that provide complementary insights without redundancy...""")
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful AI assistant, collaborating with other assistants."
             " Use the provided tools to progress towards answering the question."
             " If you are unable to fully answer, that's OK; another assistant with different tools"
             " will help where you left off. Execute what you can to make progress."
             " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"
             " or deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: ..."
             " You have access to the following tools: {tool_names}."
             " Today's date is {current_date}; treat it as 'now' ... {instrument_context}\n"
             "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:      # only commit the report on a non-tool turn
            report = result.content
        return {"messages": [result], "market_report": report}
    return market_analyst_node
```

That `if len(result.tool_calls) == 0` guard is a fix — v0.1.0 wrote `"market_report": result.content` unconditionally, so a tool-calling turn (empty content) clobbered the finished report.

**(b) Structured-output decision agents** (Research Manager, Trader, Portfolio Manager, Sentiment Analyst). The pattern is centralized in `agents/utils/structured.py`:

```python
NO_EXTERNAL_TOOLS = ("Use only the evidence provided in this prompt. Do not call external "
                     "tools or search the web; if something is missing, say so explicitly.")

def bind_structured(llm, schema, agent_name):
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning("%s: provider does not support with_structured_output (%s); "
                       "falling back to free-text generation", agent_name, exc)
        return None

def invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name):
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                raise ValueError("structured output returned no parsed result")
            return render(result)
        except Exception as exc:
            logger.warning("%s: structured-output invocation failed (%s); retrying once as free text", ...)
    response = plain_llm.invoke(prompt)
    return response.content
```

`bind_structured` is called **once at factory time**, `invoke_structured_or_freetext` per node call. `NO_EXTERNAL_TOOLS` exists because schema-only structured output binds exactly one "tool" (the schema), so a model that reaches for search emits an unknown tool call and the whole structured attempt is discarded (#1130).

**Dual representation.** Every schema in `agents/schemas.py` has a paired `render_*` function that converts the Pydantic object back to the exact markdown the rest of the system consumes:

```python
class PortfolioDecision(BaseModel):
    rating: PortfolioRating = Field(description="The final position rating. Exactly one of "
                                    "Buy / Overweight / Hold / Underweight / Sell ...")
    executive_summary: str = Field(description="A concise action plan covering entry strategy, "
                                   "position sizing, key risk levels, and time horizon. Two to four sentences.")
    investment_thesis: str = Field(description="Detailed reasoning anchored in specific evidence ...")
    price_target: float | None = None
    time_horizon: str | None = None

def render_pm_decision(decision) -> str:
    return "\n".join([f"**Rating**: {decision.rating.value}", "",
                      f"**Executive Summary**: {decision.executive_summary}", "",
                      f"**Investment Thesis**: {decision.investment_thesis}", ...])
```

Note the deliberate design comment: *"Schema field descriptions become the model's output instructions, freeing the prompt body to focus on context and the rating-scale guidance."* There is also a `_coerce_optional_float` validator that maps `{"", "none", "n/a", "null", "tbd", ...}` → `None`, because LLMs write placeholder strings into optional numeric fields (#1058).

**Context hygiene between agents** — `create_msg_delete()` in `agents/utils/agent_utils.py`:

```python
def create_msg_delete():
    def delete_messages(state):
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content=(
            f"Proceed with your assigned analysis for this workflow. "
            f"{instrument_context} The analysis date is {trade_date}."))
        return {"messages": removal_operations + [placeholder]}
    return delete_messages
```

The docstring records a real failure: the placeholder used to be a bare `"Continue"`, and some OpenAI-compatible providers took that literally and wrote an essay about the word "continue" (#888).

---

## 5. Memory and reflection — BOTH generations

### Generation 1 (v0.1.0–v0.2.3): `FinancialSituationMemory`, ChromaDB — **REMOVED**

```python
class FinancialSituationMemory:
    def __init__(self, name):
        self.client = OpenAI()
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        response = self.client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):   # list of (situation, recommendation)
        ...
        self.situation_collection.add(documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings, ids=ids)

    def get_memories(self, current_situation, n_matches=1):
        query_embedding = self.get_embedding(current_situation)
        results = self.situation_collection.query(query_embeddings=[query_embedding],
            n_results=n_matches, include=["metadatas", "documents", "distances"])
        # returns [{"matched_situation", "recommendation", "similarity_score": 1 - distance}]
```

Five *separate* collections, one per role: `bull_memory`, `bear_memory`, `trader_memory`, `invest_judge_memory`, `risk_manager_memory`. The **retrieval key** was the concatenation of all four analyst reports:

```python
curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
past_memories = memory.get_memories(curr_situation, n_matches=2)
past_memory_str = ""
for i, rec in enumerate(past_memories, 1):
    past_memory_str += rec["recommendation"] + "\n\n"
```

and the value was injected into the prompt as: *"Reflections from similar situations and lessons learned: {past_memory_str} ... You must also address reflections and learn from lessons and mistakes you made in the past."*

Writing was via `Reflector` with a long system prompt whose four numbered sections were **Reasoning / Improvement / Summary / Query** ("Extract key insights from the summary into a concise sentence of no more than 1000 tokens"), and five methods `reflect_bull_researcher / reflect_bear_researcher / reflect_trader / reflect_invest_judge / reflect_risk_manager`, each doing `memory.add_situations([(situation, result)])`.

**The fatal flaw:** the entry point was `TradingAgentsGraph.reflect_and_remember(returns_losses)` — a *manual* call the user had to make, and in `main.py` it was **commented out**: `# ta.reflect_and_remember(1000) # parameter is the position returns`. Combined with `chromadb.Client(...)` being in-memory only (no `PersistentClient`), the memory was wiped every process exit. It was, in practice, dead code. Community issues confirm: #563 "RAM-only memory", #572 "empty memory triggers fabricated past-lessons".

### Generation 2 (v0.2.4+, current): `TradingMemoryLog` — append-only markdown with **outcome-grounded deferred reflection**

`tradingagents/agents/utils/memory.py`. No embeddings, no vector DB, no BM25. Format:

```
[2026-05-10 | NVDA | Buy | pending]

DECISION:
<full rendered PM markdown>

<!-- ENTRY_END -->
```

resolved later into `[2026-05-10 | NVDA | Buy | +7.3% | +4.1% | 5d]` with an appended `REFLECTION:` section. `<!-- ENTRY_END -->` is chosen as the separator with the comment *"HTML comment: cannot appear in LLM prose output, safe as a hard delimiter."*

Key methods: `store_decision(ticker, trade_date, final_trade_decision)` (append, no LLM call, with an idempotency guard that scans for an existing pending tag), `load_entries()`, `get_pending_entries()`, `update_with_outcome(...)` / `batch_update_with_outcomes(updates)` (both atomic: write `.tmp` then `tmp_path.replace(self._log_path)`), `_apply_rotation()` (caps *resolved* entries at `memory_log_max_entries`; pending entries are never pruned).

Retrieval is **recency + ticker match, not similarity**:

```python
def get_past_context(self, ticker, n_same=5, n_cross=3) -> str:
    entries = [e for e in self.load_entries() if not e.get("pending")]
    ...
    for e in reversed(entries):
        if e["ticker"] == ticker and len(same) < n_same:   same.append(e)
        elif e["ticker"] != ticker and len(cross) < n_cross: cross.append(e)
    # "Past analyses of {ticker} (most recent first):"  -> full decision + reflection
    # "Recent cross-ticker lessons:"                    -> reflection only
```

**The reflection loop is now automatic and outcome-grounded.** In `TradingAgentsGraph.propagate()`:

1. `self._resolve_pending_entries(company_name)` runs **first**, before the graph. For each pending same-ticker entry it calls `_fetch_returns(ticker, trade_date, holding_days=5, benchmark=...)` which pulls real yfinance prices, computes `raw = (close[d] - close[0]) / close[0]`, `alpha = raw - bench_ret`, then calls the reflector, then does one batched atomic write.
2. `_resolve_benchmark(ticker)` picks the alpha baseline from a suffix map (`.T → ^N225`, `.L → ^FTSE`, `"" → SPY`).
3. The reflection prompt is now deliberately **tiny** (`graph/reflection.py`):

```python
"You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
"Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
"Cover in order:\n"
"1. Was the directional call correct? (cite the alpha figure)\n"
"2. Which part of the investment thesis held or failed?\n"
"3. One concrete lesson to apply to the next similar analysis.\n\n"
"Be specific and terse. Your output will be stored verbatim in a decision log "
"and re-read by future analysts, so every word must earn its place."
```
with the human turn being `f"Raw return: {raw_return:+.1%}\nAlpha vs {benchmark_name}: {alpha_return:+.1%}\n\nFinal Decision:\n{final_decision}"`.

4. `_run_graph` injects `past_context = self.memory_log.get_past_context(company_name)` into the initial state, and **only the Portfolio Manager reads it**:

```python
past_context = state.get("past_context", "")
lessons_line = (f"- Lessons from prior decisions and outcomes:\n{past_context}\n" if past_context else "")
```

The `if past_context else ""` is the structural fix for the hallucinated-lessons bug (#572): with no memory, the prompt has no lessons slot at all, so the model cannot invent one.

5. `store_decision(...)` runs at the end of the run, marking the new decision pending for the *next* run to resolve.

---

## 6. Config system

`tradingagents/default_config.py` — a plain dict plus a declarative env-override table.

**Two model tiers.** `deep_think_llm` (default `"gpt-5.5"`) and `quick_think_llm` (default `"gpt-5.4-mini"`). Routing in `setup.py` is explicit and **narrower than the paper claims**: deep thinking is used for exactly two nodes:

```python
research_manager_node   = create_research_manager(self.deep_thinking_llm)
portfolio_manager_node  = create_portfolio_manager(self.deep_thinking_llm)
# everything else — all analysts, both researchers, trader, all three risk debators — quick
```

The paper says *"all analyst nodes rely on deep-thinking models ... Researchers and traders use deep-thinking models"*. **The code does the opposite.** Deep models are reserved for the two synthesis/judge nodes only.

Other keys worth copying: `max_debate_rounds: 1`, `max_risk_discuss_rounds: 1`, `max_recur_limit: 100`, `checkpoint_enabled: False`, `output_language: "English"` (with the note *"Internal agent debate stays in English for reasoning quality"*), `temperature: None`, `llm_max_retries: None`, `results_dir`, `data_cache_dir`, `memory_log_path`, `memory_log_max_entries`, plus per-provider reasoning knobs (`google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`) forwarded by `_get_provider_kwargs()`.

The **`online_tools` boolean toggle from v0.1.0 is gone**, replaced by a vendor registry:

```python
"data_vendors": {"core_stock_apis": "yfinance", "technical_indicators": "yfinance",
                 "fundamental_data": "yfinance", "news_data": "yfinance",
                 "macro_data": "fred", "prediction_markets": "polymarket"},
"tool_vendors": {},   # tool-level override, takes precedence over category
```
with the explicit contract: *"The configured value is the exact vendor chain — requests are NOT silently routed to vendors you didn't choose."*

**Env overrides with type-driven coercion that fails loudly:**

```python
_ENV_OVERRIDES = {"TRADINGAGENTS_LLM_PROVIDER": "llm_provider",
                  "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "max_debate_rounds", ...}

def _coerce(value, reference):
    if isinstance(reference, bool):
        ...
        raise ValueError(f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}")
    if isinstance(reference, int) and not isinstance(reference, bool): return int(value)
    ...
```
Docstring: *"a misspelled boolean (e.g. `treu`) or non-numeric int should fail loudly at startup, not quietly misconfigure an unattended run."*

**Checkpointing** (`graph/checkpointer.py`) — per-ticker SQLite `SqliteSaver`, and the idea I most want us to steal, the **run signature folded into the thread id**:

```python
def _run_signature(self, asset_type: str) -> str:
    return "|".join(["analysts=" + ",".join(self.selected_analysts),
                     f"debate={self.config['max_debate_rounds']}",
                     f"risk={self.config['max_risk_discuss_rounds']}",
                     f"asset={asset_type}"])

def thread_id(ticker, date, signature="") -> str:
    base = f"{ticker.upper()}:{date}"
    if signature: base = f"{base}:{signature}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]
```
So a resume under a *different graph shape* cannot silently reuse an incompatible checkpoint (#1089). Checkpoints are cleared on successful completion.

---

## 7. Signal processing / final output extraction

**v0.1.0** burned an LLM call on it:
```python
messages = [("system", "You are an efficient assistant designed to analyze paragraphs or "
    "financial reports provided by a group of analysts. Your task is to extract the investment "
    "decision: SELL, BUY, or HOLD. Provide only the extracted decision ... without adding any "
    "additional text or information."), ("human", full_signal)]
return self.quick_thinking_llm.invoke(messages).content
```

**HEAD is deterministic.** `graph/signal_processing.py` now just delegates to a regex parser, keeping `SignalProcessor.process_signal(text)` only for backward compatibility. The module docstring states the principle outright: *"The deterministic heuristic ... is more than sufficient to extract that rating; no extra LLM call is needed."*

```python
RATINGS_5_TIER = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)

def parse_rating(text: str, default: str = "Hold") -> str:
    # pass 1: explicit "Rating: X" label (tolerates markdown bold)
    # pass 2: first 5-tier rating word anywhere in the text
    # else: default
```

This works *because the producer is schema-constrained*: `render_pm_decision` always emits `**Rating**: X` as line 1. Producer schema + deterministic parser beats consumer-side LLM extraction.

---

## 8. Genuinely good and reusable for our query-mining agent team

1. **Debate-then-judge for genuinely contested judgment calls.** The bull/bear + facilitator shape maps cleanly onto our real fork points: P2 taxonomy design (competing taxonomy proposals), P5 K selection (stability-peak advocate vs. overclustering-survival advocate vs. expert-intuition advocate — a natural 3-way like their risk trio), P6 merge/split adjudication, P8 governance merges. Use it where the answer is a *judgment*, never where it is a *number*.
2. **The judge must be forced to commit.** Their prompt fights the model's hedging bias explicitly: *"Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced."* Our K-selection and merge referees need exactly this anti-hedging clause, otherwise every LLM referee returns "both proposals have merit."
3. **Two-tier model routing, with deep reserved for synthesis only.** Cheap model for the many extraction/analysis nodes, expensive model for the two nodes that actually decide. For us: quick model for the 5-9 P2 research agents, per-cluster P7 naming agents, and the P1 regex-family summarizers; deep model for the taxonomy referee, the K-selection judge, and the P7 tree-audit agent.
4. **Outcome-grounded deferred reflection.** This is the strongest idea in the repo and it survived the memory rewrite while everything else was deleted. The reflection is not "what do you think you did wrong" — it is *"here is the realized number, now write 2-4 sentences."* Our analogue: after P9 produces the metrics panel, feed `(taxonomy_decision, silhouette/ARI delta, kappa achieved, % coverage, drift)` back into a decision log. The two-phase split (store pending at run end → resolve with real outcome at next run start) fits our P12 quarterly rerun *perfectly*.
5. **Bounded, terse memory entries.** *"Your output will be stored verbatim in a decision log and re-read by future analysts, so every word must earn its place."* Plus rotation that prunes resolved entries but never pending ones.
6. **Schema + render pair (dual representation).** One LLM call yields both a typed object (for metrics, dashboards, downstream agents) and human-readable markdown (for P11 reports). Field descriptions carry the output instructions. This is exactly what P7 naming and P2 label schemas need.
7. **Deterministic parse over LLM extraction** when you control the producer's schema — kills a whole class of nondeterminism in P11.
8. **Declarative node-spec table** (`ANALYST_NODE_SPECS` + `build_analyst_execution_plan`) instead of f-string node names. Node names, tool-node names, clear-node names and state keys all live in one frozen dataclass registry. Our 12 phases should have exactly this.
9. **Total routers + shared complete path maps + a drift test.** Copy `tests/test_risk_router_path_map.py` wholesale in spirit: parametrize garbage/renamed/i18n router returns and assert routability.
10. **Checkpoint thread id = hash(item + date + run signature).** Config change ⇒ new thread ⇒ no silent stale resume. Directly applicable to P0 reproducibility.
11. **Graceful structured-output degradation** (`bind_structured` returning `None`, one-shot fallback to free text) so a weak/local provider never hard-blocks a pipeline.
12. **`write_report_tree(final_state, ticker, save_path)`** in `tradingagents/reporting.py`: numbered per-stage directories (`1_analysts/`, `2_research/`, ...) plus a consolidated `complete_report.md`, shared by CLI and programmatic API so headless runs produce identical artifacts. That's our P11 in one function.
13. **`create_msg_delete()` context hygiene** — wipe the shared message channel between phases and replace it with a task-anchored placeholder.
14. **Pre-fetch instead of tool-call when the data set is known.** The Sentiment Analyst was rewritten to pre-fetch all three sources into the prompt and use no tools at all, because *"the old version had a prompt that demanded social-media analysis but the only tool available was Yahoo Finance news — which led LLMs to fabricate Reddit/X/StockTwits content under prompt pressure (verified live)."* For P7 blind naming, pre-fetching the deterministic sample of member queries (and binding **no** tools) is the structural guarantee of anti-anchoring — far stronger than instructing the agent not to look.
15. **Identity anchoring.** `resolve_instrument_context()` does one deterministic lookup at run start and threads the result into every prompt so agents cannot hallucinate what they're analyzing (#814). Our equivalent: resolve dataset id, snapshot hash, date range, row count once and inject into every node.

## What is NOT applicable

- **The whole risk-management team** has no analogue. Don't force one.
- **Single-item state.** `AgentState` carries one ticker and four prose reports. Our state must carry *artifact references* (parquet paths, embedding matrix path, cluster-assignment path, content hashes) — never the data. Their design does not scale to a million queries and shouldn't be copied.
- **Sequential analyst chaining.** Our P2 research agents and P7 naming agents should genuinely fan out in parallel; theirs cannot, because they share one `messages` channel.
- **No map-reduce.** There is no per-cluster / per-item iteration anywhere. P6/P7 need a real map step over K clusters; nothing here helps.
- **Debate over statistics.** Their agents never compute a metric inside the graph. Our P9 uniform metrics panel must be plain code with a fixed seed and sample — never an LLM debate.
- **No human-in-the-loop gates at all.** `propagate()` runs start-to-finish. We need `interrupt()` before P2 gold-label sign-off, P5 K selection, and P8 governance merges.
- **No inter-annotator agreement concept.** Nothing maps to Cohen's kappa ≥ 0.9; their "2 annotators + referee" analogue (bull/bear/judge) has no measurement of agreement, only a winner.

---

## 9. Weaknesses and anti-patterns to avoid

1. **Sequential analysts.** Latency is the *sum* of four ReAct loops when it should be the max. Cause: the shared `messages` channel forces the `Msg Clear` hack. Fix for us: give each parallel node its own private state key, fan out from a dispatcher, fan in at a barrier.
2. **Unbounded string concatenation for debate history.** `history + "\n" + argument`, forever, with no token budget, no summarization, no truncation. At `max_debate_rounds=1` it never bites; at 3+ rounds with 3 speakers it will. Use a reducer with a windowing/summarizing policy.
3. **Hand-copied state dicts.** Every node rebuilds the whole `InvestDebateState` / `RiskDebateState` and must remember to copy every sibling field. This produced a real data-loss bug (#503). Use LangGraph channel reducers (`Annotated[list, operator.add]`, or a custom merge) so nodes return **only their delta**.
4. **Manually incremented counters.** `"count": investment_debate_state["count"] + 1` in each participant. Make the counter a reducer-managed channel.
5. **Partial `path_map` on a shared router** — the original bug. Routers must be total, and `add_conditional_edges` must be given the router's full range on every edge.
6. **Stale comment vs. code** (`>= 2 * max_debate_rounds` documented as "3 rounds") — surviving since v0.1.0. In a reproducibility-critical pipeline this is a real hazard.
7. **Termination by round count only.** No convergence check, no early exit on agreement, no "the two sides now agree" detector. Cheap and predictable, but it burns a full round even when the debate settled in turn one, and caps depth even when the sides are far apart. We should add a convergence/agreement gate on top of the round cap.
8. **`max_debate_rounds: 1` default means "debate" is one turn each.** Marketing outruns behavior. Set our defaults to what we actually mean.
9. **Manual, commented-out reflection.** The v0.1 learning loop never ran for most users. Any feedback loop we build must be *automatic and inside the main entry point*, never a method the caller is expected to remember.
10. **In-memory vector store presented as long-term memory.** `chromadb.Client(...)` instead of `PersistentClient` — wiped on exit (#563). If we use a vector store, persistence must be tested, not assumed.
11. **Empty-memory hallucination.** A prompt slot that says "Here are your past reflections on mistakes: `""`" invites fabrication. The fix is structural: **omit the slot entirely** when there is nothing to put in it.
12. **Semantic retrieval keyed on a giant concatenated blob.** The v0.1 retrieval key was all four reports concatenated and embedded with `text-embedding-ada-002` — a key so diffuse that nearest-neighbour lookup is close to noise. They ultimately replaced retrieval-by-similarity with retrieval-by-recency-and-identity, which is a genuine finding: for episodic decision memory over a small corpus, **recency + exact key match beat embeddings**, at zero cost and full determinism. Don't reach for a vector DB by reflex in P12.
13. **Memory read surface shrank to one node.** Only the Portfolio Manager sees `past_context` now. Safer, but the researchers lost all learning. If we adopt the log, decide deliberately which roles read it.
14. **Prompts as inline f-strings with no registry or version.** No prompt hashing, no version pinning, no way to attribute an output change to a prompt change. For P0/P11 reproducibility we need a versioned prompt registry with hashes recorded in the run manifest.
15. **No determinism story.** `temperature: None` by default and the README concedes no setting makes output bit-identical. Our P0 demands seeds and manifests; expect to record *inputs, prompt hashes, and model ids* rather than hope for identical outputs, and keep all statistical decisions in seeded code rather than in the LLM.
16. **No cost/token budget enforcement in the graph.** `recursion_limit: 100` is the only backstop; a stuck ReAct loop can burn 100 node executions. Add a per-phase token/call budget.
17. **Analyst report is only committed when `tool_calls == 0`.** If a model never stops calling tools, the report silently stays `""` and the downstream debate proceeds on empty input. Needs an explicit max-tool-iterations per analyst with a forced-summarize fallback.
18. **Routing on prose prefixes** (`current_response.startswith("Bull")`, `latest_speaker.startswith("Aggressive")`). Control flow keyed on the first word of an LLM string, which breaks under renaming or translation — exactly the drift the path-map fix defends against. Route on an explicit enum field in state, not on prose.


---

## Recommendations carried into the design

- Steal the debate-then-judge shape only for genuinely contested judgment calls (P2 taxonomy design, P5 K selection as a natural 3-way, P6 merge/split, P8 governance) and never for statistical questions, which must stay in seeded code.
- Copy the outcome-grounded deferred reflection loop verbatim in structure: store a pending decision entry at run end, and at the start of the next run resolve it with the real measured outcome (kappa, ARI/silhouette delta, coverage, drift) before generating a terse 2-4 sentence lesson.
- Adopt the two-phase memory contract over a vector store: an append-only log with a hard delimiter, atomic tmp+replace writes, an idempotency guard, and rotation that prunes resolved but never pending entries — their own history shows recency+exact-key beat embeddings here.
- Make the 'no memory' case structurally impossible to hallucinate by omitting the lessons slot from the prompt entirely when the log is empty, rather than injecting an empty string.
- Route two model tiers the way the code does, not the way the paper claims: quick model for the many analysis/naming/extraction nodes, deep model reserved for referee and audit nodes (P2 taxonomy referee, P5 K judge, P7 tree-audit).
- Give every LLM judge an explicit anti-hedging clause that forbids defaulting to the neutral option unless evidence is genuinely balanced, mirroring their 'reserve Hold for genuinely balanced' prompt line.
- Replace their sequential analyst chain with true parallel fan-out by giving each concurrent node a private state key instead of sharing one messages channel, then fan in at a barrier node.
- Use LangGraph channel reducers for debate history and round counters so nodes return only their delta, avoiding the hand-copied-state data-loss bug (#503) their design still carries.
- Make every conditional router total and give all edges the same complete path map, then add a drift test that parametrizes empty, renamed, and translated router returns and asserts routability.
- Route control flow on an explicit enum field in state rather than on prose prefixes like startswith('Bull') — LLM-string-keyed routing breaks under renaming, i18n, and refactors.
- Fold graph-shape-affecting config (phase selection, debate rounds, alpha, K range, seed) into a hashed run signature that keys the checkpoint thread id, so a config change can never silently resume an incompatible checkpoint.
- Pair every decision schema with a render-to-markdown function so one LLM call yields both a typed artifact for metrics and human-readable prose for P11 reports, with Pydantic field descriptions carrying the output instructions.
- Extract final decisions with a deterministic parser over a schema-constrained producer instead of a second LLM call, and add nullish-string coercion validators for optional numeric fields.
- Enforce P7 anti-anchoring structurally rather than by instruction: pre-fetch the deterministic query sample into the naming prompt and bind zero tools, the same fix they applied after LLMs fabricated social data under prompt pressure.
- Add convergence detection on top of the round cap so a debate can exit early on agreement, and set debate-round defaults to what you actually mean — their default of 1 yields a single turn per side.
- Build a versioned prompt registry with hashes recorded in the run manifest, since their inline f-string prompts make it impossible to attribute an output change to a prompt change.
- Add explicit human-in-the-loop interrupts before gold-label sign-off (P2), K selection (P5), and governance merges (P8) — their pipeline has none and runs start-to-finish.
- Copy their reporting.write_report_tree pattern of numbered per-stage directories plus a consolidated report, shared by both CLI and programmatic entry points so headless runs produce identical artifacts.
- Keep pipeline state to artifact references (paths, content hashes, row counts) rather than payloads, since their prose-in-state design cannot scale to million-row query logs.
- Add per-phase token and tool-call budgets plus a forced-summarize fallback, since their only backstop is recursion_limit=100 and an analyst that never stops calling tools silently emits an empty report.

## Unverified or version-dependent

- The repo at HEAD (v0.3.1, files dated 2026-07-18) is substantially evolved beyond the arXiv paper and beyond the v0.1.x code the assignment described; several assignment premises no longer hold (FinancialSituationMemory/ChromaDB and get_memories are deleted, create_xxx_analyst no longer takes a toolkit, online_tools is gone, agents/utils/agent_utils.py no longer defines a Toolkit class). I documented both generations, but any recommendation tied to 'the current repo' should be re-checked against whatever commit we actually vendor.
- I could not diff every intermediate release; the BM25 memory generation (v0.2.x, between ChromaDB and the markdown log) is described only from CHANGELOG entries, not from its source, so my characterization of why BM25 specifically failed is inferred rather than verified.
- The paper states all analyst nodes use deep-thinking models while setup.py assigns quick_thinking_llm to every node except Research Manager and Portfolio Manager. I verified the code at both v0.1.0 and HEAD; I could not determine whether the paper describes an unreleased configuration or is simply inaccurate.
- The paper's reported results (26.62% CR, 8.21 Sharpe on AAPL over Jan 1 - Mar 29 2024) come from a ~3-month window on three tickers with gpt-4o-mini/o1-preview era models. I have not verified these independently and they are not reproducible from the current codebase; treat the performance claims as unvalidated for our purposes, though the architecture lessons stand on their own.
- Exact debate-round semantics under max_debate_rounds > 1 are inferred from reading the counter arithmetic and router, not from executing the graph; I did not run the pipeline (it needs API keys and network data vendors).
- Whether LangGraph's current version supports the parallel fan-out pattern I recommend with the exact ergonomics I describe is not verified against langgraph docs — pyproject pins langgraph>=0.4.8 but I did not read the LangGraph API reference, so the fan-out/barrier implementation details should be confirmed before we design around them.
- Library version numbers I cite come from the repo's pyproject.toml dependency floors (langchain-core>=0.3.81, langgraph>=0.4.8, langgraph-checkpoint-sqlite>=2.0.0, pydantic via langchain), not from checking the current releases of those packages.

## Sources

- https://github.com/TauricResearch/TradingAgents
- https://github.com/TauricResearch/TradingAgents/archive/refs/heads/main.zip
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/agents/utils/memory.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.1/tradingagents/agents/utils/memory.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/graph/trading_graph.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/graph/reflection.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/graph/signal_processing.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/graph/conditional_logic.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/graph/setup.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/agents/researchers/bull_researcher.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/agents/managers/research_manager.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/agents/trader/trader.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/agents/analysts/market_analyst.py
- https://raw.githubusercontent.com/TauricResearch/TradingAgents/v0.1.0/tradingagents/default_config.py
- https://arxiv.org/abs/2412.20138
- https://arxiv.org/pdf/2412.20138