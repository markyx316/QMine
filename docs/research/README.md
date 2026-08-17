# Research dossiers

Gathered 2026-08-17 by a nine-agent research fan-out before any code was written.
Each dossier was produced by an agent that fetched live documentation and, where
possible, verified API signatures against packages installed in a scratch venv.
The final dossier is an adversarial completeness pass over the other eight.

| dossier | what it settled |
|---|---|
| [LangGraph Core API (v1.2.11)](langgraph-core.md) | exact v1.x signatures, reducer semantics, the `InvalidUpdateError` trap on parallel writes |
| [Multi-Agent Orchestration Patterns](langgraph-multiagent.md) | supervisor vs swarm vs orchestrator-worker; how to keep fan-out workers context-isolated |
| [Memory Management & Context Engineering](langgraph-memory.md) | the three memory tiers and the write/select/compress/isolate framing |
| [TradingAgents Architecture Teardown](tradingagents.md) | two-tier model routing, debate topology, and the reflection-memory loop we borrowed |
| [Agent, Tool & Skill Design Best Practices](agent-best-practices.md) | tool design, Agent Skills format, structured-output repair, LLM-as-judge design |
| [Production Observability, Evaluation & Cost Control](production-ops.md) | tracing, fake-model testing, retry/rate-limit policy, batch API for bulk labelling |
| [Clustering & Embedding Stack (2026)](ml-stack.md) | sklearn 1.9 / sentence-transformers 5.x API reality, stability protocols, HDBSCAN caveats |
| [LLM Taxonomy Induction, Labeling & Naming Methods](taxonomy-llm-methods.md) | TnT-LLM, ClusterLLM, Dial-In LLM procedures; kappa pitfalls; anti-anchoring protocols |
| [Completeness critic](00-completeness-critic.md) | contradictions between dossiers and re-verified corrections |
