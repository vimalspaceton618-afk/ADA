# ADA Reasoning and Planning

This subsystem accepts immutable declarative plans and verifies dependency graphs, capability availability and deterministic execution order. Goals are explicit caller-supplied objectives; selection is limited to explicit scored alternatives and can abstain.

It does not infer hidden user intent, call an LLM, perform natural-language decomposition, execute plans, predict outcomes or certify factual correctness.

The design separates planning from acting: [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) illustrate persisted human-in-the-loop orchestration, while ADA's framework-independent contracts leave durable checkpoints to a runtime adapter. The [World Model CEM planner](../07_WORLD_MODEL/README.md) proposes candidate sequences but never executes them.

Limitations: plan verification proves graph/capability structure only, not task success, safety, resource feasibility or truth of preconditions.
