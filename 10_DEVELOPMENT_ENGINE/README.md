# ADA Development Engine

Controlled proposal and artifact lifecycle service. It records capability descriptors, validates module dependency graphs, accepts typed declarative change proposals, validates candidates, requires an explicit approval record for promotion, and supports compare-and-swap rollback.

It does **not** execute generated Python/shell, mutate production weights, install plugins, rewrite a live architecture, or deploy modules. The operation vocabulary is intentionally closed (set/remove/append_unique); candidate proposals remain data until an isolated implementation worker is supplied.

| Area | Current reference capability | Not yet implemented |
|---|---|---|
| architecture_search | proposal interface | NAS/search algorithms |
| architecture_versioning | artifact digest lineage | persistent signed registry |
| capability_detection | versioned capability cards | runtime hardware probing |
| dependency_analysis | DAG validation/order | dynamic resolver |
| development_validation | injected validator reports | hermetic compiler/test worker |
| module_composition | dependency metadata | live composition |
| module_generation | declarative operations | code-generation executor |
| rollback_engine | compare-and-swap previous digest | distributed deployment rollback |
| skill_development | versioned capability records | model skill training |
| structural_expansion | bounded proposal records | online graph/weight mutation |

Lifecycle: PROPOSED -> VALIDATED or REJECTED -> PROMOTED -> ROLLED_BACK. Parent digest must remain active at promotion. Approval identity authentication is delegated to a trusted external identity system.

Research comparisons: [SWE-agent](https://github.com/SWE-agent/SWE-agent) separates patches from execution/evaluation; [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) demonstrate persisted human-approval boundaries. ADA's [WSPPI contract](../00_FOUNDATION/architecture_contracts/ARCHITECTURE_CONTRACTS.md) additionally requires immutable production artifacts, isolated transformation workers and independent evaluation.

Limitations: the injected validator is not a secure sandbox; records are in-memory; approval metadata is not proof of human identity.
