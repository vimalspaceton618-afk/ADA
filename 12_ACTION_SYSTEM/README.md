# ADA Action System

Every external operation must pass through ToolGateway. Unregistered tools, unauthorized principals and invalid argument keys are denied. Tool specs identify version, principals, argument allowlists, risk, timeout and approval requirement. Approval binds exact call ID, tool ID, principal and canonical argument digest; tickets and call IDs are single-use. Audit records retain input/output digests and result status.

The implementation is an **in-process reference**, not a secure sandbox. Its timeout is checked after handler return and cannot preempt execution. Production adapters must execute in OS/container/VM/WASM isolation, enforce CPU/memory/time and egress limits, provide cancellation, use scoped credentials, authenticate approvers out-of-band and validate outputs.

Research: [NVIDIA NeMo Agent Toolkit security considerations](https://docs.nvidia.com/nemo/agent-toolkit/latest/resources/security-considerations.html), [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview), and [OpenAI Agents SDK tool boundaries](https://openai.github.io/openai-agents-python/tools/). These emphasize validating tool inputs, isolating credentials and instrumenting actions.

Never expose secrets to model-generated payloads. Tool output is untrusted. Do not equate a structurally matching ApprovalTicket with authenticated human approval.
