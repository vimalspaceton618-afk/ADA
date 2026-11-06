# ADA Memory

05_MEMORY/ is ADA's memory boundary between representation and downstream cognition. The stable application import is `ada.memory`.

## Architecture

ADA uses two complementary memory planes.

### 1. Neural associative memory

- `neural/energy_memory.py`: trainable Energy-Based Global Memory from the ADA source research.
- Default memory bank: 512 learnable slots of width `dim`, initialized with normal standard deviation 0.02.
- Associative retrieval: `scores = X @ M.T`, `attention = softmax(scores / temperature)`, `raw = attention @ M`.
- Feature-wise gate: `gate = sigmoid(W_mem_gate(X))`; output: `gate * raw`.
- Input/output contract: `[batch, sequence, dim]`.
- `return_aux=True` returns output, attention, raw retrieval, and gate tensors for observability and training diagnostics.

This module is the differentiable model component. The memory bank is trained by the optimizer through gradients. The source material does not specify a separate online write/erase operation, so this implementation does not invent one.

### 2. Provenance-aware record memory

- `contracts.py`: immutable records, namespaces, memory kinds, lifecycle, provenance and query contracts.
- `store.py`: bounded thread-safe process-local store with optimistic versions.
- `retrieval.py`: deterministic lexical retrieval baseline.
- `engine.py`: record creation, representation ingestion, retrieval, archive and deletion.

This is a state-management plane, not the neural memory bank. It is currently volatile and process-local.

## Quick start: neural memory

Install the optional neural dependency:

```bash
pip install -e ".[neural]"
```

Use the associative module:

```python
import torch
from ada.memory import EnergyMemory

memory = EnergyMemory(dim=768)  # ADA source default: 512 memory slots
x = torch.randn(2, 128, 768)
result = memory(x, return_aux=True)

y = result.output             # [2, 128, 768]
weights = result.attention    # [2, 128, 512]
```

The module is intentionally independent of a specific transformer/MoE block. The model layer that owns the FFN decides where the memory output is fused into its hidden state.

## Quick start: provenance records

```python
from ada.memory import MemoryEngine, MemoryKind, MemorySourceKind

memory = MemoryEngine()
record = memory.remember(
    {"text": "ADA received a verified source observation."},
    kind=MemoryKind.EPISODIC,
    source_kind=MemorySourceKind.USER,
    source_id="session-001",
    source_digest="a" * 64,  # Replace with the actual upstream SHA-256.
    namespace="research",
    tags=("ada", "observation"),
)
matches = memory.retrieve("verified source observation")
```

Representation handoff preserves the source content SHA-256 and representation metadata:

```python
from ada.perception import Modality, PerceptionEngine, PerceptionInput
from ada.representation import RepresentationEngine
from ada.memory import MemoryEngine

observed = PerceptionEngine().perceive(
    PerceptionInput(modality=Modality.TEXT, content="A source-grounded statement.")
)
projected = RepresentationEngine().transform(observed)
memory = MemoryEngine()
for item in projected.representations:
    memory.remember_representation(item, namespace="default")
```

## Integrated neural substrate block

`09_NEURAL_SUBSTRATE/expert_systems/moe_energy_memory.py` implements the PDF's integrated FFN path. It dispatches each token to one SwiGLU expert using noisy top-1 routing and a straight-through selector, computes the 512-slot Energy Memory branch in parallel, learns a per-token MoE/memory blend, and returns the residual hidden state plus auxiliary diagnostics.

```python
import torch
from ada.neural import MoEEnergyMemoryFFN

ffn = MoEEnergyMemoryFFN(d_model=768, num_experts=8, num_slots=512)
hidden, aux = ffn(torch.randn(2, 128, 768))
loss = language_model_loss + 0.01 * aux["load_balancing_loss"]
```

The block already includes its residual connection; a parent block must not add the same residual again. The caller must include `load_balancing_loss` in its training objective. The repository still does not contain the sequence mixer, full decoder/model, or a complete pretraining loop, so this is an integrated FFN/MoE component—not a complete ADA language model.

## Research references and design boundaries

- ADA source research (`Ada-src.pdf`): 512-slot associative Energy Memory, temperature-scaled softmax retrieval, feature-wise sigmoid gate.
- [Tencent Hy-Memory](https://github.com/Tencent-Hunyuan/Hy-Memory) and [Hy-Memory-30B-A3B](https://huggingface.co/tencent/Hy-Memory-30B-A3B): useful research on learned memory writing, state revision and cross-harness memory policies. The 30B-A3B writer is not a runtime dependency and is not copied into this repository.
- [R^3Mem](https://huggingface.co/papers/2502.15957): research direction for reversible context compression and hierarchical memory tokens; not implemented in this milestone.
- [S^3-Attention](https://huggingface.co/papers/2601.17702): research direction for memory-bounded long-context retrieval; not implemented in this milestone.
- [SYNAPSE](https://huggingface.co/papers/2601.02744): graph/spreading-activation direction for future associative episodic-semantic retrieval; not implemented in this milestone.

## Invariants and current limits

- Record memory carries source kind, source ID and lowercase SHA-256 source digest.
- Record content and metadata are finite JSON-compatible values and deeply immutable.
- Namespace isolation is applied to record listing and retrieval.
- Record retrieval excludes expired and deleted records; archived records require explicit opt-in.
- Optimistic version checks are available for record updates.
- Deleted records are terminal tombstones.
- No automatic fact inference, summarization, external persistence or semantic retrieval is claimed.
- The neural bank is a learned model parameter; record-store contents do not automatically modify the neural bank.

## Deferred work

Durable database adapters, vector/ANN search, learned memory writing, conflict-aware consolidation, graph construction, spreading activation, episodic reconstruction, procedural execution, memory decay, and end-to-end MoE/FFN fusion require separate implementations and tests. These are not implied by the neural read path or the current record store.

No mandatory third-party dependency is introduced. PyTorch remains optional under the existing `neural` extra.
