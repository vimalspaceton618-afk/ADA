# ADA World Model

07_WORLD_MODEL is ADA's explicit state-and-dynamics subsystem. It is separate from ADA's language-model sequence-mixing backbone and from the deterministic cognition controller.

## What this folder contains

### Symbolic state plane — no PyTorch required
- WorldFact and WorldSnapshot: immutable, provenance-linked, time-aware state assertions.
- WorldModelEngine: monotonic observation ingestion, bounded snapshot history, expiry handling and deterministic rule matching.
- TransitionRule: declarative preconditions/effects for explicitly supplied state transitions.
- WorldPrediction: explicit NO_MODEL, INAPPLICABLE and AMBIGUOUS abstention states.
- The rule-based simulator creates hypothetical snapshots and never modifies observed state.

### Learned latent plane — optional PyTorch
- environment_model/rssm.py: action-conditioned recurrent state-space model with deterministic recurrent state, grouped categorical stochastic latents, learned prior and observation-conditioned posterior.
- environment_model/model.py: feature-vector encoder, RSSM, prior observation predictor, reward head, continuation head and balanced KL / free-nats objective.
- predictive_engine/jepa.py: temporal action-conditioned latent predictor, detached target loss and EMA target-encoder helper.
- predictive_engine/cem.py: bounded Cross-Entropy Method planner over a caller-supplied transition/reward model.
- simulation_engine/rollouts.py: isolated latent rollouts and comparison of candidate action sequences.
- model_validation/metrics.py: prediction error, horizon-wise drift and expected calibration error.
- temporal_model/trajectory.py: validated episode sequences; training windows cannot cross episode boundaries.

### Structured supporting models
- causal_model/graph.py: validated acyclic dependency graphs. Edges are hypotheses/metadata, not proof of causality.
- physical_model/contracts.py: source-linked position, velocity and acceleration observations in SI units. This is not a physics engine.
- human_behavior_model/: descriptive frequencies from explicitly observed action labels. No personality, intent or individual-behavior inference is performed.
- internal_state/contracts.py: versioned runtime handles for model/latent state references, separate from observed facts.

## Research synthesis and implementation boundaries

### DreamerV3
DreamerV3 uses a recurrent state-space model with deterministic and stochastic latent state, posterior inference from observations, prior imagination conditioned on actions and model-based learning from imagined trajectories. ADA implements a compact RSSM pattern and balanced categorical KL with free-nats. It does not reproduce the full DreamerV3 actor-critic, replay, JAX stack, scalar distribution heads or large-scale training recipe.

Paper: https://arxiv.org/abs/2301.04104
Official code: https://github.com/danijar/dreamerv3

### V-JEPA 2 / V-JEPA 2-AC
V-JEPA 2 learns predictive visual representations; V-JEPA 2-AC adds action-conditioned prediction in latent space. ADA provides a vector-level action-conditioned predictor and EMA target helper. It does not reproduce patch-level video ViTs, masking, the original robot action representation, or the pretrained checkpoints.

Paper: https://arxiv.org/abs/2506.09985
Official code: https://github.com/facebookresearch/vjepa2
Hugging Face collection: https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6

### TD-MPC2
TD-MPC2 performs local trajectory optimization in the latent space of an implicit world model. ADA adopts a standalone CEM optimizer that can query a learned transition and reward model. The optimizer returns proposed actions only; it does not execute them, learn a policy or provide a safety certificate.

Paper: https://arxiv.org/abs/2310.16828
Official code: https://github.com/nicklashansen/tdmpc2

### NVIDIA Cosmos
Cosmos is a large multimodal physical-world model family. It should be integrated as an optional, separately provisioned inference adapter where hardware, model access and license conditions permit. ADA does not automatically download Cosmos weights or include Cosmos as a runtime dependency.

Official code: https://github.com/NVIDIA/cosmos
Model collection: https://huggingface.co/collections/nvidia/cosmos3

## Architecture

~~~text
03_PERCEPTION
      |
      v
04_REPRESENTATION  ---- dense vectors / temporal features
      |                                  |
      v                                  v
05_MEMORY                    07_WORLD_MODEL
      |                      +-------------------------------+
      v                      | Symbolic state ledger         |
06_COGNITION                 | World facts / time / evidence |
      |                      +-------------------------------+
      |                      | Learned latent dynamics       |
      |                      | RSSM + action-conditioned JEPA|
      |                      +-------------------------------+
      |                      | Imagined rollouts + CEM       |
      |                      +-------------------------------+
      |                                  |
      +--------------------+-------------+
                           v
                Evaluation / uncertainty
                           |
                           v
                Decision and safety verifier
                           |
                           v
                 Explicit action approval
~~~

The symbolic and learned planes are deliberately separate. A predicted latent state is not silently converted into a verified WorldFact. Observed state changes still require an observation or explicit caller policy.

## Quick start

### Symbolic state

~~~python
from ada.world_model import WorldFact, WorldModelEngine

engine = WorldModelEngine()
fact = WorldFact(
    subject_id="lamp-1",
    predicate="power",
    value="off",
    source_kind="sensor",
    source_id="sensor-17",
    source_digest="a" * 64,  # Replace with the real SHA-256 digest.
)
snapshot = engine.observe([fact])
print(snapshot.digest)
~~~

### Learned feature world model

Install PyTorch only when using learned components:

~~~bash
pip install -e ".[neural]"
~~~

The model consumes feature vectors, not raw video:

~~~python
import torch
from ada.world_model.environment_model import RSSMWorldModel

model = RSSMWorldModel(observation_dim=128, action_dim=7)
observations = torch.randn(4, 9, 128)  # [batch, time, dense observation feature]
actions = torch.randn(4, 8, 7)         # [batch, time-1, action]
rewards = torch.randn(4, 8)
terminated = torch.zeros(4, 8, dtype=torch.bool)

output = model(observations, actions, sample=True)
losses = model.loss(output, rewards=rewards, terminated=terminated)
losses["loss"].backward()
~~~

### Action-conditioned JEPA predictor

~~~python
import torch
from ada.world_model.predictive_engine import ActionConditionedJEPA

predictor = ActionConditionedJEPA(latent_dim=128, action_dim=7)
context = torch.randn(4, 8, 128)
actions = torch.randn(4, 8, 7)
target_future_latents = torch.randn(4, 8, 128)
predictions = predictor(context, actions)
loss = predictor.loss(predictions, target_future_latents)["loss"]
~~~

The future target should come from a frozen encoder or EMA target encoder and must be detached from the predictor's gradient path.

## Sequence and loss conventions

The RSSM model expects:
- observations: [B,T,O], fixed-width numeric features;
- actions: [B,T-1,A], action at t drives the transition from observation t to t+1;
- optional is_first: boolean [B,T] episode reset markers;
- optional rewards and terminated labels: [B,T-1].

At each time step, the prior predicts the next latent state from the previous posterior and action. The next observation embedding corrects that prior into a posterior. The objective combines prior feature prediction, balanced KL and optional reward/continuation supervision.

Do not randomly split adjacent frames into train and test sets. Split by episode, environment, subject/scene where applicable and collection time to avoid leakage.

## Evaluation protocol

A loss scalar alone does not validate a world model. Report:
1. one-step and horizon-wise latent prediction error;
2. reward and termination prediction error;
3. rollout drift, temporal persistence and consistency;
4. uncertainty calibration on held-out episodes;
5. counterfactual/intervention accuracy where ground truth exists;
6. closed-loop control fidelity on a defined simulator;
7. out-of-distribution and failure-case performance.

WBench (2026) evaluates multi-turn world-model behavior across video quality, setting adherence, interaction adherence, consistency and physics compliance. The World Model Benchmarks survey separates capability target, open/closed-loop protocol, prediction/outcome metrics and data source.

WBench: https://arxiv.org/abs/2605.25874
Benchmark survey: https://world-model-benchmarks.github.io/

## Relation to Ada-src.pdf

The reviewed Ada-src.pdf defines ADA's hybrid sequence architecture, associative memory and language-model experiments. It does not specify a dedicated RSSM, JEPA action predictor, learned physical dynamics model or world-model training/evaluation method. The learned dynamics modules here are an adjacent subsystem based on the cited open research; they are not claimed to have originated in the PDF.

## Compute and deployment strategy

- Symbolic state path: regular Python; no GPU.
- RSSM and CEM: optional PyTorch, small CPU-compatible dimensions for tests and prototyping.
- Video prediction: connect a separately installed pretrained feature encoder; no checkpoint is downloaded by this package.
- Cosmos or large JEPA models: run on separately provisioned GPU/accelerator infrastructure or as a remote adapter. Do not load them by default on a laptop.
- CEM: profile candidate count, horizon and rollout latency before increasing search breadth.

## Verification status and limitations

Automated tests exercise contracts, shapes, gradients, finite outputs, episode resets, episode-boundary sampling, calibration metrics, action bounds and simulation isolation. Synthetic tests verify implementation mechanics only; they do not establish predictive accuracy on physical data.

Not yet delivered:
- a trained ADA world-model checkpoint;
- a large visual/video encoder or pretrained V-JEPA integration;
- a Cosmos inference adapter;
- TD-MPC2 policy/value learning and replay;
- a calibrated uncertainty model or certified safety layer;
- a physics simulator or real actuator integration;
- empirical scores on DMC, ManiSkill, LIBERO, WBench or other benchmarks.

No external actions are executed by this subsystem.
