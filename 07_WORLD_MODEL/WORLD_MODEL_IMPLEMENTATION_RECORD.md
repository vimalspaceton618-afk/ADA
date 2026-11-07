# World Model Implementation Record

## Scope
Research-grounded extension of the symbolic world model merged in PR #4. Adds optional learned model components and support contracts. The extension remains separate from the ADA language-model decoder.

## Modules
- environment_model/rssm.py: stochastic/deterministic recurrent latent dynamics.
- environment_model/model.py: feature prediction, reward and continuation heads, balanced KL loss.
- predictive_engine/jepa.py: action-conditioned temporal predictor, target detach and EMA.
- predictive_engine/cem.py: bounded CEM action-sequence optimizer.
- simulation_engine/rollouts.py: branchable latent imagination.
- temporal_model/trajectory.py: episode-aligned data and windowing.
- model_validation/metrics.py: prediction and calibration metrics.
- causal_model/graph.py: cycle-checked dependency metadata.
- physical_model/contracts.py: SI physical observations with provenance.
- human_behavior_model/: empirical action frequencies only.
- internal_state/: model-state reference contract.

## Research choices
- DreamerV3 RSSM mechanics are used as a compact recurrent-latent baseline.
- V-JEPA 2-AC motivates action-conditioned representation prediction; the implementation operates on pooled vectors and is not the original ViT predictor.
- TD-MPC2 motivates latent rollout optimization; a generic CEM planner is provided, not the full TD-MPC2 algorithm.
- NVIDIA Cosmos remains an external deployment target, not a local dependency.

## Safety and provenance
- Imagined states are not promoted to observed WorldFact records.
- Planning returns candidate action tensors only; it never controls external devices.
- No code runs generated model output as Python.
- Human behavior module is descriptive and aggregate, not a hidden-intent or personality estimator.
- Confidence scores are not claimed to be calibrated probabilities.

## Verification
CI runs symbolic tests in the lightweight Python matrix and an isolated neural test job with CPU PyTorch. External benchmark and robot evaluation remain future empirical work.
