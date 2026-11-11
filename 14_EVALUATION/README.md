# ADA Evaluation

Evaluation reports bind a subject artifact digest to a dataset digest, evaluator identity, metrics, sample count, verdict and limitations. A policy gate returns PASS, FAIL or INCONCLUSIVE; missing required metrics, too few samples, missing suite output or an evaluator exception cannot become a pass.

Protocol: (1) version/hash the dataset and evaluator; (2) use held-out cases with contamination controls; (3) measure actual task success; (4) capture latency/throughput/memory/energy where measurable; (5) add robustness, regression, adversarial and safety suites; (6) retain report digests and limitations; (7) require all release-critical suites. A scalar model-judge score does not replace executable checks.

Research: [NVIDIA NeMo Agent Toolkit evaluation](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html) evaluates workflows on curated data and quantitative quality/reliability/latency metrics; [SWE-bench](https://swe-agent.com/0.7/usage/benchmarking/) validates repository coding tasks through executable tests; [NIST AI RMF GenAI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) informs governance, provenance and pre-deployment testing.

Metric families: capability/task success, reasoning/checker outcomes, calibration/abstention, tool authorization, knowledge retrieval/citation correctness, efficiency, reliability, safety and privacy. The repository supplies contracts and gates, not empirical benchmark results. PASS applies only to exact subject and dataset digests, thresholds and sample count in the report.
