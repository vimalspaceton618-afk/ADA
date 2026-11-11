"""Stable import facade for ADA's infrastructure and governance subsystems."""

from importlib import import_module

_MODULES = {
    "safety": "15_SAFETY_AND_GOVERNANCE",
    "data": "16_DATA_INFRASTRUCTURE",
    "training": "17_TRAINING",
    "inference": "18_INFERENCE",
    "interaction": "19_INTERACTION",
    "distributed": "20_DISTRIBUTED_INFRASTRUCTURE",
    "observability": "21_OBSERVABILITY",
    "interoperability": "22_INTEROPERABILITY",
    "deployment": "23_DEPLOYMENT",
    "testing": "24_TESTING",
    "research": "25_RESEARCH",
    "documentation": "26_DOCUMENTATION",
    "operations": "27_OPERATIONS",
}

_CHILDREN = {
    "safety": ("action_approval", "audit_logs", "data_protection", "model_integrity",
               "permission_manager", "policy_engine", "sandbox", "security_monitor"),
    "data": ("data_pipeline", "data_quality", "dataset_versioning", "evaluation_corpus",
             "processed_datasets", "raw_datasets", "synthetic_data", "training_corpus", "validation_corpus"),
    "training": ("checkpointing", "continual_training", "distributed_training", "neural_training",
                 "optimizer", "post_training", "pretraining", "representation_training", "training_monitor"),
    "inference": ("adaptive_computation", "batching", "dynamic_routing", "inference_engine",
                  "inference_monitor", "memory_access", "model_loader"),
    "interaction": ("conversation", "feedback_interface", "interaction_history",
                    "multimodal_interface", "user_context"),
    "distributed": ("accelerator_manager", "cluster_manager", "fault_tolerance", "network_transport",
                    "node_manager", "storage_manager", "synchronization", "workload_distribution"),
    "observability": ("diagnostics", "logging", "metrics", "performance_analysis", "telemetry", "tracing"),
    "interoperability": ("module_contracts", "plugin_interfaces", "protocol_definitions", "public_api", "serialization"),
    "deployment": ("cloud", "cluster", "configuration", "containerization", "local", "release_management"),
    "testing": ("integration", "reliability", "reproducibility", "safety", "stress", "system", "unit"),
    "research": ("ablation_studies", "architecture_experiments", "experiment_results", "hypotheses",
                 "literature_review", "mathematical_models", "research_publications"),
    "documentation": ("api_reference", "architecture", "decisions", "developer_guides",
                      "research_notes", "subsystem_specs"),
    "operations": ("backup_recovery", "incident_management", "lifecycle_manager", "maintenance", "system_updates"),
}

__all__ = []
for _alias, _path in _MODULES.items():
    _module = import_module(_path)
    globals()[_alias] = _module
    __all__.append(_alias)
    for _name in getattr(_module, "__all__", ()):
        if _name not in globals():
            globals()[_name] = getattr(_module, _name)
        if _name not in __all__:
            __all__.append(_name)
    for _child in _CHILDREN[_alias]:
        _child_module = import_module(f"{_path}.{_child}")
        setattr(_module, _child, _child_module)
        _export_name = f"{_alias}_{_child}"
        globals()[_export_name] = _child_module
        __all__.append(_export_name)
