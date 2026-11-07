"""Action-conditioned latent prediction and trajectory optimization."""
__all__ = ["ActionConditionedJEPA", "EMATargetEncoder", "update_ema", "CEMPlanner", "CEMPlan"]
def __getattr__(name):
    if name in {"ActionConditionedJEPA", "EMATargetEncoder", "update_ema"}:
        from . import jepa
        return getattr(jepa, name)
    if name in {"CEMPlanner", "CEMPlan"}:
        from . import cem
        return getattr(cem, name)
    raise AttributeError(name)
