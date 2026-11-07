"""Hypothetical rollouts; these functions do not execute external actions."""
__all__ = ["LatentRollout", "imagine_rollout", "compare_rollouts"]
def __getattr__(name):
    if name in __all__:
        from . import rollouts
        return getattr(rollouts, name)
    raise AttributeError(name)
