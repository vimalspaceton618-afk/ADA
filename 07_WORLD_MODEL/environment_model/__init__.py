"""Learned environment dynamics (requires the optional PyTorch extra)."""
__all__ = ["RSSMState", "RSSMSequence", "RecurrentStateSpaceModel", "RSSMWorldModel"]
def __getattr__(name):
    if name in {"RSSMState", "RSSMSequence", "RecurrentStateSpaceModel"}:
        from . import rssm
        return getattr(rssm, name)
    if name == "RSSMWorldModel":
        from .model import RSSMWorldModel
        return RSSMWorldModel
    raise AttributeError(name)
