"""Small explicit propositional-logic utilities."""
from __future__ import annotations

def modus_ponens(antecedent: bool, implication: bool):
    """Given P and (P -> Q), return Q; otherwise return None."""
    if type(antecedent) is not bool or type(implication) is not bool:
        raise TypeError("premises must be booleans")
    return True if antecedent and implication else None

def conjunction(*premises: bool) -> bool:
    if any(type(x) is not bool for x in premises):
        raise TypeError("all premises must be booleans")
    return all(premises)
