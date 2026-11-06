"""Small explicit propositional-logic utilities."""
def modus_ponens(antecedent, implication):
    if type(antecedent) is not bool or type(implication) is not bool: raise TypeError("premises must be booleans")
    return True if antecedent and implication else None
def conjunction(*premises):
    if any(type(x) is not bool for x in premises): raise TypeError("premises must be booleans")
    return all(premises)
