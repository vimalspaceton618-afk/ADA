"""Entropy summaries for non-negative candidate score distributions."""
import math
def entropy(scores):
    values=tuple(float(x) for x in scores)
    if not values or any(not math.isfinite(x) or x<0 for x in values): raise ValueError("scores must be finite and non-negative")
    total=sum(values)
    if total<=0: raise ValueError("score sum must be positive")
    return -sum((x/total)*math.log(x/total) for x in values if x>0)
def normalized_entropy(scores):
    values=tuple(scores)
    if len(values)<=1: return 0.0
    return entropy(values)/math.log(len(values))
