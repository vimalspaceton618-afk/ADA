"""Safe arithmetic primitives without eval or dynamic code execution."""
from __future__ import annotations
import math
import operator

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul,
        "/": operator.truediv, "**": operator.pow}

def calculate(left: float, op: str, right: float) -> float:
    if op not in _OPS:
        raise ValueError("unsupported operator")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (left, right)):
        raise TypeError("operands must be real numbers")
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        raise ValueError("operands must be finite")
    if op == "/" and right == 0:
        raise ZeroDivisionError("division by zero")
    if op == "**" and abs(right) > 1000:
        raise ValueError("exponent magnitude exceeds safety bound")
    result = _OPS[op](left, right)
    if not math.isfinite(float(result)):
        raise ArithmeticError("result is not finite")
    return result
