"""Finite, bounded scalar arithmetic; never evaluates source strings."""
from __future__ import annotations

import math
import operator

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul,
        "/": operator.truediv, "**": operator.pow}
_MAX_ABS_OPERAND = 1e100
_MAX_ABS_RESULT = 1e100


def calculate(left, op, right):
    if op not in _OPS:
        raise ValueError("unsupported operator")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (left, right)):
        raise TypeError("operands must be real numbers")
    try:
        left, right = float(left), float(right)
    except (OverflowError, ValueError) as exc:
        raise ValueError("operands must be finite and bounded") from exc
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("operands must be finite")
    if abs(left) > _MAX_ABS_OPERAND or abs(right) > _MAX_ABS_OPERAND:
        raise ValueError("operand magnitude exceeds safety bound")
    if op == "/" and right == 0:
        raise ZeroDivisionError("division by zero")
    if op == "**" and (abs(right) > 100 or (left == 0 and right < 0)):
        raise ValueError("exponent is outside supported bounds")
    try:
        result = _OPS[op](left, right)
    except (OverflowError, ValueError) as exc:
        raise ArithmeticError("arithmetic operation is outside supported bounds") from exc
    if not math.isfinite(result) or abs(result) > _MAX_ABS_RESULT:
        raise ArithmeticError("result exceeds finite magnitude bound")
    return result


__all__ = ["calculate"]
