"""Safe arithmetic primitives; no eval or dynamic execution."""
import math, operator
_OPS={"+":operator.add,"-":operator.sub,"*":operator.mul,"/":operator.truediv,"**":operator.pow}
def calculate(left,op,right):
    if op not in _OPS: raise ValueError("unsupported operator")
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in (left,right)): raise TypeError("operands must be real numbers")
    if not math.isfinite(float(left)) or not math.isfinite(float(right)): raise ValueError("operands must be finite")
    if op=="/" and right==0: raise ZeroDivisionError("division by zero")
    if op=="**" and abs(right)>1000: raise ValueError("exponent magnitude exceeds safety bound")
    result=_OPS[op](left,right)
    if not math.isfinite(float(result)): raise ArithmeticError("result is not finite")
    return result
