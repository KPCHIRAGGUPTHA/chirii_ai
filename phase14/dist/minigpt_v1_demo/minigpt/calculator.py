"""
Phase 11A — The calculator tool. Parses with `ast` and walks an allow-list: numbers, + - * / // % **, parentheses and
unary minus. Names, calls, attributes, strings and everything else are rejected. Never uses eval or exec.
Exponents must be whole numbers, or exactly 0.5 for a square root (Phase 11C).
"""

import ast
import math
import operator
from dataclasses import dataclass
from typing import Optional, Union

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 12
MAX_MAGNITUDE = 10 ** 30
DECIMAL_PLACES = 6
ERROR_VALUE = "error"

Number = Union[int, float]
_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}


class CalculatorError(Exception):
    """Any expression the calculator refuses or cannot compute."""


@dataclass(frozen=True)
class CalcResult:
    ok: bool
    text: str
    value: Optional[Number] = None
    error: Optional[str] = None


def _checked(value: Number) -> Number:
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
        raise CalculatorError("result out of range")
    return value


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so True and False must be excluded by exact type
        if type(node.value) not in (int, float):
            raise CalculatorError("only numbers are allowed")
        return _checked(node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_evaluate(node.operand)
        raise CalculatorError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if type(right) is float and right == 0.5:
                return _checked(left ** right)
            if type(right) is not int or not -MAX_EXPONENT <= right <= MAX_EXPONENT:
                raise CalculatorError("exponent too large")
            return _checked(left ** right)
        operation = _BINARY.get(type(node.op))
        if operation is None:
            raise CalculatorError("unsupported operator")
        return _checked(operation(left, right))
    raise CalculatorError(f"{type(node).__name__} is not allowed")


def format_value(value: Number) -> str:
    """Whole numbers have no decimal point. Others are rounded to 6 places, without trailing zeros."""
    if isinstance(value, float):
        rounded = round(value, DECIMAL_PLACES)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.{DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    return str(value)


def calculate(expression: str) -> CalcResult:
    """Never raises: any bad input gives ok=False and the text the runtime inserts as the result."""
    if not isinstance(expression, str) or not expression.strip():
        return CalcResult(False, ERROR_VALUE, error="empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        return CalcResult(False, ERROR_VALUE, error="expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _evaluate(tree)
    except ZeroDivisionError:
        return CalcResult(False, ERROR_VALUE, error="division by zero")
    except CalculatorError as error:
        return CalcResult(False, ERROR_VALUE, error=str(error))
    except (SyntaxError, ValueError, TypeError, OverflowError, MemoryError, RecursionError) as error:
        return CalcResult(False, ERROR_VALUE, error=type(error).__name__)
    return CalcResult(True, format_value(value), value=value)
