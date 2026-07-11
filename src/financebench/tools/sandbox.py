"""The arithmetic sandbox. An allow-list over a parsed AST — never ``eval``.

A model under evaluation is an **untrusted input source**. It is not an attacker in the usual sense —
it has no goals — but it produces text, that text becomes an expression, and something has to run it.
The distinction between "a model that hallucinated a weird string" and "an attacker" disappears
entirely at the point where you call ``eval()`` on it.

And a financial agent is *worse* than the usual case, because the data it reads is also untrusted. An
invoice description, a transaction memo, a supplier's line item — all of it is written by somebody
else, and all of it flows into the model's context. If a ledger row can reach an evaluator, the
attacker is whoever can add a row to the ledger.

So this module never evaluates a string. It parses it to an AST and then walks that tree, refusing
anything not on a list of things a calculator needs. The list is short, and it is the whole security
model:

- numbers, and the four operations, plus ``**``, ``%``, unary minus, and parentheses;
- a handful of named functions (``abs``, ``round``, ``min``, ``max``, ``sum``);
- nothing else. **No names. No attributes. No calls to anything not on the list. No subscripts. No
  comprehensions. No lambdas. No strings.**

That is not a blacklist of dangerous things — a blacklist is a list of the attacks you thought of.
``__import__``, ``os.system``, ``().__class__.__bases__``, ``getattr(__builtins__, ...)`` and every
other sandbox escape ever written all share one property: they need a **Name** or an **Attribute**
node. Neither is on the allow-list, so none of them parse, and the ones nobody has invented yet will
not parse either.

Arithmetic is ``Decimal``, not ``float``. This is a financial tool: ``0.1 + 0.2 != 0.3`` is a bug
report waiting to happen, and a benchmark that grades a model on a number its own calculator got
subtly wrong is grading its own rounding error.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from enum import StrEnum
from typing import Any

__all__ = [
    "MAX_EXPRESSION_CHARS",
    "RefusalKind",
    "SandboxError",
    "SandboxResult",
    "safe_eval",
]

#: A model that emits a 100 kB expression is not calculating; it is looping. Bounded before parsing,
#: because `ast.parse` on a pathological input is itself the denial of service.
MAX_EXPRESSION_CHARS = 500

#: `2 ** 999999999` is a two-token denial of service: it parses instantly, allocates forever. There is
#: no legitimate financial calculation with an exponent above this, so the exponent is bounded rather
#: than the runtime — a timeout would already be too late, because the allocation happens first.
MAX_EXPONENT = 64

#: Decimal precision. Generous for money; finite, so a repeating decimal terminates.
_PRECISION = 40

#: Nesting depth. Guards against a recursive-descent stack overflow from `((((((...))))))`.
MAX_DEPTH = 20


class RefusalKind(StrEnum):
    """*Why* the sandbox said no. The distinction is load-bearing, not decorative.

    A model that writes ``1/0`` made an arithmetic mistake. A model that writes ``__import__('os')``
    tried to execute code. Recording both as "the sandbox refused something" would bury the security
    signal in a pile of division errors and make it unfindable — which is exactly what the first
    version of this did, until a test caught it.
    """

    #: Not arithmetic at all: a name, an attribute, a string, a lambda. **This is the security event.**
    DISALLOWED_CONSTRUCT = "disallowed_construct"
    #: Arithmetic, but unbounded: a colossal exponent, a 100 kB expression, 200 levels of nesting.
    #: Also an abuse of the sandbox, and also worth knowing about.
    RESOURCE_LIMIT = "resource_limit"
    #: Ordinary arithmetic that cannot be done. Division by zero is a mistake, not an attack.
    ARITHMETIC = "arithmetic"
    #: The model emitted something that is not an expression at all.
    SYNTAX = "syntax"


class SandboxError(Exception):
    """The expression was refused. **This is a successful outcome**, not a bug.

    A refusal is the sandbox working. It is recorded as a *tool error* the model can see and recover
    from, never as a crash of the harness — and a model that gets refused and then adapts is
    demonstrating exactly the behaviour the tool benchmark is trying to measure.
    """

    def __init__(self, message: str, kind: RefusalKind = RefusalKind.DISALLOWED_CONSTRUCT) -> None:
        super().__init__(message)
        self.kind = kind

    @property
    def is_security_event(self) -> bool:
        """Did the model try to make the sandbox do something that is not arithmetic?

        Division by zero is not an attack. ``__import__`` is. A metric that cannot tell them apart
        reports a confused model as a hostile one, and buries the real signal.
        """
        return self.kind in (RefusalKind.DISALLOWED_CONSTRUCT, RefusalKind.RESOURCE_LIMIT)


@dataclass(frozen=True)
class SandboxResult:
    value: Decimal
    expression: str


#: The only functions that exist. Note what is absent: no `open`, no `getattr`, no `eval`, no
#: `__import__` — not because they are blocked, but because names are not resolvable at all.
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)
_UNARYOPS = (ast.UAdd, ast.USub)


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):  # bool is an int; a boolean is not a quantity of money
        raise SandboxError("booleans are not numbers")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    raise SandboxError(f"not a number: {type(value).__name__}")


def _eval_node(node: ast.AST, depth: int) -> Decimal:
    if depth > MAX_DEPTH:
        raise SandboxError(
            f"expression nested deeper than {MAX_DEPTH} levels", RefusalKind.RESOURCE_LIMIT
        )

    # -- a literal number
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            # A string constant is refused. It is the only way to smuggle a payload into an
            # arithmetic sandbox, and a calculator has no use for one.
            raise SandboxError(
                f"only numeric literals are allowed, got {type(node.value).__name__}"
            )
        return _to_decimal(node.value)

    # -- a + b, a * b, a ** b ...
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINOPS):
            raise SandboxError(f"operator not allowed: {type(node.op).__name__}")
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)

        if isinstance(node.op, ast.Pow):
            # Bounded BEFORE the operation: `2 ** 999999999` allocates before any timeout fires.
            if abs(right) > MAX_EXPONENT:
                raise SandboxError(
                    f"exponent {right} exceeds the limit of {MAX_EXPONENT}",
                    RefusalKind.RESOURCE_LIMIT,
                )
            return left**right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise SandboxError("division by zero", RefusalKind.ARITHMETIC)
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise SandboxError("division by zero", RefusalKind.ARITHMETIC)
            return left // right
        if right == 0:
            raise SandboxError("modulo by zero", RefusalKind.ARITHMETIC)
        return left % right

    # -- -a
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARYOPS):
            raise SandboxError(f"unary operator not allowed: {type(node.op).__name__}")
        operand = _eval_node(node.operand, depth + 1)
        return -operand if isinstance(node.op, ast.USub) else operand

    # -- abs(x), round(x, 2), min(a, b) ... and NOTHING else.
    if isinstance(node, ast.Call):
        # `func` must be a bare Name that is a key of _FUNCTIONS. An Attribute (`os.system`) never
        # gets here, because Attribute is not handled at all — it falls through to the refusal below.
        if not isinstance(node.func, ast.Name):
            raise SandboxError("only direct calls to allowed functions are permitted")
        if node.func.id not in _FUNCTIONS:
            raise SandboxError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise SandboxError("keyword arguments are not allowed")
        args: list[Any] = [_eval_node(a, depth + 1) for a in node.args]

        # `round(x, 2)` is the single most common operation a financial model will ask for, and it
        # raises TypeError if the digit count arrives as a Decimal — Python's round() requires a real
        # int there. Every argument in this sandbox is a Decimal by construction, so without this the
        # most-used tool call in the benchmark would fail every single time.
        if node.func.id == "round" and len(args) == 2:
            digits = args[1]
            if digits != digits.to_integral_value():
                raise SandboxError(
                    "round(): the number of digits must be a whole number", RefusalKind.ARITHMETIC
                )
            args[1] = int(digits)

        try:
            return _to_decimal(_FUNCTIONS[node.func.id](*args))
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"{node.func.id}(): {exc}", RefusalKind.ARITHMETIC) from exc

    # -- everything else.
    #
    # This is the whole security model. Name, Attribute, Subscript, Lambda, Comprehension, Starred,
    # JoinedStr, Await, Yield — every sandbox escape ever written needs one of these, and none of
    # them are handled. The ones nobody has invented yet need one too.
    raise SandboxError(
        f"{type(node).__name__} is not allowed in a financial expression. "
        "This sandbox evaluates arithmetic and nothing else: no names, no attributes, no imports, "
        "no subscripts, no strings."
    )


def safe_eval(expression: str) -> SandboxResult:
    """Evaluate an arithmetic expression from an untrusted source.

    Raises :class:`SandboxError` on anything that is not arithmetic — which is a *successful*
    outcome, reported back to the model as a tool error it can recover from.
    """
    if not isinstance(expression, str):
        raise SandboxError("expression must be a string", RefusalKind.SYNTAX)

    text = expression.strip()
    if not text:
        raise SandboxError("empty expression", RefusalKind.SYNTAX)
    # Bounded BEFORE parsing: ast.parse on a pathological input is itself the denial of service.
    if len(text) > MAX_EXPRESSION_CHARS:
        raise SandboxError(
            f"expression is {len(text)} characters; the limit is {MAX_EXPRESSION_CHARS}",
            RefusalKind.RESOURCE_LIMIT,
        )

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise SandboxError(f"not a valid expression: {exc.msg}", RefusalKind.SYNTAX) from exc
    except (
        ValueError,
        MemoryError,
        RecursionError,
    ) as exc:  # pragma: no cover — pathological input
        raise SandboxError(f"expression could not be parsed: {exc}", RefusalKind.SYNTAX) from exc

    try:
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            value = _eval_node(tree.body, depth=0)
    except SandboxError:
        raise
    except (InvalidOperation, DivisionByZero, ArithmeticError) as exc:
        raise SandboxError(
            f"arithmetic error: {type(exc).__name__}", RefusalKind.ARITHMETIC
        ) from exc
    except RecursionError as exc:  # pragma: no cover — MAX_DEPTH should catch this first
        raise SandboxError("expression too deeply nested", RefusalKind.RESOURCE_LIMIT) from exc

    return SandboxResult(value=value, expression=text)
