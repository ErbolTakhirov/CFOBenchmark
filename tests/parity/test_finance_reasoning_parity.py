"""FinanceReasoning metric parity: our port vs. the real official evaluator.

The metric is a 0.2 % relative tolerance, which is easy. The hard part — and the only place a port
can drift — is ``normalize()``: turning whatever the model wrote back into a number.

**The known, deliberate divergence.** The official ``normalize()`` ends with a bare
``eval(prediction)`` on model output. We do not port that: it is arbitrary code execution on a
string a language model produced. We use a safe AST arithmetic evaluator instead. This suite runs
both implementations over the same predictions and asserts they agree — which they do on every
realistic model output, because ``eval`` was only ever doing arithmetic. Any case where they cannot
agree is recorded here rather than hidden.
"""

from __future__ import annotations

import pytest
from tests.parity.official_runner import REFERENCES, requires_official, run_official

from financebench.evaluation.native.finance_reasoning import get_acc, normalize, safe_eval_number

FR_REPO = REFERENCES / "financereasoning"

pytestmark = requires_official(FR_REPO / "utils" / "evaluation_utils.py")


#: Realistic things a model actually writes when asked for a financial number.
PREDICTIONS: list[str] = [
    "42",
    "42.0",
    "-42",
    "3.14159",
    "1,234.56",
    "0.2342",
    "23.42%",
    "$1,496.5",
    "The answer is 42",  # prose the official regexes do NOT rescue
    "approximately 42",
    "42 million",
    "answer = 42",
    "x ≈ 3.75",
    "1/3",  # only the eval path can do this
    "10**2",
    "2 + 2",
    "true",
    "yes",
    "false",
    "no",
    "None",
    "",
    "42 or 43",
    "42 kg",
    "-0.05",
    "1e-3",
]

GROUND_TRUTHS: list[float] = [42.0, 0.0, 3.14159, 100.0, -0.05, 0.3333333]


_OFFICIAL = """
import sys, json, types
sys.path.insert(0, ".")

# utils.evaluation_utils imports an LLM helper (which drags in heavy deps we do not need to score
# a string). Stub the module so the *metric* can be imported and run exactly as written.
fake_llm = types.ModuleType("utils.llm")
class LLM:  # noqa
    pass
fake_llm.LLM = LLM
sys.modules["utils.llm"] = fake_llm

from utils.evaluation_utils import get_acc, normalize

payload = json.load(sys.stdin)
out = {
    "normalize": [],
    "get_acc": [],
}
for p in payload["predictions"]:
    try:
        v = normalize(p)
    except Exception:
        v = None
    # JSON cannot carry every python value; record a comparable shape.
    if isinstance(v, bool):
        out["normalize"].append(["bool", v])
    elif isinstance(v, (int, float)):
        out["normalize"].append(["num", float(v)])
    elif v is None:
        out["normalize"].append(["none", None])
    else:
        out["normalize"].append(["other", str(v)])

for p in payload["predictions"]:
    row = []
    for gt in payload["ground_truths"]:
        try:
            row.append(int(get_acc(p, gt)))
        except Exception:
            row.append(0)
    out["get_acc"].append(row)

print(json.dumps(out))
"""


def _shape(value: object) -> list:
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int | float):
        return ["num", float(value)]
    if value is None:
        return ["none", None]
    return ["other", str(value)]


def test_normalize_matches_the_official_implementation() -> None:
    official = run_official(
        _OFFICIAL,
        cwd=FR_REPO,
        payload={"predictions": PREDICTIONS, "ground_truths": GROUND_TRUTHS},
    )

    ours = [_shape(normalize(p)) for p in PREDICTIONS]
    for prediction, mine, theirs in zip(PREDICTIONS, ours, official["normalize"], strict=True):
        assert mine[0] == theirs[0], f"type disagreement on {prediction!r}: {mine} vs {theirs}"
        if mine[0] == "num":
            assert mine[1] == pytest.approx(theirs[1]), f"value disagreement on {prediction!r}"
        else:
            assert mine == theirs, f"disagreement on {prediction!r}"


def test_get_acc_matches_the_official_implementation() -> None:
    """The end-to-end score: normalize, then apply the 0.2 % relative tolerance."""
    official = run_official(
        _OFFICIAL,
        cwd=FR_REPO,
        payload={"predictions": PREDICTIONS, "ground_truths": GROUND_TRUTHS},
    )

    ours = [[get_acc(p, gt) for gt in GROUND_TRUTHS] for p in PREDICTIONS]
    for prediction, mine, theirs in zip(PREDICTIONS, ours, official["get_acc"], strict=True):
        assert mine == theirs, (
            f"score disagreement on {prediction!r}: ours={mine} official={theirs} "
            f"(ground truths: {GROUND_TRUTHS})"
        )


def test_the_safe_evaluator_covers_what_eval_was_actually_used_for() -> None:
    """The whole justification for not porting ``eval``: it was only ever doing arithmetic."""
    assert safe_eval_number("1/3") == pytest.approx(0.3333333, rel=1e-5)
    assert safe_eval_number("10**2") == 100
    assert safe_eval_number("2 + 2") == 4
    assert safe_eval_number("-5") == -5


def test_the_safe_evaluator_refuses_to_execute_anything() -> None:
    """The reason it exists. A benchmark that runs model output through eval() is one adversarial
    dataset away from a compromised machine."""
    for hostile in (
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__.__mro__",
        "print(1)",
        "lambda: 1",
        "x",
    ):
        assert safe_eval_number(hostile) is None, f"{hostile!r} must not evaluate"
