"""FinanceReasoning's native metric, ported from ``utils/evaluation_utils.py``.

The metric itself is one line:

.. code-block:: python

    def within_eps(pred, gt):
        eps = abs(gt) * 0.002
        return pred >= gt - eps and pred <= gt + eps

A **0.2 % relative tolerance**, inclusive on both sides. Note it is relative to the *gold*, so a
gold of exactly ``0`` admits only an exact ``0``.

Almost all the work is in getting a number back out of what the model wrote. The official
``normalize()`` strips currency words and symbols, takes the right-hand side of ``=`` and ``≈``,
strips ``%``, maps ``true/yes -> True`` and ``false/no -> False``, and peels units off either end
of the number.

**One deliberate deviation, and it is a security one.** The official ``normalize()`` finishes with a
bare ``eval(prediction)`` on model output. That is arbitrary code execution on a string a language
model produced; a benchmark that runs untrusted output through ``eval`` is one adversarial dataset
away from a compromised machine. Here that step is a **safe AST evaluator** instead: numeric
literals and ``+ - * / ** //`` and unary sign, nothing else — no names, no calls, no attributes,
no subscripts. It covers the cases ``eval`` actually served (``"1/3"``, ``"10**2"``, ``"-5"``) and
refuses the rest. ``tests/parity/test_finance_reasoning_parity.py`` runs both against the same
predictions and asserts they agree; where they cannot, the divergence is recorded rather than
hidden. See ``docs/research/metric_parity.md``.
"""

from __future__ import annotations

import ast
import re

from financebench.evaluation.metrics.base import Metric, register_metric
from financebench.schemas.metric import MetricResult
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "FinanceReasoningAccuracy",
    "get_acc",
    "normalize",
    "safe_eval_number",
    "within_eps",
]

#: The official relative tolerance: eps = |gt| * 0.002.
RELATIVE_TOLERANCE = 0.002


def within_eps(pred: float, gt: float) -> bool:
    """The official comparison. Inclusive on both sides; relative to the *gold*."""
    eps = abs(gt) * RELATIVE_TOLERANCE
    return gt - eps <= pred <= gt + eps


_NUMBER_RE = re.compile(r"^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$")
_SCIENTIFIC_RE = re.compile(r"^[-+]?\d+(\.\d+)?e[-]?\d+$", re.IGNORECASE)


def _is_number(text: str) -> bool:
    return bool(_NUMBER_RE.match(text))


def _is_scientific(text: str) -> bool:
    return bool(_SCIENTIFIC_RE.match(text))


# --------------------------------------------------------------------------- safe arithmetic

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def safe_eval_number(text: str) -> float | int | bool | tuple[object, ...] | None:
    """Evaluate a *numeric expression* safely. Returns ``None`` for anything that isn't one.

    This replaces the official ``eval(prediction)``. Only numeric literals, tuples of them, and the
    arithmetic operators are permitted — a name, a call, an attribute access or a subscript is
    rejected outright, so a model's output can never execute anything.

    **Tuples are permitted for one specific and unhappy reason: parity.** Python parses
    ``"1,234.56"`` as the *tuple* ``(1, 234.56)``, not as a comma-formatted number — and the
    official evaluator, which calls ``eval`` here, therefore turns a model's ``"1,234.56"`` into
    ``(1, 234.56)`` and then (in ``get_acc``) takes element ``[0]``, scoring it as **1**.

    That is an upstream bug, and it is tempting to "fix" it. We do not, because a native metric
    that disagrees with the official one is not a native metric — its numbers would not be
    comparable to any published FinanceReasoning result, which is the entire reason for
    implementing it. The quirk is reproduced exactly, recorded in docs/research/metric_parity.md,
    and pinned by tests/parity/test_finance_reasoning_parity.py.

    In practice it rarely bites here: the structured prompt asks for a bare ``numeric_value``, and
    :class:`FinanceReasoningAccuracy` prefers that float over the model's prose, so ``normalize``
    is only reached as a fallback.
    """
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    def check(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return check(node.body)
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int | float | bool)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, _ALLOWED_UNARYOPS) and check(node.operand)
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, _ALLOWED_BINOPS) and check(node.left) and check(node.right)
        if isinstance(node, ast.Tuple):  # "1,234.56" -> (1, 234.56); see the docstring
            return all(check(element) for element in node.elts)
        return False

    if not check(tree):
        return None
    try:
        # Safe by construction: the AST has been proven to contain nothing but numeric literals,
        # tuples of them, and arithmetic operators. There is no name, call, attribute or subscript
        # left for anything to execute.
        result = eval(compile(tree, "<safe>", "eval"), {"__builtins__": {}}, {})
    except (ArithmeticError, ValueError, TypeError):
        return None
    if isinstance(result, int | float | bool | tuple):
        return result
    return None


# --------------------------------------------------------------------------- normalization

_CURRENCY_WORDS = ("£", "€", "¥", "million", "billion", "thousand", "us", "usd", "rmb")


def normalize(prediction: str) -> float | int | bool | tuple[object, ...] | None:
    """Port of the official ``normalize``, minus the ``eval`` hole (see the module docstring)."""
    direct = safe_eval_number(prediction)
    if direct is not None:
        return direct

    text = prediction.strip().lower().rstrip(".")
    for word in _CURRENCY_WORDS:
        text = text.replace(word, "")

    if "=" in text:
        text = text.split("=")[-1].strip()
    if "≈" in text:
        text = text.split("≈")[-1].strip()
    text = text.replace("`", "").replace("%", "").replace("$", "").replace("°", "")

    if text in ("true", "yes", "false", "no"):
        return text in ("true", "yes")
    if "true" in text or "false" in text:
        return "true" in text

    if "approximately" in text:
        text = text.replace("approximately", "").strip()
    if " or " in text:
        text = text.split(" or ")[0]

    # Peel a unit off either end of the number: "42 kg", "kg 42", "42%", "$42".
    for pattern, group in (
        (r"[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$", r"([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$"),
        (r"[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$", r"[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$"),
        (r"[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$", r"([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$"),
        (r"[^-+\d]{1,2}(?:[\d,]*\.*\d+)$", r"[^-+\d]{1,2}((?:[\d,]*\.*\d+))$"),
    ):
        if re.match(pattern, text):
            match = re.search(group, text)
            if match:
                text = match.group(1)
            break

    if " x " in text:
        text = text.replace(" x ", "*")
    if " × " in text:
        text = text.replace(" × ", "*")
    if _is_number(text):
        text = text.replace(",", "")
    if not text:
        return None

    if _is_scientific(text):
        try:
            return float(text)
        except ValueError:
            return None
    return safe_eval_number(text)


def get_acc(prediction: object, gt: float | int | bool) -> int:
    """The official per-sample score: 1 or 0."""
    try:
        if isinstance(prediction, str):
            prediction = normalize(prediction)
        if isinstance(prediction, tuple | list):
            prediction = prediction[0] if prediction else None
        if prediction is None:
            return 0
        if isinstance(gt, bool) or isinstance(prediction, bool):
            return int(prediction == gt)
        if not isinstance(prediction, int | float):
            return 0
        return int(within_eps(float(prediction), float(gt)))
    except (TypeError, ValueError, ArithmeticError):
        return 0


# --------------------------------------------------------------------------- metric


@register_metric("finance_reasoning_accuracy")
class FinanceReasoningAccuracy(Metric):
    """FinanceReasoning's **official** numeric accuracy: within 0.2 % of the gold, relative."""

    name = "finance_reasoning_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        response = prediction.response
        if response is None or response.financial_answer is None:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=False,
                passed=False,
                details={"reason": "no parsed answer"},
            )
        answer = response.financial_answer

        gold_raw = sample.metadata.get("ground_truth", "")
        gold: float | bool
        if gold_raw.lower() in ("true", "false"):
            gold = gold_raw.lower() == "true"
        else:
            try:
                gold = float(gold_raw)
            except ValueError:
                return MetricResult(
                    sample_id=sample.sample_id,
                    metric_name=self.name,
                    value=False,
                    passed=False,
                    details={"reason": f"unparseable gold {gold_raw!r}"},
                )

        # Prefer the model's own structured numeric_value; fall back to normalizing its prose.
        predicted: object = (
            answer.numeric_value if answer.numeric_value is not None else answer.answer
        )
        correct = bool(get_acc(predicted, gold))
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=correct,
            passed=correct,
            details={
                "predicted": predicted
                if isinstance(predicted, float | int | bool)
                else str(predicted)[:120],
                "gold": gold,
                "level": sample.metadata.get("level", ""),
                "tolerance": f"{RELATIVE_TOLERANCE:.1%} relative",
            },
        )
