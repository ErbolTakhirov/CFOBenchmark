"""FinQA's native evaluator, ported faithfully from ``code/evaluate/evaluate.py``.

FinQA defines **two** native metrics, and both of them take a *predicted program*, not a
free-text answer:

- **execution accuracy** — tokenize the program, execute it against the table, round to 5
  decimals, and compare to the gold ``exe_ans`` with a plain ``==``. No tolerance.
- **program accuracy** — ``equal_program``: build a symbol map from the *gold* program's operands,
  reject any prediction that uses an operand the gold never used, expand both programs into infix
  expressions over those symbols, and compare them under sympy's ``simplify``. So a
  differently-shaped but algebraically identical program is correct, and a string comparison would
  have been wrong.

Because both are defined over programs, they are only computed for runs that used a
program-eliciting prompt profile (``program_v1``). A run that asked the model for a number has no
program to score, and this module reports *nothing* for these metrics rather than a fabricated
zero — see :func:`~financebench.evaluation.benchmark_metrics.metrics_for_run`.

For direct-answer runs there is :class:`FinQAAnswerAccuracy`, which is **ours, not FinQA's**: it
compares the model's stated final number to ``exe_ans`` within a small tolerance. It is a useful,
honest metric for direct-answer prompting — it is simply not the official one, and it is named so
that it can never be mistaken for it. (An earlier revision of this file *did* conflate the two,
reporting a tolerance-based free-text comparison under the name ``finqa_execution_accuracy``.)

Parity against the real official code is asserted in ``tests/parity/test_finqa_parity.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sympy import simplify

from financebench.evaluation.metrics.base import Metric, register_metric
from financebench.evaluation.numeric import numeric_match, parse_numeric_answer
from financebench.prompts.profiles import create_prompt_profile
from financebench.schemas.metric import MetricResult
from financebench.schemas.model_io import FinancialAnswer
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "NA",
    "FinQAAnswerAccuracy",
    "FinQAExecutionAccuracy",
    "FinQAProgramAccuracy",
    "equal_program",
    "eval_program",
    "execute_program",
    "extract_program",
    "program_tokenization",
    "str_to_num_finqa",
]

#: FinQA's own sentinel for "could not compute". Kept as the literal string the official evaluator
#: uses, so ported comparisons behave identically.
NA = "n/a"

ALL_OPS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
)
_ARITHMETIC_OPS = frozenset({"add", "subtract", "multiply", "divide", "exp", "greater"})


def str_to_num_finqa(text: str) -> float | str:
    """Port of FinQA's ``str_to_num``. Returns :data:`NA` (the string ``"n/a"``), not ``None``,
    on failure — the official code compares against that sentinel and so must we."""
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        pass
    if "%" in text:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            return NA
    if "const" in text:
        literal = text.replace("const_", "")
        if literal == "m1":
            literal = "-1"
        try:
            return float(literal)
        except ValueError:
            return NA
    return NA


def _process_row(row: Sequence[str]) -> list[float] | str:
    """Port of FinQA's ``process_row``. Negatives are encoded as ``-32 ( 32 )``, hence the split."""
    out: list[float] = []
    for cell in row:
        cleaned = cell.replace("$", "").strip().split("(")[0].strip()
        number = str_to_num_finqa(cleaned)
        if isinstance(number, str):
            return NA
        out.append(number)
    return out


def program_tokenization(original_program: str) -> list[str]:
    """Port of FinQA's ``program_tokenization``.

    ``"subtract(5829, 5735)"`` becomes ``["subtract(", "5829", "5735", ")", "EOF"]`` — four tokens
    per step plus a trailing ``EOF``, which is the shape the structure check downstream relies on.
    """
    tokens: list[str] = []
    for chunk in original_program.split(", "):
        current = ""
        for char in chunk:
            if char == ")" and current != "":
                tokens.append(current)
                current = ""
            current += char
            if char in "()":
                tokens.append(current)
                current = ""
        if current != "":
            tokens.append(current)
    tokens.append("EOF")
    return tokens


def eval_program(
    program: Sequence[str], table: Sequence[Sequence[str]] | None
) -> tuple[int, float | str]:
    """Port of FinQA's ``eval_program``: returns ``(invalid_flag, result)``.

    ``program`` is a *tokenized* program (as produced by :func:`program_tokenization`), ending in
    ``EOF``. ``result`` is a float rounded to 5 decimals, the string ``"yes"``/``"no"`` for a final
    ``greater``, or :data:`NA`. A malformed program returns ``(1, "n/a")`` rather than raising —
    matching the official code, which wraps the whole body in a bare ``except``.
    """
    invalid_flag = 0
    this_res: float | str = NA

    try:
        tokens = list(program)[:-1]  # drop EOF

        # Structure check: every 4th token is an op, every 4th+1 is a closing paren.
        for index, token in enumerate(tokens):
            if index % 4 == 0 and token.strip("(") not in ALL_OPS:
                return 1, NA
            if (index + 1) % 4 == 0 and token != ")":
                return 1, NA

        joined = "|".join(tokens)
        steps = joined.split(")")[:-1]
        res_dict: dict[int, float | str] = {}
        table_dict = {row[0]: row[1:] for row in table} if table else {}

        for index, raw_step in enumerate(steps):
            step = raw_step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1_raw, arg2_raw = args.split("|")[0].strip(), args.split("|")[1].strip()

            if op in _ARITHMETIC_OPS:
                arg1 = (
                    res_dict[int(arg1_raw.replace("#", ""))]
                    if "#" in arg1_raw
                    else str_to_num_finqa(arg1_raw)
                )
                if isinstance(arg1, str):
                    invalid_flag = 1
                    break
                arg2 = (
                    res_dict[int(arg2_raw.replace("#", ""))]
                    if "#" in arg2_raw
                    else str_to_num_finqa(arg2_raw)
                )
                if isinstance(arg2, str):
                    invalid_flag = 1
                    break

                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2  # ZeroDivisionError -> caught below, invalid
                elif op == "exp":
                    this_res = arg1**arg2
                else:  # greater
                    this_res = "yes" if arg1 > arg2 else "no"
                res_dict[index] = this_res

            else:  # table_max / table_min / table_sum / table_average
                # A table op takes a ROW LABEL, never a #N back-reference. The official code
                # would hit an unbound `num_row` here and land in its bare `except`; we reach the
                # same outcome directly. (Recorded in docs/research/metric_parity.md.)
                if "#" in arg1_raw or arg1_raw not in table_dict:
                    invalid_flag = 1
                    break
                num_row = _process_row(table_dict[arg1_raw])
                if isinstance(num_row, str):
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                else:  # table_average
                    this_res = sum(num_row) / len(num_row)
                res_dict[index] = this_res

        if this_res not in ("yes", "no", NA):
            this_res = round(float(this_res), 5)

    except (
        Exception
    ):  # the official evaluator's bare `except` — a bad program is invalid, not a crash
        invalid_flag = 1

    return invalid_flag, this_res


def execute_program(
    program: str, table: Sequence[Sequence[str]] | None = None
) -> float | str | None:
    """Convenience wrapper: execute a program *string*, returning ``None`` if it is invalid.

    Used by the dataset adapter's own tests to verify every gold program reproduces its recorded
    ``exe_ans``.
    """
    invalid, result = eval_program(program_tokenization(program), table)
    if invalid or result == NA:
        return None
    return result


def _symbol_recur(step: str, step_dict: Mapping[int, str], sym_map: Mapping[str, str]) -> str:
    """Port of the official ``symbol_recur``: expand a step into an infix expression string."""
    step = step.strip()
    op = step.split("(")[0].strip("|").strip()
    args = step.split("(")[1].strip("|").strip()
    arg1, arg2 = args.split("|")[0].strip(), args.split("|")[1].strip()

    if "table" in op:
        return sym_map[step]

    arg1_part = (
        _symbol_recur(step_dict[int(arg1.replace("#", ""))], step_dict, sym_map)
        if "#" in arg1
        else sym_map[arg1]
    )
    arg2_part = (
        _symbol_recur(step_dict[int(arg2.replace("#", ""))], step_dict, sym_map)
        if "#" in arg2
        else sym_map[arg2]
    )

    infix = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "exp": "**",
        "greater": ">",
    }[op]
    return f"( {arg1_part} {infix} {arg2_part} )"


def equal_program(gold: Sequence[str], pred: Sequence[str]) -> bool:
    """Port of FinQA's ``equal_program`` — **symbolic** program equivalence, not string equality.

    Both arguments are tokenized programs. The symbol map is built from the *gold* program, which
    is what makes step 3 below bite: the model may not invent an operand the gold never used, even
    a numerically correct one.

    1. Map each gold operand (and each whole ``table_*`` step) to a symbol ``a0``, ``a1``, …
    2. Structurally validate the prediction (op/paren positions; no forward ``#N`` references).
    3. Reject any prediction using an operand absent from the gold's symbol map.
    4. Expand both to infix over the symbols, ``simplify`` both, compare.
    """
    sym_map: dict[str, str] = {}

    gold_tokens = list(gold)[:-1]  # drop EOF
    gold_joined = "|".join(gold_tokens)
    gold_steps = gold_joined.split(")")[:-1]

    sym_index = 0
    step_dict_gold: dict[int, str] = {}
    for index, raw_step in enumerate(gold_steps):
        step = raw_step.strip()
        if len(step.split("(")) > 2:
            return False
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1, arg2 = args.split("|")[0].strip(), args.split("|")[1].strip()
        step_dict_gold[index] = step

        if "table" in op:
            if step not in sym_map:
                sym_map[step] = f"a{sym_index}"
                sym_index += 1
        else:
            for arg in (arg1, arg2):
                if "#" not in arg and arg not in sym_map:
                    sym_map[arg] = f"a{sym_index}"
                    sym_index += 1

    step_dict_pred: dict[int, str] = {}
    try:
        pred_tokens = list(pred)[:-1]  # drop EOF
        for index, token in enumerate(pred_tokens):
            if index % 4 == 0 and token.strip("(") not in ALL_OPS:
                return False
            if (index + 1) % 4 == 0 and token != ")":
                return False

        pred_joined = "|".join(pred_tokens)
        pred_steps = pred_joined.split(")")[:-1]

        for index, raw_step in enumerate(pred_steps):
            step = raw_step.strip()
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1, arg2 = args.split("|")[0].strip(), args.split("|")[1].strip()
            step_dict_pred[index] = step

            if "table" in op:
                if step not in sym_map:
                    return False
            else:
                for arg in (arg1, arg2):
                    if "#" not in arg:
                        if arg not in sym_map:
                            return False
                    elif int(arg.strip("#")) >= index:
                        return False
    except Exception:
        return False

    try:
        sym_gold = simplify(_symbol_recur(gold_steps[-1], step_dict_gold, sym_map), evaluate=False)
        sym_pred = simplify(_symbol_recur(pred_steps[-1], step_dict_pred, sym_map), evaluate=False)
    except Exception:
        return False

    return bool(sym_gold == sym_pred)


# --------------------------------------------------------------------------- model-output parsing

_PROGRAM_CHARS = set("0123456789.,()#_ abcdefghijklmnopqrstuvwxyz")


def extract_program(text: str) -> str | None:
    """Pull a FinQA program out of a model's raw output.

    **Ours, not FinQA's** — the official evaluator is handed an already-clean program by the model
    code that produced it, and has no parsing step at all. Real models wrap things in prose and
    markdown, so a run would otherwise score zero for reasons that have nothing to do with
    financial reasoning. Deliberately conservative: it looks for a line that is *entirely* a
    program, and gives up rather than guessing.
    """
    if not text:
        return None
    candidate = text.strip()
    if "```" in candidate:
        blocks = candidate.split("```")
        # Prefer the content of the first fenced block, minus any language tag.
        if len(blocks) >= 2:
            inner = blocks[1]
            if "\n" in inner:
                first, rest = inner.split("\n", 1)
                inner = rest if first.strip().isalpha() else inner
            candidate = inner.strip()

    for line in reversed([line.strip() for line in candidate.splitlines() if line.strip()]):
        lowered = line.lower()
        if "(" not in lowered or ")" not in lowered:
            continue
        if not set(lowered) <= _PROGRAM_CHARS:
            continue
        if not any(lowered.startswith(op) for op in ALL_OPS):
            continue
        return line
    return None


def _asked_for_a_program(prediction: Prediction) -> bool:
    """Did the run that produced this prediction actually ask the model for a program?

    ``prompt_version`` on the request *is* the prompt-profile name (profiles are versioned in
    their names), so the metric can tell without the run config being threaded through.
    """
    try:
        return create_prompt_profile(prediction.request.prompt_version).elicits_program
    except Exception:
        return False


def _no_answer(sample: CanonicalSample, metric: str, reason: str) -> MetricResult:
    return MetricResult(
        sample_id=sample.sample_id,
        metric_name=metric,
        value=False,
        passed=False,
        details={"reason": reason},
    )


def _looks_like_percent(answer: FinancialAnswer) -> bool:
    """Did the model present its number as a percentage?"""
    unit = (answer.unit or "").strip().lower()
    if unit in {"%", "percent", "percentage", "pct", "percentage_points"}:
        return True
    return "%" in (answer.answer or "")


def _table(sample: CanonicalSample) -> list[list[str]]:
    if not sample.context.tables:
        return []
    return [list(row) for row in sample.context.tables[0].rows]


# --------------------------------------------------------------------------- metrics


@register_metric("finqa_execution_accuracy")
class FinQAExecutionAccuracy(Metric):
    """FinQA's **official** execution accuracy: execute the predicted program, compare to
    ``exe_ans`` with a strict ``==`` after rounding to 5 decimals.

    Only meaningful for program-eliciting runs. Parity-tested.
    """

    name = "finqa_execution_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        if not _asked_for_a_program(prediction):
            return _no_answer(
                sample,
                self.name,
                "run did not use a program-eliciting prompt profile; "
                "official execution accuracy is undefined for a free-text answer",
            )
        response = prediction.response
        if response is None:
            return _no_answer(sample, self.name, "no response")

        program = extract_program(response.content)
        if program is None:
            return _no_answer(sample, self.name, "no program found in model output")

        invalid, result = eval_program(program_tokenization(program), _table(sample))
        gold = (
            sample.gold.numeric_value
            if sample.gold.numeric_value is not None
            else (sample.gold.answer.strip().casefold())
        )
        # The official comparison, verbatim: exact equality, no tolerance.
        is_match = invalid == 0 and result == gold
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=is_match,
            passed=is_match,
            details={
                "predicted_program": program,
                "executed": result,
                "gold": gold,
                "invalid_program": bool(invalid),
            },
        )


@register_metric("finqa_program_accuracy")
class FinQAProgramAccuracy(Metric):
    """FinQA's **official** program accuracy: sympy symbolic equivalence against the gold program.

    Only meaningful for program-eliciting runs. Parity-tested.
    """

    name = "finqa_program_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        if not _asked_for_a_program(prediction):
            return _no_answer(
                sample,
                self.name,
                "run did not use a program-eliciting prompt profile; there is no program to score",
            )
        response = prediction.response
        if response is None:
            return _no_answer(sample, self.name, "no response")
        if not sample.gold.program:
            return _no_answer(sample, self.name, "sample has no gold program")

        program = extract_program(response.content)
        if program is None:
            return _no_answer(sample, self.name, "no program found in model output")

        is_match = equal_program(
            program_tokenization(sample.gold.program), program_tokenization(program)
        )
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=is_match,
            passed=is_match,
            details={"predicted_program": program, "gold_program": sample.gold.program},
        )


@register_metric("finqa_answer_accuracy")
class FinQAAnswerAccuracy(Metric):
    """**Not FinQA's metric.** Compares the model's stated final number to ``exe_ans`` within a
    small tolerance.

    FinQA's official evaluator has no defined behaviour for a free-text answer — it only ever
    grades programs. So for direct-answer prompting this platform needs a metric of its own, and
    says so in the name. Numbers from it must never be compared to published FinQA leaderboard
    figures.
    """

    name = "finqa_answer_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        response = prediction.response
        if response is None or response.financial_answer is None:
            return _no_answer(sample, self.name, "no parsed answer")
        answer = response.financial_answer

        # FinQA's `greater` op yields yes/no, not a number: gold.numeric_value is None for exactly
        # those samples, so they are graded by string comparison instead.
        if sample.gold.numeric_value is None:
            predicted = answer.answer.strip().casefold()
            gold = sample.gold.answer.strip().casefold()
            is_match = predicted == gold
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=is_match,
                passed=is_match,
                details={"predicted": predicted, "gold": gold, "mode": "boolean"},
            )

        predicted_value = answer.numeric_value
        if predicted_value is None:
            parsed = parse_numeric_answer(answer.answer)
            predicted_value = parsed.resolved_value if parsed is not None else None
        if predicted_value is None:
            return _no_answer(sample, self.name, "predicted answer has no extractable number")

        gold_value: float = sample.gold.numeric_value
        is_match = numeric_match(predicted_value, gold_value, absolute_tolerance=1e-3)
        mode = "numeric"

        # FinQA's exe_ans for a percentage question is the raw division result — a FRACTION
        # (0.16417), not a percentage (16.417). A model asked for a percentage answers "16.4%",
        # which is *correct*, and would otherwise be scored as wrong by a factor of 100. So a
        # percent-flagged answer is also compared against the gold read as a percentage.
        #
        # This cannot rescue a genuinely wrong answer: 41% against a gold of 14.46% still fails,
        # because 0.41 != 0.1446 either way. It reconciles a unit convention, nothing more.
        if (
            not is_match
            and _looks_like_percent(answer)
            and numeric_match(predicted_value / 100.0, gold_value, absolute_tolerance=1e-3)
        ):
            is_match = True
            mode = "numeric_percent_reconciled"

        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=is_match,
            passed=is_match,
            details={"predicted": predicted_value, "gold": gold_value, "mode": mode},
        )
