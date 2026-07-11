"""The agent loop, and the failure it exists to catch.

Accuracy can tell you a tool-assisted model got the answer wrong. It cannot tell you WHY, and "bad at
tools" is four different defects with four different fixes:

  - it never reached for a tool          (a planner failure)
  - it reached for one that doesn't exist (a hallucination)
  - right tool, malformed arguments      (a schema failure)
  - right tool, right answer, AND THEN IT IGNORED IT

The last one is why this file exists. A model that calls the calculator, receives 40.55, and writes
"approximately 38" has not made an arithmetic error — it made a TRUST error. Accuracy sees a wrong
number and blames the sums, which is precisely the thing the model got right.
"""

from __future__ import annotations

import json

import pytest

from financebench.models.base import ModelProvider
from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.model_io import ModelRequest, ModelResponse, ModelSpec
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    GoldAnswer,
    SampleContext,
    SourceInfo,
    Table,
)
from financebench.tools.agent import run_agent

pytestmark = pytest.mark.asyncio

MODEL = ModelSpec.parse("mock/scripted")


def _sample() -> CanonicalSample:
    return CanonicalSample(
        benchmark="smoke",
        benchmark_version="1",
        split="dev",
        split_origin=SplitOrigin.GENERATED_FROZEN,
        sample_id="smoke:dev:1",
        task_family="arithmetic",
        capability_tags=("calculation", "tool_use"),
        question="What is the percentage change from 1200 to 1500?",
        context=SampleContext(
            tables=(
                Table(
                    table_id="figures",
                    rows=(("metric", "value"), ("revenue", "1500"), ("prior", "1200")),
                    header_rows=1,
                ),
            )
        ),
        gold=GoldAnswer(answer="25", answer_type=AnswerType.NUMERIC, numeric_value=25.0),
        evaluation=EvaluationSpec(),
        source=SourceInfo(license="MIT", url="https://e.com", redistributable=True),
    )


class _Scripted(ModelProvider):
    """Replays a fixed list of model outputs, one per turn."""

    name = "mock"

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.seen: list[str] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        content = self._script.pop(0) if self._script else '{"answer": "done"}'
        self.seen.append(request.messages[-1].content)
        return ModelResponse(provider="mock", model="scripted", content=content, parsed=True)

    async def aclose(self) -> None:
        return None


async def _run(script: list[str]):
    provider = _Scripted(script)
    response, trace = await run_agent(_sample(), provider=provider, model=MODEL, config=RunConfig())
    return response, trace, provider


# --------------------------------------------------------------------------- the happy path


async def test_the_model_calls_a_tool_gets_a_result_and_uses_it() -> None:
    _, trace, provider = await _run(
        [
            json.dumps({"tool": "calculator", "arguments": {"expression": "(1500-1200)/1200*100"}}),
            json.dumps({"answer": "25%", "numeric_value": 25}),
        ]
    )
    assert trace.turns == 1
    assert trace.calls[0].tool_name == "calculator"
    assert trace.calls[0].executed is True
    assert trace.calls[0].output == "25.00"
    assert trace.result_used is True

    # The tool's output really was fed back to the model, in the message it saw next.
    assert "25.00" in provider.seen[-1]


async def test_a_named_formula_goes_through_the_same_sandbox() -> None:
    """The formula registry is a convenience, not a second evaluator. There is exactly one place in
    this codebase where a string becomes a number."""
    _, trace, _ = await _run(
        [
            json.dumps(
                {
                    "tool": "formula",
                    "arguments": {"name": "gross_margin", "values": {"revenue": 1000, "cogs": 400}},
                }
            ),
            json.dumps({"answer": "60%", "numeric_value": 60}),
        ]
    )
    assert trace.calls[0].output == "60.0"
    assert trace.result_used is True


# --------------------------------------------------------------------------- THE interesting one


async def test_a_model_that_ignores_its_own_tool_is_caught() -> None:
    """The failure no end-to-end metric can see.

    The calculator returned 25.00. The model then wrote 38. Accuracy will record a wrong number and
    attribute it to arithmetic — the one thing the model got RIGHT. The trace is the only place the
    real defect is visible: it did not believe its own tool.
    """
    _, trace, _ = await _run(
        [
            json.dumps({"tool": "calculator", "arguments": {"expression": "(1500-1200)/1200*100"}}),
            json.dumps({"answer": "about 38%", "numeric_value": 38}),
        ]
    )
    assert trace.calls[0].executed is True
    assert trace.calls[0].output == "25.00"
    assert trace.result_used is False, "it called the tool, got 25, and wrote 38"


async def test_a_rounded_restatement_still_counts_as_used() -> None:
    """Deliberately generous. The failure being hunted is a model that ignored the answer entirely,
    not one that rounded it."""
    _, trace, _ = await _run(
        [
            json.dumps({"tool": "calculator", "arguments": {"expression": "108949/2687"}}),
            json.dumps({"answer": "40.55x", "numeric_value": 40.55}),
        ]
    )
    assert trace.result_used is True


# --------------------------------------------------------------------------- the failure modes


async def test_a_hallucinated_tool_is_recorded_as_a_hallucination_not_a_generic_error() -> None:
    """ "It invented a tool" and "it used a real tool badly" are different failures with different
    fixes. A single `tool_error` would hide the difference."""
    _, trace, _ = await _run(
        [
            json.dumps({"tool": "bloomberg_terminal", "arguments": {"ticker": "AAPL"}}),
            json.dumps({"answer": "25%"}),
        ]
    )
    assert trace.calls[0].tool_exists is False
    assert trace.hallucinated_tools == ("bloomberg_terminal",)


async def test_the_tool_error_goes_back_to_the_model_so_it_can_recover() -> None:
    """A model told "no column 'amt'" that then asks for 'amount' is demonstrating recovery — which
    is a measured behaviour, and a harness that crashed here would have measured nothing."""
    _, trace, provider = await _run(
        [
            json.dumps({"tool": "csv_query", "arguments": {"aggregate": "sum", "column": "amt"}}),
            json.dumps({"tool": "csv_query", "arguments": {"aggregate": "sum", "column": "value"}}),
            json.dumps({"answer": "2700", "numeric_value": 2700}),
        ]
    )
    assert trace.calls[0].executed is False
    assert "no column" in trace.calls[0].error
    assert "tool_error" in provider.seen[1], "the error must be shown to the model"

    # And it recovered.
    assert trace.calls[1].executed is True
    assert trace.calls[1].output == "2700"
    assert trace.result_used is True


async def test_a_sandbox_refusal_is_recorded_as_a_security_event_not_an_arithmetic_error() -> None:
    """A model probing the sandbox is a security event. `1/0` is not. Recording them the same way
    would make the security signal unfindable."""
    _, trace, _ = await _run(
        [
            json.dumps({"tool": "calculator", "arguments": {"expression": "__import__('os')"}}),
            json.dumps({"answer": "cannot"}),
        ]
    )
    assert trace.calls[0].executed is False
    assert trace.calls[0].security_refused is True
    assert trace.security_refusals == 1

    _, benign, _ = await _run(
        [
            json.dumps({"tool": "calculator", "arguments": {"expression": "1/0"}}),
            json.dumps({"answer": "undefined"}),
        ]
    )
    assert benign.calls[0].executed is False
    assert benign.calls[0].security_refused is False, "division by zero is not an attack"


async def test_a_model_that_never_calls_a_tool_produces_an_empty_trace() -> None:
    _, trace, _ = await _run([json.dumps({"answer": "25%", "numeric_value": 25})])
    assert trace.called_any is False
    assert trace.turns == 0
    assert trace.result_used is False


async def test_the_loop_is_bounded() -> None:
    """A model that calls the calculator forever is not reasoning, it is stuck. Hitting the cap is
    recorded rather than silently truncated into something that reads like a considered answer."""
    from financebench.tools.agent import MAX_TOOL_TURNS

    forever = [json.dumps({"tool": "calculator", "arguments": {"expression": "1+1"}})] * 50
    _, trace, _ = await _run(forever)
    assert trace.turns == MAX_TOOL_TURNS


async def test_fenced_json_is_still_parsed() -> None:
    """Small models wrap their JSON in prose and code fences. Refusing to parse that would score
    formatting and call it tool use."""
    _, trace, _ = await _run(
        [
            'Sure! Here you go:\n```json\n{"tool": "calculator", "arguments": '
            '{"expression": "2+2"}}\n```',
            'The answer is:\n{"answer": "4", "numeric_value": 4}',
        ]
    )
    assert trace.calls[0].executed is True
    assert trace.calls[0].output == "4"
    assert trace.result_used is True
