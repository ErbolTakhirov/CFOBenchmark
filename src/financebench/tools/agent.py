"""The agent loop: model asks for a tool, the tool answers, the model answers.

Bounded in both directions, and the bounds are the point.

A model that loops — calling the calculator forty times, or asking for the same table over and over —
is not reasoning, it is stuck. Left alone it will consume the whole context window and the whole
evening. So the loop is capped by turns, and a run that hits the cap is recorded as having hit it
rather than being quietly truncated into something that looks like a considered answer.

The protocol is deliberately the simplest thing that works with a small local model: the model emits
JSON, either a tool call or a final answer. Real tool-calling APIs (OpenAI's ``tools``, Anthropic's
``tool_use``) are strictly richer, and a provider that supports them can be wired to this same loop —
but a 3B model served by Ollama does not reliably produce them, and a benchmark that *requires* a
capability the model does not have measures the API, not the model.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from financebench.models.base import ModelProvider
from financebench.prompts.profiles import create_prompt_profile
from financebench.schemas.common import EvalMode
from financebench.schemas.model_io import (
    ChatMessage,
    FinancialAnswer,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    Role,
)
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import CanonicalSample
from financebench.tools.registry import ToolExecutionError, execute_tool, tools_for_sample
from financebench.tools.trace import ToolCallRecord, ToolTrace

__all__ = ["MAX_TOOL_TURNS", "run_agent"]

#: A model that has not answered after this many tool calls is not reasoning; it is stuck in a loop.
#: Hitting the cap is *recorded*, never silently truncated into something that reads like a
#: considered answer.
MAX_TOOL_TURNS = 6

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, object] | None:
    """Pull the model's JSON out of whatever it wrapped it in.

    Small models fence their JSON, prefix it with "Sure!", or trail it with an explanation. Refusing
    to parse any of that would score prose formatting and call it tool use.
    """
    for candidate in (text, *(m.group(0) for m in _JSON.finditer(text))):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _numbers(text: str) -> list[Decimal]:
    out: list[Decimal] = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return out


def _result_was_used(outputs: Sequence[str], answer: str) -> bool:
    """Did the final answer actually contain a number a tool returned?

    Deliberately generous: a model that restates `40.5467` as `40.55` has used the tool. The failure
    this is hunting is not rounding, it is a model that called the calculator, got the right number,
    and then wrote something else entirely — which accuracy records as an arithmetic failure, blaming
    the model for the one thing it got right.
    """
    stated = _numbers(answer)
    if not stated:
        return False
    for output in outputs:
        for produced in _numbers(output):
            for value in stated:
                if produced == value:
                    return True
                scale = max(abs(produced), Decimal("1"))
                if abs(produced - value) <= Decimal("0.02") * scale:
                    return True
    return False


def _system_prompt(sample: CanonicalSample) -> str:
    specs = tools_for_sample(sample)
    catalogue = "\n".join(
        f"- {spec.name}: {spec.description}\n  arguments: "
        f"{json.dumps(spec.parameters_schema.get('properties', {}))}"
        for spec in specs
    )
    return (
        "You are a financial analyst with access to tools. Use them for any calculation — they are "
        "exact, and mental arithmetic is not.\n\n"
        f"TOOLS:\n{catalogue}\n\n"
        "Reply with a single JSON object and nothing else. To call a tool:\n"
        '  {"tool": "<name>", "arguments": {...}}\n'
        "When you have the answer:\n"
        '  {"answer": "<the answer>", "numeric_value": <number or null>, '
        '"brief_explanation": "<why>"}\n\n'
        "If the material does not support an answer, reply with "
        '{"answer": "", "insufficient_information": true}. Inventing a number is worse than saying '
        "you cannot answer."
    )


def _user_prompt(sample: CanonicalSample) -> str:
    """The same context the direct run shows, rendered by the same code.

    This is load-bearing for the whole paired comparison. If the tool-assisted prompt and the direct
    prompt differed in what *facts* they showed the model, then any difference between their scores
    would be attributable to the prompt rather than to the tools — which is the one thing the
    experiment exists to isolate. Only the *system* prompt differs, and it differs by exactly one
    thing: the offer of tools.
    """
    profile = create_prompt_profile("structured_financial_v1")
    return profile.user(sample, EvalMode.CONTEXT_GIVEN, ())


async def run_agent(
    sample: CanonicalSample,
    *,
    provider: ModelProvider,
    model: ModelSpec,
    config: RunConfig,
    max_turns: int = MAX_TOOL_TURNS,
) -> tuple[ModelResponse | None, ToolTrace]:
    """Run one sample through the model↔tool loop. Returns the final response and the full trace."""
    offered = tuple(spec.name for spec in tools_for_sample(sample))
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.SYSTEM, content=_system_prompt(sample)),
        ChatMessage(role=Role.USER, content=_user_prompt(sample)),
    ]

    calls: list[ToolCallRecord] = []
    outputs: list[str] = []
    last: ModelResponse | None = None

    for turn in range(max_turns):
        request = ModelRequest(
            model=model,
            messages=tuple(messages),
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            response_format="json_object",
            prompt_version="tool_agent_v1",
            benchmark=sample.benchmark,
            benchmark_version=sample.benchmark_version,
            sample_id=sample.sample_id,
            timeout_s=config.timeout_seconds,
        )
        response = await provider.generate(request)
        last = response

        parsed = _extract_json(response.content)
        if parsed is None or "tool" not in parsed:
            break  # it answered (or produced something unparseable) — either way the loop is over

        name = str(parsed.get("tool", ""))
        raw_args = parsed.get("arguments")
        arguments = raw_args if isinstance(raw_args, dict) else {}

        started = time.perf_counter()
        record: ToolCallRecord
        try:
            output = execute_tool(name, arguments, sample)
            outputs.append(output)
            record = ToolCallRecord(
                turn=turn,
                tool_name=name,
                arguments=arguments,
                executed=True,
                output=output,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            feedback = f'{{"tool_result": {json.dumps(output)}}}'
        except ToolExecutionError as exc:
            message = str(exc)
            record = ToolCallRecord(
                turn=turn,
                tool_name=name,
                arguments=arguments,
                # "It invented a tool" and "it used a real tool badly" are different failures with
                # different fixes. A single `tool_error` would hide the difference.
                tool_exists="no such tool" not in message,
                arguments_valid="must be an object" not in message and "requires" not in message,
                executed=False,
                error=message,
                latency_ms=(time.perf_counter() - started) * 1000,
                # A model probing the sandbox is a security event; `1/0` is not.
                security_refused=message.startswith("security_refused:"),
            )
            # The error goes BACK to the model. A model that is told "no column 'amt'" and then asks
            # for 'amount' is demonstrating recovery — which is a measured behaviour, and a harness
            # that crashed here would have measured nothing.
            feedback = f'{{"tool_error": {json.dumps(message)}}}'

        calls.append(record)
        messages.append(ChatMessage(role=Role.ASSISTANT, content=response.content))
        messages.append(ChatMessage(role=Role.USER, content=feedback))

    answer_text = ""
    if last is not None:
        parsed_answer = FinancialAnswer.from_text(last.content)
        if parsed_answer is not None:
            answer_text = f"{parsed_answer.answer} {parsed_answer.brief_explanation or ''}"
            last = last.model_copy(update={"financial_answer": parsed_answer, "parsed": True})

    trace = ToolTrace(
        sample_id=sample.sample_id,
        tools_offered=offered,
        calls=tuple(calls),
        turns=len(calls),
        result_used=_result_was_used(outputs, answer_text) if outputs else False,
        final_answer=answer_text[:500],
    )
    return last, trace
