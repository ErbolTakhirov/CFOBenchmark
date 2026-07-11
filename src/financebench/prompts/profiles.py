"""Versioned prompt profiles.

A prompt profile decides *what a model is asked to produce*, which in turn decides *what can
honestly be measured*. This is not a cosmetic layer:

- ``structured_financial_v1`` asks for a final answer → you can score answer correctness.
- ``program_v1`` asks for an executable FinQA-style program → and **only then** can FinQA's and
  ConvFinQA's official *program accuracy* be computed at all. Reporting program accuracy against a
  model that was never asked for a program would be meaningless, which is why this platform did not
  report it until this profile existed.

Every profile is a pure function of the sample's **question side** — ``question``, ``context``,
``choices``, ``tools`` — plus the eval mode and any retrieved chunks. No profile may read
``sample.gold`` or ``sample.evaluation``; ``tests/security/test_gold_answer_leakage.py`` enforces
that by scrubbing gold to sentinels and asserting the rendered prompt is byte-identical.

Profiles are versioned in their name (``_v1``). Changing a prompt's text **must** mean a new
version, because the prompt is part of a run's comparability fingerprint — silently editing
``structured_financial_v1`` would make yesterday's scores incomparable with today's while looking
identical in the metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal, TypeVar

from financebench.schemas.common import DEFAULT_PROMPT_PROFILE, EvalMode
from financebench.schemas.model_io import ChatMessage, Role
from financebench.schemas.sample import CanonicalSample, Table
from financebench.utils.errors import ConfigError

__all__ = [
    "DEFAULT_PROMPT_PROFILE",
    "AnalystMemoV1",
    "ConversationV1",
    "DirectNumericV1",
    "GroundedCitationsV1",
    "ProgramV1",
    "PromptProfile",
    "RetrievedChunk",
    "StructuredFinancialV1",
    "ToolAgentV1",
    "available_prompt_profiles",
    "create_prompt_profile",
    "register_prompt_profile",
]

_JSON_ENVELOPE = (
    '{"answer": "<string>", "numeric_value": <number or null>, "unit": "<string or null>", '
    '"period": "<string or null>", '
    '"citations": [{"document_id": "...", "page": <int or null>, "table": "...", "row": "..."}], '
    '"insufficient_information": <true|false>, "confidence": <0-1 or null>, '
    '"brief_explanation": "<string>"}'
)

_REFUSAL_CLAUSE = (
    "If the provided material does not contain enough information to answer, set "
    '"insufficient_information" to true and do not guess. Inventing a plausible number is worse '
    "than saying you cannot answer."
)


class RetrievedChunk:
    """A chunk of text a retriever surfaced for a ``retrieval_required`` run.

    Carries its own provenance so the prompt can cite it and the grader can check whether the
    model's citation points at something that was actually retrieved.
    """

    __slots__ = ("document_id", "page", "score", "text")

    def __init__(
        self, *, document_id: str, text: str, page: int | None = None, score: float | None = None
    ) -> None:
        self.document_id = document_id
        self.text = text
        self.page = page
        self.score = score

    def render(self) -> str:
        location = f"{self.document_id}" + (f", page {self.page}" if self.page is not None else "")
        return f"[{location}]\n{self.text}"


def _render_table(table: Table) -> str:
    lines = [" | ".join(row) for row in table.rows]
    header = f"Table ({table.caption}):\n" if table.caption else "Table:\n"
    return header + "\n".join(lines)


def _render_context(sample: CanonicalSample, retrieved: Sequence[RetrievedChunk]) -> str:
    """The context block. In ``retrieval_required`` mode the sample's own context is *withheld* —
    the model sees only what the retriever found, which is the entire point of the mode."""
    parts: list[str] = []
    if retrieved:
        parts.append(
            "Retrieved excerpts (these are all you have; they may be incomplete or irrelevant):"
        )
        parts.extend(chunk.render() for chunk in retrieved)
        return "\n\n".join(parts)
    parts.extend(sample.context.text)
    parts.extend(_render_table(table) for table in sample.context.tables)
    return "\n\n".join(parts)


def _render_history(sample: CanonicalSample) -> str:
    if not sample.context.conversation_history:
        return ""
    turns = "\n".join(
        f"{turn.role.upper()}: {turn.content}" for turn in sample.context.conversation_history
    )
    return f"Conversation so far:\n{turns}"


class PromptProfile(ABC):
    """Turns a sample into the messages sent to a model."""

    name: ClassVar[str] = ""
    response_format: ClassVar[Literal["text", "json_object"]] = "json_object"
    #: Set on profiles that ask the model to emit an executable program rather than an answer.
    #: The FinQA/ConvFinQA program-accuracy metrics refuse to score a run whose profile is not one
    #: of these, rather than reporting a fabricated zero.
    elicits_program: ClassVar[bool] = False

    @abstractmethod
    def system(self, sample: CanonicalSample, mode: EvalMode) -> str: ...

    def user(
        self, sample: CanonicalSample, mode: EvalMode, retrieved: Sequence[RetrievedChunk]
    ) -> str:
        parts: list[str] = []
        history = _render_history(sample)
        if history:
            parts.append(history)
        context = _render_context(sample, retrieved)
        if context:
            parts.append(f"Context:\n{context}")
        if sample.choices:
            parts.append("Choices:\n" + "\n".join(sample.choices))
        parts.append(f"Question: {sample.question}")
        return "\n\n".join(parts)

    def render(
        self,
        sample: CanonicalSample,
        mode: EvalMode = EvalMode.CONTEXT_GIVEN,
        retrieved: Sequence[RetrievedChunk] = (),
    ) -> tuple[ChatMessage, ...]:
        return (
            ChatMessage(role=Role.SYSTEM, content=self.system(sample, mode)),
            ChatMessage(role=Role.USER, content=self.user(sample, mode, retrieved)),
        )


_REGISTRY: dict[str, type[PromptProfile]] = {}
_ProfileT = TypeVar("_ProfileT", bound=type[PromptProfile])


def register_prompt_profile(name: str) -> Callable[[_ProfileT], _ProfileT]:
    def decorate(cls: _ProfileT) -> _ProfileT:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def available_prompt_profiles() -> list[str]:
    return sorted(_REGISTRY)


def create_prompt_profile(name: str) -> PromptProfile:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise ConfigError(
            f"unknown prompt profile {name!r}; available: {available_prompt_profiles()}"
        ) from exc


# --------------------------------------------------------------------------- profiles


@register_prompt_profile("structured_financial_v1")
class StructuredFinancialV1(PromptProfile):
    """The default. A structured JSON answer with an explicit "I can't tell" escape hatch."""

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        source = (
            "the retrieved excerpts"
            if mode is EvalMode.RETRIEVAL_REQUIRED
            else "the provided context"
        )
        return (
            "You are a financial analyst. Answer the question using ONLY "
            f"{source}. Respond with a single JSON object and nothing else, matching this schema: "
            f"{_JSON_ENVELOPE}. "
            'Put the bare number in "numeric_value" (no commas, no currency symbol, no percent '
            'sign) and its unit in "unit" (e.g. "usd", "percent", "percentage_points", "ratio", '
            '"days"). ' + _REFUSAL_CLAUSE
        )


@register_prompt_profile("direct_numeric_v1")
class DirectNumericV1(PromptProfile):
    """Bare number, no JSON. Isolates arithmetic ability from format compliance — the gap between
    this profile's score and ``structured_financial_v1``'s *is* the formatting tax."""

    response_format: ClassVar[Literal["text", "json_object"]] = "text"

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        return (
            "You are a financial analyst. Answer with the final number ONLY — no words, no "
            "explanation, no units, no currency symbol, no thousands separators. "
            'If the question is a yes/no question, answer exactly "yes" or "no". '
            'If the context does not contain enough information, answer exactly "INSUFFICIENT".'
        )


@register_prompt_profile("grounded_citations_v1")
class GroundedCitationsV1(PromptProfile):
    """Answer **and** cite. Used wherever grounding is scored — an uncited numeric claim is
    treated as unsupported, which is the only way to measure hallucination rather than guess at it."""

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        return (
            "You are a financial analyst. Answer the question using ONLY the material provided. "
            f"Respond with a single JSON object and nothing else, matching this schema: "
            f"{_JSON_ENVELOPE}. "
            "EVERY numeric claim you make must be supported by a citation in the "
            '"citations" array, identifying the document and page it came from. '
            "An answer you cannot cite is an answer you must not give: in that case set "
            '"insufficient_information" to true. ' + _REFUSAL_CLAUSE
        )


@register_prompt_profile("analyst_memo_v1")
class AnalystMemoV1(PromptProfile):
    """Prose analysis plus a structured summary — for benchmarks (SECQUE) whose gold answers are
    multi-sentence expert judgements rather than a single number."""

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        return (
            "You are a senior financial analyst writing for an executive audience. Analyse the "
            "question using ONLY the provided material. Respond with a single JSON object and "
            f"nothing else, matching this schema: {_JSON_ENVELOPE}. "
            'Put your full analysis in "brief_explanation" (several sentences: state the figures, '
            "what they mean, and the caveats). If the question turns on a specific ratio or "
            'figure, put that number in "numeric_value". ' + _REFUSAL_CLAUSE
        )


@register_prompt_profile("conversation_v1")
class ConversationV1(PromptProfile):
    """Multi-turn. The conversation so far is rendered into the user message; the follow-up
    question usually depends on it (ConvFinQA's "and what was it in 2005?" means nothing alone)."""

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        return (
            "You are a financial analyst in an ongoing conversation. The user asks follow-up "
            "questions that depend on what was already asked and answered — resolve pronouns and "
            'ellipsis ("and in 2005?", "what was the change?") against the conversation so far. '
            "Answer the LAST question only, using ONLY the provided context. "
            f"Respond with a single JSON object and nothing else, matching this schema: "
            f"{_JSON_ENVELOPE}. " + _REFUSAL_CLAUSE
        )


@register_prompt_profile("program_v1")
class ProgramV1(PromptProfile):
    """Elicits an executable FinQA-style program instead of an answer.

    This is the profile that makes **official program accuracy computable**. The official metric
    (``equal_program``: sympy symbolic equivalence against the gold program's operand set) has
    nothing to compare unless the model actually emits a program — so this profile is a
    prerequisite for the metric, not a stylistic choice.

    The DSL below is FinQA's own, described from its operator set. The gold program is of course
    never shown — and neither is any *real* program.

    A note on the format example, which cost a bug: the obvious example to reach for is the one in
    FinQA's own paper, ``subtract(5829, 5735), divide(#0, 5735)``. That string is the **verbatim
    gold program of a real FinQA test sample** (``etr-2016-page_23.pdf-2``, answer 94), so putting
    it in the system prompt hands the model that sample's answer — a static leak that the
    scrub-equivalence test cannot see, because the prompt never varies with gold. The example below
    therefore uses deliberately synthetic operands that appear in no real sample, and
    ``tests/security/test_gold_answer_leakage.py`` asserts that no fixture sample's gold program
    is a substring of any profile's prompt.
    """

    response_format: ClassVar[Literal["text", "json_object"]] = "text"
    elicits_program: ClassVar[bool] = True

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        return (
            "You are a financial analyst. Do NOT answer with a number. Instead, write the "
            "PROGRAM that computes the answer, using this DSL and nothing else.\n"
            "\n"
            "Operations:\n"
            "  add(a, b)        subtract(a, b)    multiply(a, b)    divide(a, b)\n"
            "  exp(a, b)        greater(a, b)     -> yields yes/no\n"
            "  table_max(row, none)   table_min(row, none)\n"
            "  table_sum(row, none)   table_average(row, none)\n"
            "\n"
            "Rules:\n"
            "  - Steps are comma-separated and evaluated left to right.\n"
            "  - Refer to the result of step N (0-indexed) as #N.\n"
            "  - Arguments are numbers taken from the context, #N references, or constants\n"
            "    written as const_100, const_1000, const_m1 (= -1).\n"
            "  - table_* operations take a table ROW LABEL as their first argument.\n"
            "  - Use the numbers exactly as they appear in the context.\n"
            "\n"
            "Format illustration only — these operands are invented, do not reuse them:\n"
            "  a percentage change from AAA to BBB is written  subtract(BBB, AAA), divide(#0, AAA)\n"
            "  a two-step sum-then-share is written            add(AAA, BBB), divide(AAA, #0)\n"
            "\n"
            "Output the program on a single line. No prose, no explanation, no markdown."
        )


@register_prompt_profile("tool_agent_v1")
class ToolAgentV1(PromptProfile):
    """Tool-using agent. Scored on tool selection, arguments, and whether the tool's result was
    actually used — an agent that calls a calculator and then ignores it is a distinct failure
    from one that never calls it."""

    def system(self, sample: CanonicalSample, mode: EvalMode) -> str:
        tool_names = ", ".join(tool.name for tool in sample.tools) or "none available"
        return (
            "You are a financial analyst with tools. Prefer computing with a tool over doing "
            "arithmetic in your head — a tool result is auditable, your mental arithmetic is not. "
            f"Tools available: {tool_names}. "
            "Do not call a tool you do not need. When you have the answer, respond with a single "
            f"JSON object and nothing else, matching this schema: {_JSON_ENVELOPE}. "
            + _REFUSAL_CLAUSE
        )
