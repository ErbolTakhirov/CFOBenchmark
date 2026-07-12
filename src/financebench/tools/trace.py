"""The trace: what the model asked for, what it got back, and whether it then used it.

The last of those is the one that matters, and it is the one accuracy cannot see.

A model can call the calculator, receive `40.55`, and then write "the ratio is approximately 38".
Its answer is wrong, so accuracy records a failure — and attributes it to arithmetic, which is
exactly the thing the model got *right*. The real failure is that it did not believe its own tool.
That is a different defect with a different fix, it is invisible to every end-to-end metric, and
without a trace there is no way to see it at all.

So the trace records the whole round trip, and ``result_used`` is computed by looking for the tool's
own output in the final answer. It is a heuristic — a model could restate `40.55` as `40.6` and
genuinely have used it — so it is deliberately generous: a number that round-trips at any sensible
precision counts as used. The failure it is trying to catch is not subtle rounding, it is a model
that ignored the answer entirely.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ToolCallRecord", "ToolFailure", "ToolTrace"]


class ToolFailure(StrEnum):
    """Where a tool-assisted answer went wrong. These are not interchangeable, and collapsing them
    into "tool error" would destroy the only thing the trace is for."""

    #: It should have used a tool and did not. A model doing 12-digit arithmetic in its head.
    NO_TOOL_CALLED = "no_tool_called"
    #: It invented a tool that does not exist.
    HALLUCINATED_TOOL = "hallucinated_tool"
    #: Real tool, malformed arguments — a schema failure, not a reasoning one.
    INVALID_ARGUMENTS = "invalid_arguments"
    #: Real tool, valid arguments, and the tool said no (bad column, division by zero, refused
    #: expression). The model's plan was wrong, not its JSON.
    EXECUTION_FAILED = "execution_failed"
    #: The tool worked, returned the right number, and the model ignored it. **The interesting one.**
    RESULT_IGNORED = "result_ignored"
    #: The tool worked, the model used it, and the final answer is still wrong. Tools cannot fix
    #: reasoning.
    WRONG_ANSWER_DESPITE_TOOL = "wrong_answer_despite_tool"
    #: It tried to make the sandbox execute something that was not arithmetic.
    SECURITY_REFUSAL = "security_refusal"


class ToolCallRecord(BaseModel):
    """One tool call, from request to result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn: int
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)

    #: Did the model name a tool that exists?
    tool_exists: bool = True
    #: Did the arguments satisfy the tool's schema?
    arguments_valid: bool = True
    #: Did it run?
    executed: bool = False

    output: str = ""
    error: str = ""
    latency_ms: float = 0.0

    #: Was this call refused by the sandbox — i.e. was it an attempt to execute something that is not
    #: arithmetic? Recorded separately from an ordinary execution failure because a model probing the
    #: sandbox is a security event, and "1/0" is not.
    security_refused: bool = False


class ToolTrace(BaseModel):
    """Every tool call one sample produced, plus what became of the results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    tools_offered: tuple[str, ...] = ()
    calls: tuple[ToolCallRecord, ...] = ()
    turns: int = 0

    #: Did the final answer actually contain a number the tools returned?
    result_used: bool = False
    final_answer: str = ""

    #: Tokens and wall-clock across EVERY turn of the agent loop, not just the last one.
    #:
    #: The engine records the cost of the final response only — which is the right thing for a
    #: single-shot run, and exactly wrong for an agent: a model that took five turns to answer spent
    #: five prompts' worth of tokens, and reporting the fifth as the total makes an agent look as
    #: cheap as a direct call. "Tools cost 1.4x the tokens" is one of the questions the paired
    #: experiment exists to answer, and it cannot be answered from a number that was never summed.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def called_any(self) -> bool:
        return bool(self.calls)

    @property
    def executed_any(self) -> bool:
        return any(c.executed for c in self.calls)

    @property
    def hallucinated_tools(self) -> tuple[str, ...]:
        return tuple(c.tool_name for c in self.calls if not c.tool_exists)

    @property
    def security_refusals(self) -> int:
        return sum(1 for c in self.calls if c.security_refused)

    def to_json(self) -> dict[str, object]:
        return self.model_dump(mode="json")
