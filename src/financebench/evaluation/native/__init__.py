"""Native, benchmark-specific metrics.

Each wrapped benchmark's own official evaluation method, preserved rather than flattened into a
generic score — and, where a metric is *not* the official one, named so that it cannot be mistaken
for it (``finqa_answer_accuracy`` is ours; ``finqa_execution_accuracy`` is FinQA's). Parity against
the real official evaluators is asserted in ``tests/parity/``.

Importing this package registers every native metric implemented so far.
"""

from __future__ import annotations

from financebench.evaluation.native import convfinqa as _conv  # noqa: F401
from financebench.evaluation.native import finance_reasoning as _fr  # noqa: F401
from financebench.evaluation.native import finqa as _finqa  # noqa: F401
from financebench.evaluation.native import secque as _secque  # noqa: F401
from financebench.evaluation.native import smb_cfo as _smb  # noqa: F401
from financebench.evaluation.native import tatqa as _tatqa  # noqa: F401
from financebench.evaluation.native import tool_use as _tools  # noqa: F401
from financebench.evaluation.native.convfinqa import (
    ConvFinQAExecutionAccuracy,
    ConvFinQAProgramAccuracy,
    ConvFinQATurnAccuracy,
)
from financebench.evaluation.native.finance_reasoning import FinanceReasoningAccuracy
from financebench.evaluation.native.finqa import (
    FinQAAnswerAccuracy,
    FinQAExecutionAccuracy,
    FinQAProgramAccuracy,
    execute_program,
)
from financebench.evaluation.native.tatqa import TatQAExactMatch, TatQAF1, TatQAScaleAccuracy

__all__ = [
    "ConvFinQAExecutionAccuracy",
    "ConvFinQAProgramAccuracy",
    "ConvFinQATurnAccuracy",
    "FinQAAnswerAccuracy",
    "FinQAExecutionAccuracy",
    "FinQAProgramAccuracy",
    "FinanceReasoningAccuracy",
    "TatQAExactMatch",
    "TatQAF1",
    "TatQAScaleAccuracy",
    "execute_program",
]
