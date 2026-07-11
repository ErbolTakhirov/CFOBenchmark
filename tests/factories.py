"""Shared builders for test fixtures.

Kept deliberately small: enough to construct a valid :class:`CanonicalSample` without every test
re-typing the eight required fields, and — importantly — a ``scrub_gold`` helper the gold-leakage
suite uses to prove the answer key cannot reach a model.
"""

from __future__ import annotations

from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    Evidence,
    GoldAnswer,
    SourceInfo,
)

__all__ = ["SCRUBBED_ANSWER", "SCRUBBED_NUMERIC", "make_sample", "scrub_gold"]

#: Sentinel values that appear nowhere in any real prompt. If a rendered request differs at all
#: after substituting these in, gold is reaching the model.
SCRUBBED_ANSWER = "ZZZ_SCRUBBED_GOLD_ANSWER_ZZZ"
SCRUBBED_NUMERIC = -987654321.123


def make_sample(
    *,
    sample_id: str = "smoke:dev:1",
    benchmark: str = "smoke",
    split: str = "dev",
    question: str = "What is 2 + 2?",
    gold_answer: str = "4",
    gold_numeric_value: float | None = 4.0,
    task_family: str = "arithmetic",
    capability_tags: tuple[str, ...] = ("calculation",),
) -> CanonicalSample:
    """A minimal valid :class:`CanonicalSample`."""
    return CanonicalSample(
        benchmark=benchmark,
        benchmark_version="1",
        split=split,
        split_origin=SplitOrigin.GENERATED_FROZEN,
        sample_id=sample_id,
        task_family=task_family,
        capability_tags=capability_tags,
        question=question,
        gold=GoldAnswer(
            answer=gold_answer,
            answer_type=AnswerType.NUMERIC,
            numeric_value=gold_numeric_value,
        ),
        evaluation=EvaluationSpec(),
        source=SourceInfo(license="public-domain", url="local", redistributable=True),
    )


def scrub_gold(sample: CanonicalSample) -> CanonicalSample:
    """Return ``sample`` with **everything on the evaluator's side of the fence** replaced.

    Gold answer, gold numeric value, gold program, acceptable answers, gold evidence, and the
    grading tolerances in :class:`EvaluationSpec` all become sentinels. The question, context,
    choices and tools — the *question side* — are left untouched.

    The gold-leakage suite renders a request from both the original and the scrubbed sample and
    asserts they are byte-identical. Any difference means something on the evaluator's side of the
    fence is reaching the model.
    """
    return sample.model_copy(
        update={
            "gold": GoldAnswer(
                answer=SCRUBBED_ANSWER,
                answer_type=sample.gold.answer_type,
                numeric_value=SCRUBBED_NUMERIC,
                unit=SCRUBBED_ANSWER,
                scale=sample.gold.scale,
                currency=SCRUBBED_ANSWER,
                evidence=(Evidence(text_snippet=SCRUBBED_ANSWER, page=999_999),),
                program=SCRUBBED_ANSWER,
                acceptable_answers=(SCRUBBED_ANSWER,),
            ),
            "evaluation": EvaluationSpec(
                absolute_tolerance=123456.0,
                relative_tolerance=654321.0,
                requires_citation=not sample.evaluation.requires_citation,
                should_refuse=not sample.evaluation.should_refuse,
            ),
        }
    )
