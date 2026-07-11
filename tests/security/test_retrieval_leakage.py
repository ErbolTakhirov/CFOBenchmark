"""The retriever must never see the answer.

A retrieval benchmark whose retriever can peek at the gold evidence measures nothing at all — and
it is *easier* to commit accidentally than prompt leakage, because the gold evidence is sitting
right there on the sample and using it would appear to work beautifully.

Two guarantees, in the same shape as the gold-leakage suite:

1. **Structural.** :class:`Retriever`'s whole interface is ``retrieve(query: str)``. It is never
   handed a :class:`CanonicalSample`, so it has nothing to reach ``sample.gold`` *from*.
2. **Scrub-equivalence.** Replace a sample's gold evidence with sentinels, retrieve again, and the
   pages come back identical. If retrieval cannot change when the gold changes, the gold is not
   being used.

Plus the one that caught a live bug: in ``retrieval_required`` mode, an **empty** retrieval must not
fall back to the sample's own context — because for FinanceBench the sample's context *is the gold
evidence text*. That fallback would have handed the answer to the model on precisely the questions
where the retriever failed hardest, inflating the retrieval score exactly where the retriever was
worst.
"""

from __future__ import annotations

import inspect

import pytest

from financebench.prompts.profiles import RetrievedChunk, create_prompt_profile
from financebench.retrieval.corpus import Page, PageCorpus
from financebench.retrieval.retriever import BM25Retriever, Retriever
from financebench.schemas.common import AnswerType, EvalMode, SplitOrigin
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    Evidence,
    GoldAnswer,
    SampleContext,
    SourceInfo,
)

_PAGES = [
    Page(document_id="ACME_2020_10K", page=1, text="Table of contents. Item 1. Business."),
    Page(
        document_id="ACME_2020_10K",
        page=42,
        text="Consolidated Statement of Cash Flows. Purchases of property, plant and equipment "
        "(1,577). Net cash used in investing activities (2,410).",
    ),
    Page(document_id="ACME_2020_10K", page=43, text="Notes to the financial statements. Leases."),
]


def _corpus() -> PageCorpus:
    return PageCorpus(list(_PAGES))


def _sample(*, gold_answer: str = "$1577.00", evidence_page: int = 42) -> CanonicalSample:
    return CanonicalSample(
        benchmark="financebench",
        benchmark_version="test",
        split="open_source",
        split_origin=SplitOrigin.PUBLIC_SUBSET,
        sample_id="financebench:open_source:x1",
        task_family="financebench_numeric",
        capability_tags=("evidence_grounding",),
        question="What is the FY2020 capital expenditure for ACME?",
        context=SampleContext(text=("Purchases of property, plant and equipment (1,577)",)),
        gold=GoldAnswer(
            answer=gold_answer,
            answer_type=AnswerType.NUMERIC,
            numeric_value=1577.0,
            evidence=(
                Evidence(
                    document_id="ACME_2020_10K",
                    page=evidence_page,
                    text_snippet="Purchases of property, plant and equipment (1,577)",
                ),
            ),
        ),
        evaluation=EvaluationSpec(),
        source=SourceInfo(license="CC BY-NC-4.0", url="local", redistributable=False),
        metadata={"doc_name": "ACME_2020_10K"},
    )


# --------------------------------------------------------------------------- 1. structural


def test_the_retriever_interface_cannot_receive_a_sample() -> None:
    """The guarantee is the signature. No sample in, no gold to find."""
    signature = inspect.signature(Retriever.retrieve)
    parameters = set(signature.parameters)

    assert parameters == {"self", "query", "top_k"}
    assert "sample" not in parameters
    assert "gold" not in parameters
    assert "evidence" not in parameters

    annotations = {
        name: str(p.annotation) for name, p in signature.parameters.items() if name != "self"
    }
    assert "CanonicalSample" not in str(annotations)


# --------------------------------------------------------------------------- 2. scrub-equivalence


def test_retrieval_is_identical_when_the_gold_evidence_is_scrubbed() -> None:
    """Change the answer and the evidence completely; the retrieved pages must not move."""
    retriever = BM25Retriever(_corpus())

    original = _sample()
    scrubbed = original.model_copy(
        update={
            "gold": GoldAnswer(
                answer="ZZZ_SCRUBBED",
                answer_type=AnswerType.NUMERIC,
                numeric_value=-987654321.0,
                evidence=(
                    Evidence(
                        document_id="ZZZ_NOT_A_DOCUMENT",
                        page=999,
                        text_snippet="ZZZ_SCRUBBED_EVIDENCE",
                    ),
                ),
            )
        }
    )

    a = retriever.retrieve(original.question, top_k=3)
    b = retriever.retrieve(scrubbed.question, top_k=3)
    assert a.chunk_ids == b.chunk_ids


def test_the_query_is_the_question_not_the_answer() -> None:
    """The retriever is given what a *user* would type. If it were given the gold answer, or the
    gold evidence snippet, retrieval would be trivial and the benchmark would measure nothing."""
    from financebench.retrieval.pipeline import RetrievalPipeline

    source = inspect.getsource(RetrievalPipeline.retrieve_for)
    # Strip the docstring: it *discusses* gold in prose, which is not the same as touching it.
    body = source.split('"""')[-1]

    assert "sample.question" in body
    for forbidden in ("sample.gold", ".gold.", "evidence", "justification"):
        assert forbidden not in body, f"the retriever query must not touch {forbidden}"


# --------------------------------------------------------------------------- 3. the empty-retrieval
# fallback that WAS a live leak


def test_empty_retrieval_does_not_fall_back_to_the_samples_own_context() -> None:
    """The bug this test exists for.

    Falling back to ``sample.context`` when retrieval returns nothing looks harmless. For
    FinanceBench the sample's context IS the gold evidence text — so the fallback would have handed
    the answer to the model on exactly the questions where the retriever failed completely, and the
    retrieval score would have been inflated most precisely where retrieval was worst.
    """
    sample = _sample()
    profile = create_prompt_profile("structured_financial_v1")

    rendered = profile.render(sample, EvalMode.RETRIEVAL_REQUIRED, ())
    prompt = "\n".join(message.content for message in rendered)

    # The sample's context (which is the gold evidence) must be nowhere in the prompt.
    for block in sample.context.text:
        assert block not in prompt
    assert "1,577" not in prompt
    assert "No relevant excerpts were retrieved" in prompt


def test_context_given_mode_still_shows_the_context() -> None:
    """Guards the guard: if the withholding were unconditional, context_given would be broken and
    every test above would pass vacuously."""
    sample = _sample()
    profile = create_prompt_profile("structured_financial_v1")

    rendered = profile.render(sample, EvalMode.CONTEXT_GIVEN, ())
    prompt = "\n".join(message.content for message in rendered)
    assert "1,577" in prompt


def test_retrieval_mode_shows_only_what_was_retrieved() -> None:
    sample = _sample()
    profile = create_prompt_profile("structured_financial_v1")

    chunks = (
        RetrievedChunk(
            document_id="OTHER_2019_10K", text="Some entirely unrelated page.", page=7, score=1.0
        ),
    )
    rendered = profile.render(sample, EvalMode.RETRIEVAL_REQUIRED, chunks)
    prompt = "\n".join(message.content for message in rendered)

    assert "Some entirely unrelated page." in prompt
    assert "1,577" not in prompt, "the sample's own (gold) context must never appear"


# --------------------------------------------------------------------------- retrieval is graded
# only afterwards


def test_gold_is_used_only_to_grade_and_only_after_the_fact() -> None:
    from financebench.retrieval.metrics import score_retrieval

    retriever = BM25Retriever(_corpus())
    sample = _sample()
    result = retriever.retrieve(sample.question, top_k=3)

    score = score_retrieval(sample, result)
    assert score.n_gold_pages == 1
    assert isinstance(score.page_hit, bool)


@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_retrieval_is_deterministic(top_k: int) -> None:
    """A retriever whose output wobbles between runs makes every downstream score unreproducible."""
    retriever = BM25Retriever(_corpus())
    question = _sample().question
    first = retriever.retrieve(question, top_k=top_k)
    second = retriever.retrieve(question, top_k=top_k)
    assert first.chunk_ids == second.chunk_ids
