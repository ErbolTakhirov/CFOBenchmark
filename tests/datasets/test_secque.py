"""SECQUE: the benchmark that most tempts an evaluation suite to lie.

Its gold is an expert's prose. There is no exact-match metric and there cannot be one, so the
temptation is to invent a number — a benchmark with no score looks broken. These tests pin the two
honest alternatives instead: grade the parts that are genuinely checkable, and say *not applicable*
about the parts that are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from financebench.datasets.base import create_dataset
from financebench.datasets.secque import SECQUE_CATEGORIES, SecqueAdapter
from financebench.evaluation.benchmark_metrics import metrics_for_run, preferred_metric_name
from financebench.prompts.renderer import render_messages
from financebench.schemas.common import AnswerType
from financebench.schemas.manifest import AdapterStatus
from financebench.utils.errors import DatasetLoadError

DATA_DIR = Path("data/downloads/secque")

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "SECQUE_benchmark_Ratio.jsonl").is_file(),
    reason="secque not prepared; run `financebench prepare secque`",
)


def _samples():
    return list(SecqueAdapter().load("test"))


# --------------------------------------------------------------------------- the data is the data


def test_all_565_tasks_load_with_the_official_category_counts() -> None:
    """565 = Analysis 72 + Comparison 220 + Ratio 188 + Risk 85. If any of these drift, the data
    changed underneath us and the scores stop being comparable with anything."""
    from collections import Counter

    samples = _samples()
    assert len(samples) == 565

    counts = Counter(s.metadata["category"] for s in samples)
    assert dict(counts) == {"Analysis": 72, "Comparison": 220, "Ratio": 188, "Risk": 85}


def test_every_task_has_a_question_a_context_and_an_expert_answer() -> None:
    for sample in _samples():
        assert sample.question.strip(), f"{sample.sample_id} has no question"
        assert sample.context.text and sample.context.text[0].strip(), "no SEC context"
        assert sample.gold.answer.strip(), "no expert reference answer"


def test_the_filing_is_identified_for_every_single_task() -> None:
    """The context header names the filer, in three different shapes — and only one of them is
    obvious. A regex that handled just the first left 236 of 565 tasks with no company at all, which
    would have silently made the wrong-filing check "not applicable" on 42 % of the benchmark."""
    samples = _samples()
    assert all(s.metadata["company"] for s in samples), "every task names a company"
    assert len({s.metadata["company"] for s in samples}) > 10


def test_a_changed_upstream_file_is_refused_rather_than_silently_evaluated(tmp_path: Path) -> None:
    """A benchmark is only reproducible if the bytes are the same bytes. Data that changed under us
    would produce scores incomparable with every run made before it — so the checksum refuses."""
    for name in ("Analysis", "Comparison", "Ratio", "Risk"):
        src = DATA_DIR / f"SECQUE_benchmark_{name}.jsonl"
        (tmp_path / f"SECQUE_benchmark_{name}.jsonl").write_bytes(src.read_bytes())

    tampered = tmp_path / "SECQUE_benchmark_Ratio.jsonl"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="checksum mismatch"):
        SecqueAdapter(data_dir=tmp_path).load("test")


def test_the_category_files_are_strata_not_splits() -> None:
    """Reporting "SECQUE: 87 %" from the 72-sample Analysis file alone would be a four-fold
    overstatement of coverage. There is exactly one split."""
    with pytest.raises(DatasetLoadError, match="strata, not splits"):
        SecqueAdapter().load("Ratio")


# --------------------------------------------------------------------------- the gold stays out


def test_the_expert_answer_never_reaches_the_model() -> None:
    """The whole benchmark rests on this. The expert's answer is the answer key; it lives on
    `sample.gold`, which no prompt profile may read."""
    for sample in _samples()[:40]:
        prompt = "\n".join(m.content for m in render_messages(sample))
        assert sample.gold.answer not in prompt
        # And the context the model DOES get is the SEC excerpt, which it needs.
        assert sample.context.text[0][:200] in prompt


def test_the_gold_is_typed_as_text_because_it_is_prose() -> None:
    """Typing an expert's paragraph as NUMERIC would invite an exact-match metric that cannot exist.
    Inventing one is how this benchmark gets faked."""
    for sample in _samples()[:20]:
        assert sample.gold.answer_type is AnswerType.TEXT
        assert sample.gold.numeric_value is None


# --------------------------------------------------------------------------- honest labelling


def test_the_categories_are_labelled_by_how_numeric_they_actually_are() -> None:
    """Risk is narrative. Running a numeric metric over it and reporting 0.0 would measure this
    metric's blindness and print it as the model's fault."""
    assert SECQUE_CATEGORIES["Ratio"] == "numeric"
    assert SECQUE_CATEGORIES["Comparison"] == "numeric"
    assert SECQUE_CATEGORIES["Analysis"] == "mixed"
    assert SECQUE_CATEGORIES["Risk"] == "narrative"

    by_numeracy = {s.metadata["category"]: s.metadata["numeracy"] for s in _samples()}
    assert by_numeracy["Risk"] == "narrative"


def test_the_preferred_metric_is_the_one_a_fluent_answer_cannot_talk_past() -> None:
    """Of everything deterministic here, the hallucination detector is the only check a persuasive
    model cannot argue with. A number that is not in the document is not a matter of opinion."""
    assert preferred_metric_name("secque") == "secque_unsupported_numeric_claim"
    names = {m.name for m in metrics_for_run("secque")}
    assert {
        "secque_numeric_agreement",
        "secque_comparison_direction",
        "secque_filing_identification",
    } <= names


def test_the_manifest_says_there_is_no_exact_match_metric_and_why() -> None:
    manifest = SecqueAdapter().manifest()
    assert manifest.status is AdapterStatus.FULLY_SUPPORTED
    assert manifest.license == "MIT"

    limits = " ".join(manifest.known_limitations)
    assert "there cannot be one" in limits
    assert "passed=None" in limits
    assert "STRATA" in limits


def test_the_adapter_is_registered() -> None:
    assert isinstance(create_dataset("secque"), SecqueAdapter)


def test_any_prefix_is_stratified_across_the_four_categories() -> None:
    """The categories are wildly unbalanced (Comparison 220, Ratio 188, Risk 85, Analysis 72) and
    they measure DIFFERENT THINGS — Ratio is arithmetic, Risk is an essay.

    Emitted file-by-file, `--max-samples 80` returned 72 Analysis and 8 Comparison: zero Ratio, zero
    Risk — and reported it as "SECQUE". That is not a sampling artefact, it is a different benchmark
    wearing SECQUE's name, and it would have been completely invisible in the output.
    """
    from collections import Counter

    samples = _samples()
    assert len(samples) == 565, "interleaving must not drop or duplicate a single task"

    for n in (40, 80, 120):
        counts = Counter(s.metadata["category"] for s in samples[:n])
        assert len(counts) == 4, f"a {n}-sample prefix must cover all four categories"
        assert min(counts.values()) >= n // 8, f"a {n}-sample prefix is lopsided: {dict(counts)}"


def test_the_order_is_deterministic_so_a_run_is_reproducible() -> None:
    first = [s.sample_id for s in _samples()[:20]]
    second = [s.sample_id for s in _samples()[:20]]
    assert first == second
