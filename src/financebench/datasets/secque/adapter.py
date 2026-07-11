"""SECQUE — 565 expert-written financial-analysis tasks over real SEC filings.

This is the benchmark that most tempts a platform to lie, and it is worth being explicit about why.

Its gold is a **narrative expert answer**, not a number. There is no exact-match to fall back on. So
there are exactly three things you can do with it, and only two of them are honest:

1. Grade the parts that *are* deterministically checkable — the numbers, the ratio, the direction of a
   comparison, the company, the period. That is real, and it is what ``evaluation/native/secque.py``
   does. It is called **diagnostics**, not a score, because it is not one.
2. Have a *calibrated* judge grade the analytical quality, and report the calibration alongside it so a
   reader can decide whether to believe it (``evaluation/judge/``).
3. Invent a number. This is the tempting one — a benchmark with no score looks broken — and it is
   the reason SECQUE is where an evaluation suite usually starts lying.

The categories are not interchangeable, and this matters more than it looks:

- **Ratio** (188) — the reference contains a formula, its operands, and a final value. Almost fully
  checkable deterministically.
- **Comparison** (220) — the reference contains figures per year and a direction of travel. Largely
  checkable.
- **Analysis** (72) — a number, then an interpretation of it. Half checkable.
- **Risk** (85) — *"cybersecurity, fraud, and data protection…"*. **Essentially no numbers at all.**

So a deterministic metric applied to the Risk split would be scoring an essay on its arithmetic. Those
samples return ``passed=None`` — *not applicable* — and never a zero. A zero would say "the model
failed"; the truth is "this metric cannot see this question".

Data: HuggingFace ``nogabenyoash/SecQue`` @ ``894196b8``, **MIT**, public, no auth. Four JSONL files,
565 rows, every field populated. Checksums pinned below: the benchmark is only reproducible if the
bytes are the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from financebench.datasets.base import DatasetAdapter, register_dataset
from financebench.datasets.downloader import download_file
from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.manifest import AdapterStatus, DatasetManifest
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    Evidence,
    GoldAnswer,
    SampleContext,
    SourceInfo,
)
from financebench.utils.errors import DatasetLoadError

__all__ = ["SECQUE_CATEGORIES", "SecqueAdapter"]

#: The dataset revision. Pinned: "SECQUE" without a revision is not a benchmark, it is a moving target.
_PINNED_REVISION = "894196b8764a9326be8d51d5689b63e68462dae2"
_BASE_URL = f"https://huggingface.co/datasets/nogabenyoash/SecQue/resolve/{_PINNED_REVISION}"
_DEFAULT_DATA_DIR = Path("data/downloads/secque")

#: file -> (sha256, expected row count). Both are checked. A dataset that silently changed under us is
#: a dataset whose scores are not comparable with yesterday's, and the whole point of the evaluator
#: fingerprint is that such a change must be impossible to miss.
_FILES: dict[str, tuple[str, int]] = {
    "SECQUE_benchmark_Analysis.jsonl": (
        "2faaf8931a7908a364c3d89d21b7fdb9d215587379b9082fdf5d781bad1a09ea",
        72,
    ),
    "SECQUE_benchmark_Comparison.jsonl": (
        "9eea0ca52d3f94ccb46f10646bcc41a06707eeeecb9de4a1b207943f698ffce4",
        220,
    ),
    "SECQUE_benchmark_Ratio.jsonl": (
        "b88be41b83f23eb15634ac266c2c03d594ed274f7074bcc7f932cd4a041ebd9a",
        188,
    ),
    "SECQUE_benchmark_Risk.jsonl": (
        "bbf17a2051db1dba15d58b9d4904bd47414a8f0fc16d23f3af3b1ad2b60c8a38",
        85,
    ),
}

#: How numeric each category actually is. Consulted by the diagnostics, which refuse to grade the
#: arithmetic of an essay.
SECQUE_CATEGORIES: dict[str, str] = {
    "Ratio": "numeric",  # formula + operands + a final value
    "Comparison": "numeric",  # figures per year + a direction of travel
    "Analysis": "mixed",  # a number, then an interpretation of it
    "Risk": "narrative",  # essentially no numbers at all
}

_TOTAL = 565

#: The context's first line names the filer. It comes in three shapes, and only one of them was
#: obvious — the other two cover **236 of the 565 tasks**, so a regex that handled the first alone
#: would have left the filing check silently "not applicable" on 42 % of the benchmark:
#:
#:   Apple Inc. 10-K form for the fiscal year ended 2023-09-30, page 29:
#:   GENERAL MILLS INC 10-K form, page 42:                                   <- no date at all
#:   BANK OF AMERICA CORP /DE/ 10-Q form for quarterly period ended 2024-06-30, page 11:
#:
#: The period is optional because for some filings the data simply does not carry one. An absent
#: period is recorded as absent; it is not guessed from the filing's other fields.
_HEADER = re.compile(
    r"^(?P<company>.+?)\s+10-[KQ]\s+form"
    r"(?:\s+for\s+(?:the\s+fiscal\s+year|quarterly\s+period)\s+ended\s+(?P<period>[\d-]+))?"
    r"\s*,\s*page",
    re.IGNORECASE,
)


def _company_and_period(context: str) -> tuple[str, str]:
    """Read the filer and the period off the context's own first line.

    Deliberately parsed from the *context* — which the model also sees — rather than from the gold
    answer. A "wrong company" check whose expectation came from the answer key would be reaching
    across the fence for something the question already told everybody.
    """
    match = _HEADER.match(context.lstrip())
    if not match:
        return "", ""
    # The period group is optional — some filings carry no date in the header. An absent period stays
    # absent rather than being guessed from elsewhere.
    return match.group("company").strip(), (match.group("period") or "").strip()


@register_dataset("secque")
class SecqueAdapter(DatasetAdapter):
    name = "secque"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    # -- preparation --------------------------------------------------------

    def prepare(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        for filename in _FILES:
            target = self._data_dir / filename
            if not target.is_file():
                download_file(f"{_BASE_URL}/{filename}", target, max_bytes=200 * 1024 * 1024)
        self._verify()

    def _verify(self) -> None:
        """Checksum every file. A benchmark is only reproducible if the bytes are the same bytes."""
        for filename, (expected_sha, expected_rows) in _FILES.items():
            path = self._data_dir / filename
            if not path.is_file():
                raise DatasetLoadError(
                    f"secque file missing: {path}. Run `financebench prepare secque`."
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_sha:
                raise DatasetLoadError(
                    f"secque checksum mismatch for {filename}.\n"
                    f"  expected {expected_sha}\n  got      {actual}\n"
                    f"The upstream data changed, or the download is corrupt. Either way the scores "
                    f"would not be comparable with any run made before it — so this refuses rather "
                    f"than quietly evaluating a different benchmark under the same name."
                )
            rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            if rows != expected_rows:
                raise DatasetLoadError(
                    f"secque {filename}: expected {expected_rows} rows, found {rows}"
                )

    # -- loading ------------------------------------------------------------

    def load(self, split: str) -> Sequence[CanonicalSample]:
        """SECQUE ships a single ``test`` split. The four category files are not splits — they are
        strata within it, and treating them as splits would let somebody report "SECQUE: 87 %" from
        the 72-sample Analysis file alone."""
        if split != "test":
            raise DatasetLoadError(
                f"secque has no split {split!r}; it ships exactly one: 'test' (565 tasks). "
                "The four category files are strata, not splits."
            )
        self._verify()
        samples = list(self._samples())
        if len(samples) != _TOTAL:
            raise DatasetLoadError(f"secque: expected {_TOTAL} tasks, built {len(samples)}")
        return samples

    def _samples(self) -> Iterator[CanonicalSample]:
        """All 565, in a **deterministic stratified interleave** — round-robin across the four
        categories rather than one file after another.

        This is not cosmetic. The categories are wildly unbalanced (Comparison 220, Ratio 188,
        Risk 85, Analysis 72) and they measure *different things*: Ratio is arithmetic, Risk is an
        essay. Emitted file-by-file, ``--max-samples 80`` returns **72 Analysis and 8 Comparison** —
        zero Ratio, zero Risk — and reports it as "SECQUE". That is not a small sampling artefact; it
        is a different benchmark wearing SECQUE's name, and it would have been invisible in the
        output.

        Interleaved, any prefix is approximately balanced: 80 samples is roughly 20 of each. The order
        is fully deterministic (sorted by QID within each stratum), so a run is reproducible and a
        frozen manifest can pin exactly these ids.
        """
        by_category: dict[str, list[CanonicalSample]] = {}
        for filename in sorted(_FILES):
            path = self._data_dir / filename
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                sample = self._to_sample(json.loads(line))
                by_category.setdefault(sample.metadata["category"], []).append(sample)

        for group in by_category.values():
            group.sort(key=lambda s: s.metadata["qid"])

        # Round-robin. A stratum that runs out simply stops contributing; the rest carry on, so all
        # 565 are still emitted and only their ORDER changes.
        streams = [by_category[key] for key in sorted(by_category)]
        for index in range(max((len(s) for s in streams), default=0)):
            for stream in streams:
                if index < len(stream):
                    yield stream[index]

    def _to_sample(self, record: dict[str, Any]) -> CanonicalSample:
        qid = str(record["QID"]).strip()
        category = str(record["question_type"]).strip()
        question = str(record["Question"]).strip()
        reference = str(record["ground_truth_answer"]).strip()
        context = str(record["context_markdown_with_headers"])
        accession = str(record.get("accession_number", "")).strip()

        company, period = _company_and_period(context)
        pages = tuple(
            int(p) for p in re.findall(r"\d+", str(record.get("page_number", ""))) if p.isdigit()
        )

        return CanonicalSample(
            benchmark="secque",
            benchmark_version=f"hf@{_PINNED_REVISION[:8]}",
            split="test",
            split_origin=SplitOrigin.OFFICIAL,
            sample_id=f"secque:test:{qid}",
            task_family=f"secque_{category.lower()}",
            # NOT tagged `analysis`/`insight`, and that is deliberate.
            #
            # Those tags feed the ANALYTICAL_INSIGHT capability dimension, which is scored from the
            # benchmark's preferred metric — here, the hallucination detector. The first live run
            # reported `analysis_score: 0.9`, which reads as "this model is a good financial analyst".
            # What it actually meant was "90% of its answers contain no invented figures", while the
            # same run showed it agreeing with the expert's numbers **8%** of the time and discussing
            # the **wrong company 41%** of the time.
            #
            # Analytical QUALITY is exactly the thing this platform cannot currently measure: the
            # judge failed its calibration (41% false-positive rate — it waves through answers about
            # the wrong company). So SECQUE contributes to the dimensions its diagnostics genuinely
            # cover — grounding, and reading a filing — and the analytical dimension stays
            # NOT_EVALUATED rather than being quietly filled in with a number that means something
            # else entirely.
            capability_tags=("evidence_grounding", "table_text"),
            question=question,
            # The SEC excerpt IS the model's context. The expert answer is NOT — it lives on `gold`,
            # which no prompt profile may read (tests/security/test_gold_answer_leakage.py).
            context=SampleContext(text=(context,)),
            gold=GoldAnswer(
                answer=reference,
                # TEXT, not NUMERIC — and that is a statement of fact, not a shrug. The gold is an
                # expert's prose. Typing it as numeric would invite an exact-match metric that cannot
                # exist, and inventing one is how this benchmark gets faked.
                answer_type=AnswerType.TEXT,
                evidence=tuple(
                    Evidence(document_id=accession or company, page=page) for page in pages
                ),
            ),
            evaluation=EvaluationSpec(),
            source=SourceInfo(
                license="MIT",
                url=f"https://huggingface.co/datasets/nogabenyoash/SecQue/tree/{_PINNED_REVISION}",
                redistributable=True,
            ),
            language="en",
            metadata={
                "category": category,
                "numeracy": SECQUE_CATEGORIES.get(category, "unknown"),
                "company": company,
                "period": period,
                "accession_number": accession,
                "item": str(record.get("item", "")).strip(),
                "page_number": str(record.get("page_number", "")).strip(),
                "qid": qid,
            },
        )

    # -- manifest -----------------------------------------------------------

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="secque",
            official_source="nogabenyoash/SecQue (HuggingFace)",
            paper_url="https://arxiv.org/abs/2504.04596",
            repository_url="https://github.com/EnvCommons/SECQUE",
            version_or_commit=_PINNED_REVISION,
            download_method="https (HuggingFace, pinned revision, sha256-checked)",
            official_splits=("test",),
            local_splits=("test",),
            license="MIT",
            redistribution_status="redistributable",
            expected_files=tuple(sorted(_FILES)),
            status=AdapterStatus.FULLY_SUPPORTED,
            status_tested_at="2026-07-11T00:00:00Z",
            known_limitations=(
                "SECQUE's gold is an expert's PROSE answer, not a number. There is no exact-match "
                "metric and there cannot be one. This platform therefore reports two things and "
                "never pretends they are one: deterministic DIAGNOSTICS (the numbers, the ratio, the "
                "direction, the company, the period) and, separately, a CALIBRATED JUDGE score whose "
                "calibration is published next to it.",
                "The four categories are not interchangeable. Ratio (188) and Comparison (220) are "
                "largely checkable deterministically; Analysis (72) is half checkable; Risk (85) is "
                "essentially narrative and contains almost no numbers. Deterministic metrics return "
                "passed=None on samples they cannot see — never 0.0. Scoring an essay on its "
                "arithmetic and calling the result 'accuracy' would be the single easiest lie this "
                "benchmark affords.",
                "Contexts are large (~8k tokens). On a small local model this is the dominant cost, "
                "and a truncated context is a different benchmark — so truncation is never silent.",
                "All 565 public tasks are the 'test' split. The category files are STRATA, not "
                "splits: reporting 'SECQUE' from the 72-sample Analysis file alone would be a "
                "four-fold overstatement of coverage.",
            ),
        )
