"""The evaluator fingerprint — what makes two runs comparable, and what makes them not.

A benchmark's scores are only meaningful relative to the code that produced them. Last session
proved that the hard way: fixing the answer parser moved qwen2.5:3b's FinQA score from 5 % to 15 %
**on the same cached model responses**. Nothing about the model changed. If those two numbers had
been allowed to sit next to each other on a leaderboard, the leaderboard would have been lying.

So every run records an **evaluator fingerprint**: a hash over the versioned pieces of the
evaluation pipeline that can move a score without the model doing anything differently —

- the answer-parser version,
- the prompt-profile versions (what the model was *asked* for),
- the metric implementation versions,
- the dataset adapter versions (which data, pinned to which upstream commit),
- the scoring configuration (capability weights, gate thresholds).

Two runs with different fingerprints are **not comparable**, and `compare`/`leaderboard` say so
rather than quietly averaging them. The point is not to prevent the pipeline from improving — it is
to make an improvement *visible* instead of retroactively rewriting history.

Bumping a `*_VERSION` constant below is therefore a deliberate act: it declares "this change can
move a score, and old runs must not be compared to new ones without being re-scored."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = [
    "DATASET_ADAPTER_VERSIONS",
    "METRIC_VERSIONS",
    "PARSER_VERSION",
    "RETRIEVAL_VERSION",
    "SCORING_VERSION",
    "EvaluatorFingerprint",
    "current_fingerprint",
]

#: The structured-answer parser (`FinancialAnswer.from_text` + `evaluation/numeric.py`).
#: v2: tolerate `"insufficient_information": null` — a strict bool was discarding ~half of all
#: valid answers and blaming the model for it. This bump is why old runs must be re-scored.
PARSER_VERSION = "2"

#: Per-metric implementation versions. Bump when a metric's *behaviour* changes.
METRIC_VERSIONS: dict[str, str] = {
    "exact_match": "1",
    "finqa_execution_accuracy": "1",  # official, parity-tested
    "finqa_program_accuracy": "1",  # official, parity-tested
    "finqa_answer_accuracy": "2",  # ours; v2 reconciles FinQA's fraction-vs-percent convention
    "tatqa_exact_match": "2",  # v2 restores the official int-vs-float scale semantics
    "tatqa_f1": "2",
    "tatqa_scale_accuracy": "1",
    "finance_reasoning_accuracy": "1",  # official, parity-tested
    "financebench_answer_accuracy": "1",  # ours — FinanceBench ships no evaluator
    # v3: a match now needs the LEADING DIGITS to agree, not merely a 0.5% window after scaling.
    # A SECQUE filing carries 733 numbers; times nine scale factors, the candidate set was so dense
    # that an invented 987,654,321 was "supported" by the filing's 983 (983e6 is 0.47% away). The
    # detector got WEAKER the more numbers a document had — backwards, and worst exactly where
    # hallucination matters most. v3 is STRICTER: every unsupported-claim rate measured before it is
    # an understatement.
    "financebench_unsupported_numeric_claim": "3",
    "financebench_citation_accuracy": "1",
    "smb_cfo_accuracy": "1",  # gold from a Python oracle, never an LLM
    # v2 reads refusal from the SUBSTANCE of the answer. v1 read only the `insufficient_information`
    # flag, so a model that correctly declined in its own words was recorded as having hallucinated
    # — the metric was measuring schema compliance and reporting it as dangerous invention. Every
    # v1 refusal number is wrong, which is exactly what a fingerprint bump is for.
    "smb_cfo_refusal_correctness": "2",
    "smb_cfo_injection_resistance": "1",
    "convfinqa_turn_accuracy": "1",  # ours; ConvFinQA's official metrics grade programs
    "convfinqa_execution_accuracy": "1",  # official (FinQA's parity-tested executor)
    "convfinqa_program_accuracy": "1",  # official
    # SECQUE Layer A. DIAGNOSTICS, not a quality score — and named so in the metric docstrings.
    # SECQUE's gold is an expert's prose; there is no exact-match metric and there cannot be one.
    "secque_numeric_agreement": "1",
    "secque_unsupported_numeric_claim": "1",  # uses grounding v3 (leading-digit match)
    "secque_comparison_direction": "1",
    "secque_filing_identification": "1",
    # Tool use. `tool_result_utilization` is the one that matters: a model that calls the calculator,
    # receives 40.55, and writes "approximately 38" has made a TRUST error, not an arithmetic one —
    # and every end-to-end metric misattributes it to the sums, which is the thing it got right.
    "tool_selection_accuracy": "1",
    "tool_execution_success": "1",
    "tool_result_utilization": "1",
    "tool_security_rejection": "1",
}

#: Dataset adapters, pinned to the upstream commit their data comes from. A locally *generated*
#: dataset is pinned to its generator version instead: regenerate SMB-CFO with different oracles and
#: every SMB-CFO score in the repo becomes incomparable with the ones before it, which is precisely
#: the thing this dict exists to make visible.
DATASET_ADAPTER_VERSIONS: dict[str, str] = {
    "finqa": "official@0f16e286",
    "tatqa": "official@870accc4",
    "finance_reasoning": "official@b0fe6455",
    "financebench": "open_source@cc39aeb4",
    "convfinqa": "official@cf3eed2d",
    "secque": "hf@894196b8",
    "smb_cfo": "generated@1",
    "smoke": "in-repo@1",
}

#: The retrieval pipeline. It belongs in the fingerprint because it moves every
#: ``retrieval_required`` score without the model changing at all — which is exactly what happened.
#:
#: v2: ``document_scoped`` now actually narrows the corpus to the filing the question names. It used
#: to leave the retriever searching all 12,013 pages and merely paste the document's name onto the
#: front of the query, so a run artifact stamped ``document_scoped: true`` while nothing had been
#: scoped at all. The label described a setting the code never entered.
#:
#: The cost of that bug, measured: BM25 page recall @10 in the "document-scoped" run was **4.0 %** —
#: which is *precisely* the open-corpus number, because that is what it was actually doing. With the
#: corpus genuinely narrowed it is **18.7 %**, nearly five times better. Every v1 retrieval number
#: describes a setting nobody asked for.
RETRIEVAL_VERSION = "2"

#: Capability weights, gate thresholds, and how a per-sample result reaches a capability score.
#: Changing any of them moves every verdict.
#:
#: v2: a NOT-APPLICABLE result is excluded from the rollup instead of being scored as 0.0, and a
#: dimension is scored by the metric that measures it. Both bugs invented failures the model never
#: committed: on the real FinanceBench run, document grounding was reported as 0.151 when the truth
#: over the 89 gradable questions is 0.254. Every v1 capability score is wrong, low, and not
#: comparable with a v2 one — which is precisely what a scoring-version bump is for.
SCORING_VERSION = "2"


@dataclass(frozen=True)
class EvaluatorFingerprint:
    """Everything about *our* code that can move a score without the model changing."""

    parser_version: str
    prompt_profiles: dict[str, str]
    metric_versions: dict[str, str]
    dataset_adapters: dict[str, str]
    retrieval_version: str
    scoring_version: str
    scoring_config_hash: str

    @property
    def digest(self) -> str:
        """A short, stable hash of the whole fingerprint."""
        payload = json.dumps(
            {
                "parser": self.parser_version,
                "prompts": dict(sorted(self.prompt_profiles.items())),
                "metrics": dict(sorted(self.metric_versions.items())),
                "datasets": dict(sorted(self.dataset_adapters.items())),
                "retrieval": self.retrieval_version,
                "scoring": self.scoring_version,
                "scoring_config": self.scoring_config_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "parser_version": self.parser_version,
            "prompt_profiles": self.prompt_profiles,
            "metric_versions": self.metric_versions,
            "dataset_adapters": self.dataset_adapters,
            "retrieval_version": self.retrieval_version,
            "scoring_version": self.scoring_version,
            "scoring_config_hash": self.scoring_config_hash,
        }

    def comparable_with(self, other: EvaluatorFingerprint) -> bool:
        return self.digest == other.digest


def _scoring_config_hash() -> str:
    """Hash the capability weights and gate thresholds — the numbers that decide every verdict."""
    from financebench.evaluation.capability_map import CAPABILITY_WEIGHTS
    from financebench.evaluation.gates import GATE_THRESHOLDS

    payload = json.dumps(
        {
            "weights": {k.value: v for k, v in sorted(CAPABILITY_WEIGHTS.items())},
            "gates": dict(sorted(GATE_THRESHOLDS.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def current_fingerprint() -> EvaluatorFingerprint:
    """The fingerprint of the evaluation pipeline as it exists right now."""
    from financebench.prompts.profiles import available_prompt_profiles, create_prompt_profile

    # A prompt profile is versioned in its *name* (`structured_financial_v1`), so the name is the
    # version. Its system text is hashed too, because editing a prompt without renaming it would
    # otherwise change what the model was asked while claiming nothing had moved.
    profiles: dict[str, str] = {}
    for name in available_prompt_profiles():
        profile = create_prompt_profile(name)
        text = f"{profile.response_format}|{profile.elicits_program}"
        profiles[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    return EvaluatorFingerprint(
        parser_version=PARSER_VERSION,
        prompt_profiles=profiles,
        metric_versions=dict(METRIC_VERSIONS),
        dataset_adapters=dict(DATASET_ADAPTER_VERSIONS),
        retrieval_version=RETRIEVAL_VERSION,
        scoring_version=SCORING_VERSION,
        scoring_config_hash=_scoring_config_hash(),
    )
