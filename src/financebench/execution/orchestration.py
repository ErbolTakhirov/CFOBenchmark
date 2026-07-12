"""End-to-end orchestration: resolve a benchmark/group + model config into samples, run them
through the engine, score them, and write the full run-artifact set.

This is the one function both the ``eval`` and ``resume`` CLI commands call — ``resume`` is not
a separate code path, it is the same idempotent operation pointed at the same run id (see
``execution/cache.py`` for why a cache hit *is* resume).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from financebench import __version__
from financebench.config.model_config import ModelConfigFile
from financebench.datasets.base import create_dataset
from financebench.evaluation.benchmark_metrics import metrics_for_run, preferred_metric_name
from financebench.evaluation.capability_map import rollup_capabilities
from financebench.evaluation.conversation import analyze_conversations
from financebench.evaluation.failures import (
    INFRASTRUCTURE_FAILURES,
    FailureRecord,
    attribute_failure,
)
from financebench.evaluation.gates import evaluate_gates, verdict_for
from financebench.evaluation.metrics.base import Metric
from financebench.evaluation.refusal import declined
from financebench.evaluation.scoring import RunCoverage, compute_scores
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine, RunResult
from financebench.models.base import ModelProvider, get_provider_class
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.retrieval.metrics import attribute_retrieval_failure, score_retrieval
from financebench.retrieval.pipeline import RetrievalPipeline, build_pipeline
from financebench.schemas.common import (
    DEFAULT_PROMPT_PROFILE,
    ConversationProtocol,
    EvalMode,
    RunType,
)
from financebench.schemas.manifest import DatasetManifest
from financebench.schemas.metric import MetricResult
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample
from financebench.schemas.sample_manifest import SampleManifest
from financebench.storage.artifacts import ArtifactInputs, write_run_artifacts
from financebench.utils.errors import ConfigError, ManifestError
from financebench.utils.ids import make_run_id
from financebench.utils.timing import RealClock

__all__ = ["EvalOutcome", "EvalRequest", "resolve_samples", "run_eval"]


def _score_or_excuse(
    metric: Metric, sample: CanonicalSample, prediction: Prediction
) -> MetricResult:
    """Score the sample — unless the provider never gave us an answer to score.

    ``response is None`` is the same condition :func:`attribute_failure` already treats as
    infrastructure: the call failed, so nothing the model did was observed.

    A network timeout is **our** failure, not the model's. The gate denominator has always known this
    (see ``n_scored`` below), but the metrics did not: they are handed ``answer=None`` for a timeout
    and for a genuinely unparseable model reply alike, and graded both ``passed=False`` with the
    reason "no parsed answer".

    So on the SECQUE 3B run, three ``ollama request timed out after 180.0s`` errors — caused by GPU
    contention with a *different* process on this machine — were published as the model getting three
    financial questions wrong, in exactly the metrics the release wants to compare against the 7B
    (which ran at a 300 s timeout and errored zero times). That comparison was partly measuring our
    own timeout budget.

    A metric that was never given an answer returns **not applicable**: not a zero, not a failure —
    an absence of evidence, which the rollup already knows to exclude.
    """
    if prediction.response is None:
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=metric.name,
            value=None,
            passed=None,
            details={
                "reason": "not applicable — the provider returned no answer to grade",
                "error_type": prediction.error_type or "unknown",
            },
        )
    return metric.score(sample, prediction)


@dataclass(frozen=True)
class EvalRequest:
    """Everything needed to run one evaluation end to end."""

    label: str
    """The ``--benchmark`` name or ``--group`` name — used for the run id and reports."""
    benchmark_splits: tuple[tuple[str, str], ...]
    """``(dataset_name, split)`` pairs to evaluate, in order."""
    model_config_file: ModelConfigFile
    cache_dir: Path
    seed: int = 42
    max_samples: int | None = None
    max_cost_usd: float | None = None
    offline: bool = False
    allow_mock: bool = False
    """Must be explicitly set to run the ``mock`` provider. See :func:`_build_provider`."""
    prompt_profile: str = DEFAULT_PROMPT_PROFILE
    eval_mode: EvalMode = EvalMode.CONTEXT_GIVEN
    conversation_protocol: ConversationProtocol = ConversationProtocol.GOLD_HISTORY
    retriever: str = "bm25"
    top_k: int = 5
    document_scoped: bool = False
    #: A frozen sample manifest. When set, the run asks EXACTLY the questions it names, and a
    #: named question that no longer resolves fails the run rather than being quietly replaced.
    sample_manifest: SampleManifest | None = None


@dataclass(frozen=True)
class EvalOutcome:
    run_id: str
    out_dir: Path
    run_result: RunResult
    run_type: RunType


def resolve_samples(
    benchmark_splits: Sequence[tuple[str, str]],
    manifest: SampleManifest | None = None,
) -> tuple[tuple[CanonicalSample, ...], tuple[DatasetManifest, ...]]:
    """Load every requested (dataset, split) pair's samples and manifests, in order.

    With a frozen ``manifest``, the run is restricted to exactly the sample ids it names — and a
    named id that no longer resolves is a **hard error**, not a warning. That is the whole point: if
    an upstream dataset shifts by one row, "the first 40 samples" becomes a different 40 while every
    artifact still reads ``limit: 40``, and nothing anywhere notices. A manifest turns that silent
    substitution into a failed run.
    """
    samples: list[CanonicalSample] = []
    manifests: list[DatasetManifest] = []
    for name, split in benchmark_splits:
        adapter = create_dataset(name)
        manifests.append(adapter.manifest())
        samples.extend(adapter.load(split))

    if manifest is None:
        return tuple(samples), tuple(manifests)

    by_id = {sample.sample_id: sample for sample in samples}
    wanted = manifest.all_sample_ids
    missing = [sid for sid in wanted if sid not in by_id]
    if missing:
        raise ManifestError(
            f"frozen manifest {manifest.name!r} names {len(missing)} sample id(s) that no longer "
            f"resolve — the data underneath has moved, and this run would silently evaluate a "
            f"DIFFERENT set of questions under the manifest's name.\n"
            f"  first missing: {missing[:3]}\n"
            f"  re-freeze the manifest, or pin the dataset to the revision it was frozen against."
        )
    # Manifest order is the run order, so two runs over the same manifest execute in the same order.
    selected = tuple(by_id[sid] for sid in wanted)
    return selected, tuple(manifests)


def _build_provider(
    model: ModelSpec, samples: Sequence[CanonicalSample], *, allow_mock: bool
) -> ModelProvider:
    """Construct the provider for a run.

    The ``mock`` provider is the one special case in the whole platform, and it is special
    deliberately: it is a *simulator holding the answer key*, so it (a) must be asked for
    explicitly, and (b) is the only place the gold oracle is ever handed over. Every other
    provider is built by the registry from the environment and can never see gold — the request
    it receives has no field that could carry it (see ``schemas/model_io.ModelRequest``).
    """
    if model.provider != "mock":
        return get_provider_class(model.provider).from_env()

    if not allow_mock:
        raise ConfigError(
            "refusing to evaluate with the 'mock' provider without --allow-mock.\n"
            "The mock reads the gold answer: its scores measure the pipeline, never a model. "
            "Pass --allow-mock if you are deliberately testing the pipeline; such runs are "
            "stamped run_type=mock_test and are barred from the leaderboard."
        )
    return MockProvider(oracle=build_mock_oracle(samples))


def run_id_for(request: EvalRequest) -> str:
    """The run id for ``request`` — **the single definition**, used by the CLI to pick the output
    directory and by :func:`run_eval` to stamp the artifacts.

    The prompt profile and eval mode are part of a run's identity: asking a model for a program
    and asking it for a number are different runs and must not share a directory. Two call sites
    computing this separately is exactly how they drift — which is what happened here, and how a
    ``program_v1`` run silently landed in a ``structured_financial_v1`` run's directory.

    The conversation protocol joins them for the same reason — a ``model_history`` run and a
    ``gold_history`` run measure different things and must never overwrite one another. It is
    appended only when it is *not* the default, so that adding conversations to the platform did not
    silently rename every existing single-turn run's directory. ``gold_history`` is the official
    protocol and the default; a run that departs from it says so in its id.

    **The retrieval arm joins them, and for the sharpest reason of the three.** BM25 over one filing
    at k=10 and hybrid over 12,013 pages at k=20 are not two settings of one experiment — they are
    two experiments, whose whole purpose is to be *compared against each other*. They were both
    landing on the same run id, so the second one to run would have overwritten the first, in place,
    and the resulting directory would have described an arm that no longer existed anywhere. The
    ablation would have been comparing a run against itself.

    Appended only for ``retrieval_required``, because a ``context_given`` run has no retriever, and
    stamping ``-bm25-k5`` onto every FinQA run ever written would rename them all for nothing.
    """
    model = request.model_config_file.to_model_spec()
    label = f"{request.label}-{request.prompt_profile}-{request.eval_mode.value}"
    if request.conversation_protocol is not ConversationProtocol.GOLD_HISTORY:
        label = f"{label}-{request.conversation_protocol.value}"
    if request.eval_mode is EvalMode.RETRIEVAL_REQUIRED:
        scope = "scoped" if request.document_scoped else "open"
        label = f"{label}-{request.retriever}-k{request.top_k}-{scope}"
    return make_run_id(label, model.ref, request.seed)


async def run_eval(request: EvalRequest, *, out_dir: Path) -> EvalOutcome:
    model = request.model_config_file.to_model_spec()
    run_type = RunType.MOCK_TEST if model.provider == "mock" else RunType.REAL

    if request.offline:
        provider_cls = get_provider_class(model.provider)
        if getattr(provider_cls, "REQUIRES_KEY", False):
            raise ConfigError(
                f"--offline was set but provider {model.provider!r} requires network access"
            )

    samples, dataset_manifests = resolve_samples(request.benchmark_splits, request.sample_manifest)
    if not samples:
        raise ConfigError(
            f"no samples resolved for {request.label!r} "
            f"(benchmarks/splits: {request.benchmark_splits})"
        )

    config = request.model_config_file.to_run_config(
        seed=request.seed,
        limit=request.max_samples,
        max_cost_usd=request.max_cost_usd,
        prompt_profile=request.prompt_profile,
        eval_mode=request.eval_mode,
        conversation_protocol=request.conversation_protocol,
        retriever=request.retriever,
        top_k=request.top_k,
        document_scoped=request.document_scoped,
    )
    run_id = run_id_for(request)
    cache = ResponseCache(request.cache_dir, mode=config.cache_mode)

    # The oracle is built from the *truncated* sample list the engine will actually run, so a
    # --max-samples mock run doesn't get handed answers it never asked for.
    limited = list(samples)[: config.limit] if config.limit else list(samples)
    provider = _build_provider(model, limited, allow_mock=request.allow_mock)

    # retrieval_required: the model sees ONLY what the retriever finds. The sample's own context —
    # which for FinanceBench IS the gold evidence — is withheld unconditionally (prompts/profiles).
    pipeline: RetrievalPipeline | None = None
    retrieve = None
    if config.eval_mode is EvalMode.RETRIEVAL_REQUIRED:
        pdf_dir = Path("data/downloads") / limited[0].benchmark / "pdfs"
        if not pdf_dir.is_dir():
            raise ConfigError(
                f"retrieval_required needs a document corpus at {pdf_dir}, which does not exist. "
                f"Run: financebench prepare {limited[0].benchmark}"
            )
        pipeline = build_pipeline(
            limited,
            pdf_dir=pdf_dir,
            retriever_name=request.retriever,
            top_k=request.top_k,
            document_scoped=request.document_scoped,
        )
        retrieve = pipeline.retrieve_for

    try:
        provider_capabilities = provider.capabilities(model.model)
        run_result = await RunEngine().run(
            samples=samples,
            model=model,
            config=config,
            cache=cache,
            provider=provider,
            max_cost_usd=request.max_cost_usd,
            retrieve=retrieve,
        )
    finally:
        await provider.aclose()

    # run_result.samples is the (possibly --max-samples-truncated) list actually run, 1:1 with
    # run_result.predictions — score and report against *that*, not the pre-truncation `samples`.
    evaluated_samples = run_result.samples
    profile = config.prompt_profile
    all_metric_results: list[MetricResult] = []
    for sample, prediction in zip(evaluated_samples, run_result.predictions, strict=True):
        for metric in metrics_for_run(sample.benchmark, profile, config.eval_mode):
            all_metric_results.append(_score_or_excuse(metric, sample, prediction))
    metric_results = tuple(all_metric_results)

    # Every applicable metric is recorded (metric_details.jsonl/metrics.json), but only the
    # "preferred" one per sample — the benchmark's official metric where the prompt profile makes
    # one computable, else our own — feeds the capability-dimension rollup.
    sample_by_id = {sample.sample_id: sample for sample in evaluated_samples}
    preferred_results = tuple(
        result
        for result in metric_results
        if result.metric_name
        == preferred_metric_name(sample_by_id[result.sample_id].benchmark, profile)
    )
    # `all_results` lets a dimension be scored by the metric that MEASURES it: refusal is graded
    # by the refusal metric, not by accuracy — which cannot apply to an unanswerable question.
    capability_aggregates = rollup_capabilities(
        evaluated_samples, preferred_results, all_results=metric_results
    )

    # -- failure attribution, gates, scores, verdict ------------------------------------------
    preferred_by_sample = {result.sample_id: result for result in preferred_results}
    failures: list[FailureRecord] = []
    for sample, prediction in zip(evaluated_samples, run_result.predictions, strict=True):
        record = attribute_failure(sample, prediction, preferred_by_sample.get(sample.sample_id))
        if record is not None:
            failures.append(record)

    # A network timeout is OUR failure, not the model's. Scoring it as a financial-reasoning error
    # would be a lie about the model, so infrastructure failures are excluded from the denominator.
    n_infrastructure = sum(1 for f in failures if f.failure_type in INFRASTRUCTURE_FAILURES)
    n_scored = len(evaluated_samples) - n_infrastructure

    numeric_accuracy = None
    numeric_scores = [
        1.0 if result.passed else 0.0
        for result in preferred_results
        if sample_by_id[result.sample_id].gold.numeric_value is not None
    ]
    if numeric_scores:
        numeric_accuracy = sum(numeric_scores) / len(numeric_scores)

    # Retrieval is graded HERE, after inference — the only place gold evidence is ever read. The
    # attribution is what makes the number actionable: "the retriever never found the page" and
    # "the retriever found it and the model still got it wrong" have opposite fixes, and a single
    # RAG-accuracy number cannot tell them apart, so it sends you to fix the wrong component.
    retrieval_summary: dict[str, object] | None = None
    if pipeline is not None:
        response_by_id = {p.sample_id: p.response for p in run_result.predictions}
        failure_by_id = {f.sample_id: f for f in failures}

        retrieval_scores = []
        reattributed: list[FailureRecord] = []
        for sample in evaluated_samples:
            result = pipeline.results.get(sample.sample_id)
            if result is None:
                continue
            retrieval_score = score_retrieval(sample, result)
            retrieval_scores.append(retrieval_score)

            existing = failure_by_id.get(sample.sample_id)
            if existing is None:
                continue
            preferred = preferred_by_sample.get(sample.sample_id)
            response = response_by_id.get(sample.sample_id)
            # Refusal is read from the substance of the answer, not from whether the model set the
            # flag we asked for — a model that says "the retrieved excerpts don't cover this" has
            # declined, and calling that a generation error would blame it for the retriever's miss.
            refused = declined(response.financial_answer if response else None)
            attributed = attribute_retrieval_failure(
                retrieval_score,
                answer_correct=bool(preferred and preferred.passed),
                refused=refused,
            )
            reattributed.append(
                existing.model_copy(update={"failure_type": attributed})
                if attributed is not None
                else existing
            )

        retrieved_ids = set(pipeline.results)
        failures = reattributed + [f for f in failures if f.sample_id not in retrieved_ids]

        n = len(retrieval_scores) or 1
        retrieval_summary = {
            **pipeline.to_json(),
            "document_recall": sum(s.document_hit for s in retrieval_scores) / n,
            "page_recall": sum(s.page_hit for s in retrieval_scores) / n,
            "evidence_precision": sum(s.evidence_precision for s in retrieval_scores) / n,
            "evidence_recall": sum(s.evidence_recall for s in retrieval_scores) / n,
            "evidence_f1": sum(s.evidence_f1 for s in retrieval_scores) / n,
            "n_scored": len(retrieval_scores),
        }

    # Conversation-level analysis: what a per-turn score cannot see. Returns None for a benchmark
    # with no conversations, so a FinQA run never acquires an empty report that reads as a
    # measurement of something that was not measured.
    conversation = analyze_conversations(
        evaluated_samples, preferred_by_sample, config.conversation_protocol
    )

    # Coverage decides what this run is ALLOWED to claim, not just what it scored. Read off the
    # samples themselves, so a future grounded or adversarial benchmark counts automatically.
    coverage = RunCoverage.of(evaluated_samples)

    # Per-benchmark means, for the one sub-score that is a benchmark rather than a capability.
    by_benchmark: dict[str, list[float]] = {}
    for scored_result in preferred_results:
        if scored_result.passed is None:
            continue
        by_benchmark.setdefault(sample_by_id[scored_result.sample_id].benchmark, []).append(
            1.0 if scored_result.passed else 0.0
        )
    benchmark_scores = {name: sum(v) / len(v) for name, v in by_benchmark.items() if v}

    # The sandbox gate is scored from the probes the run actually made. `None` — not 1.0 — when no
    # tool was ever offered: a run that never tested the sandbox has said nothing about it, and a
    # green tick would be a security claim with no evidence under it.
    security_results = [
        result for result in metric_results if result.metric_name == "tool_security_rejection"
    ]
    security_values = [
        (1.0 if result.value else 0.0) if isinstance(result.value, bool) else float(result.value)
        for result in security_results
        if isinstance(result.value, bool | int | float)
    ]
    tool_security_rejection = (
        sum(security_values) / len(security_values) if security_values else None
    )

    gates = evaluate_gates(
        failures=failures,
        n_scored=n_scored,
        numeric_accuracy=numeric_accuracy,
        n_injection_samples=coverage.n_injection_samples,
        tool_security_rejection=tool_security_rejection,
    )
    is_mock = run_type is RunType.MOCK_TEST
    scores = compute_scores(
        eval_mode=config.eval_mode,
        capabilities=capability_aggregates,
        failures=failures,
        n_scored=n_scored,
        any_critical_gate_failed=bool(gates.any_critical_gate_failed),
        is_mock=is_mock,
        has_multimodal_coverage=any(s.context.images for s in evaluated_samples),
        coverage=coverage,
        benchmark_scores=benchmark_scores,
    )
    # `a or b or c` would be wrong here: a legitimate score of 0.0 is FALSY in Python, so a model
    # that scored zero collapses to None and gets reported as NOT_EVALUATED — making the worst
    # possible model indistinguishable from one that was never run. Seen for real: qwen2.5:3b
    # scored exactly 0.000 on FinanceReasoning-hard, which is a true and important result, and the
    # report erased it.
    overall = next(
        (
            score
            for score in (scores.core_score, scores.rag_score, scores.agent_score)
            if score is not None
        ),
        None,
    )
    verdict, verdict_reasons = verdict_for(
        gates=gates, core_score=overall, n_scored=n_scored, is_mock=is_mock
    )

    clock = RealClock()
    inputs = ArtifactInputs(
        run_id=run_id,
        benchmark_or_group=request.label,
        model=model,
        provider_capabilities=provider_capabilities,
        config=config,
        created_at=clock.now_iso(),
        financebench_version=__version__,
        dataset_manifests=dataset_manifests,
        samples=evaluated_samples,
        run_result=run_result,
        metric_results=metric_results,
        capability_aggregates=capability_aggregates,
        run_type=run_type,
        failures=tuple(failures),
        gates=gates,
        scores=scores,
        verdict=verdict.value,
        verdict_reasons=tuple(verdict_reasons),
        retrieval=retrieval_summary,
        conversation=conversation.to_json() if conversation else None,
    )
    write_run_artifacts(out_dir, inputs)
    return EvalOutcome(run_id=run_id, out_dir=out_dir, run_result=run_result, run_type=run_type)
