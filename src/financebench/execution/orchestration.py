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
from financebench.evaluation.failures import (
    INFRASTRUCTURE_FAILURES,
    FailureRecord,
    attribute_failure,
)
from financebench.evaluation.gates import evaluate_gates, verdict_for
from financebench.evaluation.scoring import compute_scores
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine, RunResult
from financebench.models.base import ModelProvider, get_provider_class
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.common import DEFAULT_PROMPT_PROFILE, EvalMode, RunType
from financebench.schemas.manifest import DatasetManifest
from financebench.schemas.metric import MetricResult
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.sample import CanonicalSample
from financebench.storage.artifacts import ArtifactInputs, write_run_artifacts
from financebench.utils.errors import ConfigError
from financebench.utils.ids import make_run_id
from financebench.utils.timing import RealClock

__all__ = ["EvalOutcome", "EvalRequest", "resolve_samples", "run_eval"]


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


@dataclass(frozen=True)
class EvalOutcome:
    run_id: str
    out_dir: Path
    run_result: RunResult
    run_type: RunType


def resolve_samples(
    benchmark_splits: Sequence[tuple[str, str]],
) -> tuple[tuple[CanonicalSample, ...], tuple[DatasetManifest, ...]]:
    """Load every requested (dataset, split) pair's samples and manifests, in order."""
    samples: list[CanonicalSample] = []
    manifests: list[DatasetManifest] = []
    for name, split in benchmark_splits:
        adapter = create_dataset(name)
        manifests.append(adapter.manifest())
        samples.extend(adapter.load(split))
    return tuple(samples), tuple(manifests)


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
    """
    model = request.model_config_file.to_model_spec()
    return make_run_id(
        f"{request.label}-{request.prompt_profile}-{request.eval_mode.value}",
        model.ref,
        request.seed,
    )


async def run_eval(request: EvalRequest, *, out_dir: Path) -> EvalOutcome:
    model = request.model_config_file.to_model_spec()
    run_type = RunType.MOCK_TEST if model.provider == "mock" else RunType.REAL

    if request.offline:
        provider_cls = get_provider_class(model.provider)
        if getattr(provider_cls, "REQUIRES_KEY", False):
            raise ConfigError(
                f"--offline was set but provider {model.provider!r} requires network access"
            )

    samples, dataset_manifests = resolve_samples(request.benchmark_splits)
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
    )
    run_id = run_id_for(request)
    cache = ResponseCache(request.cache_dir, mode=config.cache_mode)

    # The oracle is built from the *truncated* sample list the engine will actually run, so a
    # --max-samples mock run doesn't get handed answers it never asked for.
    limited = list(samples)[: config.limit] if config.limit else list(samples)
    provider = _build_provider(model, limited, allow_mock=request.allow_mock)
    try:
        provider_capabilities = provider.capabilities(model.model)
        run_result = await RunEngine().run(
            samples=samples,
            model=model,
            config=config,
            cache=cache,
            provider=provider,
            max_cost_usd=request.max_cost_usd,
        )
    finally:
        await provider.aclose()

    # run_result.samples is the (possibly --max-samples-truncated) list actually run, 1:1 with
    # run_result.predictions — score and report against *that*, not the pre-truncation `samples`.
    evaluated_samples = run_result.samples
    profile = config.prompt_profile
    all_metric_results: list[MetricResult] = []
    for sample, prediction in zip(evaluated_samples, run_result.predictions, strict=True):
        for metric in metrics_for_run(sample.benchmark, profile):
            all_metric_results.append(metric.score(sample, prediction))
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
    capability_aggregates = rollup_capabilities(evaluated_samples, preferred_results)

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

    gates = evaluate_gates(failures=failures, n_scored=n_scored, numeric_accuracy=numeric_accuracy)
    is_mock = run_type is RunType.MOCK_TEST
    scores = compute_scores(
        eval_mode=config.eval_mode,
        capabilities=capability_aggregates,
        failures=failures,
        n_scored=n_scored,
        any_critical_gate_failed=bool(gates.any_critical_gate_failed),
        is_mock=is_mock,
        has_multimodal_coverage=any(s.context.images for s in evaluated_samples),
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
    )
    write_run_artifacts(out_dir, inputs)
    return EvalOutcome(run_id=run_id, out_dir=out_dir, run_result=run_result, run_type=run_type)
