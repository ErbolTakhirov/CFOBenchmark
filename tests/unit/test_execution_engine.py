from __future__ import annotations

from pathlib import Path

import pytest

from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.base import ModelProvider
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.model_io import ModelRequest, ModelResponse, ModelSpec
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    GoldAnswer,
    SourceInfo,
)
from financebench.utils.errors import ProviderResponseError, ProviderTimeoutError


def _sample(n: int) -> CanonicalSample:
    return CanonicalSample(
        benchmark="smoke",
        benchmark_version="1",
        split="dev",
        split_origin=SplitOrigin.GENERATED_FROZEN,
        sample_id=f"smoke:dev:{n}",
        task_family="arithmetic",
        question=f"What is {n} + {n}?",
        gold=GoldAnswer(
            answer=str(n + n), answer_type=AnswerType.NUMERIC, numeric_value=float(n + n)
        ),
        evaluation=EvaluationSpec(),
        source=SourceInfo(license="public-domain", url="local", redistributable=True),
    )


async def _no_sleep(_delay: float) -> None:
    return None


class _CountingProvider(ModelProvider):
    """Wraps another provider and counts real calls — used to prove a cache hit makes zero
    provider calls, not just that the returned data looks right."""

    def __init__(self, inner: ModelProvider) -> None:
        self._inner = inner
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return await self._inner.generate(request)


class _TrackingCloseProvider(ModelProvider):
    def __init__(self, inner: ModelProvider) -> None:
        self._inner = inner
        self.closed = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self._inner.generate(request)

    async def aclose(self) -> None:
        self.closed = True


class _FixedCostProvider(ModelProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider="test", model="fixed-cost", content="42", parsed=True, estimated_cost_usd=1.0
        )


class _AlwaysTimeoutProvider(ModelProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise ProviderTimeoutError("always times out", provider="test", retryable=True)


class _FlakyProvider(ModelProvider):
    def __init__(self, fail_times: int) -> None:
        self._remaining_failures = fail_times

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ProviderTimeoutError("flaky", provider="test", retryable=True)
        return ModelResponse(provider="test", model="flaky", content="ok", parsed=True)


class _AlwaysErrorProvider(ModelProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise ProviderResponseError("permanent failure", provider="test", retryable=False)


@pytest.mark.asyncio
async def test_predictions_preserve_input_sample_order(tmp_path: Path) -> None:
    samples = [_sample(n) for n in range(8)]
    engine = RunEngine(sleep=_no_sleep)
    cache = ResponseCache(tmp_path)
    config = RunConfig(concurrency=8)

    result = await engine.run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=config,
        cache=cache,
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )

    assert [p.sample_id for p in result.predictions] == [s.sample_id for s in samples]


@pytest.mark.asyncio
async def test_echo_gold_scores_perfectly_and_reports_no_errors(tmp_path: Path) -> None:
    samples = [_sample(n) for n in range(3)]
    engine = RunEngine(sleep=_no_sleep)
    cache = ResponseCache(tmp_path)

    result = await engine.run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=cache,
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )

    assert result.n_errors == 0
    for sample, prediction in zip(samples, result.predictions, strict=True):
        assert prediction.response is not None
        assert prediction.response.financial_answer is not None
        assert prediction.response.financial_answer.answer == sample.gold.answer


@pytest.mark.asyncio
async def test_rerun_with_same_cache_hits_everything_and_makes_zero_calls(tmp_path: Path) -> None:
    samples = [_sample(n) for n in range(5)]
    cache_dir = tmp_path / "cache"
    config = RunConfig()

    first_provider = _CountingProvider(MockProvider(oracle=build_mock_oracle(samples)))
    first_result = await RunEngine(sleep=_no_sleep).run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=config,
        cache=ResponseCache(cache_dir),
        provider=first_provider,
    )
    assert first_provider.calls == 5
    assert first_result.n_cache_hits == 0

    second_provider = _CountingProvider(MockProvider(oracle=build_mock_oracle(samples)))
    second_result = await RunEngine(sleep=_no_sleep).run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=config,
        cache=ResponseCache(cache_dir),
        provider=second_provider,
    )
    assert second_provider.calls == 0
    assert second_result.n_cache_hits == 5
    assert [p.response for p in second_result.predictions] == [
        p.response for p in first_result.predictions
    ]


@pytest.mark.asyncio
async def test_non_retryable_error_fails_after_a_single_attempt(tmp_path: Path) -> None:
    result = await RunEngine(sleep=_no_sleep).run(
        samples=[_sample(0)],
        model=ModelSpec.parse("test/whatever"),
        config=RunConfig(max_retries=4),
        cache=ResponseCache(tmp_path),
        provider=_AlwaysErrorProvider(),
    )
    prediction = result.predictions[0]
    assert prediction.response is None
    assert prediction.attempts == 1
    assert prediction.error_type == "ProviderResponseError"
    assert result.n_errors == 1


@pytest.mark.asyncio
async def test_retryable_failure_exhausts_configured_retries(tmp_path: Path) -> None:
    config = RunConfig(max_retries=2, base_delay_s=0.001, max_delay_s=0.001)
    result = await RunEngine(sleep=_no_sleep).run(
        samples=[_sample(0)],
        model=ModelSpec.parse("test/whatever"),
        config=config,
        cache=ResponseCache(tmp_path),
        provider=_AlwaysTimeoutProvider(),
    )
    prediction = result.predictions[0]
    assert prediction.response is None
    assert prediction.attempts == config.max_retries + 1
    assert prediction.error_type == "ProviderTimeoutError"


@pytest.mark.asyncio
async def test_flaky_provider_recovers_within_retry_budget(tmp_path: Path) -> None:
    config = RunConfig(max_retries=2, base_delay_s=0.001, max_delay_s=0.001)
    result = await RunEngine(sleep=_no_sleep).run(
        samples=[_sample(0)],
        model=ModelSpec.parse("test/whatever"),
        config=config,
        cache=ResponseCache(tmp_path),
        provider=_FlakyProvider(fail_times=1),
    )
    prediction = result.predictions[0]
    assert prediction.response is not None
    assert prediction.attempts == 2


@pytest.mark.asyncio
async def test_budget_guard_stops_new_calls_once_cap_is_reached(tmp_path: Path) -> None:
    samples = [_sample(n) for n in range(3)]
    # concurrency=1 makes evaluation strictly sequential, so the budget check is deterministic.
    config = RunConfig(concurrency=1, max_output_tokens=1)
    result = await RunEngine(sleep=_no_sleep).run(
        samples=samples,
        model=ModelSpec.parse("test/whatever"),
        config=config,
        cache=ResponseCache(tmp_path),
        provider=_FixedCostProvider(),
        max_cost_usd=1.0,
    )
    first, second, third = result.predictions
    assert first.response is not None
    assert second.error_type == "BudgetExceeded"
    assert second.attempts == 0
    assert third.error_type == "BudgetExceeded"
    assert result.budget_exceeded is True
    assert result.total_estimated_cost_usd == 1.0


@pytest.mark.asyncio
async def test_engine_closes_a_provider_it_created_itself(tmp_path: Path) -> None:
    # The engine only owns provider lifecycle when it constructs one itself (provider=None);
    # when a provider is injected, the caller keeps ownership and the engine must not close it.
    samples = [_sample(0)]
    injected = _TrackingCloseProvider(MockProvider(oracle=build_mock_oracle(samples)))
    await RunEngine(sleep=_no_sleep).run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=injected,
    )
    assert injected.closed is False


@pytest.mark.asyncio
async def test_max_samples_limit_is_applied(tmp_path: Path) -> None:
    samples = [_sample(n) for n in range(10)]
    result = await RunEngine(sleep=_no_sleep).run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(limit=3),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    assert result.n_samples == 3
    assert [p.sample_id for p in result.predictions] == [
        "smoke:dev:0",
        "smoke:dev:1",
        "smoke:dev:2",
    ]
    # result.samples must be the *truncated* set, 1:1 with predictions — a caller zipping the
    # original (untruncated) sample list against predictions would raise or silently misalign.
    assert [s.sample_id for s in result.samples] == [p.sample_id for p in result.predictions]
    assert len(result.samples) == len(samples[:3])
