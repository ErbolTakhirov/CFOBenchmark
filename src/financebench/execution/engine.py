"""The async run engine.

Each sample becomes exactly one model request in Milestone 1 (no cross-sample dependency yet —
ConvFinQA-style turn chaining, where a model's own prior answer feeds the next turn, is a
Milestone 4 extension of this engine, not a redesign of it). Samples run concurrently under a
bounded semaphore; ``asyncio.gather`` preserves input order in its results regardless of
completion order, so the returned predictions are byte-stable without needing a separate
collect-then-reorder step.

The response cache is consulted before every call and populated after every success — a cache
hit *is* how a resumed/re-run command avoids redoing work (see ``execution/cache.py``), not a
separate mechanism. A failed call is never cached, so a transient error doesn't get pinned.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from financebench import __version__
from financebench.execution.cache import ResponseCache
from financebench.execution.retry import RateLimiter, Sleeper, backoff_delay
from financebench.models import create_provider
from financebench.models.base import ModelProvider
from financebench.prompts.renderer import PROMPT_VERSION, render_messages
from financebench.schemas.model_io import ModelRequest, ModelResponse, ModelSpec
from financebench.schemas.prediction import Prediction
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import CanonicalSample
from financebench.utils.errors import ProviderError
from financebench.utils.timing import Clock, RealClock

__all__ = ["RunEngine", "RunResult", "build_request"]


def build_request(
    sample: CanonicalSample,
    model: ModelSpec,
    config: RunConfig,
) -> ModelRequest:
    """Assemble the :class:`ModelRequest` sent to a provider for ``sample``.

    This is a pure function of the sample's *question side* — ``question``, ``context``,
    ``choices``, ``tools`` — and the run config. It never reads ``sample.gold``,
    ``sample.evaluation`` (which holds grading tolerances), or anything else on the evaluator's
    side of the fence. ``tests/security/test_gold_answer_leakage.py`` pins that down by scrubbing
    a sample's gold to sentinel values and asserting the request it produces is byte-identical.
    """
    return ModelRequest(
        model=model,
        messages=render_messages(sample),
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        prompt_version=PROMPT_VERSION,
        benchmark=sample.benchmark,
        benchmark_version=sample.benchmark_version,
        sample_id=sample.sample_id,
        timeout_s=config.timeout_seconds,
    )


@dataclass(frozen=True)
class RunResult:
    """In-memory outcome of a run. Persisting this to ``runs/{run_id}/`` is a separate concern
    (``storage/artifacts.py``) — the engine only runs samples and reports what happened.

    ``samples`` is the (possibly ``config.limit``-truncated) list actually run — 1:1 and in the
    same order as ``predictions``. Callers must score/report against *this* list, not whatever
    superset they originally passed to :meth:`RunEngine.run`, or a ``--max-samples`` run will
    zip predictions against the wrong samples.
    """

    samples: tuple[CanonicalSample, ...]
    predictions: tuple[Prediction, ...]
    n_samples: int
    n_errors: int
    n_cache_hits: int
    total_estimated_cost_usd: float | None = None
    total_tokens: int | None = None
    budget_exceeded: bool = False


@dataclass
class _RunContext:
    """Per-run invariants plus mutable cost accumulators (mutated cooperatively; see the budget
    docstring below on overshoot)."""

    model: ModelSpec
    config: RunConfig
    provider: ModelProvider
    cache: ResponseCache
    limiter: RateLimiter
    max_cost_usd: float | None
    total_cost: float = 0.0
    total_tokens: int = 0
    priced_any: bool = False
    budget_exceeded: bool = False


class RunEngine:
    """Runs a set of samples against a model, returning a :class:`RunResult`.

    A :class:`Clock`, a rate limiter's sleep function, and/or a pre-built provider can be
    injected for deterministic, offline tests.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
    ) -> None:
        self._clock = clock or RealClock()
        self._sleep = sleep or asyncio.sleep

    async def run(
        self,
        *,
        samples: Sequence[CanonicalSample],
        model: ModelSpec,
        config: RunConfig,
        cache: ResponseCache,
        provider: ModelProvider | None = None,
        requests_per_second: float | None = None,
        max_cost_usd: float | None = None,
    ) -> RunResult:
        owns_provider = provider is None
        provider_instance = provider or create_provider(model.provider)
        ctx = _RunContext(
            model=model,
            config=config,
            provider=provider_instance,
            cache=cache,
            limiter=RateLimiter(requests_per_second, self._sleep),
            max_cost_usd=max_cost_usd,
        )
        limited = list(samples)[: config.limit] if config.limit else list(samples)
        try:
            semaphore = asyncio.Semaphore(max(1, config.concurrency))

            async def guarded(sample: CanonicalSample) -> Prediction:
                async with semaphore:
                    return await self._run_sample(ctx, sample)

            # asyncio.gather preserves input order in its results regardless of completion
            # order, so this is already the deterministic ordering byte-stability needs.
            predictions = list(await asyncio.gather(*(guarded(sample) for sample in limited)))
        finally:
            if owns_provider:
                await provider_instance.aclose()

        n_errors = sum(1 for p in predictions if p.response is None)
        n_cache_hits = sum(1 for p in predictions if p.cache_hit)
        return RunResult(
            samples=tuple(limited),
            predictions=tuple(predictions),
            n_samples=len(predictions),
            n_errors=n_errors,
            n_cache_hits=n_cache_hits,
            total_estimated_cost_usd=round(ctx.total_cost, 8) if ctx.priced_any else None,
            total_tokens=ctx.total_tokens or None,
            budget_exceeded=ctx.budget_exceeded,
        )

    # -- per-sample execution ------------------------------------------------
    async def _run_sample(self, ctx: _RunContext, sample: CanonicalSample) -> Prediction:
        request = build_request(sample, ctx.model, ctx.config)

        # Best-effort budget guard: stop issuing *new* calls once spend reaches the cap.
        # In-flight calls may overshoot by at most the concurrency width; pair with
        # --max-samples for a hard cap.
        if ctx.max_cost_usd is not None and ctx.total_cost >= ctx.max_cost_usd:
            ctx.budget_exceeded = True
            return self._prediction(
                sample,
                request,
                error="max_cost_usd budget reached before this sample was attempted",
                error_type="BudgetExceeded",
                attempts=0,
            )

        cached = ctx.cache.get(request)
        if cached is not None:
            return self._prediction(sample, request, response=cached, attempts=0, cache_hit=True)

        response, error, attempts, retry_wait_ms = await self._call_with_retries(ctx, request)
        if response is not None:
            self._record_cost(ctx, response)
            ctx.cache.put(
                request,
                response,
                financebench_version=__version__,
                written_at=self._clock.now_iso(),
            )
            return self._prediction(
                sample, request, response=response, attempts=attempts, retry_wait_ms=retry_wait_ms
            )
        return self._prediction(
            sample,
            request,
            error=str(error) if error is not None else "unknown provider failure",
            error_type=type(error).__name__ if error is not None else None,
            attempts=attempts,
            retry_wait_ms=retry_wait_ms,
        )

    def _prediction(
        self,
        sample: CanonicalSample,
        request: ModelRequest,
        *,
        response: ModelResponse | None = None,
        error: str | None = None,
        error_type: str | None = None,
        attempts: int = 1,
        cache_hit: bool = False,
        retry_wait_ms: float = 0.0,
    ) -> Prediction:
        return Prediction(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            split=sample.split,
            request=request,
            created_at=self._clock.now_iso(),
            response=response,
            error=error,
            error_type=error_type,
            attempts=attempts,
            cache_hit=cache_hit,
            retry_wait_ms=retry_wait_ms,
        )

    def _record_cost(self, ctx: _RunContext, response: ModelResponse) -> None:
        usage = response.token_usage
        if usage and usage.total_tokens:
            ctx.total_tokens += usage.total_tokens
        if response.estimated_cost_usd is not None:
            ctx.total_cost += response.estimated_cost_usd
            ctx.priced_any = True

    async def _call_with_retries(
        self, ctx: _RunContext, request: ModelRequest
    ) -> tuple[ModelResponse | None, BaseException | None, int, float]:
        attempts = 0
        total_wait = 0.0
        start = self._clock.monotonic()
        while True:
            attempts += 1
            await ctx.limiter.acquire()
            try:
                response = await ctx.provider.generate(request)
                return response, None, attempts, total_wait * 1000.0
            except ProviderError as exc:
                if not (exc.retryable and attempts <= ctx.config.max_retries):
                    return None, exc, attempts, total_wait * 1000.0
                delay = backoff_delay(
                    ctx.config, attempts, exc.retry_after, sample_id=request.sample_id
                )
                elapsed = self._clock.monotonic() - start
                if ctx.config.deadline_s is not None and elapsed + delay > ctx.config.deadline_s:
                    return None, exc, attempts, total_wait * 1000.0
                await self._sleep(delay)
                total_wait += delay
            except Exception as exc:  # record any failure, never hide it
                return None, exc, attempts, total_wait * 1000.0
