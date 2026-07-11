"""The async run engine.

Each sample becomes exactly one model request. Samples run concurrently under a bounded semaphore,
and predictions come back in input order regardless of completion order, so a run's artifacts are
byte-stable.

**Conversations are the one exception, and they are why the unit of concurrency is a *group*, not a
sample.** A ConvFinQA turn is not independent of the turn before it — turn 1 is literally *"and what
was it in 2005?"*, which means nothing on its own. Under the ``model_history`` protocol, turn N's
prompt contains the model's own answer to turn N-1, so that answer must **exist** before turn N can
even be built. Firing every turn at once with ``asyncio.gather`` would produce a prompt containing
an answer the model had not yet given.

So samples are partitioned into conversation groups (a single-turn benchmark yields groups of one,
and behaves exactly as before): **sequential within a conversation, parallel across conversations.**
That is the only ordering constraint the data actually imposes, and imposing more would just make
runs slower for no benefit.

The response cache is consulted before every call and populated after every success — a cache
hit *is* how a resumed/re-run command avoids redoing work (see ``execution/cache.py``), not a
separate mechanism. A failed call is never cached, so a transient error doesn't get pinned.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from financebench import __version__
from financebench.execution.cache import ResponseCache
from financebench.execution.retry import RateLimiter, Sleeper, backoff_delay
from financebench.models import create_provider
from financebench.models.base import ModelProvider
from financebench.prompts.profiles import RetrievedChunk, create_prompt_profile
from financebench.schemas.common import ConversationProtocol, EvalMode
from financebench.schemas.model_io import FinancialAnswer, ModelRequest, ModelResponse, ModelSpec
from financebench.schemas.prediction import Prediction
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import CanonicalSample, ConversationTurn
from financebench.utils.errors import ConfigError, ProviderError
from financebench.utils.timing import Clock, RealClock

__all__ = [
    "RunEngine",
    "RunResult",
    "build_request",
    "conversation_groups",
    "model_answer_text",
]


def build_request(
    sample: CanonicalSample,
    model: ModelSpec,
    config: RunConfig,
    *,
    retrieved: Sequence[RetrievedChunk] = (),
) -> ModelRequest:
    """Assemble the :class:`ModelRequest` sent to a provider for ``sample``.

    This is a pure function of the sample's *question side* — ``question``, ``context``,
    ``choices``, ``tools`` — the run config, and (in ``retrieval_required`` mode) whatever the
    retriever found. It never reads ``sample.gold``, ``sample.evaluation`` (which holds grading
    tolerances), or anything else on the evaluator's side of the fence.
    ``tests/security/test_gold_answer_leakage.py`` pins that down by scrubbing a sample's gold to
    sentinel values and asserting the request it produces is byte-identical.
    """
    profile = create_prompt_profile(config.prompt_profile)
    return ModelRequest(
        model=model,
        messages=profile.render(sample, config.eval_mode, retrieved),
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        response_format=profile.response_format,
        tools=sample.tools if config.eval_mode is EvalMode.TOOL_ASSISTED else (),
        # The profile name *is* the prompt version — profiles are versioned in their names, so a
        # changed prompt is a changed name, and a changed name changes the cache key.
        prompt_version=config.prompt_profile,
        benchmark=sample.benchmark,
        benchmark_version=sample.benchmark_version,
        sample_id=sample.sample_id,
        timeout_s=config.timeout_seconds,
    )


#: An indexed sample — its position in the run's input list, so predictions can be restored to
#: input order after conversations have been run out of order relative to one another.
_Indexed = tuple[int, CanonicalSample]


def conversation_groups(samples: Sequence[CanonicalSample]) -> list[list[_Indexed]]:
    """Partition samples into units that must run **sequentially**, keeping their input positions.

    A sample with no ``conversation_id`` is its own group of one, so every single-turn benchmark
    keeps exactly the fully-parallel behaviour it had before conversations existed.

    Turns within a conversation are ordered by ``turn_index``, not by the order they happened to
    arrive in. Relying on input order would work right up until a stratified manifest or a shuffled
    split handed them over out of order — and then the model would be shown a "conversation so far"
    that ran backwards, which is the sort of bug that produces a plausible score and no error.
    """
    groups: dict[str, list[_Indexed]] = {}
    ordered: list[list[_Indexed]] = []

    for index, sample in enumerate(samples):
        conversation_id = sample.metadata.get("conversation_id", "")
        if not conversation_id:
            ordered.append([(index, sample)])
            continue
        key = f"{sample.benchmark}:{conversation_id}"
        if key not in groups:
            groups[key] = []
            ordered.append(groups[key])
        groups[key].append((index, sample))

    for group in ordered:
        group.sort(key=lambda item: _turn_index(item[1]))
    return ordered


def _turn_index(sample: CanonicalSample) -> int:
    try:
        return int(sample.metadata.get("turn_index", "0"))
    except ValueError:
        return 0


def _assistant_slots(sample: CanonicalSample) -> int:
    return sum(1 for turn in sample.context.conversation_history if turn.role == "assistant")


def validate_conversations(
    samples: Sequence[CanonicalSample], protocol: ConversationProtocol
) -> None:
    """Under ``model_history``, refuse a run whose conversations are missing their opening turns.

    Turn 5's prompt needs the model's own answers to turns 0-4. If turns 0-4 were not in the run —
    because a manifest sampled individual turns, or a split was shuffled — those answers do not
    exist, and there are exactly two things the engine could do: quietly substitute the *gold* prior
    answers, or say so.

    Quietly substituting gold is the temptation, and it is the same bug as falling back to the gold
    evidence text when retrieval finds nothing: it would repair the prompt precisely where the
    conversation is most broken, and the ``model_history`` score — whose entire purpose is to expose
    error propagation — would be inflated by the gold answers it exists to do without.

    So it raises. A protocol you cannot honour is a configuration error, not a rounding error.
    """
    if protocol is not ConversationProtocol.MODEL_HISTORY:
        return
    for group in conversation_groups(samples):
        head = group[0][1]
        if _assistant_slots(head) == 0:
            continue  # starts at turn 0 (or is single-turn): the chain can be built
        raise ConfigError(
            f"conversation_protocol=model_history needs whole conversations, but "
            f"{head.sample_id!r} is turn {_turn_index(head)} and its earlier turns are not in this "
            "run — the model's own prior answers, which this protocol exists to feed forward, do "
            "not exist. Run complete conversations, or use conversation_protocol=gold_history "
            "(which grades each turn against the gold history and needs no chain)."
        )


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def model_answer_text(prediction: Prediction) -> str:
    """What the model said, in the shape a *prior turn* appears in — a bare answer, as the gold
    history has it, not the JSON envelope it arrived in.

    A failed or unparseable turn becomes an explicit marker rather than an empty string. Under
    ``model_history`` the model must see that its previous turn produced nothing; blanking it would
    hide the very failure the protocol is there to propagate.
    """
    response = prediction.response
    if response is None or response.financial_answer is None:
        return "(no answer)"
    answer = response.financial_answer
    if answer.insufficient_information:
        return "(insufficient information)"
    if answer.numeric_value is not None:
        return _format_number(answer.numeric_value)
    return answer.answer.strip()[:200] or "(no answer)"


def with_model_history(sample: CanonicalSample, answers: Sequence[str]) -> CanonicalSample:
    """Replace the gold prior answers in a turn's history with the model's own.

    The *questions* stay as they are — they are the dataset, not a prediction. Only the assistant
    side is rewritten, and the rewritten turns carry no ``turn_program`` or ``turn_answer``: those
    are gold, and under this protocol the model is entitled to none of it.
    """
    history = sample.context.conversation_history
    if not history:
        return sample

    slots = _assistant_slots(sample)
    if slots != len(answers):  # pragma: no cover — validate_conversations rules this out up front
        raise ConfigError(
            f"{sample.sample_id!r} expects {slots} prior model answer(s) but {len(answers)} were "
            "produced; the conversation chain is broken."
        )

    rebuilt: list[ConversationTurn] = []
    spoken = 0
    for turn in history:
        if turn.role != "assistant":
            rebuilt.append(turn)
            continue
        rebuilt.append(ConversationTurn(role="assistant", content=answers[spoken]))
        spoken += 1

    context = sample.context.model_copy(update={"conversation_history": tuple(rebuilt)})
    return sample.model_copy(update={"context": context})


def _reparse(response: ModelResponse) -> ModelResponse:
    """Re-derive the structured answer from the provider's **raw** content.

    ``ModelResponse.content`` is ground truth — it is exactly what the model said.
    ``financial_answer`` is a *parse* of that, and a parse is a derived value that our code owns
    and keeps improving.

    Caching the parse alongside the content froze old parses in place: a fix to the extractor would
    only take effect for samples that had never been run, and applying it to everything else would
    mean paying for the inference all over again. Worse, it *hid* the bug — a cached run kept
    reporting the broken parse no matter how many times it was re-scored, so the fix looked like it
    had done nothing.

    So the parse is redone on every cache read. Inference is the expensive, cacheable part;
    interpreting what came back is cheap, and must always reflect today's code.
    """
    answer = FinancialAnswer.from_text(response.content)
    if answer == response.financial_answer:
        return response
    return response.model_copy(update={"financial_answer": answer, "parsed": answer is not None})


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


#: Resolves the retrieved context for one sample. Takes a sample and returns the chunks the model
#: will actually see, plus whatever the evaluator needs afterwards. The engine treats it as opaque:
#: it does not know or care how retrieval happened, only that the model gets what came back.
RetrievalFn = Callable[[CanonicalSample], tuple[tuple[RetrievedChunk, ...], object]]


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
    retrieve: RetrievalFn | None = None
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
        retrieve: RetrievalFn | None = None,
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
            retrieve=retrieve,
        )
        limited = list(samples)[: config.limit] if config.limit else list(samples)
        validate_conversations(limited, config.conversation_protocol)
        groups = conversation_groups(limited)

        try:
            semaphore = asyncio.Semaphore(max(1, config.concurrency))

            async def guarded(group: list[_Indexed]) -> list[tuple[int, Prediction]]:
                async with semaphore:
                    return await self._run_conversation(ctx, group)

            # The semaphore is held for a whole conversation, so `concurrency` still bounds the
            # number of in-flight requests: N conversations advancing one turn at a time.
            completed = await asyncio.gather(*(guarded(group) for group in groups))

            # Conversations finish in whatever order they finish. Predictions are restored to the
            # run's *input* order, which is what makes the artifacts byte-stable.
            slots: list[Prediction | None] = [None] * len(limited)
            for group_result in completed:
                for index, prediction in group_result:
                    slots[index] = prediction
            predictions = [p for p in slots if p is not None]
            assert len(predictions) == len(limited), "every sample must produce a prediction"
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

    # -- conversation execution ----------------------------------------------
    async def _run_conversation(
        self, ctx: _RunContext, group: list[_Indexed]
    ) -> list[tuple[int, Prediction]]:
        """Run one conversation's turns in order. A group of one is just a sample.

        This is the only place the two conversation protocols differ, and the difference is a single
        substitution:

        - ``gold_history`` — the sample is run exactly as the adapter built it, carrying the **gold**
          prior conversation. Each turn is graded in isolation; a wrong answer at turn 1 cannot
          contaminate turn 3, because turn 3 never sees it.
        - ``model_history`` — the assistant side of the history is replaced with what **this model**
          actually said. A wrong answer at turn 1 is now in turn 3's prompt, and stays there. That
          is not a flaw in the protocol; it is the measurement.

        The gap between a model's two scores is the number worth reporting, and it exists only
        because these are run separately and never averaged together.
        """
        results: list[tuple[int, Prediction]] = []
        spoken: list[str] = []
        chaining = ctx.config.conversation_protocol is ConversationProtocol.MODEL_HISTORY

        for index, sample in group:
            prepared = with_model_history(sample, spoken) if chaining else sample
            prediction = await self._run_sample(ctx, prepared)
            results.append((index, prediction))
            if chaining:
                spoken.append(model_answer_text(prediction))
        return results

    # -- per-sample execution ------------------------------------------------
    async def _run_sample(self, ctx: _RunContext, sample: CanonicalSample) -> Prediction:
        # In retrieval_required mode the model sees ONLY what the retriever found — the sample's own
        # context is withheld, which is the entire point of the mode. The retriever is never handed
        # the sample's gold (see retrieval/retriever.py).
        retrieved: tuple[RetrievedChunk, ...] = ()
        if ctx.retrieve is not None:
            # The engine only needs the chunks the model will see. What the retriever *found* is
            # recorded by the pipeline itself and graded after the run — the engine must not be in
            # the business of knowing about gold evidence.
            retrieved, _ = ctx.retrieve(sample)

        request = build_request(sample, ctx.model, ctx.config, retrieved=retrieved)

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
            cached_response = _reparse(cached)
            # Account for a cache hit's tokens too. A fully-resumed run was otherwise reporting
            # `tokens=None, cost=None`, which reads as broken instrumentation rather than "these
            # were already paid for". The tokens WERE spent: a run describes what it took to
            # produce its answers, not what it cost to fetch them back off disk.
            self._record_cost(ctx, cached_response)
            return self._prediction(
                sample, request, response=cached_response, attempts=0, cache_hit=True
            )

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
