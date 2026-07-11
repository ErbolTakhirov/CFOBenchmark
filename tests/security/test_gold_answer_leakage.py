"""Proof that the answer key cannot reach a model.

If gold leaks into the prompt, every number this platform produces is meaningless. So this suite
does not merely check that today's code happens not to leak — it pins down two properties that make
leaking *structurally impossible*:

1. **No channel exists.** ``ModelRequest`` has no field that can carry gold, and ``extra="forbid"``
   means one cannot be added dynamically.
2. **Scrub-equivalence.** Take a real sample, replace its entire gold answer, gold program,
   acceptable answers, gold evidence and grading tolerances with sentinels, then render the request
   again — and get back *the same bytes*. If the request cannot change when the answer changes, the
   answer is not in the request. This is a much stronger statement than "grep the prompt for the
   gold string", which would produce false alarms (a span-extraction answer legitimately appears in
   its own context) and false comfort (a paraphrased or reformatted gold would slip through).

The suite runs against real FinQA samples as well as synthetic ones, so it covers a real adapter's
output and not just a hand-built fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.factories import SCRUBBED_ANSWER, make_sample, scrub_gold

from financebench.datasets.finqa.adapter import FinQAAdapter
from financebench.execution.engine import build_request
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.prompts.renderer import render_messages
from financebench.schemas.model_io import ModelRequest, ModelSpec
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import CanonicalSample

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "finqa"

MODEL = ModelSpec.parse("openai/gpt-4o-mini")
CONFIG = RunConfig()


def _real_finqa_samples(limit: int = 8) -> list[CanonicalSample]:
    return list(FinQAAdapter(data_dir=FIXTURE_DIR).load("test"))[:limit]


def _all_samples() -> list[CanonicalSample]:
    return [make_sample(), *_real_finqa_samples()]


# --------------------------------------------------------------------------- 1. no channel exists


def test_model_request_has_no_field_that_could_carry_gold() -> None:
    """The structural guarantee. A request literally has nowhere to put an answer key."""
    forbidden = {"gold", "gold_answer", "simulation_context", "oracle", "answer", "solution"}
    assert forbidden.isdisjoint(ModelRequest.model_fields)


def test_model_request_forbids_extra_fields_so_gold_cannot_be_smuggled_in() -> None:
    """Without ``extra="forbid"``, the structural guarantee above would be bypassable."""
    assert ModelRequest.model_config.get("extra") == "forbid"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModelRequest(
            model=MODEL,
            messages=(),
            prompt_version="v1",
            benchmark="smoke",
            benchmark_version="1",
            sample_id="smoke:dev:1",
            simulation_context={"gold_answer": "4"},  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------- 2. scrub-equivalence


@pytest.mark.parametrize("sample", _all_samples(), ids=lambda s: s.sample_id)
def test_rendered_messages_are_identical_when_gold_is_scrubbed(sample: CanonicalSample) -> None:
    """Change the answer completely; the prompt must not move by a single byte."""
    assert render_messages(sample) == render_messages(scrub_gold(sample))


@pytest.mark.parametrize("sample", _all_samples(), ids=lambda s: s.sample_id)
def test_full_request_is_byte_identical_when_gold_is_scrubbed(sample: CanonicalSample) -> None:
    """The same property for the whole serialized request — the object that actually gets sent,
    written to ``predictions.jsonl``, and hashed into the response-cache key."""
    original = build_request(sample, MODEL, CONFIG).model_dump_json()
    scrubbed = build_request(scrub_gold(sample), MODEL, CONFIG).model_dump_json()
    assert original == scrubbed


@pytest.mark.parametrize("sample", _all_samples(), ids=lambda s: s.sample_id)
def test_sentinel_gold_never_appears_in_the_serialized_request(sample: CanonicalSample) -> None:
    """A direct corollary, stated directly: after scrubbing, the sentinel is nowhere in the wire
    format. (The un-scrubbed gold string is *not* asserted absent, because for span-extraction
    tasks the answer legitimately appears inside its own source context — asserting otherwise
    would be a false alarm. Scrubbing is what makes the question answerable at all.)"""
    payload = build_request(scrub_gold(sample), MODEL, CONFIG).model_dump_json()
    assert SCRUBBED_ANSWER not in payload
    assert "987654321" not in payload


def test_scrub_gold_actually_changes_the_sample() -> None:
    """Guards the guard: if ``scrub_gold`` silently did nothing, every test above would pass
    vacuously."""
    sample = make_sample()
    scrubbed = scrub_gold(sample)
    assert scrubbed.gold != sample.gold
    assert scrubbed.evaluation != sample.evaluation
    assert scrubbed.question == sample.question  # the question side is untouched
    assert scrubbed.context == sample.context


# --------------------------------------------------------------------------- the mock is the
# ONLY thing that ever sees gold, and only by explicit hand-off


def test_the_only_gold_oracle_is_built_explicitly_and_keyed_by_sample() -> None:
    samples = _real_finqa_samples(3)
    oracle = build_mock_oracle(samples)
    assert set(oracle) == {s.sample_id for s in samples}
    for sample in samples:
        assert oracle[sample.sample_id].gold_answer == sample.gold.answer


@pytest.mark.asyncio
async def test_a_real_provider_receives_a_request_containing_no_answer_key() -> None:
    """End-to-end: capture exactly what a non-mock provider is handed."""
    captured: list[ModelRequest] = []

    class CapturingProvider(MockProvider):
        async def generate(self, request: ModelRequest):  # type: ignore[no-untyped-def]
            captured.append(request)
            return await super().generate(request)

    sample = _real_finqa_samples(1)[0]
    # Deliberately give this provider the oracle — and show that it *still* cannot see gold in the
    # request, because the oracle is a separate object the request knows nothing about.
    provider = CapturingProvider(oracle=build_mock_oracle([sample]))
    await provider.generate(build_request(sample, ModelSpec.parse("mock/echo-gold"), CONFIG))

    (request,) = captured
    payload = json.loads(request.model_dump_json())
    assert "simulation_context" not in payload
    assert not any("gold" in key.lower() for key in payload)
