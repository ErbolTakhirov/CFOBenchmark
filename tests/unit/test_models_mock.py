from __future__ import annotations

import pytest
from tests.factories import make_sample

from financebench.models.mock import GoldOracleEntry, MockProvider, build_mock_oracle
from financebench.schemas.model_io import ChatMessage, ModelRequest, ModelSpec, Role
from financebench.utils.errors import ProviderResponseError, ProviderTimeoutError

SAMPLE_ID = "smoke:dev:1"

#: The answer key is handed to the provider explicitly, at construction — it is *not* reachable
#: through the request, which has no field it could ride in. See ``models/mock.py``.
ORACLE = {SAMPLE_ID: GoldOracleEntry(gold_answer="12.5%", gold_numeric_value=12.5, unit="percent")}


def _request(profile: str) -> ModelRequest:
    return ModelRequest(
        model=ModelSpec.parse(f"mock/{profile}"),
        messages=(ChatMessage(role=Role.USER, content="What was the revenue increase?"),),
        prompt_version="v1",
        benchmark="smoke",
        benchmark_version="1",
        sample_id=SAMPLE_ID,
    )


def _mock() -> MockProvider:
    return MockProvider(oracle=ORACLE)


@pytest.mark.asyncio
async def test_echo_gold_returns_the_gold_answer_verbatim() -> None:
    response = await _mock().generate(_request("echo-gold"))
    assert response.parsed is True
    assert response.financial_answer is not None
    assert response.financial_answer.answer == "12.5%"
    assert response.financial_answer.numeric_value == 12.5


@pytest.mark.asyncio
async def test_formatting_noise_preserves_the_numeric_value_but_not_the_plain_string() -> None:
    response = await _mock().generate(_request("formatting-noise"))
    answer = response.financial_answer
    assert answer is not None
    assert answer.numeric_value == 12.5
    assert answer.answer != "12.5%"
    assert "%" in answer.answer


@pytest.mark.asyncio
async def test_always_wrong_never_matches_gold_numeric_value() -> None:
    response = await _mock().generate(_request("always-wrong"))
    answer = response.financial_answer
    assert answer is not None
    assert answer.numeric_value != 12.5


@pytest.mark.asyncio
async def test_refuse_sets_insufficient_information() -> None:
    response = await _mock().generate(_request("refuse"))
    answer = response.financial_answer
    assert answer is not None
    assert answer.insufficient_information is True


@pytest.mark.asyncio
async def test_error_profile_raises_non_retryable() -> None:
    with pytest.raises(ProviderResponseError) as excinfo:
        await _mock().generate(_request("error"))
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_timeout_profile_raises_retryable() -> None:
    with pytest.raises(ProviderTimeoutError) as excinfo:
        await _mock().generate(_request("timeout"))
    assert excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_unknown_profile_raises_non_retryable_response_error() -> None:
    with pytest.raises(ProviderResponseError) as excinfo:
        await _mock().generate(_request("not-a-real-profile"))
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_a_mock_without_an_oracle_cannot_answer() -> None:
    """The safe failure mode: no oracle handed over, nothing to echo.

    This is the whole point of constructor injection — an accidentally-constructed mock (e.g. via
    the generic ``create_provider``) has no path to the answer key, so it cannot manufacture a
    correct-looking prediction.
    """
    response = await MockProvider().generate(_request("echo-gold"))
    assert response.financial_answer is not None
    assert response.financial_answer.answer == ""


@pytest.mark.asyncio
async def test_oracle_is_keyed_by_sample_id_so_answers_cannot_cross_samples() -> None:
    provider = MockProvider(oracle=ORACLE)
    other = ModelRequest(
        model=ModelSpec.parse("mock/echo-gold"),
        messages=(ChatMessage(role=Role.USER, content="different sample"),),
        prompt_version="v1",
        benchmark="smoke",
        benchmark_version="1",
        sample_id="smoke:dev:999",
    )
    response = await provider.generate(other)
    assert response.financial_answer is not None
    assert response.financial_answer.answer == ""


def test_build_mock_oracle_maps_sample_ids_to_their_own_gold() -> None:
    samples = [
        make_sample(sample_id="smoke:dev:1", gold_answer="10", gold_numeric_value=10.0),
        make_sample(sample_id="smoke:dev:2", gold_answer="20", gold_numeric_value=20.0),
    ]
    oracle = build_mock_oracle(samples)
    assert oracle["smoke:dev:1"].gold_numeric_value == 10.0
    assert oracle["smoke:dev:2"].gold_numeric_value == 20.0


@pytest.mark.asyncio
async def test_token_usage_and_latency_are_populated() -> None:
    response = await _mock().generate(_request("echo-gold"))
    assert response.token_usage is not None
    assert response.token_usage.total_tokens is not None
    assert response.token_usage.total_tokens > 0
    assert response.latency_ms == 5.0


def test_capabilities_reports_text_and_json_mode() -> None:
    caps = MockProvider().capabilities("echo-gold")
    assert caps.text is True
    assert caps.json_mode is True
    assert caps.vision is False
