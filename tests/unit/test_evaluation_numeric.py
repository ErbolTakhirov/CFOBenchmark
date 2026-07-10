from __future__ import annotations

import pytest

from financebench.evaluation.numeric import numeric_match, parse_numeric_answer
from financebench.schemas.common import Scale


def _resolved(text: str, **kwargs: object) -> float:
    parsed = parse_numeric_answer(text, **kwargs)  # type: ignore[arg-type]
    assert parsed is not None, f"expected a parse for {text!r}"
    return parsed.resolved_value


# --------------------------------------------------------------------------- basic literals


def test_plain_integer() -> None:
    assert _resolved("42") == 42.0


def test_plain_decimal() -> None:
    assert _resolved("42.5") == 42.5


def test_comma_grouped_thousands_en_locale() -> None:
    assert _resolved("1,234,567") == 1234567.0


def test_decimal_comma_eu_locale() -> None:
    assert _resolved("1.234.567,89", locale_hint="eu") == pytest.approx(1234567.89)


def test_invalid_locale_hint_raises() -> None:
    with pytest.raises(ValueError, match="locale_hint"):
        parse_numeric_answer("100", locale_hint="fr")


# --------------------------------------------------------------------------- sign handling


def test_parenthesized_number_is_negative() -> None:
    assert _resolved("(1,234)") == -1234.0


def test_explicit_minus_sign() -> None:
    assert _resolved("-1234") == -1234.0


def test_explicit_plus_sign() -> None:
    assert _resolved("+1234") == 1234.0


def test_parenthesized_currency_is_negative() -> None:
    parsed = parse_numeric_answer("($1,234)")
    assert parsed is not None
    assert parsed.resolved_value == -1234.0
    assert parsed.currency == "usd"


# --------------------------------------------------------------------------- percent / pp / bps
# The mission is explicit: 5%, 5 percentage points, and 500 basis points must never be silently
# treated as interchangeable — each gets its own distinct `unit` tag.


def test_percent_sets_unit_and_flag() -> None:
    parsed = parse_numeric_answer("12.5%")
    assert parsed is not None
    assert parsed.resolved_value == 12.5
    assert parsed.is_percentage is True
    assert parsed.unit == "percent"


def test_percentage_points_are_a_distinct_unit_from_percent() -> None:
    parsed = parse_numeric_answer("5 percentage points")
    assert parsed is not None
    assert parsed.unit == "percentage_point"
    assert parsed.is_percentage is False


def test_basis_points_are_not_auto_converted_to_percent() -> None:
    parsed = parse_numeric_answer("500 bps")
    assert parsed is not None
    assert parsed.unit == "basis_points"
    assert parsed.is_basis_points is True
    # 500 bps numerically *equals* 5% as a rate, but this parser must not perform that
    # conversion itself — it reports the raw written number, tagged, and nothing more.
    assert parsed.resolved_value == 500.0


def test_basis_points_spelled_out() -> None:
    parsed = parse_numeric_answer("120 basis points")
    assert parsed is not None
    assert parsed.unit == "basis_points"
    assert parsed.resolved_value == 120.0


# --------------------------------------------------------------------------- currency


def test_dollar_symbol_prefix() -> None:
    parsed = parse_numeric_answer("$1,234.56")
    assert parsed is not None
    assert parsed.currency == "usd"
    assert parsed.resolved_value == pytest.approx(1234.56)


def test_currency_code_suffix() -> None:
    parsed = parse_numeric_answer("1234 USD")
    assert parsed is not None
    assert parsed.currency == "usd"
    assert parsed.resolved_value == 1234.0


def test_euro_symbol() -> None:
    parsed = parse_numeric_answer("€500")
    assert parsed is not None
    assert parsed.currency == "eur"


# --------------------------------------------------------------------------- scale (K/M/B)
# The mission is explicit: $5 million and $5 must never be treated as equivalent.


def test_five_million_dollars_is_not_five_dollars() -> None:
    five_million = parse_numeric_answer("$5 million")
    five = parse_numeric_answer("$5")
    assert five_million is not None
    assert five is not None
    assert five_million.resolved_value == 5_000_000.0
    assert five.resolved_value == 5.0
    assert five_million.scale == Scale.MILLION
    assert five.scale == Scale.UNIT


def test_scale_word_with_space() -> None:
    assert _resolved("1.2 billion") == 1_200_000_000.0


def test_scale_suffix_attached_no_space() -> None:
    parsed = parse_numeric_answer("$1.2M")
    assert parsed is not None
    assert parsed.resolved_value == 1_200_000.0
    assert parsed.currency == "usd"


def test_lowercase_k_suffix() -> None:
    assert _resolved("25k") == 25_000.0


def test_bn_suffix() -> None:
    assert _resolved("3.5bn") == 3_500_000_000.0


def test_equivalent_phrasings_resolve_to_the_same_value() -> None:
    # "1.2 billion" and "1,200 million" are genuinely the same quantity — unlike the percent-vs-
    # percentage-point-vs-bps case, collapsing these is correct, not a semantic conflation.
    assert _resolved("1.2 billion") == _resolved("1,200 million")


# --------------------------------------------------------------------------- unparseable input


def test_empty_string_is_none() -> None:
    assert parse_numeric_answer("") is None


def test_whitespace_only_is_none() -> None:
    assert parse_numeric_answer("   ") is None


def test_non_numeric_prose_is_none() -> None:
    assert parse_numeric_answer("insufficient information") is None


# --------------------------------------------------------------------------- numeric_match


def test_numeric_match_exact_equality() -> None:
    assert numeric_match(100.0, 100.0) is True


def test_numeric_match_no_tolerance_rejects_close_but_unequal() -> None:
    assert numeric_match(100.001, 100.0) is False


def test_numeric_match_absolute_tolerance() -> None:
    assert numeric_match(100.00001, 100.0, absolute_tolerance=1e-4) is True
    assert numeric_match(100.01, 100.0, absolute_tolerance=1e-4) is False


def test_numeric_match_relative_tolerance_within_finance_reasoning_bound() -> None:
    # FinanceReasoning's own tolerance: within 0.2% relative error.
    assert numeric_match(100.1, 100.0, relative_tolerance=0.002) is True
    assert numeric_match(101.0, 100.0, relative_tolerance=0.002) is False


def test_numeric_match_relative_tolerance_skipped_when_gold_is_zero() -> None:
    assert numeric_match(0.001, 0.0, relative_tolerance=0.5) is False
    assert numeric_match(0.0, 0.0, relative_tolerance=0.5) is True  # exact-equality path


def test_numeric_match_finqa_style_rounding_via_absolute_tolerance() -> None:
    # FinQA's execution accuracy rounds to 5 decimals before comparing.
    assert numeric_match(1.0000049, 1.0, absolute_tolerance=5e-6) is True
    assert numeric_match(1.00001, 1.0, absolute_tolerance=5e-6) is False
