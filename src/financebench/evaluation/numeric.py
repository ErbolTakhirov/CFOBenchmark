"""Numeric answer parsing and tolerance-based comparison.

Every downstream native metric (FinQA execution accuracy, TAT-QA numeracy F1, FinanceReasoning's
0.2%-tolerance check, ...) depends on this module being right, which is why it's built before any
real dataset adapter (see the Milestone 2 build order in the project plan).

**Deliberately out of scope**: period/date normalization (``FY2025`` vs. ``2025 Q1``) is a
different concern from number parsing and is not handled here. **Deliberately not "solved"**:
whether ``"1.234"`` means one-point-two-three-four (US decimal) or one-thousand-two-hundred-
thirty-four (European thousands grouping) is genuinely ambiguous without knowing the source
locale — this module accepts an explicit ``locale_hint`` (``"en"`` or ``"eu"``) rather than
guessing, and defaults to ``"en"`` to match the English-language SEC-filing sources of FinQA,
TAT-QA, and FinanceReasoning.

Per the mission's explicit warnings, this module never silently equates: 5% and 5 percentage
points; 5% and 500 basis points; $5 million and $5. Each is parsed with its own distinct
``unit``/``scale`` tag — collapsing them is a decision for a task-aware caller to make
explicitly, not something this parser does implicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from financebench.schemas.common import Scale

__all__ = ["ParsedNumber", "numeric_match", "parse_numeric_answer"]

_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "usd",
    "€": "eur",
    "£": "gbp",
    "¥": "jpy",
    "₽": "rub",
}
_CURRENCY_CODE_RE = re.compile(r"(?i)\b(usd|eur|gbp|jpy|rub|cny)\b")

_BASIS_POINTS_RE = re.compile(r"(?i)\b(bps|basis\s+points?)\b")
_PERCENTAGE_POINT_RE = re.compile(r"(?i)\b(percentage\s+points?|p\.?p\.?)\b")

_SCALE_WORDS: dict[str, Scale] = {
    "k": Scale.THOUSAND,
    "thousand": Scale.THOUSAND,
    "thousands": Scale.THOUSAND,
    "m": Scale.MILLION,
    "mm": Scale.MILLION,
    "million": Scale.MILLION,
    "millions": Scale.MILLION,
    "b": Scale.BILLION,
    "bn": Scale.BILLION,
    "billion": Scale.BILLION,
    "billions": Scale.BILLION,
}
# No leading `\b`: scale suffixes are commonly attached directly to the digits ("$1.2M", "25k")
# with no word boundary between the last digit and the letter (both are `\w`).
_SCALE_SUFFIX_RE = re.compile(r"(?i)(thousands?|millions?|mm|billions?|bn|k|m|b)\s*$")
_SCALE_MULTIPLIER: dict[Scale, float] = {
    Scale.UNIT: 1.0,
    Scale.THOUSAND: 1e3,
    Scale.MILLION: 1e6,
    Scale.BILLION: 1e9,
}


@dataclass(frozen=True)
class ParsedNumber:
    """The result of parsing one numeric answer string.

    ``resolved_value`` is ``raw_value`` with ``scale`` already multiplied in (e.g. ``"$1.2
    million"`` -> ``raw_value=1.2, scale=MILLION, resolved_value=1_200_000.0``) — use it for
    tolerance comparisons. ``raw_value``/``scale`` are kept separately because that's the
    canonical sample schema's own convention (``GoldAnswer.numeric_value`` + ``.scale``).
    """

    resolved_value: float
    raw_value: float
    scale: Scale
    unit: str | None
    currency: str | None
    is_percentage: bool
    is_basis_points: bool
    raw_text: str


def parse_numeric_answer(text: str, *, locale_hint: str = "en") -> ParsedNumber | None:
    """Parse a numeric answer out of ``text``, or return ``None`` if no number is found.

    ``locale_hint`` must be ``"en"`` (comma = thousands separator, period = decimal separator)
    or ``"eu"`` (period = thousands separator, comma = decimal separator) — there is no
    auto-detected default; ambiguous input without a specified locale is a caller error, not
    something to guess at.
    """
    if locale_hint not in ("en", "eu"):
        raise ValueError(f"unsupported locale_hint {locale_hint!r}; expected 'en' or 'eu'")

    original = text
    working = text.strip()
    if not working:
        return None

    is_negative = False
    if working.startswith("(") and working.endswith(")"):
        is_negative = True
        working = working[1:-1].strip()
    if working.startswith("-"):
        is_negative = True
        working = working[1:].strip()
    elif working.startswith("+"):
        working = working[1:].strip()

    unit: str | None = None
    is_basis_points = False
    is_percentage = False

    bps_match = _BASIS_POINTS_RE.search(working)
    if bps_match:
        is_basis_points = True
        unit = "basis_points"
        working = (working[: bps_match.start()] + working[bps_match.end() :]).strip()

    if unit is None:
        pp_match = _PERCENTAGE_POINT_RE.search(working)
        if pp_match:
            unit = "percentage_point"
            working = (working[: pp_match.start()] + working[pp_match.end() :]).strip()

    if "%" in working:
        is_percentage = True
        if unit is None:
            unit = "percent"
        working = working.replace("%", "").strip()

    currency: str | None = None
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in working:
            currency = code
            working = working.replace(symbol, "").strip()
            break
    if currency is None:
        code_match = _CURRENCY_CODE_RE.search(working)
        if code_match:
            currency = code_match.group(1).lower()
            working = (working[: code_match.start()] + working[code_match.end() :]).strip()

    scale = Scale.UNIT
    scale_match = _SCALE_SUFFIX_RE.search(working)
    if scale_match:
        scale = _SCALE_WORDS[scale_match.group(1).lower()]
        working = working[: scale_match.start()].strip()

    if not working:
        return None

    cleaned = (
        working.replace(",", "").replace(" ", "")
        if locale_hint == "en"
        else working.replace(".", "").replace(" ", "").replace(",", ".")
    )
    try:
        raw_value = float(cleaned)
    except ValueError:
        return None

    if is_negative:
        raw_value = -abs(raw_value)

    if unit is None and currency is not None:
        unit = currency

    return ParsedNumber(
        resolved_value=raw_value * _SCALE_MULTIPLIER[scale],
        raw_value=raw_value,
        scale=scale,
        unit=unit,
        currency=currency,
        is_percentage=is_percentage,
        is_basis_points=is_basis_points,
        raw_text=original,
    )


def numeric_match(
    predicted: float,
    gold: float,
    *,
    relative_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
) -> bool:
    """Whether ``predicted`` matches ``gold`` within the given tolerance.

    With both tolerances ``None`` this is exact equality. Absolute tolerance is checked first
    (matches FinQA-style "round to N decimals" comparisons when expressed as, e.g.,
    ``absolute_tolerance=5e-6``); relative tolerance is skipped when ``gold == 0`` (undefined,
    not silently treated as a match).
    """
    if predicted == gold:
        return True
    within_absolute = absolute_tolerance is not None and abs(predicted - gold) <= absolute_tolerance
    within_relative = (
        relative_tolerance is not None
        and gold != 0
        and abs(predicted - gold) / abs(gold) <= relative_tolerance
    )
    return within_absolute or within_relative
