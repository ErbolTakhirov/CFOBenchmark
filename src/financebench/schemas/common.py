"""Shared enums and small value types used across every schema module.

This module is the dependency leaf of the schema package — it imports nothing else from
``financebench`` so that every other schema module can build on top of it without risking a
circular import.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DEFAULT_PROMPT_PROFILE",
    "SCHEMA_VERSION",
    "AnswerType",
    "EvalMode",
    "Language",
    "RunType",
    "Scale",
    "SplitOrigin",
    "TranslationProvenance",
]

#: Version of the canonical sample/prediction/metric/run schemas defined in this package.
#: Bump this (and add a migration note in docs/reproducibility.md) on any breaking field change.
SCHEMA_VERSION = "1.0"

#: The prompt profile a run uses unless told otherwise. Lives here, in the schema leaf, rather than
#: in ``prompts/profiles.py`` so that ``RunConfig`` can default to it without the schema layer
#: having to import the prompt layer (which imports the schema layer right back).
DEFAULT_PROMPT_PROFILE = "structured_financial_v1"

#: An ISO-639-1-ish language code, e.g. "en", "ru". Kept as a plain string rather than a closed
#: enum since new benchmarks may introduce languages we don't want to enumerate up front.
Language = str


class SplitOrigin(StrEnum):
    """Where a sample's split assignment actually comes from.

    Never mix an ``official`` split with a locally re-derived one without this label — a
    leaderboard comparing "finqa:test" scores across two runs is only meaningful if both drew
    from the same split origin.
    """

    OFFICIAL = "official"
    DERIVED_LOCAL = "derived_local"
    GENERATED_FROZEN = "generated_frozen"
    PUBLIC_SUBSET = "public_subset"
    USER_SUPPLIED = "user_supplied"


class TranslationProvenance(StrEnum):
    """How a non-English (or non-source-language) sample came to be in that language.

    Required on every bilingual EN/RU sample so a report never presents a machine-translated
    question as if it were an official-language original.
    """

    OFFICIAL_LANGUAGE = "official_language"
    HUMAN_VERIFIED_TRANSLATION = "human_verified_translation"
    MACHINE_TRANSLATED_DERIVED = "machine_translated_derived"


class AnswerType(StrEnum):
    """The shape of a gold (or predicted) answer, driving which metric applies."""

    NUMERIC = "numeric"
    TEXT = "text"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    PROGRAM = "program"
    REFUSAL = "refusal"


class RunType(StrEnum):
    """Whether a run evaluated a real model or merely exercised the pipeline.

    ``MOCK_TEST`` runs are produced by the ``mock`` provider, which is a *simulator with access to
    an answer oracle* — it proves the pipeline works, never that a model can do anything. Such runs
    are barred from the leaderboard and from the Finance Capability Index (see
    ``docs/research/validity_threats.md``).
    """

    REAL = "real"
    MOCK_TEST = "mock_test"


class EvalMode(StrEnum):
    """What a run actually measures.

    Model ability, retrieval ability and agent ability are *different things*; averaging them into
    one number tells you nothing about any of them. The mode is part of ``RunConfig``, so it lands
    in the run id and the response-cache key — two modes can never silently collide or share a
    cached answer.
    """

    #: The relevant context is handed to the model. Measures financial reasoning.
    CONTEXT_GIVEN = "context_given"
    #: The model gets a corpus and must retrieve its own evidence. Measures the RAG system.
    RETRIEVAL_REQUIRED = "retrieval_required"
    #: The model may call sandboxed tools. Measures tool selection, arguments, and use of results.
    TOOL_ASSISTED = "tool_assisted"


class Scale(StrEnum):
    """The magnitude multiplier implied by a numeric answer's presentation.

    Deliberately separate from *unit* (e.g. "percent", "usd", "ratio", "days") — ``12.5`` with
    ``unit=percent, scale=unit`` means 12.5%, not 12.5% * 1000. Adapters normalize each source
    benchmark's own scale/unit conflation (e.g. TAT-QA treats "percent" as a scale option) into
    this separated representation.
    """

    UNIT = "unit"
    THOUSAND = "thousand"
    MILLION = "million"
    BILLION = "billion"
