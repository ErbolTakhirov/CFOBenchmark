"""Renders a :class:`CanonicalSample` into the messages sent to a model.

This is a thin dispatcher over the profile registry in ``prompts/profiles.py`` — the profile
decides what the model is asked to produce, and this module just resolves the name and calls it.
Keeping the indirection here (rather than having the engine import a concrete profile) is what
lets a run select its prompt by name from a config, and lets the leakage suite iterate over every
registered profile without knowing what they are.
"""

from __future__ import annotations

from collections.abc import Sequence

from financebench.prompts.profiles import (
    DEFAULT_PROMPT_PROFILE,
    RetrievedChunk,
    create_prompt_profile,
)
from financebench.schemas.common import EvalMode
from financebench.schemas.model_io import ChatMessage
from financebench.schemas.sample import CanonicalSample

__all__ = ["prompt_system_text", "render_messages"]


def render_messages(
    sample: CanonicalSample,
    *,
    profile_name: str = DEFAULT_PROMPT_PROFILE,
    mode: EvalMode = EvalMode.CONTEXT_GIVEN,
    retrieved: Sequence[RetrievedChunk] = (),
) -> tuple[ChatMessage, ...]:
    """Render ``sample`` into the message list for ``profile_name`` under ``mode``.

    Reads only the sample's question side. Never ``sample.gold``, never ``sample.evaluation`` —
    see ``tests/security/test_gold_answer_leakage.py``.
    """
    return create_prompt_profile(profile_name).render(sample, mode, retrieved)


def prompt_system_text(
    sample: CanonicalSample,
    *,
    profile_name: str = DEFAULT_PROMPT_PROFILE,
    mode: EvalMode = EvalMode.CONTEXT_GIVEN,
) -> str:
    """The system prompt a profile would use — for hashing into ``prompt_manifest.json``."""
    return create_prompt_profile(profile_name).system(sample, mode)
