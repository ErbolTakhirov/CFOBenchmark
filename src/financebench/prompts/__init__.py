"""Prompt rendering.

``profiles.py`` holds the versioned prompt profiles — what the model is *asked* for (a number, a
citation-bearing answer, an executable program), which decides what can honestly be measured.
``renderer.py`` is the thin dispatcher over them.
"""

from __future__ import annotations

from financebench.prompts.profiles import (
    DEFAULT_PROMPT_PROFILE,
    PromptProfile,
    RetrievedChunk,
    available_prompt_profiles,
    create_prompt_profile,
    register_prompt_profile,
)
from financebench.prompts.renderer import prompt_system_text, render_messages

__all__ = [
    "DEFAULT_PROMPT_PROFILE",
    "PromptProfile",
    "RetrievedChunk",
    "available_prompt_profiles",
    "create_prompt_profile",
    "prompt_system_text",
    "register_prompt_profile",
    "render_messages",
]
