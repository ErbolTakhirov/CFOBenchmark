"""Sandboxed financial tools a model may call, and the loop that lets it."""

from __future__ import annotations

from financebench.tools.agent import MAX_TOOL_TURNS, run_agent
from financebench.tools.registry import (
    FINANCE_TOOLS,
    FORMULAS,
    ToolExecutionError,
    execute_tool,
    tool_manifest,
    tools_for_sample,
)
from financebench.tools.sandbox import SandboxError, safe_eval
from financebench.tools.trace import ToolCallRecord, ToolFailure, ToolTrace

__all__ = [
    "FINANCE_TOOLS",
    "FORMULAS",
    "MAX_TOOL_TURNS",
    "SandboxError",
    "ToolCallRecord",
    "ToolExecutionError",
    "ToolFailure",
    "ToolTrace",
    "execute_tool",
    "run_agent",
    "safe_eval",
    "tool_manifest",
    "tools_for_sample",
]
