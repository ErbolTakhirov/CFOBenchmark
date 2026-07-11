"""The finance tools a model may call, and the registry that describes them to it.

Every tool here is **deterministic from the benchmark's own inputs**. That is the design constraint,
and it is what makes the tool score reproducible: a run today and a run in a year produce the same
numbers, because nothing in this file asks the outside world anything.

That rules out the tools a finance-agent benchmark most obviously wants — live market prices, current
FX, a filings API — and the exclusion is deliberate. A benchmark whose scores depend on what a
third-party endpoint happened to return on a Tuesday is not a benchmark; it is a snapshot of a
Tuesday. External tools may exist, but they are marked ``external_non_reproducible`` and are excluded
from the default score (``tools/external.py``).

**Currency conversion uses the rates supplied in the sample**, not a live rate. A model asked to
convert an invoice must use the ledger's own FX table, exactly as an accountant would; if it goes
looking for today's rate instead, it has answered a different question.

Every tool returns a **string** — the same channel a real tool-calling API gives a model. It never
returns a Python object, because the model cannot receive one, and pretending otherwise would test a
path that does not exist in production.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from financebench.schemas.sample import CanonicalSample, Table
from financebench.schemas.tooling import ToolSpec
from financebench.tools.sandbox import SandboxError, safe_eval

__all__ = [
    "FINANCE_TOOLS",
    "FormulaSpec",
    "ToolExecutionError",
    "ToolImpl",
    "execute_tool",
    "tools_for_sample",
]

#: A tool's output is capped. A model that receives 200 kB of CSV back has not been helped, and a
#: tool that can flood the context window is a denial of service against the model's own reasoning.
MAX_OUTPUT_CHARS = 4000


class ToolExecutionError(Exception):
    """The tool could not do what was asked.

    This is **reported back to the model**, not raised at the harness. A model that asks for a column
    that does not exist, gets told so, and then asks for the right one is demonstrating precisely the
    recovery behaviour the benchmark is trying to measure — and a harness that crashed instead would
    have measured nothing.
    """


@dataclass(frozen=True)
class FormulaSpec:
    """A named financial formula, so a model can ask for one by name rather than rederiving it."""

    name: str
    expression: str
    inputs: tuple[str, ...]
    description: str


#: The formulas a CFO actually asks for. Offered by name so a model that knows *what* it needs but
#: misremembers the algebra can still get the right answer — and so that when it gets the number
#: wrong anyway, we know the formula was not the reason.
FORMULAS: dict[str, FormulaSpec] = {
    "gross_margin": FormulaSpec(
        "gross_margin", "(revenue - cogs) / revenue * 100", ("revenue", "cogs"), "Gross margin %"
    ),
    "net_margin": FormulaSpec(
        "net_margin", "net_income / revenue * 100", ("net_income", "revenue"), "Net margin %"
    ),
    "current_ratio": FormulaSpec(
        "current_ratio",
        "current_assets / current_liabilities",
        ("current_assets", "current_liabilities"),
        "Current ratio",
    ),
    "quick_ratio": FormulaSpec(
        "quick_ratio",
        "(current_assets - inventory) / current_liabilities",
        ("current_assets", "inventory", "current_liabilities"),
        "Quick ratio",
    ),
    "interest_coverage": FormulaSpec(
        "interest_coverage",
        "ebit / interest_expense",
        ("ebit", "interest_expense"),
        "Interest coverage ratio",
    ),
    "debt_to_equity": FormulaSpec(
        "debt_to_equity",
        "total_debt / total_equity",
        ("total_debt", "total_equity"),
        "Debt-to-equity",
    ),
    "roe": FormulaSpec(
        "roe",
        "net_income / total_equity * 100",
        ("net_income", "total_equity"),
        "Return on equity %",
    ),
    "roa": FormulaSpec(
        "roa",
        "net_income / total_assets * 100",
        ("net_income", "total_assets"),
        "Return on assets %",
    ),
    "percent_change": FormulaSpec(
        "percent_change", "(new - old) / old * 100", ("new", "old"), "Percentage change"
    ),
    "cagr": FormulaSpec(
        "cagr", "((end / start) ** (1 / years) - 1) * 100", ("end", "start", "years"), "CAGR %"
    ),
    "runway_months": FormulaSpec(
        "runway_months", "cash / monthly_burn", ("cash", "monthly_burn"), "Months of runway"
    ),
    "working_capital": FormulaSpec(
        "working_capital",
        "current_assets - current_liabilities",
        ("current_assets", "current_liabilities"),
        "Working capital",
    ),
}


def _decimal(value: Any, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ToolExecutionError(f"{field!r} is not a number: {value!r}") from exc


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated at {MAX_OUTPUT_CHARS} characters]"


# --------------------------------------------------------------------------- the tools


def _calculator(args: dict[str, Any], sample: CanonicalSample) -> str:
    """Arithmetic, through the AST sandbox. Never `eval`."""
    expression = args.get("expression")
    if not isinstance(expression, str):
        raise ToolExecutionError("calculator requires an 'expression' string")
    try:
        return str(safe_eval(expression).value)
    except SandboxError as exc:
        # The sandbox refusing is the sandbox WORKING. The model is told what was refused so it can
        # adapt — that recovery is a measured behaviour, not an incident.
        #
        # The PREFIX carries the classification, because that is all the agent loop can see. A model
        # that wrote `1/0` made a mistake; one that wrote `__import__` executed code. Marking both
        # the same way buried the security signal under a pile of division errors — a test caught it.
        prefix = "security_refused" if exc.is_security_event else "error"
        raise ToolExecutionError(f"{prefix}: {exc}") from exc


def _formula(args: dict[str, Any], sample: CanonicalSample) -> str:
    name = str(args.get("name", "")).strip()
    spec = FORMULAS.get(name)
    if spec is None:
        raise ToolExecutionError(
            f"unknown formula {name!r}. Available: {', '.join(sorted(FORMULAS))}"
        )
    values = args.get("values")
    if not isinstance(values, dict):
        raise ToolExecutionError(f"formula {name!r} requires a 'values' object")

    missing = [i for i in spec.inputs if i not in values]
    if missing:
        raise ToolExecutionError(f"formula {name!r} needs {missing} — got {sorted(values)}")

    # Substitute into the expression, then hand it to the SAME sandbox. The formula registry is a
    # convenience, not a second evaluator: there is exactly one place in this codebase where a string
    # becomes a number, and it is the AST walker.
    expression = spec.expression
    for key in sorted(spec.inputs, key=len, reverse=True):  # longest first: `revenue` before `rev`
        expression = expression.replace(key, f"({_decimal(values[key], key)})")
    try:
        return str(safe_eval(expression).value)
    except SandboxError as exc:
        raise ToolExecutionError(f"formula {name!r}: {exc}") from exc


def _percent_change(args: dict[str, Any], sample: CanonicalSample) -> str:
    old = _decimal(args.get("old"), "old")
    new = _decimal(args.get("new"), "new")
    if old == 0:
        raise ToolExecutionError("percentage change from zero is undefined")
    return str((new - old) / old * 100)


def _basis_points(args: dict[str, Any], sample: CanonicalSample) -> str:
    """bps <-> percent. A dedicated tool because 250 bps vs 2.5 % is a catastrophic-error class in
    this platform's own failure taxonomy, and a model that reaches for a converter is a model that
    knows the trap exists."""
    if "percent" in args:
        return str(_decimal(args["percent"], "percent") * 100)
    if "basis_points" in args:
        return str(_decimal(args["basis_points"], "basis_points") / 100)
    raise ToolExecutionError("provide either 'percent' or 'basis_points'")


def _find_table(sample: CanonicalSample, table_id: str | None) -> Table:
    tables = sample.context.tables
    if not tables:
        raise ToolExecutionError("this question has no tables")
    if table_id:
        for table in tables:
            if table.table_id == table_id:
                return table
        raise ToolExecutionError(
            f"no table {table_id!r}. Available: {[t.table_id for t in tables]}"
        )
    return tables[0]


def _table_lookup(args: dict[str, Any], sample: CanonicalSample) -> str:
    table = _find_table(sample, args.get("table_id"))
    row_query = str(args.get("row", "")).strip().casefold()
    if not row_query:
        raise ToolExecutionError("table_lookup requires a 'row' label to search for")

    matches = [row for row in table.rows if any(row_query in str(c).casefold() for c in row[:2])]
    if not matches:
        labels = [row[0] for row in table.rows[:25] if row]
        raise ToolExecutionError(f"no row matching {row_query!r}. Row labels include: {labels}")
    return _clip("\n".join(" | ".join(row) for row in matches[:10]))


def _csv_query(args: dict[str, Any], sample: CanonicalSample) -> str:
    """Filter / group / aggregate over a table. One tool rather than three, because a model that has
    to chain three calls to answer "what did I spend on payroll in March" will fail on the chaining
    rather than on the finance, and we would learn nothing about the finance."""
    table = _find_table(sample, args.get("table_id"))
    if not table.rows:
        raise ToolExecutionError("that table is empty")

    header = [str(c).strip() for c in table.rows[0]]
    body = [list(r) for r in table.rows[1:]]

    def column_index(name: str) -> int:
        target = name.strip().casefold()
        for index, column in enumerate(header):
            if column.casefold() == target:
                return index
        raise ToolExecutionError(f"no column {name!r}. Columns: {header}")

    rows = body
    where = args.get("where")
    if isinstance(where, dict):
        for column, wanted in where.items():
            index = column_index(str(column))
            needle = str(wanted).strip().casefold()
            rows = [r for r in rows if index < len(r) and needle in str(r[index]).casefold()]

    aggregate = str(args.get("aggregate", "")).strip().lower()
    if not aggregate:
        return _clip(
            " | ".join(header) + "\n" + "\n".join(" | ".join(map(str, r)) for r in rows[:40])
        )

    column = args.get("column")
    if not column:
        raise ToolExecutionError(f"aggregate {aggregate!r} needs a 'column'")
    index = column_index(str(column))

    values: list[Decimal] = []
    for row in rows:
        if index >= len(row):
            continue
        raw = str(row[index]).replace(",", "").replace("$", "").strip()
        raw = raw.strip("()") if raw.startswith("(") and raw.endswith(")") else raw
        try:
            values.append(Decimal(raw))
        except InvalidOperation:
            continue  # a non-numeric cell is skipped, not fatal: headers and notes live in tables too

    if aggregate == "count":
        return str(len(rows))
    if not values:
        raise ToolExecutionError(f"column {column!r} has no numeric values in the matching rows")
    if aggregate == "sum":
        return str(sum(values))
    if aggregate == "mean":
        return str(sum(values) / len(values))
    if aggregate == "min":
        return str(min(values))
    if aggregate == "max":
        return str(max(values))
    raise ToolExecutionError(f"unknown aggregate {aggregate!r}: use sum|mean|min|max|count")


def _convert_currency(args: dict[str, Any], sample: CanonicalSample) -> str:
    """Convert using the rates **supplied in the sample** — never a live rate.

    An accountant converting an invoice uses the rate the ledger records, not today's. A model that
    goes looking for a live rate has answered a different question, and a benchmark that let it would
    produce a score that changes when the market moves.
    """
    amount = _decimal(args.get("amount"), "amount")
    source = str(args.get("from", "")).strip().upper()
    target = str(args.get("to", "")).strip().upper()

    rates: dict[str, Decimal] = {}
    for table in sample.context.tables:
        if "rate" not in (table.table_id + (table.caption or "")).lower():
            continue
        for row in table.rows[1:]:
            if len(row) >= 2:
                try:
                    rates[str(row[0]).strip().upper()] = Decimal(str(row[1]).strip())
                except InvalidOperation:
                    continue

    if not rates:
        raise ToolExecutionError(
            "this ledger supplies no exchange rates, so the conversion cannot be done. "
            "Do not use an external rate — the answer must come from the books."
        )
    if source not in rates and source != target:
        raise ToolExecutionError(f"no rate for {source!r} in this ledger. Rates: {sorted(rates)}")

    rate = Decimal(1) if source == target else rates[source]
    return str(amount * rate)


def _document_search(args: dict[str, Any], sample: CanonicalSample) -> str:
    query = str(args.get("query", "")).strip().casefold()
    if not query:
        raise ToolExecutionError("document_search requires a 'query'")
    hits = [
        line
        for block in sample.context.text
        for line in block.splitlines()
        if query in line.casefold()
    ]
    if not hits:
        raise ToolExecutionError(f"no lines matching {query!r} in the supplied documents")
    return _clip("\n".join(hits[:20]))


def _date_diff(args: dict[str, Any], sample: CanonicalSample) -> str:
    def parse(field: str) -> date:
        raw = str(args.get(field, "")).strip()
        try:
            return date.fromisoformat(raw)
        except ValueError as exc:
            raise ToolExecutionError(
                f"{field} must be an ISO date (YYYY-MM-DD), got {raw!r}"
            ) from exc

    return str((parse("to") - parse("from")).days)


# --------------------------------------------------------------------------- the registry

ToolImpl = Callable[[dict[str, Any], CanonicalSample], str]


@dataclass(frozen=True)
class _Tool:
    spec: ToolSpec
    run: ToolImpl


def _spec(
    name: str, description: str, properties: dict[str, Any], required: Sequence[str]
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters_schema={
            "type": "object",
            "properties": properties,
            "required": list(required),
        },
    )


_NUM = {"type": "number"}
_STR = {"type": "string"}

FINANCE_TOOLS: dict[str, _Tool] = {
    "calculator": _Tool(
        _spec(
            "calculator",
            "Evaluate an arithmetic expression exactly (decimal, not floating point). "
            "Supports + - * / % ** and abs/round/min/max/sum. Example: '(1500-1200)/1200*100'.",
            {"expression": _STR},
            ["expression"],
        ),
        _calculator,
    ),
    "formula": _Tool(
        _spec(
            "formula",
            "Apply a named financial formula. Available: " + ", ".join(sorted(FORMULAS)) + ".",
            {"name": _STR, "values": {"type": "object"}},
            ["name", "values"],
        ),
        _formula,
    ),
    "percent_change": _Tool(
        _spec(
            "percent_change",
            "Percentage change from an old value to a new one.",
            {"old": _NUM, "new": _NUM},
            ["old", "new"],
        ),
        _percent_change,
    ),
    "basis_points": _Tool(
        _spec(
            "basis_points",
            "Convert between percent and basis points. Give either 'percent' or 'basis_points'.",
            {"percent": _NUM, "basis_points": _NUM},
            [],
        ),
        _basis_points,
    ),
    "table_lookup": _Tool(
        _spec(
            "table_lookup",
            "Find rows in a table of the question's context by their label.",
            {"row": _STR, "table_id": _STR},
            ["row"],
        ),
        _table_lookup,
    ),
    "csv_query": _Tool(
        _spec(
            "csv_query",
            "Filter, group and aggregate a table. 'where' filters by column values; 'aggregate' is "
            "one of sum|mean|min|max|count over 'column'. Omit 'aggregate' to see matching rows.",
            {
                "table_id": _STR,
                "where": {"type": "object"},
                "aggregate": _STR,
                "column": _STR,
            },
            [],
        ),
        _csv_query,
    ),
    "convert_currency": _Tool(
        _spec(
            "convert_currency",
            "Convert an amount using the exchange rates SUPPLIED IN THIS QUESTION's ledger. "
            "Never uses a live market rate.",
            {"amount": _NUM, "from": _STR, "to": _STR},
            ["amount", "from", "to"],
        ),
        _convert_currency,
    ),
    "document_search": _Tool(
        _spec(
            "document_search",
            "Search the supplied documents for lines containing a phrase.",
            {"query": _STR},
            ["query"],
        ),
        _document_search,
    ),
    "date_diff": _Tool(
        _spec(
            "date_diff",
            "Days between two ISO dates (YYYY-MM-DD). Useful for ageing receivables.",
            {"from": _STR, "to": _STR},
            ["from", "to"],
        ),
        _date_diff,
    ),
}


def tools_for_sample(sample: CanonicalSample) -> tuple[ToolSpec, ...]:
    """Which tools a sample offers the model.

    Every sample is offered the same set. Tailoring the toolbox per question would leak the answer's
    shape — "we gave you a currency converter" is a very strong hint that the question is about
    currency — and the benchmark is supposed to measure whether the model can *choose* a tool.
    """
    return tuple(tool.spec for tool in FINANCE_TOOLS.values())


def execute_tool(name: str, arguments: dict[str, Any], sample: CanonicalSample) -> str:
    """Run one tool call. Raises :class:`ToolExecutionError`, which is shown to the model."""
    tool = FINANCE_TOOLS.get(name)
    if tool is None:
        # A hallucinated tool. The model is told, precisely, so that "it invented a tool" and "it
        # used a real tool badly" stay distinguishable — they are different failures with different
        # fixes, and a single "tool error" would hide the difference.
        raise ToolExecutionError(
            f"no such tool: {name!r}. Available tools: {', '.join(sorted(FINANCE_TOOLS))}"
        )
    if not isinstance(arguments, dict):
        raise ToolExecutionError(
            f"{name}: arguments must be an object, got {type(arguments).__name__}"
        )
    return _clip(tool.run(arguments, sample))


def tool_manifest() -> str:
    """The toolbox, as JSON — for the run artifacts, so a reader can see exactly what was offered."""
    return json.dumps(
        [tool.spec.model_dump(mode="json") for tool in FINANCE_TOOLS.values()], indent=2
    )
