"""The oracles. Every SMB-CFO gold answer is computed here, by Python, over the books.

**No LLM ever produces a gold answer.** A benchmark whose answers were written by a model inherits
that model's mistakes and then grades other models against them — which is not evaluation, it is
agreement-measurement. An oracle is a function you can read, disagree with, and test.

Everything is `Decimal`. Not fussiness: a benchmark that computes its own gold in binary floating
point will produce answers ending in `...0000001`, then grade a model wrong for saying the round
number that a human would. The money is exact, and the tolerance in the metric exists for the
*model's* rounding, not for ours.

Each oracle returns a :class:`OracleResult` carrying not just the number but its **unit, currency,
scale, period, and the exact rows it came from** — so a failure can be attributed (wrong period?
wrong currency? right number, wrong scale?) rather than just marked wrong.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from financebench.datasets.smb_cfo.business import Business, Invoice

__all__ = [
    "OracleResult",
    "accounts_payable_pressure",
    "accounts_receivable_total",
    "budget_variance",
    "cash_gap_date",
    "cash_runway_months",
    "current_cash_balance",
    "customer_concentration",
    "duplicate_transactions",
    "expense_growth",
    "gross_margin",
    "missing_periods",
    "monthly_burn",
    "normalize_currency",
    "operating_margin",
    "overdue_receivables",
    "receivable_prioritization",
    "revenue_growth",
    "scenario_collect_receivables",
    "scenario_cut_expenses",
    "scenario_lose_customer",
    "scenario_revenue_drop",
    "supplier_concentration",
    "tax_inclusive_vs_exclusive",
]

_CENTS = Decimal("0.01")


@dataclass(frozen=True)
class OracleResult:
    """A gold answer, plus everything needed to grade a wrong one precisely."""

    value: Decimal | str | None
    unit: str  # "currency" | "months" | "percent" | "percentage_points" | "date" | "count" | "text"
    currency: str | None = None
    scale: str = "unit"
    period: str | None = None
    #: The ledger rows the answer is derived from. This is what makes a "show your evidence"
    #: question checkable, and what a grounding metric checks against.
    evidence_ids: tuple[str, ...] = ()
    #: Set when the question genuinely cannot be answered from the data. The model is *supposed* to
    #: refuse; answering anyway is the most dangerous failure there is.
    unanswerable: bool = False
    note: str = ""
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def numeric(self) -> float | None:
        return float(self.value) if isinstance(self.value, Decimal) else None


def _q(value: Decimal) -> Decimal:
    return value.quantize(_CENTS)


def normalize_currency(amount: Decimal, currency: str, business: Business) -> Decimal:
    """Convert to base currency **using the rate supplied in the books**.

    A model that applies a rate it "knows" is hallucinating: rates move, and the only correct rate
    is the one the ledger states. If the currency has no rate in the data, the conversion cannot be
    done — and that is an unanswerable question, not an excuse to guess.
    """
    rate = business.fx_rates.get(currency)
    if rate is None:
        raise KeyError(currency)
    return _q(amount * rate)


# --------------------------------------------------------------------------- cash


def current_cash_balance(business: Business) -> OracleResult:
    """Opening balance plus every transaction up to `as_of`. The fundamental identity."""
    total = business.opening_balance
    used: list[str] = []
    for txn in business.transactions:
        if txn.txn_date > business.as_of:
            continue
        total += normalize_currency(txn.amount, txn.currency, business)
        used.append(txn.txn_id)
    return OracleResult(
        value=_q(total),
        unit="currency",
        currency=business.base_currency,
        period=f"as of {business.as_of.isoformat()}",
        evidence_ids=tuple(used),
        detail={"opening_balance": str(business.opening_balance)},
    )


def _monthly_net(business: Business) -> dict[str, Decimal]:
    by_month: dict[str, Decimal] = defaultdict(Decimal)
    for txn in business.transactions:
        if txn.txn_date > business.as_of:
            continue
        by_month[f"{txn.txn_date:%Y-%m}"] += normalize_currency(txn.amount, txn.currency, business)
    return dict(by_month)


def monthly_burn(business: Business, months: int = 3) -> OracleResult:
    """Average NET monthly cash movement over the last `months` complete months.

    Net, not gross expenses. "Burn" is what the bank balance actually loses in a month — a business
    with 100k of costs and 95k of revenue burns 5k, not 100k. Reporting gross expenses as burn
    overstates the danger by twentyfold, and a runway computed from it is nonsense.

    A **positive** result here means the business is cash-generative, and burn is reported as
    negative. That sign convention is stated in the question.
    """
    by_month = _monthly_net(business)
    complete = sorted(by_month)[:-1] if len(by_month) > 1 else sorted(by_month)
    recent = complete[-months:]
    if not recent:
        return OracleResult(
            value=None, unit="currency", unanswerable=True, note="no complete months"
        )

    average = sum((by_month[m] for m in recent), Decimal(0)) / Decimal(len(recent))
    return OracleResult(
        value=_q(average),
        unit="currency",
        currency=business.base_currency,
        period=f"{recent[0]}..{recent[-1]}",
        detail={"months_used": ",".join(recent)},
    )


def cash_runway_months(business: Business, months: int = 3) -> OracleResult:
    """How many months until the cash runs out at the recent net burn rate.

    If the business is cash-generative the runway is **infinite**, and the honest answer is to say
    so — not to divide by a positive number and report a nonsense figure. A model that reports a
    finite runway for a profitable business has misunderstood the question.
    """
    cash = current_cash_balance(business).value
    burn = monthly_burn(business, months=months)
    assert isinstance(cash, Decimal)

    if burn.value is None:
        return OracleResult(value=None, unit="months", unanswerable=True, note="burn unknown")
    assert isinstance(burn.value, Decimal)

    # Already out of money. `cash / abs(burn)` would return a NEGATIVE number of months here, which
    # is not a shorter runway — it is nonsense, and a model that "matched" it would be matching a
    # broken gold. A business with no cash has no runway: the answer is zero, and the gap is now.
    if cash <= 0:
        return OracleResult(
            value=Decimal("0.0"),
            unit="months",
            note="the cash balance is already zero or negative; the runway is exhausted",
            detail={"cash": str(cash), "net_monthly_burn": str(burn.value)},
        )

    if burn.value >= 0:
        return OracleResult(
            value="infinite",
            unit="months",
            note="the business is cash-generative over the period; there is no runway to run out of",
            detail={"net_monthly": str(burn.value), "cash": str(cash)},
        )

    runway = cash / abs(burn.value)
    return OracleResult(
        value=runway.quantize(Decimal("0.1")),
        unit="months",
        period=burn.period,
        detail={"cash": str(cash), "net_monthly_burn": str(burn.value)},
    )


def cash_gap_date(business: Business, months: int = 3) -> OracleResult:
    """The month in which cash first goes negative, projecting the recent net burn forward."""
    runway = cash_runway_months(business, months=months)
    if runway.unanswerable:
        return OracleResult(value=None, unit="date", unanswerable=True, note=runway.note)
    if runway.value == "infinite":
        return OracleResult(
            value="never",
            unit="date",
            note="cash is growing; there is no cash-gap date",
        )
    assert isinstance(runway.value, Decimal)

    # Already negative: the gap is not in the future, it is now.
    if runway.value <= 0:
        return OracleResult(
            value="already negative",
            unit="date",
            note="the cash balance is already zero or below; the gap is not a forecast",
        )

    whole = int(runway.value)
    cursor = business.as_of
    for _ in range(whole + 1):
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return OracleResult(
        value=f"{cursor:%Y-%m}",
        unit="date",
        detail={"runway_months": str(runway.value)},
    )


# --------------------------------------------------------------------------- receivables / payables


def _open_invoices(business: Business) -> list[Invoice]:
    return [i for i in business.invoices if not i.paid and i.issue_date <= business.as_of]


def accounts_receivable_total(business: Business, *, gross: bool = True) -> OracleResult:
    """What customers owe.

    ``gross=True`` (the default, and what a customer actually owes) includes tax. The net figure is
    what the business *earns*; the gross is what lands in the bank. Confusing them is one of the
    benchmark's deliberate traps because it is one of the commonest real errors.
    """
    open_invoices = _open_invoices(business)
    total = sum(
        (
            normalize_currency(i.gross if gross else i.amount, i.currency, business)
            for i in open_invoices
        ),
        Decimal(0),
    )
    return OracleResult(
        value=_q(total),
        unit="currency",
        currency=business.base_currency,
        period=f"as of {business.as_of.isoformat()}",
        evidence_ids=tuple(i.invoice_id for i in open_invoices),
        detail={"basis": "gross (tax-inclusive)" if gross else "net (tax-exclusive)"},
    )


def overdue_receivables(business: Business) -> OracleResult:
    """Unpaid invoices whose due date has already passed."""
    overdue = [i for i in _open_invoices(business) if i.due_date < business.as_of]
    total = sum((normalize_currency(i.gross, i.currency, business) for i in overdue), Decimal(0))
    return OracleResult(
        value=_q(total),
        unit="currency",
        currency=business.base_currency,
        period=f"as of {business.as_of.isoformat()}",
        evidence_ids=tuple(i.invoice_id for i in overdue),
        detail={"count": str(len(overdue))},
    )


def receivable_prioritization(business: Business, top_n: int = 3) -> OracleResult:
    """Which overdue invoices to chase first — largest and most overdue.

    Ranked by amount x days overdue, because chasing a small invoice that is very late is usually
    worth less than a large one that is moderately late.
    """
    overdue = [i for i in _open_invoices(business) if i.due_date < business.as_of]
    if not overdue:
        return OracleResult(value=None, unit="text", unanswerable=True, note="nothing is overdue")

    def weight(inv: Invoice) -> Decimal:
        days = Decimal((business.as_of - inv.due_date).days)
        return normalize_currency(inv.gross, inv.currency, business) * days

    ranked = sorted(overdue, key=lambda i: (-weight(i), i.invoice_id))[:top_n]
    return OracleResult(
        value=", ".join(i.invoice_id for i in ranked),
        unit="text",
        evidence_ids=tuple(i.invoice_id for i in ranked),
        detail={"ranked_by": "amount x days overdue"},
    )


def accounts_payable_pressure(business: Business, days: int = 30) -> OracleResult:
    """What we owe suppliers, due within `days`."""
    horizon = business.as_of + timedelta(days=days)
    due = [p for p in business.payables if not p.paid and p.due_date <= horizon]
    total = sum((normalize_currency(p.amount, p.currency, business) for p in due), Decimal(0))
    return OracleResult(
        value=_q(total),
        unit="currency",
        currency=business.base_currency,
        period=f"next {days} days",
        evidence_ids=tuple(p.payment_id for p in due),
    )


# --------------------------------------------------------------------------- margins & growth


def _revenue_expense(business: Business) -> tuple[Decimal, Decimal, Decimal]:
    revenue = Decimal(0)
    cogs = Decimal(0)
    opex = Decimal(0)
    for txn in business.transactions:
        if txn.txn_date > business.as_of:
            continue
        amount = normalize_currency(txn.amount, txn.currency, business)
        if amount > 0:
            revenue += amount
        elif txn.category == "Cost of goods sold":
            cogs += -amount
        else:
            opex += -amount
    return revenue, cogs, opex


def gross_margin(business: Business) -> OracleResult:
    """(Revenue - COGS) / Revenue. A PERCENT, not a percentage point."""
    revenue, cogs, _ = _revenue_expense(business)
    if revenue == 0:
        return OracleResult(value=None, unit="percent", unanswerable=True, note="no revenue")
    margin = (revenue - cogs) / revenue * 100
    return OracleResult(
        value=margin.quantize(Decimal("0.1")),
        unit="percent",
        period=f"{business.period_start:%Y-%m}..{business.as_of:%Y-%m}",
        detail={"revenue": str(_q(revenue)), "cogs": str(_q(cogs))},
    )


def operating_margin(business: Business) -> OracleResult:
    """(Revenue - COGS - Opex) / Revenue, as a percent."""
    revenue, cogs, opex = _revenue_expense(business)
    if revenue == 0:
        return OracleResult(value=None, unit="percent", unanswerable=True, note="no revenue")
    margin = (revenue - cogs - opex) / revenue * 100
    return OracleResult(
        value=margin.quantize(Decimal("0.1")),
        unit="percent",
        period=f"{business.period_start:%Y-%m}..{business.as_of:%Y-%m}",
        detail={"revenue": str(_q(revenue)), "cogs": str(_q(cogs)), "opex": str(_q(opex))},
    )


def _monthly_flow(business: Business, *, inflow: bool) -> dict[str, Decimal]:
    by_month: dict[str, Decimal] = defaultdict(Decimal)
    for txn in business.transactions:
        if txn.txn_date > business.as_of:
            continue
        amount = normalize_currency(txn.amount, txn.currency, business)
        if (amount > 0) == inflow:
            by_month[f"{txn.txn_date:%Y-%m}"] += abs(amount)
    return dict(by_month)


def _growth(by_month: dict[str, Decimal], unit_label: str) -> OracleResult:
    months = sorted(by_month)
    if len(months) < 2:
        return OracleResult(
            value=None, unit="percent", unanswerable=True, note="fewer than two months of data"
        )
    # Use the last two COMPLETE months. The current month is partial, and comparing a half-month to
    # a full one manufactures a 50% collapse that never happened.
    previous, latest = months[-3], months[-2]
    if by_month[previous] == 0:
        return OracleResult(
            value=None, unit="percent", unanswerable=True, note="base month is zero"
        )
    change = (by_month[latest] - by_month[previous]) / by_month[previous] * 100
    return OracleResult(
        value=change.quantize(Decimal("0.1")),
        unit="percent",
        period=f"{previous} -> {latest}",
        detail={unit_label: f"{previous}={by_month[previous]}, {latest}={by_month[latest]}"},
    )


def revenue_growth(business: Business) -> OracleResult:
    return _growth(_monthly_flow(business, inflow=True), "revenue")


def expense_growth(business: Business) -> OracleResult:
    return _growth(_monthly_flow(business, inflow=False), "expenses")


def budget_variance(business: Business, category: str) -> OracleResult:
    """Actual monthly spend vs budget for a category, in currency and as a percent."""
    budget = business.monthly_budget.get(category)
    if budget is None:
        return OracleResult(
            value=None,
            unit="currency",
            unanswerable=True,
            note=f"no budget is set for {category!r} — the data does not support this question",
        )
    spend_by_month: dict[str, Decimal] = defaultdict(Decimal)
    for txn in business.transactions:
        if txn.category == category and txn.txn_date <= business.as_of and txn.amount < 0:
            spend_by_month[f"{txn.txn_date:%Y-%m}"] += -normalize_currency(
                txn.amount, txn.currency, business
            )
    if not spend_by_month:
        return OracleResult(value=None, unit="currency", unanswerable=True, note="no spend")

    months = sorted(spend_by_month)[:-1] or sorted(spend_by_month)
    average = sum((spend_by_month[m] for m in months), Decimal(0)) / Decimal(len(months))
    variance = average - budget
    return OracleResult(
        value=_q(variance),
        unit="currency",
        currency=business.base_currency,
        period=f"average of {months[0]}..{months[-1]}",
        detail={"budget": str(budget), "actual_average": str(_q(average))},
    )


# --------------------------------------------------------------------------- concentration


def _concentration(totals: dict[str, Decimal], label: str) -> OracleResult:
    grand = sum(totals.values(), Decimal(0))
    if grand == 0:
        return OracleResult(value=None, unit="percent", unanswerable=True, note="no revenue")
    top_name, top_value = max(totals.items(), key=lambda kv: (kv[1], kv[0]))
    share = top_value / grand * 100
    return OracleResult(
        value=share.quantize(Decimal("0.1")),
        unit="percent",
        detail={label: top_name, "amount": str(_q(top_value)), "total": str(_q(grand))},
    )


def customer_concentration(business: Business) -> OracleResult:
    """The largest customer's share of revenue. A concentration risk, and a percent."""
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for invoice in business.invoices:
        if invoice.issue_date > business.as_of:
            continue
        customer = business.customer(invoice.customer_id)
        if customer:
            totals[customer.name] += normalize_currency(invoice.amount, invoice.currency, business)
    return _concentration(dict(totals), "top_customer")


def supplier_concentration(business: Business) -> OracleResult:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for txn in business.transactions:
        if txn.amount < 0 and txn.counterparty != "Internal" and txn.txn_date <= business.as_of:
            totals[txn.counterparty] += -normalize_currency(txn.amount, txn.currency, business)
    return _concentration(dict(totals), "top_supplier")


# --------------------------------------------------------------------------- scenarios


def scenario_collect_receivables(business: Business, fraction: Decimal) -> OracleResult:
    """Cash after collecting `fraction` of what is owed."""
    cash = current_cash_balance(business).value
    receivable = accounts_receivable_total(business).value
    assert isinstance(cash, Decimal) and isinstance(receivable, Decimal)
    return OracleResult(
        value=_q(cash + receivable * fraction),
        unit="currency",
        currency=business.base_currency,
        detail={
            "cash_now": str(cash),
            "receivables": str(receivable),
            "collected_fraction": str(fraction),
        },
    )


def scenario_revenue_drop(business: Business, fraction: Decimal, months: int = 3) -> OracleResult:
    """Runway if revenue fell by `fraction` and costs held.

    The trap: revenue falling 20% does not make burn 20% worse — it makes it worse by 20% *of
    revenue*, which against a small net margin can be catastrophic. A model that scales the burn
    rate rather than the revenue line gets a comfortingly wrong answer.
    """
    by_month_in = _monthly_flow(business, inflow=True)
    by_month_out = _monthly_flow(business, inflow=False)
    months_list = sorted(set(by_month_in) | set(by_month_out))[:-1][-months:]
    if not months_list:
        return OracleResult(value=None, unit="months", unanswerable=True, note="not enough data")

    revenue = sum((by_month_in.get(m, Decimal(0)) for m in months_list), Decimal(0)) / len(
        months_list
    )
    expenses = sum((by_month_out.get(m, Decimal(0)) for m in months_list), Decimal(0)) / len(
        months_list
    )
    new_net = revenue * (Decimal(1) - fraction) - expenses

    cash = current_cash_balance(business).value
    assert isinstance(cash, Decimal)
    if cash <= 0:
        return OracleResult(
            value=Decimal("0.0"), unit="months", note="the cash balance is already exhausted"
        )
    if new_net >= 0:
        return OracleResult(value="infinite", unit="months", note="still cash-generative")
    return OracleResult(
        value=(cash / abs(new_net)).quantize(Decimal("0.1")),
        unit="months",
        detail={
            "monthly_revenue": str(_q(revenue)),
            "monthly_expenses": str(_q(expenses)),
            "new_monthly_net": str(_q(new_net)),
        },
    )


def scenario_lose_customer(business: Business) -> OracleResult:
    """Revenue lost if the largest customer left — in currency, over the period."""
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for invoice in business.invoices:
        if invoice.issue_date > business.as_of:
            continue
        customer = business.customer(invoice.customer_id)
        if customer:
            totals[customer.name] += normalize_currency(invoice.amount, invoice.currency, business)
    if not totals:
        return OracleResult(value=None, unit="currency", unanswerable=True, note="no revenue")
    name, value = max(totals.items(), key=lambda kv: (kv[1], kv[0]))
    return OracleResult(
        value=_q(value),
        unit="currency",
        currency=business.base_currency,
        detail={"customer": name},
    )


def scenario_cut_expenses(business: Business, fraction: Decimal, months: int = 3) -> OracleResult:
    """Runway if expenses were cut by `fraction`."""
    by_month_in = _monthly_flow(business, inflow=True)
    by_month_out = _monthly_flow(business, inflow=False)
    months_list = sorted(set(by_month_in) | set(by_month_out))[:-1][-months:]
    if not months_list:
        return OracleResult(value=None, unit="months", unanswerable=True, note="not enough data")

    revenue = sum((by_month_in.get(m, Decimal(0)) for m in months_list), Decimal(0)) / len(
        months_list
    )
    expenses = sum((by_month_out.get(m, Decimal(0)) for m in months_list), Decimal(0)) / len(
        months_list
    )
    new_net = revenue - expenses * (Decimal(1) - fraction)

    cash = current_cash_balance(business).value
    assert isinstance(cash, Decimal)
    if cash <= 0:
        return OracleResult(
            value=Decimal("0.0"), unit="months", note="the cash balance is already exhausted"
        )
    if new_net >= 0:
        return OracleResult(value="infinite", unit="months", note="cash-generative after the cut")
    return OracleResult(
        value=(cash / abs(new_net)).quantize(Decimal("0.1")),
        unit="months",
        detail={"new_monthly_net": str(_q(new_net))},
    )


# --------------------------------------------------------------------------- data quality


def duplicate_transactions(business: Business) -> OracleResult:
    """Transactions that look like double entries: same date, amount and counterparty."""
    seen: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for txn in business.transactions:
        seen[(txn.txn_date.isoformat(), str(txn.amount), txn.counterparty)].append(txn.txn_id)
    duplicates = sorted(txn_id for ids in seen.values() if len(ids) > 1 for txn_id in ids[1:])
    return OracleResult(
        value=Decimal(len(duplicates)),
        unit="count",
        evidence_ids=tuple(duplicates),
        detail={"duplicate_ids": ",".join(duplicates) or "none"},
    )


def missing_periods(business: Business, category: str = "Payroll") -> OracleResult:
    """Months with no entry for a category that appears every other month.

    Payroll and rent are paid every month. A month without one is a **gap in the data**, not a month
    the business didn't pay its staff — and a model that reports the burn rate without noticing is
    reporting a number that is wrong for a reason it never mentioned.
    """
    months_seen = {f"{t.txn_date:%Y-%m}" for t in business.transactions if t.category == category}
    expected: list[str] = []
    cursor = business.period_start
    while cursor <= business.as_of:
        expected.append(f"{cursor:%Y-%m}")
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    missing = sorted(set(expected[:-1]) - months_seen)
    return OracleResult(
        value=Decimal(len(missing)),
        unit="count",
        detail={"missing_months": ",".join(missing) or "none", "category": category},
    )


def tax_inclusive_vs_exclusive(business: Business) -> OracleResult:
    """The difference between gross and net receivables — i.e. the tax sitting in the AR balance."""
    gross = accounts_receivable_total(business, gross=True).value
    net = accounts_receivable_total(business, gross=False).value
    assert isinstance(gross, Decimal) and isinstance(net, Decimal)
    return OracleResult(
        value=_q(gross - net),
        unit="currency",
        currency=business.base_currency,
        detail={"gross": str(gross), "net": str(net), "tax_rate": str(business.tax_rate)},
    )
