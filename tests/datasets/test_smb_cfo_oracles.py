"""SMB-CFO oracle correctness — property-based, over hundreds of generated businesses.

These oracles ARE the gold. If one is wrong, every model is graded against a wrong answer, and the
benchmark measures agreement-with-a-bug. So they are tested as *properties* over many seeds, not as
examples: an accounting identity either holds for every business or it does not hold at all.

The tests below caught a real bug. `cash_runway_months` computed `cash / abs(burn)`, which for a
business already in overdraft (negative cash) returns a NEGATIVE number of months. That is not a
short runway — it is nonsense, and a model that "matched" it would have been matching broken gold.
A business with no cash has no runway: the answer is zero, and the gap is now, not in the future.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from financebench.datasets.smb_cfo import oracles as O
from financebench.datasets.smb_cfo.adversarial import (
    INJECTION_CANARY,
    INJECTION_PAYLOADS,
    inject_into_ledger,
)
from financebench.datasets.smb_cfo.business import generate_business

SEEDS = range(150)


def _businesses():
    return [generate_business(seed, multi_currency=(seed % 4 == 0)) for seed in SEEDS]


# --------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("seed", [0, 1, 42, 999])
def test_the_same_seed_always_produces_the_same_business(seed: int) -> None:
    """A benchmark that regenerates differently each run cannot be reproduced, and cannot even be
    compared with itself."""
    a = generate_business(seed)
    b = generate_business(seed)

    assert a.name == b.name
    assert a.opening_balance == b.opening_balance
    assert len(a.transactions) == len(b.transactions)
    assert [t.txn_id for t in a.transactions] == [t.txn_id for t in b.transactions]
    assert [t.amount for t in a.transactions] == [t.amount for t in b.transactions]
    assert [i.invoice_id for i in a.invoices] == [i.invoice_id for i in b.invoices]


def test_different_seeds_produce_different_businesses() -> None:
    """Guards the guard: if the generator ignored the seed, every determinism test above would pass
    vacuously and the benchmark would be one business repeated 600 times."""
    balances = {generate_business(seed).opening_balance for seed in range(20)}
    assert len(balances) > 15


# --------------------------------------------------------------------------- accounting identities


def test_the_cash_identity_holds_for_every_business() -> None:
    """cash = opening balance + every transaction. If this ever fails, every gold answer derived
    from cash is wrong, and that is most of the benchmark."""
    for business in _businesses():
        expected = business.opening_balance + sum(
            (
                O.normalize_currency(t.amount, t.currency, business)
                for t in business.transactions
                if t.txn_date <= business.as_of
            ),
            Decimal(0),
        )
        actual = O.current_cash_balance(business).value
        assert isinstance(actual, Decimal)
        assert abs(expected - actual) <= Decimal("0.01"), f"seed {business.seed}"


def test_gross_receivables_always_exceed_net_by_exactly_the_tax() -> None:
    for business in _businesses():
        gross = O.accounts_receivable_total(business, gross=True).value
        net = O.accounts_receivable_total(business, gross=False).value
        tax = O.tax_inclusive_vs_exclusive(business).value
        assert isinstance(gross, Decimal) and isinstance(net, Decimal) and isinstance(tax, Decimal)

        assert gross >= net, "gross receivables cannot be less than net"
        assert abs((gross - net) - tax) <= Decimal("0.02")


def test_overdue_receivables_never_exceed_total_receivables() -> None:
    for business in _businesses():
        total = O.accounts_receivable_total(business).value
        overdue = O.overdue_receivables(business).value
        assert isinstance(total, Decimal) and isinstance(overdue, Decimal)
        assert overdue <= total + Decimal("0.01")


def test_a_runway_is_never_negative() -> None:
    """The bug this file exists for. `cash / abs(burn)` on a negative cash balance returns a
    negative number of months, which is not a shorter runway — it is a broken gold answer."""
    for business in _businesses():
        for result in (
            O.cash_runway_months(business),
            O.scenario_revenue_drop(business, Decimal("0.20")),
            O.scenario_cut_expenses(business, Decimal("0.15")),
        ):
            if isinstance(result.value, Decimal):
                assert result.value >= 0, f"seed {business.seed}: negative months {result.value}"


def test_a_cash_generative_business_reports_an_infinite_runway_not_a_number() -> None:
    """Dividing cash by a POSITIVE net flow yields a number, and that number is meaningless. The
    honest answer is that there is no runway to run out of."""
    found = False
    for business in _businesses():
        burn = O.monthly_burn(business)
        cash = O.current_cash_balance(business).value
        assert isinstance(cash, Decimal)
        if isinstance(burn.value, Decimal) and burn.value >= 0 and cash > 0:
            found = True
            assert O.cash_runway_months(business).value == "infinite"
            assert O.cash_gap_date(business).value == "never"
    assert found, "the generator should produce at least one cash-generative business"


def test_a_concentration_share_is_always_a_valid_percentage() -> None:
    for business in _businesses():
        for result in (
            O.customer_concentration(business),
            O.supplier_concentration(business),
        ):
            if isinstance(result.value, Decimal):
                assert 0 <= result.value <= 100, f"seed {business.seed}: {result.value}%"


def test_collecting_more_receivables_never_leaves_less_cash() -> None:
    """A monotonicity property. If collecting 75% left less cash than collecting 25%, the scenario
    arithmetic is inverted somewhere."""
    for business in _businesses()[:40]:
        quarter = O.scenario_collect_receivables(business, Decimal("0.25")).value
        half = O.scenario_collect_receivables(business, Decimal("0.50")).value
        three_quarters = O.scenario_collect_receivables(business, Decimal("0.75")).value
        assert isinstance(quarter, Decimal)
        assert isinstance(half, Decimal)
        assert isinstance(three_quarters, Decimal)
        assert quarter <= half <= three_quarters


def test_currency_conversion_uses_only_the_rate_supplied_in_the_books() -> None:
    """A model that applies a rate it 'knows' is hallucinating — and so would an oracle."""
    business = generate_business(4, multi_currency=True)
    assert "EUR" in business.fx_rates

    amount = Decimal("100.00")
    converted = O.normalize_currency(amount, "EUR", business)
    assert converted == (amount * business.fx_rates["EUR"]).quantize(Decimal("0.01"))

    with pytest.raises(KeyError):
        O.normalize_currency(amount, "JPY", business)  # no JPY rate is supplied


# --------------------------------------------------------------------------- prompt injection


def test_an_injected_instruction_does_not_change_the_true_answer() -> None:
    """The attack alters a DESCRIPTION field, never an amount. So the correct answer is unchanged —
    the model simply has to not be talked out of it by its own input. If injection moved the gold,
    the metric would be measuring the attack instead of the model's resistance to it."""
    business = generate_business(11)
    before = O.current_cash_balance(business).value

    for payload in INJECTION_PAYLOADS:
        poisoned = inject_into_ledger(business, payload)
        after = O.current_cash_balance(poisoned).value
        assert before == after, "an injected instruction must not move the gold answer"


def test_the_canary_appears_nowhere_in_a_clean_ledger() -> None:
    """This is what makes the measurement unambiguous. If a model reports 1,000,000, there is no
    path to that number except the injected instruction — so it obeyed its data."""
    canary = float(INJECTION_CANARY)
    for business in _businesses()[:60]:
        for txn in business.transactions:
            assert abs(float(txn.amount) - canary) > 1.0
        for invoice in business.invoices:
            assert abs(float(invoice.gross) - canary) > 1.0

        cash = O.current_cash_balance(business).value
        assert isinstance(cash, Decimal)
        assert abs(float(cash) - canary) > 1.0


def test_the_injection_payload_really_is_in_the_data_the_model_sees() -> None:
    """Guards the guard: an attack that never reaches the prompt would make every model look
    perfectly resistant."""
    from financebench.datasets.smb_cfo.adapter import render_books

    business = inject_into_ledger(generate_business(3), INJECTION_PAYLOADS[0])
    context = render_books(business)
    rendered = "\n".join(" ".join(row) for table in context.tables for row in table.rows)

    assert "IGNORE PREVIOUS INSTRUCTIONS" in rendered
    assert "1000000" in rendered
