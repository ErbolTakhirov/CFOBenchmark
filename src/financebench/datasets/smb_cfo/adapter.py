"""SMB-CFO: the deterministic, uncontaminated, bilingual small-business CFO benchmark.

Every gold answer here is produced by a Python oracle over generated books (`oracles.py`). No LLM
writes a gold answer, ever.

Splits:

- ``public`` — ~300 standard cases across 24 task families, EN.
- ``adversarial`` — ~150 traps and prompt injections.
- ``bilingual`` — ~100 EN/RU **paired** cases: the same business, the same oracle, the same gold
  value, asked in two languages. A gap between them is therefore a gap in the model, not in the
  phrasing — which is the only way an EN/RU number means anything.
- ``smoke`` — a tiny stratified slice for fast checks.

The **private** variant is generated from a secret seed supplied via ``SMB_CFO_SECRET_SEED``. The
seed is never committed; only its hash goes into the run metadata. That is what makes an
uncontaminated benchmark stay uncontaminated: the public cases will eventually leak into somebody's
training data, and when they do, the private ones will still be clean, and the gap between them
becomes the measurement.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from financebench.datasets.base import DatasetAdapter, register_dataset
from financebench.datasets.smb_cfo.adversarial import AdversarialCase, build_adversarial
from financebench.datasets.smb_cfo.business import Business, generate_business
from financebench.datasets.smb_cfo.oracles import OracleResult
from financebench.datasets.smb_cfo.tasks import TASK_FAMILIES, TaskSpec
from financebench.schemas.common import AnswerType, SplitOrigin, TranslationProvenance
from financebench.schemas.manifest import AdapterStatus, DatasetManifest
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    Evidence,
    GoldAnswer,
    SampleContext,
    SourceInfo,
    Table,
)
from financebench.utils.errors import DatasetLoadError

__all__ = ["SECRET_SEED_ENV", "SmbCfoAdapter", "render_books"]

SECRET_SEED_ENV = "SMB_CFO_SECRET_SEED"

_VERSION = "1"

#: The oracles compute exact Decimals. The model is allowed to round like a human would, so grading
#: uses a small relative tolerance. This exists for the MODEL's rounding, never for ours.
_RELATIVE_TOLERANCE = 0.01

_SPLIT_SIZES = {
    "public": 300,
    "adversarial": 150,
    "bilingual": 100,
    "smoke": 12,
}


def _rows(table: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(cell) for cell in row) for row in table)


def _customer_name(business: Business, customer_id: str) -> str:
    customer = business.customer(customer_id)
    return customer.name if customer else customer_id


def _supplier_name(business: Business, supplier_id: str) -> str:
    supplier = business.supplier(supplier_id)
    return supplier.name if supplier else supplier_id


def render_books(business: Business, *, conflicting_summary: bool = False) -> SampleContext:
    """Render a business's books as the model sees them: tables, like a real export.

    The summary line is deliberately *first*, because that is where a real export puts it — and it is
    what a lazy reader quotes instead of computing. In the `conflicting_totals` trap it is
    deliberately wrong, and the transactions below it are the truth.
    """
    from financebench.datasets.smb_cfo import oracles as O

    true_cash = O.current_cash_balance(business).value
    assert isinstance(true_cash, Decimal)
    stated = true_cash * Decimal("1.15") if conflicting_summary else true_cash

    summary = Table(
        table_id="summary",
        caption="Summary (as reported by the accounting system)",
        rows=_rows(
            [
                ["Field", "Value"],
                ["Business", business.name],
                ["Base currency", business.base_currency],
                ["Period", f"{business.period_start:%Y-%m-%d} to {business.period_end:%Y-%m-%d}"],
                ["As of", business.as_of.isoformat()],
                ["Opening balance", f"{business.opening_balance}"],
                ["Reported closing balance", f"{stated.quantize(Decimal('0.01'))}"],
                ["Tax rate", f"{business.tax_rate * 100}%"],
            ]
        ),
    )

    fx = Table(
        table_id="fx_rates",
        caption="Exchange rates supplied with this ledger (use ONLY these)",
        rows=_rows(
            [["Currency", f"Rate to {business.base_currency}"]]
            + [[c, str(r)] for c, r in sorted(business.fx_rates.items())]
        ),
    )

    ledger = Table(
        table_id="transactions",
        caption="Bank transactions (inflow positive, outflow negative)",
        rows=_rows(
            [["txn_id", "date", "description", "amount", "currency", "category", "counterparty"]]
            + [
                [
                    t.txn_id,
                    t.txn_date.isoformat(),
                    t.description,
                    str(t.amount),
                    t.currency,
                    t.category,
                    t.counterparty,
                ]
                for t in business.transactions
            ]
        ),
    )

    invoices = Table(
        table_id="invoices",
        caption="Invoices issued to customers (amount is NET of tax; tax stated separately)",
        rows=_rows(
            [
                [
                    "invoice_id",
                    "customer",
                    "issue_date",
                    "due_date",
                    "amount_net",
                    "tax",
                    "currency",
                    "paid",
                ]
            ]
            + [
                [
                    i.invoice_id,
                    _customer_name(business, i.customer_id),
                    i.issue_date.isoformat(),
                    i.due_date.isoformat(),
                    str(i.amount),
                    str(i.tax),
                    i.currency,
                    "yes" if i.paid else "no",
                ]
                for i in business.invoices
            ]
        ),
    )

    payables = Table(
        table_id="payables",
        caption="Unpaid supplier bills",
        rows=_rows(
            [["payment_id", "supplier", "due_date", "amount", "currency", "category"]]
            + [
                [
                    p.payment_id,
                    _supplier_name(business, p.supplier_id),
                    p.due_date.isoformat(),
                    str(p.amount),
                    p.currency,
                    p.category,
                ]
                for p in business.payables
            ]
        ),
    )

    budget = Table(
        table_id="budget",
        caption="Monthly budget by category",
        rows=_rows(
            [["category", "monthly_budget"]]
            + [[c, str(v)] for c, v in sorted(business.monthly_budget.items())]
        ),
    )

    return SampleContext(tables=(summary, fx, ledger, invoices, payables, budget))


def _gold(result: OracleResult) -> GoldAnswer:
    if result.unanswerable:
        return GoldAnswer(
            answer="INSUFFICIENT_INFORMATION",
            answer_type=AnswerType.REFUSAL,
            unit=result.unit,
        )
    value = result.value
    numeric = float(value) if isinstance(value, Decimal) else None
    return GoldAnswer(
        answer=str(value),
        answer_type=AnswerType.NUMERIC if numeric is not None else AnswerType.TEXT,
        numeric_value=numeric,
        unit=result.unit,
        currency=result.currency,
        evidence=tuple(Evidence(row=row_id) for row_id in result.evidence_ids[:40]),
    )


@register_dataset("smb_cfo")
class SmbCfoAdapter(DatasetAdapter):
    name = "smb_cfo"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._secret_seed = os.environ.get(SECRET_SEED_ENV)

    def prepare(self) -> None:
        """Nothing to download. The benchmark generates itself from a seed — which is exactly why
        it cannot be in anyone's training data."""
        return None

    def available_splits(self) -> tuple[str, ...]:
        return ("public", "adversarial", "bilingual", "smoke", "private")

    def load(self, split: str) -> Sequence[CanonicalSample]:
        if split not in self.available_splits():
            raise DatasetLoadError(
                f"smb_cfo has no split {split!r}; available: {list(self.available_splits())}"
            )
        if split == "private":
            if not self._secret_seed:
                raise DatasetLoadError(
                    f"the private split needs a secret seed in ${SECRET_SEED_ENV}. It is never "
                    "committed — only its hash is recorded in the run metadata. Without it there "
                    "is no private benchmark, and pretending otherwise would defeat the point."
                )
            base = int(hashlib.sha256(self._secret_seed.encode()).hexdigest()[:8], 16)
            return self._build(base, 200, "private", bilingual=True, adversarial_every=3)

        if split == "public":
            return self._build(1_000_000, _SPLIT_SIZES["public"], split)
        if split == "adversarial":
            return self._build(2_000_000, _SPLIT_SIZES["adversarial"], split, adversarial_every=1)
        if split == "bilingual":
            return self._build(3_000_000, _SPLIT_SIZES["bilingual"], split, bilingual=True)
        return self._build(4_000_000, _SPLIT_SIZES["smoke"], split)

    def _build(
        self,
        base_seed: int,
        target: int,
        split: str,
        *,
        bilingual: bool = False,
        adversarial_every: int = 0,
    ) -> list[CanonicalSample]:
        samples: list[CanonicalSample] = []
        index = 0
        while len(samples) < target:
            seed = base_seed + index
            index += 1
            multi_currency = seed % 4 == 0
            business = generate_business(seed, multi_currency=multi_currency)

            if adversarial_every and index % adversarial_every == 0:
                case = build_adversarial(business, index)
                samples.append(self._to_sample(case.business, case.spec, split, seed, case, "en"))
                continue

            family = TASK_FAMILIES[index % len(TASK_FAMILIES)]
            spec = family.build(business)
            if spec is None:
                continue

            samples.append(self._to_sample(business, spec, split, seed, None, "en"))
            if bilingual and len(samples) < target:
                # The SAME business, the SAME oracle, the SAME gold. Only the language differs — so
                # any gap is the model's, not the question's.
                samples.append(self._to_sample(business, spec, split, seed, None, "ru"))
        return samples[:target]

    def _to_sample(
        self,
        business: Business,
        spec: TaskSpec,
        split: str,
        seed: int,
        case: AdversarialCase | None,
        language: str,
    ) -> CanonicalSample:
        result = spec.result
        conflicting = spec.family == "trap_conflicting_totals"
        question = spec.question_ru if language == "ru" else spec.question_en

        should_refuse = bool(case and case.should_refuse) or result.unanswerable
        injection = bool(case and case.injection)

        return CanonicalSample(
            benchmark="smb_cfo",
            benchmark_version=f"generated@v{_VERSION}",
            split=split,
            split_origin=SplitOrigin.GENERATED_FROZEN,
            sample_id=f"smb_cfo:{split}:{seed}-{spec.family}-{language}",
            task_family=spec.family,
            capability_tags=(
                *spec.capability_tags,
                *(("bilingual",) if language == "ru" else ()),
            ),
            language=language,
            translation_provenance=(
                TranslationProvenance.OFFICIAL_LANGUAGE if language == "ru" else None
            ),
            question=question,
            context=render_books(business, conflicting_summary=conflicting),
            gold=_gold(result),
            evaluation=EvaluationSpec(
                relative_tolerance=_RELATIVE_TOLERANCE,
                should_refuse=should_refuse,
            ),
            source=SourceInfo(
                license="Apache-2.0 (generated by this project)",
                url="https://github.com/ErbolTakhirov/CFOBench",
                redistributable=True,
            ),
            metadata={
                "seed": str(seed),
                "business": business.name,
                "difficulty": spec.difficulty,
                "unit": result.unit,
                "currency": result.currency or "",
                "period": result.period or "",
                "unanswerable": "true" if result.unanswerable else "false",
                "prompt_injection": "true" if injection else "false",
                "oracle_note": result.note,
                "oracle_detail": ";".join(f"{k}={v}" for k, v in sorted(result.detail.items()))[
                    :400
                ],
            },
        )

    def manifest(self) -> DatasetManifest:
        seed_hash = (
            hashlib.sha256(self._secret_seed.encode()).hexdigest()[:16]
            if self._secret_seed
            else None
        )
        return DatasetManifest(
            name="smb_cfo",
            official_source="this project (generated)",
            repository_url="https://github.com/ErbolTakhirov/CFOBench",
            version_or_commit=f"generated@v{_VERSION}",
            download_method=None,
            official_splits=("public", "adversarial", "bilingual", "smoke", "private"),
            local_splits=("public", "adversarial", "bilingual", "smoke"),
            license="Apache-2.0",
            redistribution_status="redistributable",
            expected_files=(),
            status=AdapterStatus.FULLY_SUPPORTED,
            status_tested_at="2026-07-11T00:00:00Z",
            known_limitations=(
                "EVERY gold answer is computed by a Python oracle over generated books. No LLM ever "
                "produces a gold answer — a benchmark whose answers were written by a model "
                "inherits that model's mistakes and then grades other models against them.",
                "This is the ONLY uncontaminated benchmark here. FinQA/TAT-QA/FinanceBench all "
                "predate every current model's training cutoff. A model cannot have memorised a "
                "business generated from a seed. A large gap between a model's FinQA score and its "
                "SMB-CFO score is itself the contamination signal.",
                "It is also the ONLY source of Russian coverage in this platform. The RU questions "
                "are PAIRED with the EN ones — same business, same oracle, same gold value — so an "
                "EN/RU gap is a gap in the model, not in the phrasing.",
                "The businesses are synthetic. They are realistic in structure (ledger, invoices "
                "with due dates, FX, tax, budgets, duplicates, gaps) but they are not real "
                "companies, and a model that does well here has not been shown to do well on a "
                "real company's messy books.",
                f"The private split requires ${SECRET_SEED_ENV}. The seed is never committed; only "
                f"its hash is recorded. Seed hash currently configured: {seed_hash or 'none'}.",
            ),
        )
