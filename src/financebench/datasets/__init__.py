"""Dataset adapters: the ``DatasetAdapter`` contract, the registry, and every built-in adapter.

Importing this package registers all of them, so CLI code should ``import financebench.datasets``
rather than a specific adapter module — the registry is then always fully populated.

``smoke`` is a pipeline fixture, **not a benchmark**. It exists to prove the CLI/engine/cache/
metrics/artifacts chain works offline, and its scores are never evidence of financial capability.
"""

from __future__ import annotations

from financebench.datasets import convfinqa as _conv  # noqa: F401  (registers "convfinqa")
from financebench.datasets import finance_reasoning as _fr  # noqa: F401  (registers it)
from financebench.datasets import financebench as _fb  # noqa: F401  (registers "financebench")
from financebench.datasets import finqa as _finqa  # noqa: F401  (registers "finqa")
from financebench.datasets import smb_cfo as _smb  # noqa: F401  (registers "smb_cfo")
from financebench.datasets import smoke as _smoke  # noqa: F401  (registers "smoke")
from financebench.datasets import tatqa as _tatqa  # noqa: F401  (registers "tatqa")
from financebench.datasets.base import (
    DatasetAdapter,
    available_datasets,
    create_dataset,
    get_dataset_class,
    register_dataset,
)
from financebench.datasets.convfinqa import ConvFinQAAdapter
from financebench.datasets.finance_reasoning import FinanceReasoningAdapter
from financebench.datasets.financebench import FinanceBenchAdapter
from financebench.datasets.finqa import FinQAAdapter
from financebench.datasets.smb_cfo import SmbCfoAdapter
from financebench.datasets.smoke import SmokeDatasetAdapter
from financebench.datasets.tatqa import TatQAAdapter

__all__ = [
    "ConvFinQAAdapter",
    "DatasetAdapter",
    "FinQAAdapter",
    "FinanceBenchAdapter",
    "FinanceReasoningAdapter",
    "SmbCfoAdapter",
    "SmokeDatasetAdapter",
    "TatQAAdapter",
    "available_datasets",
    "create_dataset",
    "get_dataset_class",
    "register_dataset",
]
