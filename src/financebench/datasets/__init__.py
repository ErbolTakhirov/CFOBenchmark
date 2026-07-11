"""Dataset adapters: the ``DatasetAdapter`` contract, registry, and the built-in ``smoke`` fixture.

Importing this package registers every built-in adapter (currently just ``smoke``; real
benchmark adapters land in Milestone 2+) — CLI code should ``import financebench.datasets`` (not
a specific adapter module) so the registry is always fully populated.
"""

from __future__ import annotations

from financebench.datasets import (
    finance_reasoning as _fr,  # noqa: F401  (registers "finance_reasoning")
)
from financebench.datasets import finqa as _finqa  # noqa: F401  (import registers "finqa")
from financebench.datasets import smoke as _smoke  # noqa: F401
from financebench.datasets import (
    tatqa as _tatqa,  # noqa: F401  (registers "tatqa")  (import registers "smoke")
)
from financebench.datasets.base import (
    DatasetAdapter,
    available_datasets,
    create_dataset,
    get_dataset_class,
    register_dataset,
)
from financebench.datasets.finqa import FinQAAdapter
from financebench.datasets.smoke import SmokeDatasetAdapter

__all__ = [
    "DatasetAdapter",
    "FinQAAdapter",
    "SmokeDatasetAdapter",
    "available_datasets",
    "create_dataset",
    "get_dataset_class",
    "register_dataset",
]
