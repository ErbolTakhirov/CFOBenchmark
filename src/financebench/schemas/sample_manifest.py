"""A frozen sample manifest: the exact questions a run is allowed to ask, listed by id.

Why this exists, and why ``--max-samples`` is not enough.

``--max-samples 40`` is a head-truncation of the adapter's file order. It happens to be
*deterministic* — the same 40 ids come back every time, on every model, in every mode, and that was
verified — so a paired comparison built on it is not wrong. But it is fragile in three ways that
matter for a release:

1. **It cannot express a set.** A stratified sample, a hand-audited subset, "the 89 FinanceBench
   questions that are gradable" — none of these is a prefix of a file, so none can be named.
2. **On a `--group` it truncates the CONCATENATION**, not each benchmark. ``--group release --max-samples
   150`` over six benchmarks does not give you 25 of each; it gives you the first 150 rows of the
   first benchmark or two, and reports the result under the group's name.
3. **It is silent when the data moves.** If an upstream dataset inserts a row, "the first 40" is a
   different 40, and every artifact still says ``limit: 40``. Nothing anywhere would notice.

A manifest names the questions. If one of them stops resolving, the run **fails** rather than
quietly evaluating a different set and publishing it under the same name — which is the only
behaviour that makes "we ran the same questions on both models" a claim rather than a hope.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from financebench.utils.errors import ConfigError, ManifestError

__all__ = [
    "ManifestBenchmark",
    "SampleManifest",
    "load_sample_manifest",
    "sample_id_set_hash",
]


def sample_id_set_hash(sample_ids: list[str] | tuple[str, ...]) -> str:
    """A stable hash of a SET of sample ids — order-independent, because order is not identity."""
    payload = "\n".join(sorted(set(sample_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ManifestBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    split: str
    #: The exact sample ids. Not a count — a list.
    sample_ids: tuple[str, ...]

    @property
    def id_hash(self) -> str:
        return sample_id_set_hash(self.sample_ids)


class SampleManifest(BaseModel):
    """The frozen input set of a release evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str | None = None
    created_at: str | None = None
    #: The commit the manifest was frozen at. Advisory — recorded so a reader can find the adapters
    #: that produced these ids.
    frozen_at_commit: str | None = None
    benchmarks: tuple[ManifestBenchmark, ...]

    @property
    def all_sample_ids(self) -> tuple[str, ...]:
        return tuple(sid for b in self.benchmarks for sid in b.sample_ids)

    @property
    def id_hash(self) -> str:
        """One hash over every id in the manifest. Two runs sharing this hash asked the same
        questions; two runs that do not, did not, whatever their `--max-samples` said."""
        return sample_id_set_hash(self.all_sample_ids)

    @property
    def benchmark_splits(self) -> tuple[tuple[str, str], ...]:
        return tuple((b.name, b.split) for b in self.benchmarks)


def load_sample_manifest(path: str | Path) -> SampleManifest:
    """Load and validate a frozen sample manifest (JSON or YAML)."""
    file_path = Path(path)
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read sample manifest at {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid manifest at {file_path}: {exc}") from exc
    try:
        manifest = SampleManifest.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid sample manifest at {file_path}:\n{exc}") from exc
    if not manifest.benchmarks:
        raise ManifestError(f"sample manifest {file_path} names no benchmarks")
    for benchmark in manifest.benchmarks:
        if not benchmark.sample_ids:
            raise ManifestError(
                f"sample manifest {file_path}: benchmark {benchmark.name!r} names no samples"
            )
        duplicates = len(benchmark.sample_ids) - len(set(benchmark.sample_ids))
        if duplicates:
            raise ManifestError(
                f"sample manifest {file_path}: benchmark {benchmark.name!r} lists "
                f"{duplicates} duplicate sample id(s) — a question asked twice is not two questions"
            )
    return manifest
