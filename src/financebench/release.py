"""Build a release: the frozen manifest, the checksums, and the gate report that decides whether
this release may be tagged at all.

The release manifest answers one question — *"can somebody else get these numbers?"* — and it answers
it by naming everything that could change them: the datasets and their hashes, the exact sample ids,
the model digests and quantization, the runtime versions, the prompt and parser and metric versions,
the scoring config, the seeds, the retrieval index, the hardware, and the commit.

A number that cannot be reproduced is an anecdote. The manifest is what makes it a measurement.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financebench import __version__
from financebench.evaluation.fingerprint import current_fingerprint
from financebench.schemas.sample_manifest import load_sample_manifest
from financebench.utils.gitmeta import git_commit

__all__ = [
    "GateOutcome",
    "build_release",
    "check_release_gates",
    "sha256_file",
    "validate_release_manifest",
]

#: The schema the release manifest must satisfy. It is a real JSON Schema validated by a real
#: validator — not a hand-rolled key check, which is a validator that passes everything and proves
#: nothing.
SCHEMA_PATH = Path("schemas/release_manifest.schema.json")


def validate_release_manifest(manifest: dict[str, Any]) -> list[str]:
    """Every way this manifest violates the schema. Empty list = valid.

    A release manifest that does not satisfy its own published schema is worse than no schema: it
    tells a reader the file has a contract, and then breaks it.
    """
    import jsonschema

    if not SCHEMA_PATH.is_file():
        return [f"schema not found at {SCHEMA_PATH}"]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(manifest), key=lambda e: list(e.absolute_path))
    ]


def _dirty_excluding_release() -> bool:
    """Uncommitted SOURCE changes. The release directory is excluded, because `build_release` is
    writing into it as this is called, and a flag that is permanently true is a flag nobody reads."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", ":!release/"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _ollama_model_info(model: str) -> dict[str, str]:
    """Digest and quantization, from the runtime that will actually serve the model.

    Not from a config file. A config file says what we *asked* for; `ollama show` says what is
    loaded. A release that records the former has recorded an intention.
    """
    info: dict[str, str] = {}
    try:
        shown = subprocess.run(
            ["ollama", "show", model], capture_output=True, text=True, timeout=20, check=False
        ).stdout
        for line in shown.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in {"quantization", "parameters", "architecture"}:
                info[parts[0]] = parts[1]
        listed = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
        for line in listed.splitlines():
            parts = line.split()
            if parts and parts[0] == model:
                info["digest"] = parts[1]
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _hardware() -> dict[str, Any]:
    gpu = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        gpu = out or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "gpu": gpu,
        # Stated plainly because it is the single biggest confound in every latency number here:
        # qwen2.5:7b is 4.7 GB of weights on a 4 GB card, so it spills to CPU. Its latencies are a
        # measurement of THIS machine, not a general claim about 7B inference.
        "note": (
            "The 7B model does not fit in this GPU's 4 GB and partly runs on CPU. Latency "
            "comparisons between the 3B and 7B measure this hardware, not the models in general."
        ),
    }


def build_release(
    version: str,
    *,
    runs_dir: Path,
    run_ids: list[str],
    manifests: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    """Assemble ``release_manifest.json`` from the runs that make up this release."""
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    models: dict[str, dict[str, str]] = {}
    for run_id in run_ids:
        run_path = runs_dir / run_id
        env_path = run_path / "environment.json"
        if not env_path.is_file():
            continue
        env = json.loads(env_path.read_text(encoding="utf-8"))
        config = json.loads((run_path / "run_config.json").read_text(encoding="utf-8"))
        coverage = json.loads((run_path / "coverage.json").read_text(encoding="utf-8"))
        model_ref = str(env["model_ref"])
        model_name = model_ref.split("/", 1)[-1]
        if model_ref not in models:
            models[model_ref] = _ollama_model_info(model_name)

        sample_ids = [
            json.loads(line)["sample_id"]
            for line in (run_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runs.append(
            {
                "run_id": run_id,
                "model_ref": model_ref,
                "provider": env["provider"],
                "run_type": env["run_type"],
                "benchmark_or_group": env["benchmark_or_group"],
                "eval_mode": config.get("eval_mode"),
                "prompt_profile": config.get("prompt_profile"),
                "seed": config.get("seed"),
                "temperature": config.get("temperature"),
                "retriever": config.get("retriever"),
                "top_k": config.get("top_k"),
                "document_scoped": config.get("document_scoped"),
                "evaluator_fingerprint": env.get("evaluator_fingerprint", {}).get("digest"),
                "n_samples": len(sample_ids),
                "sample_id_set_hash": hashlib.sha256(
                    "\n".join(sorted(set(sample_ids))).encode("utf-8")
                ).hexdigest()[:16],
                "coverage": coverage,
            }
        )

    fingerprint = current_fingerprint()
    manifest = {
        "release": version,
        "financebench_version": __version__,
        "repository_commit": git_commit(),
        # Dirty *excluding the release directory*, which this very function is in the middle of
        # writing. Including it makes the flag permanently true and the gate permanently red — and a
        # gate that can never go green teaches people to ignore gates. What the flag is FOR is
        # "were there uncommitted SOURCE changes when this was built", because those are what make a
        # release irreproducible from the commit it names.
        "repository_dirty": _dirty_excluding_release(),
        "evaluator_fingerprint": fingerprint.to_json(),
        "sample_manifests": [
            {
                "path": str(p),
                "name": load_sample_manifest(p).name,
                "id_hash": load_sample_manifest(p).id_hash,
                "n_samples": len(load_sample_manifest(p).all_sample_ids),
                "sha256": sha256_file(p),
            }
            for p in manifests
            if p.is_file()
        ],
        "models": models,
        "runs": runs,
        "hardware": _hardware(),
    }
    (out_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class GateOutcome:
    name: str
    passed: bool | None  # None = NOT APPLICABLE. It is not a pass.
    detail: str

    @property
    def label(self) -> str:
        if self.passed is None:
            return "NOT APPLICABLE"
        return "PASS" if self.passed else "FAIL"


def check_release_gates(out_dir: Path, *, runs_dir: Path) -> list[GateOutcome]:
    """Every mandatory release gate. A FAIL here means the release is NOT tagged."""
    gates: list[GateOutcome] = []

    def run(cmd: list[str]) -> tuple[int, str]:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, (proc.stdout + proc.stderr)

    code, out = run([".venv/bin/ruff", "format", "--check", "."])
    gates.append(GateOutcome("ruff format", code == 0, out.strip().splitlines()[-1] if out else ""))
    code, out = run([".venv/bin/ruff", "check", "."])
    gates.append(GateOutcome("ruff check", code == 0, out.strip().splitlines()[-1] if out else ""))
    code, out = run([".venv/bin/mypy", "src/financebench"])
    gates.append(GateOutcome("mypy", code == 0, out.strip().splitlines()[-1] if out else ""))

    for label, path in (
        ("primary tests", "tests"),
        ("security tests", "tests/security"),
        ("parity tests", "tests/parity"),
    ):
        code, out = run([".venv/bin/pytest", path, "-q"])
        last = out.strip().splitlines()[-1] if out else ""
        passed = code == 0
        if label == "parity tests" and "skipped" in last:
            # A skipped parity test proves NOTHING. It is the exact failure mode that let the parity
            # suite quietly stop testing anything for a whole milestone: /tmp was cleared, the
            # reference evaluators vanished, and 17 tests went green-with-skips.
            passed = False
            last = f"{last}  <-- SKIPS ARE NOT PASSES"
        gates.append(GateOutcome(label, passed, last))

    # The release directory is EXCLUDED, because this very command writes into it — the manifest and
    # the checksums are generated a few lines above. A gate that fails because of the artifact the
    # gate-runner just produced is not measuring anything.
    code, out = run(["git", "status", "--porcelain", "--", ":!release/"])
    gates.append(GateOutcome("clean working tree", not out.strip(), out.strip()[:120] or "clean"))

    remote_code, remote = run(["git", "rev-parse", "origin/main"])
    _, local_head = run(["git", "rev-parse", "HEAD"])
    gates.append(
        GateOutcome(
            "HEAD == origin/main",
            local_head.strip() == remote.strip() if remote_code == 0 else None,
            f"local {local_head.strip()[:8]} vs remote {remote.strip()[:8]}",
        )
    )

    manifest_path = out_dir / "release_manifest.json"
    gates.append(
        GateOutcome(
            "release manifest exists",
            manifest_path.is_file(),
            str(manifest_path),
        )
    )
    if manifest_path.is_file():
        errors = validate_release_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        gates.append(
            GateOutcome(
                "release manifest validates against its schema",
                not errors,
                f"{len(errors)} violation(s): {errors[:2]}" if errors else "valid",
            )
        )
    else:
        gates.append(
            GateOutcome("release manifest validates against its schema", None, "no manifest")
        )

    # The manifest exists to answer "can somebody else get these numbers?". A manifest that records
    # a DIRTY working tree cannot be reproduced from its own commit, and says so.
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        dirty = bool(payload.get("repository_dirty"))
        gates.append(
            GateOutcome(
                "release built from a clean commit",
                not dirty,
                "dirty tree — this release cannot be reproduced from its commit"
                if dirty
                else str(payload.get("repository_commit", ""))[:12],
            )
        )
    gates.append(
        GateOutcome(
            "checksums exist",
            (out_dir / "checksums.txt").is_file(),
            str(out_dir / "checksums.txt"),
        )
    )

    # No mock run may appear in the public leaderboard.
    leaderboard = Path("reports/leaderboard.json")
    mock_rows = 0
    if leaderboard.is_file():
        rows = json.loads(leaderboard.read_text(encoding="utf-8"))
        mock_rows = sum(1 for r in rows if r.get("run_type") != "real")
    gates.append(
        GateOutcome("no mock rows on leaderboard", mock_rows == 0, f"{mock_rows} mock row(s)")
    )

    code, out = run([".venv/bin/python", "scripts/secret_scan_repo.py"])
    gates.append(GateOutcome("no secrets", code == 0, out.strip().splitlines()[-1] if out else ""))

    # ---------------------------------------------------------------- the gates that actually matter
    #
    # Everything above checks that the CODE is healthy. None of it checks that the EVALUATION was
    # finished, and a release that tags on a green test suite while half its runs are still executing
    # would be exactly the dishonesty this project exists to prevent. A benchmark whose gates only
    # test the benchmark's own source code is testing the wrong thing.

    # The runs THIS RELEASE PUBLISHES — read from the manifest that was just written, not from
    # whatever happens to be lying in runs/.
    #
    # This distinction is the gate's whole meaning. An old exploratory run, scored by an evaluator
    # from three commits ago, is not a release blocker — it is an old run. Gating on "every directory
    # under runs/" would either block the release forever or, far worse, push somebody to delete
    # inconvenient history to make a gate go green. The question is whether every run the release
    # PUTS ITS NAME TO was scored by the same evaluator, because those are the numbers that will sit
    # next to each other and be compared.
    published = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
    ids = [str(r["run_id"]) for r in published]

    fingerprints = {str(r.get("evaluator_fingerprint")) for r in published}
    fingerprints.discard("None")
    gates.append(
        GateOutcome(
            "one evaluator fingerprint across published runs",
            len(fingerprints) == 1,
            f"{len(published)} published run(s), {len(fingerprints)} distinct fingerprint(s): "
            f"{sorted(fingerprints)}"[:110],
        )
    )

    # A mock run reads the gold answers. It measures the pipeline, never a model, and it may not be
    # published under any circumstances.
    mocks = [r["run_id"] for r in published if r.get("run_type") != "real"]
    gates.append(
        GateOutcome(
            "no mock run is published",
            not mocks,
            f"{len(mocks)} mock run(s) in the release: {mocks[:2]}" if mocks else "none",
        )
    )

    # The paired tool experiment: all FOUR variants, or the comparison does not exist.
    wanted = {
        "3B direct": ("tool_paired_v1", "context_given", "3b"),
        "3B tools": ("tool_paired_v1", "tool_assisted", "3b"),
        "7B direct": ("tool_paired_v1", "context_given", "7b"),
        "7B tools": ("tool_paired_v1", "tool_assisted", "7b"),
    }
    missing = [
        label
        for label, (group, mode, model) in wanted.items()
        if not any(group in r and mode in r and model in r for r in ids)
    ]
    gates.append(
        GateOutcome(
            "paired direct-vs-tools run complete",
            not missing,
            f"missing: {missing}" if missing else "all 4 variants present",
        )
    )

    # The release group is the ONLY run that can produce an FCI (it covers SMB-CFO + grounding +
    # refusal together). Without it the report has no headline.
    release_runs = [r for r in ids if "release_v0_1" in r]
    gates.append(
        GateOutcome(
            "release-group run complete (both models)",
            len(release_runs) >= 2,
            f"{len(release_runs)} of 2 present",
        )
    )

    # The retrieval ablation: retrieval metrics for every arm, AND at least one generated arm beyond
    # the two we already had — otherwise "does better retrieval improve answers?" is unanswered.
    ablation = Path("reports/retrieval_ablation.json")
    cells = 0
    if ablation.is_file():
        cells = len(json.loads(ablation.read_text(encoding="utf-8")).get("cells", []))
    generated_arms = [r for r in ids if "retrieval_required" in r]
    gates.append(
        GateOutcome(
            "retrieval ablation complete",
            cells >= 18 and len(generated_arms) >= 2,
            f"{cells} cells, {len(generated_arms)} generated arm(s)",
        )
    )

    for label, artifact in (
        ("report generated", Path("release") / out_dir.name / "report.md"),
        ("manual validity review", Path("release") / out_dir.name / "manual_validity_review.md"),
        ("limitations documented", Path("docs/known_limitations.md")),
        ("reproduction guide", Path("release") / out_dir.name / "reproduction.md"),
    ):
        gates.append(GateOutcome(label, artifact.is_file(), str(artifact)))

    return gates
