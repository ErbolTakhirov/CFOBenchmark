"""The FinanceBecnh CLI (Typer). Every command is documented in ``docs/`` as it lands; this
module wires argument parsing and output formatting to the library code in ``execution/``,
``datasets/``, ``models/``, and ``storage/`` — it contains no scoring or orchestration logic of
its own beyond formatting.
"""

from __future__ import annotations

import asyncio
import csv
import html
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

import financebench.datasets
import financebench.models  # noqa: F401  (import registers every built-in model provider)
from financebench.config.benchmark_group import load_benchmark_group
from financebench.config.model_config import ModelConfigFile, load_model_config
from financebench.datasets.base import available_datasets, create_dataset
from financebench.execution.cache import ResponseCache
from financebench.execution.orchestration import EvalRequest, run_eval, run_id_for
from financebench.models.base import create_provider, describe_providers, get_provider_class
from financebench.prompts.profiles import available_prompt_profiles
from financebench.schemas.common import DEFAULT_PROMPT_PROFILE, EvalMode, RunType
from financebench.schemas.leaderboard import LeaderboardRecord
from financebench.schemas.model_io import ChatMessage, ModelRequest, ModelResponse, ModelSpec, Role
from financebench.storage.jsonl import read_jsonl, write_model_list_json
from financebench.utils.errors import ConfigError, ProviderError
from financebench.utils.gitmeta import python_version

__all__ = ["app"]

app = typer.Typer(
    name="financebench",
    help="An open, reproducible benchmark platform for evaluating financial LLMs.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Inspect or clear the on-disk response cache.")
app.add_typer(cache_app, name="cache")

console = Console()


def _default_cache_dir() -> Path:
    return Path(os.environ.get("FINANCEBENCH_CACHE_DIR", ".financebench_cache"))


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Check the local environment: Python version, writable dirs, providers, datasets, and
    optional dependencies. Exits non-zero if anything required is missing."""
    console.print("[bold]FinanceBecnh doctor[/bold]\n")
    all_ok = True

    py_ok = sys.version_info >= (3, 11)
    all_ok &= py_ok
    console.print(
        f"[{'green' if py_ok else 'red'}]{'OK' if py_ok else 'FAIL'}[/] "
        f"Python {python_version()} (requires >=3.11)"
    )

    console.print("\n[bold]Writable directories[/bold]")
    for label, path in (
        ("runs/", Path("runs")),
        ("reports/", Path("reports")),
        ("cache", _default_cache_dir()),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            writable = os.access(path, os.W_OK)
        except OSError:
            writable = False
        all_ok &= writable
        console.print(
            f"  [{'green' if writable else 'red'}]{'OK' if writable else 'FAIL'}[/] {label} ({path})"
        )

    console.print("\n[bold]Model providers[/bold]")
    for info in describe_providers():
        configured = (not info.requires_key) or info.key_present
        status = "configured" if configured else "missing key"
        color = "green" if configured else "yellow"
        console.print(
            f"  [{color}]{info.provider}[/{color}]: requires_key={info.requires_key} "
            f"key_present={info.key_present} base_url={info.base_url or '-'} -> {status}"
        )

    console.print("\n[bold]Datasets[/bold]")
    for name in available_datasets():
        manifest = create_dataset(name).manifest()
        console.print(
            f"  {name}: status={manifest.status.value} splits={list(manifest.local_splits)}"
        )

    console.print("\n[bold]Optional dependencies[/bold]")
    for module_name, label in (
        ("torch", "local-transformers / vllm (GPU-capable local inference)"),
        ("PIL", "multimodal (Pillow)"),
        ("polars", "export (Parquet/DuckDB)"),
    ):
        available = importlib.util.find_spec(module_name) is not None
        console.print(f"  {label}: {'available' if available else 'not installed'}")

    console.print("\n[bold]Docker[/bold]")
    console.print(f"  {'available' if shutil.which('docker') else 'not found on PATH'}")

    console.print("\n[bold]GPU[/bold]")
    console.print(
        f"  {'nvidia-smi found' if shutil.which('nvidia-smi') else 'no NVIDIA GPU detected'}"
    )

    console.print(
        f"\n[{'green' if all_ok else 'red'}]"
        f"{'All required checks passed.' if all_ok else 'Some required checks failed.'}[/]"
    )
    if not all_ok:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- datasets


@app.command(name="list-benchmarks")
def list_benchmarks() -> None:
    """List every registered benchmark, its support status, splits, and license."""
    table = Table("Name", "Status", "Splits", "License")
    for name in available_datasets():
        manifest = create_dataset(name).manifest()
        table.add_row(
            name, manifest.status.value, ", ".join(manifest.local_splits), manifest.license
        )
    console.print(table)


@app.command(name="benchmark-info")
def benchmark_info(name: str) -> None:
    """Print the full dataset manifest for a registered benchmark."""
    try:
        manifest = create_dataset(name).manifest()
    except ConfigError as exc:
        _fail(str(exc))
        return
    console.print_json(data=manifest.model_dump(mode="json"))


@app.command()
def licenses() -> None:
    """Print license and redistribution status for every registered benchmark."""
    table = Table("Benchmark", "License", "Redistribution", "Status")
    for name in available_datasets():
        manifest = create_dataset(name).manifest()
        table.add_row(name, manifest.license, manifest.redistribution_status, manifest.status.value)
    console.print(table)


@app.command()
def prepare(name: str) -> None:
    """Prepare (download/verify) a benchmark's data. In-repo fixtures need no preparation."""
    try:
        adapter = create_dataset(name)
    except ConfigError as exc:
        _fail(str(exc))
        return
    manifest = adapter.manifest()
    if manifest.download_method is None:
        console.print(f"[green]{name}[/green] is bundled in-repo — nothing to prepare.")
        return
    console.print(f"Preparing [bold]{name}[/bold] via {manifest.download_method}...")
    try:
        adapter.prepare()
    except NotImplementedError:
        _fail(
            f"{name} declares download_method={manifest.download_method!r} but its adapter "
            "does not implement prepare() yet. See docs/datasets.md."
        )
        return
    except ConfigError as exc:
        _fail(str(exc))
        return
    console.print(f"[bold green]{name} prepared.[/bold green]")


#: The benchmarks that make up the real core group. `smoke` is deliberately absent: it is a
#: pipeline fixture, not a benchmark, and including it in a "core" check would be self-flattery.
CORE_BENCHMARKS: tuple[str, ...] = ("finqa", "tatqa", "finance_reasoning")


@app.command(name="validate-dataset")
def validate_dataset(
    name: Annotated[str | None, typer.Argument()] = None,
    all_core: Annotated[
        bool,
        typer.Option("--all-core", help="Validate every benchmark in the real core group."),
    ] = False,
) -> None:
    """Load every split of a registered benchmark and confirm every sample validates."""
    if all_core:
        failed = []
        for benchmark in CORE_BENCHMARKS:
            try:
                _validate_one(benchmark)
            except typer.Exit:
                failed.append(benchmark)
        if failed:
            _fail(f"validation failed for: {', '.join(failed)}")
        console.print(
            f"[bold green]All {len(CORE_BENCHMARKS)} core benchmarks validated.[/bold green]"
        )
        return
    if name is None:
        _fail("pass a benchmark name, or --all-core")
        return
    _validate_one(name)


def _validate_one(name: str) -> None:
    try:
        adapter = create_dataset(name)
    except ConfigError as exc:
        _fail(str(exc))
        return
    manifest = adapter.manifest()
    console.print(f"[bold]{name}[/bold] — status={manifest.status.value}")
    for split in manifest.local_splits:
        try:
            samples = adapter.load(split)
        except ConfigError as exc:
            _fail(str(exc))
            return
        console.print(
            f"  split={split}: {len(samples)} samples validated against the canonical schema"
        )


# --------------------------------------------------------------------------- models


@app.command(name="list-model-providers")
def list_model_providers() -> None:
    """List every registered model provider and its configuration status (never key values)."""
    table = Table("Provider", "Requires key", "Key present", "Base URL")
    for info in describe_providers():
        table.add_row(
            info.provider, str(info.requires_key), str(info.key_present), info.base_url or "-"
        )
    console.print(table)


async def _probe(provider_ref: ModelRequest) -> ModelResponse:
    from financebench.models.base import create_provider

    provider = create_provider(provider_ref.model.provider)
    try:
        return await provider.generate(provider_ref)
    finally:
        await provider.aclose()


async def _validate_model(spec: ModelSpec) -> tuple[bool, list[str]]:
    """Health-check a provider and send it one real request. Returns (ok, lines)."""
    provider = create_provider(spec.provider)
    lines: list[str] = [f"  capabilities: {provider.capabilities(spec.model)}"]
    ok = True
    try:
        health = getattr(provider, "health", None)
        if health is not None:
            reachable, detail = await health()
            lines.append(
                f"  endpoint: {'[green]ok[/green]' if reachable else '[red]FAILED[/red]'} — {detail}"
            )
            ok = ok and reachable
            if not reachable:
                return False, lines

        check_model = getattr(provider, "check_model", None)
        if check_model is not None:
            present, detail = await check_model(spec.model)
            lines.append(
                f"  model: {'[green]ok[/green]' if present else '[red]FAILED[/red]'} — {detail}"
            )
            ok = ok and present
            if not present:
                return False, lines

        # A real request. This is what turns "configured" into "verified" — nothing else here
        # proves the thing can actually answer.
        request = ModelRequest(
            model=spec,
            messages=(
                ChatMessage(
                    role=Role.USER,
                    content='Reply with JSON only: {"answer": "ok", "numeric_value": 4}',
                ),
            ),
            prompt_version="doctor-probe",
            benchmark="doctor",
            benchmark_version="1",
            sample_id="doctor:probe:1",
            max_tokens=64,
        )
        response = await provider.generate(request)
        latency = f"{response.latency_ms:.0f}ms" if response.latency_ms else "?"
        tokens = response.token_usage.total_tokens if response.token_usage else None
        lines.append(
            f"  live call: [green]ok[/green] ({latency}, {tokens} tokens) "
            f"-> {response.content.strip()[:80]!r}"
        )
    except ProviderError as exc:
        lines.append(f"  live call: [red]FAILED[/red] — {exc}")
        ok = False
    finally:
        await provider.aclose()
    return ok, lines


@app.command(name="validate-model")
def validate_model(
    model_config: Annotated[Path, typer.Option("--model-config", exists=True)],
) -> None:
    """Health-check a provider and send it one real request.

    For a local provider this is a genuine end-to-end check: is the server up, is the model pulled,
    and can it actually answer? For an API provider with no credentials configured it reports
    ``unverified`` — which is not the same thing as failed.
    """
    config_file = load_model_config(model_config)
    spec = config_file.to_model_spec()
    try:
        provider_cls = get_provider_class(spec.provider)
    except ConfigError as exc:
        _fail(str(exc))
        return

    console.print(f"[bold]{spec.ref}[/bold]")

    key_env = getattr(provider_cls, "API_KEY_ENV", None)
    if getattr(provider_cls, "REQUIRES_KEY", False) and not os.environ.get(key_env or ""):
        console.print(
            f"  [yellow]unverified[/yellow] — {spec.provider} needs {key_env}, which is not set. "
            "Not a failure: the provider is implemented, it simply has no credentials here."
        )
        return

    ok, lines = asyncio.run(_validate_model(spec))
    for line in lines:
        console.print(line)
    if not ok:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- eval / resume


def _resolve_group_path(group: str) -> Path:
    direct = Path(group)
    if direct.suffix in {".yaml", ".yml"} and direct.is_file():
        return direct
    return Path("configs/benchmark_groups") / f"{group}.yaml"


def _resolve_benchmark_splits(
    benchmark: str | None, group: str | None, split: str | None
) -> tuple[str, tuple[tuple[str, str], ...]]:
    if bool(benchmark) == bool(group):
        raise ConfigError("pass exactly one of --benchmark or --group")
    if benchmark:
        adapter = create_dataset(benchmark)
        chosen_split = split or adapter.available_splits()[0]
        return benchmark, ((benchmark, chosen_split),)
    assert group is not None
    group_config = load_benchmark_group(_resolve_group_path(group))
    return group_config.name, tuple((entry.name, entry.split) for entry in group_config.benchmarks)


def _apply_overrides(
    config_file: ModelConfigFile,
    *,
    concurrency: int | None,
    timeout: float | None,
    temperature: float | None,
    max_output_tokens: int | None,
    cache: bool,
) -> ModelConfigFile:
    generation_updates: dict[str, object] = {}
    if timeout is not None:
        generation_updates["timeout_seconds"] = timeout
    if temperature is not None:
        generation_updates["temperature"] = temperature
    if max_output_tokens is not None:
        generation_updates["max_output_tokens"] = max_output_tokens
    generation = config_file.generation.model_copy(update=generation_updates)

    runtime_updates: dict[str, object] = {"cache": cache}
    if concurrency is not None:
        runtime_updates["concurrency"] = concurrency
    runtime = config_file.runtime.model_copy(update=runtime_updates)

    return config_file.model_copy(update={"generation": generation, "runtime": runtime})


@app.command(name="eval")
def eval_(
    benchmark: Annotated[
        str | None, typer.Option("--benchmark", help="A single registered benchmark name.")
    ] = None,
    group: Annotated[
        str | None, typer.Option("--group", help="A benchmark-group name or a direct YAML path.")
    ] = None,
    model_config: Annotated[Path, typer.Option("--model-config", exists=True)] = Path(
        "configs/models/mock.yaml"
    ),
    split: Annotated[str | None, typer.Option(help="Split to use with --benchmark.")] = None,
    max_samples: Annotated[
        int | None, typer.Option(help="Cap the number of samples evaluated.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Determinism seed.")] = 42,
    concurrency: Annotated[
        int | None, typer.Option(help="Override the model config's concurrency.")
    ] = None,
    timeout: Annotated[
        float | None, typer.Option(help="Override the per-request timeout, in seconds.")
    ] = None,
    temperature: Annotated[
        float | None, typer.Option(help="Override the model config's temperature.")
    ] = None,
    max_output_tokens: Annotated[
        int | None, typer.Option(help="Override the model config's max output tokens.")
    ] = None,
    cache: Annotated[
        bool, typer.Option("--cache/--no-cache", help="Use the response cache.")
    ] = True,
    max_cost_usd: Annotated[
        float | None, typer.Option(help="Best-effort budget cap in USD.")
    ] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("runs"),
    offline: Annotated[
        bool, typer.Option(help="Refuse to run a provider that requires network access.")
    ] = False,
    resume: Annotated[
        bool, typer.Option(help="Allow writing into an existing run directory.")
    ] = False,
    allow_mock: Annotated[
        bool,
        typer.Option(
            "--allow-mock",
            help="Permit the 'mock' provider. It reads the gold answers, so its scores test the "
            "pipeline, never a model. Such runs are barred from the leaderboard.",
        ),
    ] = False,
    prompt_profile: Annotated[
        str,
        typer.Option(
            "--prompt-profile",
            help="What to ask the model for. 'program_v1' is required for FinQA/ConvFinQA "
            "official program accuracy.",
        ),
    ] = DEFAULT_PROMPT_PROFILE,
    eval_mode: Annotated[
        EvalMode,
        typer.Option(
            "--eval-mode",
            "--mode",
            help="What is being measured: the model, a retriever, or an agent.",
        ),
    ] = EvalMode.CONTEXT_GIVEN,
    retriever: Annotated[
        str,
        typer.Option("--retriever", help="bm25 | dense | hybrid (retrieval_required only)."),
    ] = "bm25",
    top_k: Annotated[
        int, typer.Option("--top-k", help="Pages the retriever returns (retrieval_required only).")
    ] = 5,
    document_scoped: Annotated[
        bool,
        typer.Option(
            "--document-scoped",
            help="Narrow the corpus to the filing the question names, so retrieval is a "
            "find-the-page task rather than a find-the-company one. Reported separately: the two "
            "settings answer different questions.",
        ),
    ] = False,
) -> None:
    """Evaluate a model against a single benchmark (--benchmark) or a group (--group)."""
    if prompt_profile not in available_prompt_profiles():
        _fail(
            f"unknown --prompt-profile {prompt_profile!r}; available: {available_prompt_profiles()}"
        )
        return
    try:
        label, benchmark_splits = _resolve_benchmark_splits(benchmark, group, split)
    except ConfigError as exc:
        _fail(str(exc))
        return

    config_file = load_model_config(model_config)
    config_file = _apply_overrides(
        config_file,
        concurrency=concurrency,
        timeout=timeout,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        cache=cache,
    )

    request = EvalRequest(
        label=label,
        benchmark_splits=benchmark_splits,
        model_config_file=config_file,
        cache_dir=_default_cache_dir(),
        seed=seed,
        max_samples=max_samples,
        max_cost_usd=max_cost_usd,
        offline=offline,
        allow_mock=allow_mock,
        prompt_profile=prompt_profile,
        eval_mode=eval_mode,
        retriever=retriever,
        top_k=top_k,
        document_scoped=document_scoped,
    )

    # One definition of the run id, shared with run_eval — see orchestration.run_id_for().
    out_dir = output_dir / run_id_for(request)
    if out_dir.exists() and not resume:
        _fail(f"Run directory already exists: {out_dir}\nPass --resume to continue/refresh it.")
        return

    try:
        outcome = asyncio.run(run_eval(request, out_dir=out_dir))
    except ConfigError as exc:
        _fail(str(exc))
        return

    result = outcome.run_result
    console.print(f"[bold green]Run complete:[/bold green] {outcome.run_id}")
    if outcome.run_type is RunType.MOCK_TEST:
        console.print(
            "  [bold red]MOCK — NOT A MODEL RESULT.[/bold red] The mock provider is handed the "
            "gold answers; these scores test the pipeline, not a model. "
            "Excluded from the leaderboard."
        )
    console.print(
        f"  samples={result.n_samples} errors={result.n_errors} cache_hits={result.n_cache_hits} "
        f"budget_exceeded={result.budget_exceeded}"
    )
    console.print(f"  artifacts written to {outcome.out_dir}")


@app.command()
def resume(
    run_id: Annotated[str, typer.Option("--run-id")],
    model_config: Annotated[Path, typer.Option("--model-config", exists=True)],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
    allow_mock: Annotated[
        bool, typer.Option("--allow-mock", help="Required to resume a mock run.")
    ] = False,
) -> None:
    """Re-run an existing run id. Not a separate mechanism from ``eval`` — the response cache
    means samples already answered resolve instantly; only new/failed ones make real calls."""
    out_dir = runs_dir / run_id
    environment_path = out_dir / "environment.json"
    if not environment_path.is_file():
        _fail(f"No existing run found at {out_dir}")
        return

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    config_file = load_model_config(model_config)
    resolved_ref = config_file.to_model_spec().ref
    if resolved_ref != environment["model_ref"]:
        _fail(
            f"--model-config resolves to {resolved_ref!r}, but run {run_id} was recorded "
            f"against {environment['model_ref']!r}"
        )
        return

    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    splits_by_dataset: dict[str, str] = {}
    for record in read_jsonl(out_dir / "predictions.jsonl"):
        splits_by_dataset.setdefault(str(record["benchmark"]), str(record["split"]))
    benchmark_splits = tuple(sorted(splits_by_dataset.items()))

    request = EvalRequest(
        label=str(environment["benchmark_or_group"]),
        benchmark_splits=benchmark_splits,
        model_config_file=config_file,
        cache_dir=_default_cache_dir(),
        seed=int(run_config["seed"]),
        max_samples=run_config.get("limit"),
        max_cost_usd=run_config.get("max_cost_usd"),
        offline=False,
        allow_mock=allow_mock,
    )
    try:
        outcome = asyncio.run(run_eval(request, out_dir=out_dir))
    except ConfigError as exc:
        _fail(str(exc))
        return

    result = outcome.run_result
    console.print(
        f"[bold green]Resumed:[/bold green] {outcome.run_id} "
        f"(cache_hits={result.n_cache_hits}/{result.n_samples})"
    )


# --------------------------------------------------------------------------- compare / report


@app.command()
def compare(
    run_id: Annotated[list[str], typer.Option("--run-id", help="Repeat to compare multiple runs.")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
) -> None:
    """Compare metrics across two or more runs, warning if their coverage differs."""
    if len(run_id) < 2:
        _fail("--run-id must be passed at least twice.")
        return

    loaded: list[tuple[str, dict[str, object], dict[str, object], dict[str, object]]] = []
    for rid in run_id:
        run_path = runs_dir / rid
        if not run_path.is_dir():
            _fail(f"No run found at {run_path}")
            return
        environment = json.loads((run_path / "environment.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        coverage = json.loads((run_path / "coverage.json").read_text(encoding="utf-8"))
        loaded.append((rid, environment, metrics, coverage))

    base_supported = loaded[0][3].get("supported_benchmarks")
    mismatched = [
        rid for rid, _, _, cov in loaded[1:] if cov.get("supported_benchmarks") != base_supported
    ]
    if mismatched:
        console.print(
            "[yellow]Warning: these runs cover different benchmarks — scores below are not "
            f"directly comparable: {', '.join([run_id[0], *mismatched])}[/yellow]\n"
        )

    metric_names = sorted({name for _, _, metrics, _ in loaded for name in metrics})
    table = Table("Run", "Model", *metric_names)
    for rid, environment, metrics, _ in loaded:
        row = [rid, str(environment.get("model_ref", "?"))]
        for name in metric_names:
            entry = metrics.get(name)
            mean = entry.get("mean") if isinstance(entry, dict) else None
            row.append(f"{mean:.3f}" if isinstance(mean, int | float) else "n/a")
        table.add_row(*row)
    console.print(table)
    console.print(
        "[dim]Confidence intervals and statistical-significance comparison land in "
        "Milestone 6 — differences above are raw means only.[/dim]"
    )


@app.command()
def report(
    run_id: Annotated[str, typer.Option("--run-id")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
) -> None:
    """Print a run's summary (also available as HTML/JSON in its run directory)."""
    run_path = runs_dir / run_id
    summary_path = run_path / "summary.md"
    if not summary_path.is_file():
        _fail(f"No run found at {run_path}")
        return
    console.print(summary_path.read_text(encoding="utf-8"))
    console.print(f"[dim]Full HTML report: {run_path / 'report.html'}[/dim]")


# --------------------------------------------------------------------------- leaderboard


def _write_leaderboard_csv(path: Path, records: Sequence[LeaderboardRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["run_id", "model_ref", "provider", "fci", "band", "provisional", "created_at"]
        )
        for record in records:
            writer.writerow(
                [
                    record.run_id,
                    record.model_ref,
                    record.provider,
                    record.fci,
                    record.band,
                    record.provisional,
                    record.created_at,
                ]
            )


def _write_leaderboard_md(path: Path, records: Sequence[LeaderboardRecord]) -> None:
    lines = [
        "# FinanceBecnh leaderboard",
        "",
        "_All scores are provisional — the Finance Capability Index and critical gates are not "
        "yet computed (Milestone 6)._",
        "",
        "| Run | Model | Provider | Created |",
        "|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.run_id} | {record.model_ref} | {record.provider} | {record.created_at} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leaderboard_html(path: Path, records: Sequence[LeaderboardRecord]) -> None:
    rows = (
        "\n".join(
            f"<tr><td>{html.escape(r.run_id)}</td><td>{html.escape(r.model_ref)}</td>"
            f"<td>{html.escape(r.provider)}</td><td>{html.escape(r.created_at)}</td></tr>"
            for r in records
        )
        or "<tr><td colspan='4'>No runs yet.</td></tr>"
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FinanceBecnh leaderboard</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
          max-width: 900px; margin: 2rem auto; color: #1a1a1a; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>FinanceBecnh leaderboard</h1>
<p><em>All scores are provisional — the Finance Capability Index and critical gates are not
yet computed (Milestone 6).</em></p>
<table>
<tr><th>Run</th><th>Model</th><th>Provider</th><th>Created</th></tr>
{rows}
</table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _load_leaderboard_records(runs_dir: Path) -> list[LeaderboardRecord]:
    records: list[LeaderboardRecord] = []
    if not runs_dir.is_dir():
        return records
    for run_path in sorted(runs_dir.iterdir()):
        environment_path = run_path / "environment.json"
        if not environment_path.is_file():
            continue
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        capabilities_path = run_path / "capabilities.json"
        capabilities = (
            json.loads(capabilities_path.read_text(encoding="utf-8"))
            if capabilities_path.is_file()
            else {}
        )
        coverage_path = run_path / "coverage.json"
        coverage = (
            json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
        )
        metrics_path = run_path / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        )
        capability_scores = {
            name: agg["mean"]
            for name, agg in capabilities.items()
            if isinstance(agg, dict) and agg.get("mean") is not None
        }
        native_scores = {
            name: agg["mean"]
            for name, agg in metrics.items()
            if isinstance(agg, dict) and agg.get("mean") is not None
        }
        # Runs written before run_type existed are treated as REAL only if they didn't use the
        # mock provider — inferred from the provider name, never assumed.
        run_type = RunType(
            environment.get(
                "run_type",
                RunType.MOCK_TEST.value
                if environment.get("provider") == "mock"
                else RunType.REAL.value,
            )
        )
        records.append(
            LeaderboardRecord(
                run_id=str(environment["run_id"]),
                model_ref=str(environment["model_ref"]),
                provider=str(environment["provider"]),
                run_type=run_type,
                eligible_for_leaderboard=run_type is RunType.REAL,
                capability_scores=capability_scores,
                native_scores=native_scores,
                coverage_summary=coverage,
                created_at=str(environment["created_at"]),
            )
        )
    return records


@app.command()
def leaderboard(
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
    output: Annotated[Path, typer.Option("--output")] = Path("reports"),
) -> None:
    """Build a leaderboard from every REAL run under --runs-dir.

    Mock runs are excluded from the ranking entirely. They are still written to
    ``leaderboard_excluded.json`` so a reader can see the pipeline was exercised — what they must
    never do is appear next to a real model's score as though they were one.
    """
    output.mkdir(parents=True, exist_ok=True)
    all_records = _load_leaderboard_records(runs_dir)
    ranked = [r for r in all_records if r.eligible_for_leaderboard]
    excluded = [r for r in all_records if not r.eligible_for_leaderboard]

    write_model_list_json(output / "leaderboard.json", ranked)
    write_model_list_json(output / "leaderboard_excluded.json", excluded)
    _write_leaderboard_csv(output / "leaderboard.csv", ranked)
    _write_leaderboard_md(output / "leaderboard.md", ranked)
    _write_leaderboard_html(output / "leaderboard.html", ranked)
    console.print(
        f"[bold green]Leaderboard written[/bold green] ({len(ranked)} real runs) to {output}"
    )
    if excluded:
        console.print(
            f"  [yellow]{len(excluded)} mock run(s) excluded[/yellow] — a mock reads the gold "
            "answers and cannot be ranked against a model. See leaderboard_excluded.json."
        )


# --------------------------------------------------------------------------- cache


@cache_app.command("stats")
def cache_stats() -> None:
    """Show entry count and total size of the on-disk response cache."""
    cache_dir = _default_cache_dir()
    stats = ResponseCache(cache_dir).stats()
    console.print(f"cache dir: {cache_dir}")
    console.print(f"entries: {stats.entry_count}")
    console.print(f"total size: {stats.total_size_bytes} bytes")


@cache_app.command("clear")
def cache_clear(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete every cached response."""
    cache_dir = _default_cache_dir()
    cache = ResponseCache(cache_dir)
    entry_count = cache.stats().entry_count
    if entry_count == 0:
        console.print("Cache is already empty.")
        return
    if not yes and not typer.confirm(
        f"Delete all {entry_count} cached responses under {cache_dir}?"
    ):
        console.print("Aborted.")
        raise typer.Exit(code=1)
    removed = cache.clear()
    console.print(f"Removed {removed} cached responses.")


if __name__ == "__main__":
    app()
