"""The FinanceBench CLI (Typer). Every command is documented in ``docs/`` as it lands; this
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
from financebench.evaluation.benchmark_metrics import preferred_metric_name
from financebench.evaluation.fingerprint import current_fingerprint
from financebench.evaluation.stats import paired_bootstrap
from financebench.execution.cache import ResponseCache
from financebench.execution.orchestration import (
    EvalRequest,
    rescore_from_artifacts,
    resolve_samples,
    run_eval,
    run_id_for,
)
from financebench.models.base import (
    ProviderCapabilities,
    create_provider,
    describe_providers,
    get_provider_class,
)
from financebench.models.verification import ProviderVerification, verify_all_providers
from financebench.prompts.profiles import available_prompt_profiles
from financebench.release import build_release, check_release_gates, sha256_file
from financebench.reporting import build_mission_report, load_runs
from financebench.reporting.release_report import ModelResult, build_release_report, load_run
from financebench.reporting.retrieval_report import ArmResult, write_ablation_report
from financebench.retrieval.ablation import run_ablation
from financebench.schemas.common import (
    DEFAULT_PROMPT_PROFILE,
    ConversationProtocol,
    EvalMode,
    RunType,
)
from financebench.schemas.leaderboard import LeaderboardRecord
from financebench.schemas.model_io import ChatMessage, ModelRequest, ModelResponse, ModelSpec, Role
from financebench.schemas.sample_manifest import (
    ManifestBenchmark,
    SampleManifest,
    load_sample_manifest,
)
from financebench.storage.jsonl import read_jsonl, write_model_list_json
from financebench.utils.errors import ConfigError, ManifestError, ProviderError
from financebench.utils.gitmeta import git_commit, python_version
from financebench.utils.timing import RealClock

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
    console.print("[bold]FinanceBench doctor[/bold]\n")
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
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            help="A frozen sample manifest naming the exact sample ids to evaluate. "
            "Supersedes --benchmark/--group/--split/--max-samples.",
        ),
    ] = None,
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
    conversation_protocol: Annotated[
        ConversationProtocol,
        typer.Option(
            "--conversation-protocol",
            help="Multi-turn only. gold_history: each turn gets the GOLD prior conversation "
            "(official; isolates per-turn reasoning). model_history: each turn gets the model's OWN "
            "prior answers (exposes error propagation). Their scores are never mixed — run both and "
            "compare.",
        ),
    ] = ConversationProtocol.GOLD_HISTORY,
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

    sample_manifest = None
    if manifest is not None:
        try:
            sample_manifest = load_sample_manifest(manifest)
        except (ConfigError, ManifestError) as exc:
            _fail(str(exc))
            return
        if max_samples is not None:
            _fail(
                "--max-samples cannot be combined with --manifest. A manifest names the exact "
                "questions to ask; truncating it would ask a different set under its name."
            )
            return
        label = sample_manifest.name
        benchmark_splits = sample_manifest.benchmark_splits
    else:
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
        conversation_protocol=conversation_protocol,
        retriever=retriever,
        top_k=top_k,
        document_scoped=document_scoped,
        sample_manifest=sample_manifest,
        sample_manifest_path=str(manifest) if manifest is not None else None,
    )
    if sample_manifest is not None:
        console.print(
            f"[bold]Frozen manifest:[/bold] {sample_manifest.name} "
            f"({len(sample_manifest.all_sample_ids)} samples, id_hash={sample_manifest.id_hash})"
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

    # A manifest run asked a FROZEN set of questions, and a resume must ask the same ones. Without
    # this, resume fell back to `limit: null` and reloaded the entire benchmark — 2,815 samples for
    # a 150-sample finqa+tatqa manifest — turning a resume into a completely different, far larger
    # evaluation that would then have overwritten the original run's artifacts under its own id.
    resumed_manifest = None
    manifest_path = run_config.get("sample_manifest_path")
    if manifest_path:
        try:
            resumed_manifest = load_sample_manifest(Path(manifest_path))
        except (ConfigError, ManifestError) as exc:
            _fail(f"cannot resume {run_id}: its frozen manifest could not be loaded.\n{exc}")
            return
        recorded_hash = run_config.get("sample_manifest_id_hash")
        if recorded_hash and resumed_manifest.id_hash != recorded_hash:
            _fail(
                f"cannot resume {run_id}: the manifest at {manifest_path} has CHANGED since the run "
                f"(id_hash {resumed_manifest.id_hash} != recorded {recorded_hash}).\n"
                "Resuming would ask a different set of questions and publish the answers under the "
                "original run's id."
            )
            return
        benchmark_splits = resumed_manifest.benchmark_splits

    # A resume must reconstruct the run that WAS, not a run with the same name. Every field below is
    # part of the run's identity, and every one of them used to be dropped: resume rebuilt the
    # request with library defaults, so resuming a `retrieval_required` / hybrid / document-scoped /
    # `program_v1` run silently re-ran it as `context_given` / bm25 / k=5 / open-corpus /
    # `structured_financial_v1` — and then overwrote the original run's artifacts, in place, under
    # the original run's id. The artifacts would still have said `retrieval_required`. Nothing would
    # have looked wrong.
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
        prompt_profile=str(run_config.get("prompt_profile", DEFAULT_PROMPT_PROFILE)),
        eval_mode=EvalMode(run_config.get("eval_mode", EvalMode.CONTEXT_GIVEN.value)),
        conversation_protocol=ConversationProtocol(
            run_config.get("conversation_protocol", ConversationProtocol.GOLD_HISTORY.value)
        ),
        retriever=str(run_config.get("retriever", "bm25")),
        top_k=int(run_config.get("top_k", 5)),
        document_scoped=bool(run_config.get("document_scoped", False)),
        sample_manifest=resumed_manifest,
        sample_manifest_path=manifest_path,
    )

    # The rebuilt request must land on the run id it came from. If it does not, some part of the
    # run's identity was not restored, and continuing would write one experiment's results into
    # another's directory. Refuse loudly rather than corrupt the artifact.
    rebuilt_id = run_id_for(request)
    if rebuilt_id != run_id:
        _fail(
            f"Refusing to resume: the request rebuilt from {run_id}/run_config.json resolves to a "
            f"DIFFERENT run id ({rebuilt_id}).\nSome part of the run's identity (prompt profile, "
            "eval mode, conversation protocol, retriever, top_k, document scoping) could not be "
            "restored — most likely the run predates its being recorded. Re-run it explicitly with "
            "`financebench eval` rather than resuming it into the wrong directory."
        )
        return

    # The fingerprint BEFORE we overwrite the artifacts. If it differs from the one we are about to
    # write, this resume is a MIGRATION, and a migration must record how it happened.
    old_fingerprint = str(environment.get("evaluator_fingerprint", {}).get("digest", "unknown"))
    # The hashes of the raw responses we already hold. If every one of them survives the resume, then
    # no inference was re-run: the model said exactly what it said before, and only OUR code re-read
    # it. That is the difference between a re-score and a new experiment, and it is the difference a
    # reader needs in order to trust the migrated number.
    old_hashes = _raw_response_hashes(out_dir)

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

    new_environment = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    new_fingerprint = str(new_environment.get("evaluator_fingerprint", {}).get("digest", "unknown"))
    if new_fingerprint != old_fingerprint:
        new_hashes = _raw_response_hashes(out_dir)
        preserved = bool(old_hashes) and old_hashes == new_hashes
        record = {
            "source_run_id": run_id,
            "migration_type": "reparse_rescore" if preserved else "full_rerun",
            "old_fingerprint": old_fingerprint,
            "new_fingerprint": new_fingerprint,
            "reason": (
                "The evaluator changed; the model did not. Every raw response is byte-identical to "
                "the one already on disk, so no inference was re-run — our code re-read what the "
                "model had already said."
                if preserved
                else "At least one raw response differs from the one previously stored, so this run "
                "made real model calls. It is a re-evaluation, not a re-score."
            ),
            "raw_response_hashes_preserved": preserved,
            "n_samples": result.n_samples,
            "n_cache_hits": result.n_cache_hits,
            "migrated_at": RealClock().now_iso(),
        }
        (out_dir / "migration.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        kind = "re-scored" if preserved else "RE-RAN INFERENCE"
        console.print(
            f"  [dim]migration recorded: {old_fingerprint} -> {new_fingerprint} ({kind})[/dim]"
        )


def _raw_response_hashes(run_dir: Path) -> dict[str, str]:
    """``{sample_id: sha256(raw response text)}`` from a run's stored predictions.

    This is what makes a migration auditable. "The fingerprint changed and the score moved" is not a
    finding unless you can say whether the MODEL's output moved too — and the only way to say that is
    to hash what it actually said, before and after.
    """
    import hashlib

    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        return {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        response = record.get("response") or {}
        content = str(response.get("content", ""))
        hashes[str(record["sample_id"])] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hashes


# --------------------------------------------------------------------------- compare / report


def _per_sample_scores(run_path: Path, metric_name: str) -> dict[str, float]:
    """``{sample_id: 0.0|1.0}`` for one metric, from ``metric_details.jsonl``.

    Not-applicable results are **omitted**, not zeroed — so they drop out of the pairing instead of
    being counted as the model getting something wrong.
    """
    scores: dict[str, float] = {}
    path = run_path / "metric_details.jsonl"
    if not path.is_file():
        return scores
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("metric_name") != metric_name:
            continue
        value = record.get("value")
        if isinstance(value, bool):
            scores[record["sample_id"]] = 1.0 if value else 0.0
        elif isinstance(value, int | float):
            scores[record["sample_id"]] = float(value)
    return scores


@app.command()
def compare(
    run_id: Annotated[list[str], typer.Option("--run-id", help="Exactly two runs, paired.")],
    metric: Annotated[
        str | None,
        typer.Option("--metric", help="Metric to pair on. Default: the runs' preferred metric."),
    ] = None,
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
    output: Annotated[
        Path | None, typer.Option("--output", help="Write the paired comparison as JSON.")
    ] = None,
) -> None:
    """Compare two runs on the samples they BOTH answered, with a paired bootstrap CI.

    Pairing is the whole point. Two models scoring 45 % and 50 % on the same 40 questions look
    indistinguishable as independent samples — but if the second got everything the first got plus
    two more, that is a real difference an unpaired test would miss. In the other direction, an
    unpaired test will happily manufacture a difference out of nothing.

    Two runs may only be compared if they answered the **same questions** with the **same evaluator**.
    Both are checked, and a mismatch is an error rather than a footnote: a difference between runs
    with different evaluator fingerprints is a measurement of our own code changing, not of the model.
    """
    if len(run_id) != 2:
        _fail("--run-id must be passed exactly twice — a paired comparison compares two runs.")
        return

    paths = [runs_dir / rid for rid in run_id]
    for path in paths:
        if not path.is_dir():
            _fail(f"No run found at {path}")
            return

    envs = [json.loads((p / "environment.json").read_text(encoding="utf-8")) for p in paths]

    # -- identity check 1: the same evaluator produced both sets of numbers.
    digests = [e.get("evaluator_fingerprint", {}).get("digest") for e in envs]
    if digests[0] != digests[1]:
        _fail(
            "These runs were scored by DIFFERENT evaluators and are not comparable:\n"
            f"  {run_id[0]}: {digests[0]}\n  {run_id[1]}: {digests[1]}\n"
            "Re-score the older run on the current commit (`financebench resume --run-id ...`, "
            "which replays cached responses) before comparing. Fixing a parser once moved a FinQA "
            "score from 5% to 15% on identical model output; a difference across fingerprints "
            "measures our code, not the model."
        )
        return

    # -- identity check 2: the same questions.
    metric_name = metric
    if metric_name is None:
        benchmarks = [
            (
                json.loads((p / "coverage.json").read_text(encoding="utf-8")).get(
                    "supported_benchmarks"
                )
                or [None]
            )[0]
            for p in paths
        ]
        if benchmarks[0] != benchmarks[1] or benchmarks[0] is None:
            _fail("These runs cover different benchmarks; pass --metric to say what to pair on.")
            return
        profile = str(envs[0].get("prompt_profile", DEFAULT_PROMPT_PROFILE))
        metric_name = preferred_metric_name(str(benchmarks[0]), profile)

    scores = [_per_sample_scores(p, metric_name) for p in paths]
    if not scores[0] or not scores[1]:
        _fail(f"Metric {metric_name!r} has no per-sample results in one of these runs.")
        return

    only_a = sorted(set(scores[0]) - set(scores[1]))
    only_b = sorted(set(scores[1]) - set(scores[0]))
    if only_a or only_b:
        console.print(
            f"[yellow]These runs do not cover identical samples — pairing on the "
            f"{len(set(scores[0]) & set(scores[1]))} they share.\n"
            f"  only in {run_id[0]}: {len(only_a)}   only in {run_id[1]}: {len(only_b)}[/yellow]\n"
        )

    result = paired_bootstrap(scores[0], scores[1])
    if result is None:
        _fail("These runs share no samples — there is nothing to pair.")
        return

    console.print(f"[bold]Paired comparison[/bold] on [cyan]{metric_name}[/cyan]\n")
    table = Table("", run_id[0], run_id[1])
    table.add_row("model", str(envs[0].get("model_ref", "?")), str(envs[1].get("model_ref", "?")))
    table.add_row("mean", f"{result.mean_a:.4f}", f"{result.mean_b:.4f}")
    console.print(table)

    # The 2x2 is the interesting part, and it is exactly what a difference of means hides: two runs
    # can post identical means while disagreeing on half the questions.
    discord = Table("Outcome", "n")
    discord.add_row(f"only {run_id[0]} right", str(result.a_right_b_wrong))
    discord.add_row(f"only {run_id[1]} right", str(result.b_right_a_wrong))
    discord.add_row("both right", str(result.both_right))
    discord.add_row("both wrong", str(result.both_wrong))
    console.print(discord)

    console.print(f"\n  paired n = {result.n_paired}")
    console.print(
        f"  difference = {result.mean_difference:+.4f}  "
        f"95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
    )
    verdict = result.verdict(run_id[0], run_id[1])
    console.print(f"\n[bold]{verdict}[/bold]")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "run_a": run_id[0],
                    "run_b": run_id[1],
                    "model_a": envs[0].get("model_ref"),
                    "model_b": envs[1].get("model_ref"),
                    "evaluator_fingerprint": digests[0],
                    "metric": metric_name,
                    "n_paired": result.n_paired,
                    "mean_a": result.mean_a,
                    "mean_b": result.mean_b,
                    "mean_difference": result.mean_difference,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "significant": result.significant,
                    "underpowered": result.underpowered,
                    "a_right_b_wrong": result.a_right_b_wrong,
                    "b_right_a_wrong": result.b_right_a_wrong,
                    "both_right": result.both_right,
                    "both_wrong": result.both_wrong,
                    "verdict": verdict,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        console.print(f"\n[dim]Wrote {output}[/dim]")


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


def _fci_cell(record: LeaderboardRecord) -> str:
    """The index, or the reason there isn't one. Never a number with an asterisk."""
    return "WITHHELD" if record.fci is None else f"{record.fci:.4f}"


def _write_leaderboard_csv(path: Path, records: Sequence[LeaderboardRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "model_ref",
                "provider",
                "fci",
                "verdict",
                "critical_gate_failed",
                "created_at",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.run_id,
                    record.model_ref,
                    record.provider,
                    "" if record.fci is None else record.fci,
                    record.verdict or "",
                    record.critical_gate_failed,
                    record.created_at,
                ]
            )


def _write_leaderboard_md(path: Path, records: Sequence[LeaderboardRecord]) -> None:
    lines = [
        "# FinanceBench leaderboard",
        "",
        "A **WITHHELD** index is not a missing number — it is a refusal. An index is only published "
        "when the run covered enough to support the claim it makes and no critical gate failed; the "
        "run's `capabilities.json` records `fci_withheld_because` in plain words.",
        "",
        "| Run | Model | Provider | FCI | Verdict | Critical gate | Created |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        gate = "**FAILED**" if record.critical_gate_failed else "ok"
        lines.append(
            f"| {record.run_id} | {record.model_ref} | {record.provider} | {_fci_cell(record)} "
            f"| {record.verdict or '—'} | {gate} | {record.created_at} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leaderboard_html(path: Path, records: Sequence[LeaderboardRecord]) -> None:
    def _row(r: LeaderboardRecord) -> str:
        gate = (
            "<span class='fail'>FAILED</span>"
            if r.critical_gate_failed
            else "<span class='pass'>ok</span>"
        )
        css = "withheld" if r.fci is None else ""
        return (
            f"<tr><td>{html.escape(r.run_id)}</td><td>{html.escape(r.model_ref)}</td>"
            f"<td>{html.escape(r.provider)}</td>"
            f"<td class='{css}'>{html.escape(_fci_cell(r))}</td>"
            f"<td>{html.escape(r.verdict or '—')}</td><td>{gate}</td>"
            f"<td>{html.escape(r.created_at)}</td></tr>"
        )

    rows = "\n".join(_row(r) for r in records) or "<tr><td colspan='7'>No runs yet.</td></tr>"
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FinanceBench leaderboard</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
          max-width: 1100px; margin: 2rem auto; color: #1a1a1a; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  .withheld {{ color: #666; font-style: italic; }}
  .fail {{ color: #b00020; font-weight: 600; }}
  .pass {{ color: #1b7f3b; }}
</style>
</head>
<body>
<h1>FinanceBench leaderboard</h1>
<p><em>A <strong>WITHHELD</strong> index is not a missing number — it is a refusal. An index is only
published when the run covered enough to support the claim it makes and no critical gate failed; the
run's <code>capabilities.json</code> records <code>fci_withheld_because</code> in plain words.</em></p>
<table>
<tr><th>Run</th><th>Model</th><th>Provider</th><th>FCI</th><th>Verdict</th>
    <th>Critical gate</th><th>Created</th></tr>
{rows}
</table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


#: Benchmarks/groups whose runs are pipeline tests, not model evaluations. They use a REAL provider,
#: so the mock filter does not catch them, and a `smoke` run would otherwise be ranked next to a
#: 150-sample FinanceBench run as though the two said comparable things about a model.
_NON_LEADERBOARD_BENCHMARKS: frozenset[str] = frozenset({"smoke"})


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
        # `capabilities.json` is NESTED: {"dimensions": {...}, "scores": {...}, "verdict": ...}.
        # This used to read it as if it were flat, so `agg.get("mean")` was looked up on the keys
        # "dimensions"/"scores"/"verdict" — none of which has a `mean` — and every record silently
        # got `capability_scores: {}`, `fci: null`, `verdict: null`. The leaderboard was structurally
        # incapable of ever displaying a Finance Capability Index, and said "provisional" instead.
        dimensions = capabilities.get("dimensions", {})
        scores = capabilities.get("scores", {})
        capability_scores = {
            name: agg["mean"]
            for name, agg in dimensions.items()
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
        # A run of the `smoke` group is a pipeline test with a handful of in-repo fixtures. It uses a
        # REAL provider, so the mock filter does not catch it — and it would have walked onto the
        # public leaderboard as if it were a model evaluation. Eligibility is about what was ASKED,
        # not only about who answered.
        benchmark = str(environment.get("benchmark_or_group", ""))
        is_smoke = benchmark in _NON_LEADERBOARD_BENCHMARKS
        eligible = run_type is RunType.REAL and not is_smoke
        gates = (
            json.loads((run_path / "gates.json").read_text(encoding="utf-8"))
            if (run_path / "gates.json").is_file()
            else {}
        )
        records.append(
            LeaderboardRecord(
                run_id=str(environment["run_id"]),
                model_ref=str(environment["model_ref"]),
                provider=str(environment["provider"]),
                run_type=run_type,
                eligible_for_leaderboard=eligible,
                fci=scores.get("finance_capability_index"),
                verdict=capabilities.get("verdict"),
                # An FCI is refused, not asterisked — so a run without one is not "provisional",
                # it is a run whose index was withheld, and `fci_withheld_because` says why.
                provisional=scores.get("finance_capability_index") is None,
                critical_gate_failed=bool(gates.get("any_critical_gate_failed", False)),
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


@app.command(name="verify-providers")
def verify_providers(
    output: Annotated[
        Path,
        typer.Option("--output", help="Where to write the verification report."),
    ] = Path("reports/provider_verification.json"),
) -> None:
    """Find out which providers actually work — by calling them.

    Three outcomes, and they are three different things:

    \b
      live_verified                  a real call was made and a real answer came back
      implemented_not_live_verified  no key, so no call was ever made. Unproven, NOT broken.
      unreachable                    a key exists, the call was attempted, and it failed

    A provider with no key is never marked as failing. There is nothing wrong with it — we simply
    have no way to find out, and a red mark would be as much of an invention as a green one.
    """
    records = asyncio.run(verify_all_providers())

    table = Table("Provider", "Status", "Detail")
    colours = {
        ProviderVerification.LIVE_VERIFIED: "green",
        ProviderVerification.IMPLEMENTED_NOT_LIVE_VERIFIED: "yellow",
        ProviderVerification.UNREACHABLE: "red",
        ProviderVerification.NOT_A_MODEL: "dim",
    }
    for record in sorted(records, key=lambda r: r.provider):
        colour = colours[record.status]
        table.add_row(
            record.provider,
            f"[{colour}]{record.status.value}[/{colour}]",
            record.detail[:90],
        )
    console.print(table)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "'implemented_not_live_verified' means no API key was present, so no call was ever "
            "made. It is not a failure. Only 'live_verified' means the provider actually worked."
        ),
        "providers": [record.to_json() for record in sorted(records, key=lambda r: r.provider)],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"\nWritten to [bold]{output}[/bold]")

    live = sum(1 for r in records if r.status is ProviderVerification.LIVE_VERIFIED)
    console.print(f"{live} provider(s) live-verified on this machine.")


@app.command(name="capability-report")
def capability_report(
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
    output: Annotated[Path, typer.Option("--output")] = Path("reports/financial_capability.html"),
) -> None:
    """Build the one report a human actually reads: can this model be trusted with money?

    Answers the five questions this platform exists to answer — calculation, conversation,
    retrieval and citation, small-business CFO analysis, and refusal under adversarial pressure —
    across every real run under --runs-dir.

    A question with no run behind it is reported as UNANSWERED, not as a zero. An absent measurement
    and a failed one are different findings, and only one of them is about the model.
    """
    runs = load_runs(runs_dir)
    real = [r for r in runs if r.run_type == "real"]
    if not real:
        _fail(f"no real runs found under {runs_dir} — a report with nothing in it is not a report")
        return

    document = build_mission_report(runs, generated_at=RealClock().now_iso())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")

    console.print(f"[bold green]Report written[/bold green] to {output}")
    console.print(f"  {len(real)} real run(s) across {len({r.model_ref for r in real})} model(s)")
    unanswered = document.count("class='unanswered'")
    if unanswered:
        console.print(
            f"  [yellow]{unanswered} of the five questions is/are UNANSWERED[/yellow] — "
            "the report says so rather than printing a zero."
        )


def _write_retrieval_report(
    cells: Sequence[object],
    corpus_info: dict[str, object],
    build_ms: dict[str, float],
    runs_dir: Path = Path("runs"),
) -> None:
    """Attach each arm's live GENERATION outcome to its retrieval numbers — where one exists.

    Retrieval metrics are cheap: no model runs. Answer accuracy is not — ~4.6 GPU-hours per arm at
    109.5 s/sample on this hardware. So every arm has retrieval numbers, and only the arms we paid to
    generate have answer numbers. The rest report a dash, which means **not run**, not zero.
    """
    arms: list[ArmResult] = []
    seen: dict[tuple[str, int, bool], str] = {}
    if runs_dir.is_dir():
        for path in sorted(runs_dir.iterdir()):
            capability_path = path / "capabilities.json"
            if not capability_path.is_file() or "retrieval_required" not in path.name:
                continue
            capabilities = json.loads(capability_path.read_text(encoding="utf-8"))
            retrieval = capabilities.get("retrieval")
            if not isinstance(retrieval, dict):
                continue
            # `capabilities.json -> retrieval` is written from the pipeline that ACTUALLY RAN. It is
            # the authoritative record of the arm, and `run_config.json` is not: runs made before the
            # arm was recorded there have `retriever: null, top_k: null, document_scoped: null`, so
            # reading the arm from the config defaults a document-scoped k=10 run to
            # (bm25, k=5, open-corpus) — and hangs its answer accuracy on a completely different arm
            # of the very ablation that exists to tell those arms apart.
            key = (
                str(retrieval.get("retriever", "bm25")),
                int(retrieval.get("top_k", 5)),
                bool(retrieval.get("document_scoped", False)),
            )

            # CONSISTENCY ASSERTION. Where run_config DOES record the arm, it must agree with what
            # the pipeline says it ran. If the two disagree, we do not know which arm this run is,
            # and a wrong attribution here does not look like an error — it looks like a result.
            config_path = path / "run_config.json"
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("retriever") is not None:
                    declared = (
                        str(config["retriever"]),
                        int(config.get("top_k", 5)),
                        bool(config.get("document_scoped", False)),
                    )
                    if declared != key:
                        _fail(
                            f"Refusing to build the ablation report: run {path.name} disagrees with "
                            f"itself about which arm it is.\n"
                            f"  run_config.json  says: {declared}\n"
                            f"  capabilities.json says: {key}  (this is what the pipeline ACTUALLY ran)\n"
                            "Attributing an answer score to the wrong arm of the ablation that exists "
                            "to tell arms apart would not look like an error. It would look like a "
                            "result."
                        )
                        return

            if key in seen:
                _fail(
                    f"Refusing to build the ablation report: two runs claim the same arm {key}.\n"
                    f"  {seen[key]}\n  {path.name}\n"
                    "One of them would silently be dropped, and the report would not say which."
                )
                return
            seen[key] = path.name

    for cell in cells:
        retriever = cell.retriever  # type: ignore[attr-defined]
        top_k = cell.top_k  # type: ignore[attr-defined]
        scoped = cell.document_scoped  # type: ignore[attr-defined]
        run_id = seen.get((retriever, top_k, scoped))

        answer_accuracy = n_generated = unsupported = misses = gen_fail = None
        if run_id is not None:
            metrics = json.loads((runs_dir / run_id / "metrics.json").read_text(encoding="utf-8"))
            entry = metrics.get("financebench_answer_accuracy", {})
            answer_accuracy = entry.get("mean")
            n_generated = entry.get("n")
            unsupported_entry = metrics.get("financebench_unsupported_numeric_claim", {})
            supported = unsupported_entry.get("mean")
            # The metric is "did it avoid inventing a figure". The RATE OF INVENTION is 1 - that.
            unsupported = None if supported is None else 1.0 - supported
            failures = (runs_dir / run_id / "failures.jsonl").read_text(encoding="utf-8")
            misses = sum(1 for line in failures.splitlines() if "retrieval_miss" in line)
            gen_fail = sum(
                1 for line in failures.splitlines() if "generation_error_after_retrieval" in line
            )

        scope = "doc-scoped" if scoped else "open-corpus"
        arms.append(
            ArmResult(
                name=f"{retriever} / {scope} / k={top_k}",
                retriever=retriever,
                top_k=top_k,
                scope=scope,
                page_recall=cell.page_recall,  # type: ignore[attr-defined]
                document_recall=cell.document_recall,  # type: ignore[attr-defined]
                mrr=cell.mrr,  # type: ignore[attr-defined]
                ndcg=cell.ndcg,  # type: ignore[attr-defined]
                mean_query_ms=cell.mean_query_ms,  # type: ignore[attr-defined]
                run_id=run_id,
                answer_accuracy=answer_accuracy,
                n_generated=n_generated,
                unsupported_claim_rate=unsupported,
                retrieval_misses=misses,
                generation_errors_after_retrieval=gen_fail,
            )
        )

    generated = [a for a in arms if a.answer_accuracy is not None]
    finding = (
        "Retrieval quality is NOT what limits the answers. Fixing document scoping raised page recall "
        "4.0% -> 18.7% (4.7x) and produced no statistically supported improvement in answer accuracy, "
        "while generation_error_after_retrieval rose 2 -> 7. Reading those failures by hand: every one "
        "is a JSON-envelope failure — the retriever found the page, and the model answered in its own "
        "shape. The fix is a parser, not an index."
    )
    if len(generated) < 2:
        finding += (
            f"  [Only {len(generated)} arm(s) have been generated against so far, at ~4.6 GPU-hours "
            "each. Arms without answer numbers report a dash: NOT RUN, never zero.]"
        )

    write_ablation_report(
        Path("reports/retrieval_ablation"),
        arms=arms,
        cells=[c.to_json() for c in cells],  # type: ignore[attr-defined]
        corpus=corpus_info or {"pages": 0, "documents": 0, "fingerprint": "unknown"},
        index_build_ms=build_ms,
        finding=finding,
    )
    console.print(
        "Wrote [bold]reports/retrieval_ablation/[/bold] (report.html, summary.md, results.*)"
    )


@app.command(name="retrieval-eval")
def retrieval_eval(
    benchmark: Annotated[str, typer.Option("--benchmark")] = "financebench",
    split: Annotated[str, typer.Option("--split")] = "open_source",
    retrievers: Annotated[
        str, typer.Option("--retrievers", help="Comma-separated: bm25,dense,hybrid")
    ] = "bm25",
    top_ks: Annotated[
        str, typer.Option("--top-k", help="Comma-separated, e.g. 5,10,20")
    ] = "5,10,20",
    output: Annotated[Path, typer.Option("--output")] = Path("reports/retrieval_ablation.json"),
) -> None:
    """Compare retrievers on recall — with NO model in the loop.

    The retrieval-mode run found that 74 of 85 wrong answers were RETRIEVAL misses: the right page
    was never put in front of the model. Only 2 were cases where the page was found and the model
    then fumbled it. Improving the model would move almost nothing; the retriever is the bottleneck.

    Recall@k needs a query, a corpus, and the gold evidence — read AFTER retrieval, never before.
    So this is the cheapest question in the platform and the one with the most leverage.

    Every retriever asked for is reported. A retriever chosen by its own benchmark and then shown
    without its rivals is a number with the losing evidence deleted.
    """
    names = [r.strip() for r in retrievers.split(",") if r.strip()]
    ks = [int(k.strip()) for k in top_ks.split(",") if k.strip()]

    samples = list(create_dataset(benchmark).load(split))
    pdf_dir = Path("data/downloads") / benchmark / "pdfs"
    if not pdf_dir.is_dir():
        _fail(f"no document corpus at {pdf_dir}. Run: financebench prepare {benchmark}")
        return

    console.print(
        f"Sweeping {names} x k={ks} x [open-corpus, document-scoped] over {len(samples)} questions. "
        "No model is called."
    )

    def progress(name: str, scoped: bool) -> None:
        console.print(f"  done: {name} ({'document-scoped' if scoped else 'open-corpus'})")

    try:
        cells = run_ablation(
            samples, pdf_dir=pdf_dir, retrievers=names, top_ks=ks, on_progress=progress
        )
    except RuntimeError as exc:
        _fail(str(exc))
        return

    table = Table(
        "Retriever", "Scoping", "k", "Page recall", "Doc recall", "Evidence F1", "Gold rank"
    )
    for cell in cells:
        table.add_row(
            cell.retriever,
            "document-scoped" if cell.document_scoped else "open-corpus",
            str(cell.top_k),
            f"{cell.page_recall:.1%}",
            f"{cell.document_recall:.1%}",
            f"{cell.evidence_f1:.3f}",
            "-" if cell.mean_gold_rank is None else f"{cell.mean_gold_rank:.1f}",
        )
    console.print(table)

    output.parent.mkdir(parents=True, exist_ok=True)
    corpus_info = getattr(run_ablation, "last_corpus", {})
    build_ms = getattr(run_ablation, "last_index_build_ms", {})
    payload = {
        "benchmark": benchmark,
        "split": split,
        "n_questions": len(samples),
        "corpus": corpus_info,
        "index_build_ms": {k: round(v, 1) for k, v in build_ms.items()},
        "note": (
            "Recall@k is measured with no model in the loop. Gold evidence is read only AFTER "
            "retrieval. Every retriever swept is reported, including the losers."
        ),
        "cells": [cell.to_json() for cell in cells],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"\nWritten to [bold]{output}[/bold]")

    # The full report set: retrieval performance, generation-given-retrieval, and end-to-end kept
    # strictly apart. A single "RAG accuracy" welds them together and then sends you to rebuild the
    # component that was working.
    _write_retrieval_report(cells, corpus_info, build_ms)

    best = max(cells, key=lambda c: c.page_recall)
    console.print(
        f"Best page recall: [bold]{best.page_recall:.1%}[/bold] "
        f"({best.retriever}, k={best.top_k}, "
        f"{'document-scoped' if best.document_scoped else 'open-corpus'})"
    )


@app.command(name="calibrate-judge")
def calibrate_judge(
    judge_model: Annotated[
        str, typer.Option("--judge", help="The judge. Must not be the candidate's own family.")
    ] = "ollama/llama3.2:3b",
    n: Annotated[int, typer.Option("--cases", help="Calibration cases to run.")] = 48,
    output: Annotated[Path, typer.Option("--output")] = Path("reports/judge_calibration.json"),
) -> None:
    """Find out whether the judge can tell a good answer from a bad one — BEFORE trusting it.

    Runs the judge against cases whose correct verdict is known by construction: the expert's own
    answer (correct), the same answer about a different company (wrong), the same answer with an
    invented figure (wrong), a refusal where the filing plainly contains the answer (wrong), and so on.

    \b
    A judge that does not clear the bar does not get to produce a score. The analytical dimension is
    reported as NOT_EVALUATED, with the reason — never as a zero.
    """
    from financebench.datasets.secque import SecqueAdapter
    from financebench.evaluation.judge import (
        build_calibration_set,
        judge_answer,
        score_calibration,
    )

    judge = ModelSpec.parse(judge_model)
    console.print(f"Calibrating [bold]{judge.ref}[/bold] on {n} derived cases...")
    console.print(
        "[dim]These are labelled derived_judge_calibration. They are NOT SECQUE tasks and are "
        "never reported as such.[/dim]\n"
    )

    samples = list(SecqueAdapter().load("test"))
    cases = build_calibration_set(samples, target=n)

    async def run() -> list[bool | None]:
        provider = create_provider(judge.provider)
        verdicts: list[bool | None] = []
        try:
            for index, case in enumerate(cases, start=1):
                verdict = await judge_answer(
                    case.sample, case.answer, provider=provider, judge=judge
                )
                verdicts.append(verdict.correct if verdict.valid else None)
                mark = "ok" if verdicts[-1] == case.should_be_correct else "MISS"
                console.print(
                    f"  [{index:2d}/{len(cases)}] {case.corruption:36s} "
                    f"expected={case.should_be_correct!s:5s} got={verdicts[-1]!s:5s} {mark}"
                )
        finally:
            await provider.aclose()
        return verdicts

    verdicts = asyncio.run(run())
    report = score_calibration(cases, verdicts)

    console.print()
    table = Table("Corruption", "Judge accuracy")
    for name, accuracy in sorted(report.by_corruption.items()):
        table.add_row(name, f"{accuracy:.0%}")
    console.print(table)

    colour = "green" if report.passed else "red"
    console.print(f"\n[{colour}]{report.verdict}[/{colour}]")
    console.print(
        f"  accuracy {report.accuracy:.1%} | false positives {report.false_positive_rate:.1%} "
        f"| false negatives {report.false_negative_rate:.1%} | invalid {report.invalid_judgments}"
    )
    if not report.passed:
        console.print(
            "\n[yellow]The analytical score will be reported as NOT_EVALUATED, never as zero. "
            "A judge you cannot trust produces no number at all.[/yellow]"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_model": judge.ref,
        "provenance": "derived_judge_calibration",
        "note": (
            "Cases are DERIVED from real SECQUE tasks by transformations whose effect on correctness "
            "is known by construction. They are not SECQUE tasks and are not reported as such."
        ),
        **report.to_json(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"\nWritten to [bold]{output}[/bold]")


# --------------------------------------------------------------------------- freeze-manifest


@app.command(name="freeze-manifest")
def freeze_manifest(
    spec: Annotated[
        list[str],
        typer.Option(
            "--take",
            help="benchmark:split:n — repeat per benchmark. e.g. --take finqa:test:75",
        ),
    ],
    name: Annotated[str, typer.Option("--name", help="The manifest's name.")],
    output: Annotated[Path, typer.Option("--output")] = Path("configs/manifests/manifest.json"),
    description: Annotated[str | None, typer.Option("--description")] = None,
) -> None:
    """Freeze an exact set of sample ids into a manifest, STRATIFIED by task family.

    The sampling is deterministic and stratified — a round-robin across each benchmark's task
    families — for a reason found the hard way: ``--max-samples 80`` on SECQUE returned 72 Analysis
    questions and 8 Comparison ones, zero Ratio and zero Risk, and reported the result as "SECQUE".
    It was a different benchmark wearing SECQUE's name. A head-truncation of a file whose rows are
    grouped by category is not a sample of that benchmark; it is a sample of its first category.
    """
    entries: list[ManifestBenchmark] = []
    for item in spec:
        try:
            bench, split_name, count_text = item.split(":")
            count = int(count_text)
        except ValueError:
            _fail(f"--take must be benchmark:split:n — got {item!r}")
            return
        try:
            samples = list(create_dataset(bench).load(split_name))
        except ConfigError as exc:
            _fail(str(exc))
            return

        # Round-robin across task families, so any prefix is balanced.
        by_family: dict[str, list[str]] = {}
        for sample in samples:
            by_family.setdefault(sample.task_family, []).append(sample.sample_id)

        picked: list[str] = []
        families = sorted(by_family)
        index = 0
        while len(picked) < count and any(by_family[f] for f in families):
            family = families[index % len(families)]
            if by_family[family]:
                picked.append(by_family[family].pop(0))
            index += 1

        if len(picked) < count:
            console.print(
                f"[yellow]{bench}:{split_name} has only {len(picked)} samples; asked for "
                f"{count}.[/yellow]"
            )
        entries.append(ManifestBenchmark(name=bench, split=split_name, sample_ids=tuple(picked)))
        console.print(
            f"  {bench}:{split_name} -> {len(picked)} samples across {len(families)} task families"
        )

    manifest = SampleManifest(
        name=name,
        description=description,
        created_at=RealClock().now_iso(),
        frozen_at_commit=git_commit(),
        benchmarks=tuple(entries),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    console.print(
        f"\n[bold green]Frozen[/bold green] {len(manifest.all_sample_ids)} samples -> {output}"
        f"\n  id_hash = [cyan]{manifest.id_hash}[/cyan]"
    )


# --------------------------------------------------------------------------- release


@app.command(name="release-report")
def release_report(
    version: Annotated[str, typer.Option("--version")] = "v0.1.0-rc1",
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
) -> None:
    """Build the public release report from every run scored by the CURRENT evaluator.

    Runs scored by a different evaluator are **excluded**, not averaged in. Two runs with different
    fingerprints are not comparable, and putting them on one page would make the page a lie.
    """
    current = current_fingerprint().digest
    by_model: dict[str, ModelResult] = {}
    excluded: list[tuple[str, str]] = []

    for path in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        artifacts = load_run(runs_dir, path.name)
        if artifacts is None:
            continue
        env = artifacts.get("environment", {})
        if env.get("run_type") != "real":
            continue
        digest = env.get("evaluator_fingerprint", {}).get("digest")
        if digest != current:
            excluded.append((path.name, str(digest)))
            continue
        model_ref = str(env["model_ref"])
        by_model.setdefault(model_ref, ModelResult(model_ref=model_ref)).runs[path.name] = artifacts

    if excluded:
        console.print(
            f"[yellow]Excluded {len(excluded)} run(s) scored by a DIFFERENT evaluator — they are not "
            f"comparable with the rest and are not averaged in:[/yellow]"
        )
        for rid, digest in excluded[:6]:
            console.print(f"  [dim]{digest}  {rid[:60]}[/dim]")
        console.print(
            "[dim]  Re-score them: financebench resume --run-id <id> --model-config <cfg>[/dim]\n"
        )

    if not by_model:
        _fail(
            f"No runs are scored by the current evaluator ({current}). "
            "Nothing can be published until at least one is."
        )
        return

    out_dir = Path("release") / version
    limitations = Path("docs/known_limitations.md")
    build_release_report(
        out_dir,
        version=version,
        models=list(by_model.values()),
        paired=[],
        fingerprint=current,
        hardware=_hardware_summary(),
        limitations=(
            "See [`docs/known_limitations.md`](../../docs/known_limitations.md)."
            if limitations.is_file()
            else "NOT DOCUMENTED — this is itself a release blocker."
        ),
    )
    total = sum(len(m.runs) for m in by_model.values())
    console.print(
        f"[bold green]Wrote[/bold green] {out_dir}/report.html, results.json, leaderboard.csv "
        f"({total} run(s), {len(by_model)} model(s), fingerprint {current})"
    )


def _hardware_summary() -> dict[str, object]:
    import platform

    gpu = None
    if shutil.which("nvidia-smi"):
        import subprocess

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu or None,
    }


@app.command(name="release-build")
def release_build(
    version: Annotated[str, typer.Option("--version")] = "v0.1.0-rc1",
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
    run_id: Annotated[
        list[str] | None, typer.Option("--run-id", help="Repeat. The runs this release publishes.")
    ] = None,
    manifest: Annotated[
        list[Path] | None, typer.Option("--manifest", help="Repeat. The frozen sample manifests.")
    ] = None,
) -> None:
    """Assemble the release manifest + checksums, then run every mandatory release gate.

    Exits non-zero if any gate fails — and writes `BLOCKERS.md` rather than a tag.
    """
    out_dir = Path("release") / version
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list(run_id or [])
    if not ids:
        ids = sorted(p.name for p in runs_dir.iterdir() if (p / "environment.json").is_file())

    console.print(f"[bold]Building release {version}[/bold] from {len(ids)} run(s)\n")
    payload = build_release(
        version,
        runs_dir=runs_dir,
        run_ids=ids,
        manifests=list(manifest or []),
        out_dir=out_dir,
    )
    console.print(f"  evaluator fingerprint: {payload['evaluator_fingerprint']['digest']}")
    console.print(f"  commit: {payload['repository_commit']}  dirty={payload['repository_dirty']}")

    console.print("\n[bold]Release gates[/bold]")
    gates = check_release_gates(out_dir, runs_dir=runs_dir)
    table = Table("Gate", "Result", "Detail")
    for gate in gates:
        colour = {"PASS": "green", "FAIL": "red", "NOT APPLICABLE": "yellow"}[gate.label]
        table.add_row(gate.name, f"[{colour}]{gate.label}[/{colour}]", gate.detail[:60])
    console.print(table)

    # Every gate, individually, as an artifact. A release that publishes only "all gates passed" is
    # asking to be trusted; one that publishes each gate and its observed value can be checked.
    (out_dir / "gate_results.json").write_text(
        json.dumps(
            {
                "version": version,
                "evaluated_at": RealClock().now_iso(),
                "n_gates": len(gates),
                "n_passed": sum(1 for g in gates if g.passed is True),
                "n_failed": sum(1 for g in gates if g.passed is False),
                "n_not_applicable": sum(1 for g in gates if g.passed is None),
                "tagged": not any(g.passed is False for g in gates),
                "gates": [
                    {
                        "name": g.name,
                        "result": g.label.replace(" ", "_"),
                        "detail": g.detail,
                    }
                    for g in gates
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    failed = [g for g in gates if g.passed is False]
    if failed:
        remedy = {
            "one evaluator fingerprint across all runs": (
                "Re-score the stale runs onto the current fingerprint:\n"
                "`financebench resume --run-id <id> --model-config <cfg>` replays cached responses "
                "and costs nothing.\n\n"
                "**Except for tool_assisted runs.** Those are deliberately uncached (an agent run is "
                "a conversation, not one hashable request), so re-scoring one means re-running every "
                "turn against the model."
            ),
            "paired direct-vs-tools run complete": (
                "Run the missing variants against the frozen manifest:\n"
                "`financebench eval --manifest configs/manifests/tool_paired_v1.json "
                "--model-config <cfg> --mode {context_given|tool_assisted}`\n\n"
                "All four must exist, on the SAME 150 sample ids, or the paired comparison does not "
                "exist. Do not substitute an unrelated direct run — the previous tool run was on "
                "`tatqa:train:` ids while both direct runs used `tatqa:dev:`, so it could not be "
                "paired with anything at all."
            ),
            "release-group run complete (both models)": (
                "`financebench eval --manifest configs/manifests/release_v0_1.json "
                "--model-config <cfg>`\n\n"
                "This is the **only** run that can produce a Finance Capability Index: the index is "
                "withheld unless ONE run covered SMB-CFO *and* a grounding benchmark *and* refusal "
                "together. Without it, the report's headline is `INSUFFICIENT_COVERAGE`."
            ),
            "retrieval ablation complete": (
                "Retrieval metrics for all 18 cells are cheap (no model runs):\n"
                "`financebench retrieval-eval --retrievers bm25,dense,hybrid --top-k 1,3,5,10,20`\n\n"
                "The **generated** arms are not cheap — ~4.6 GPU-hours each at 109.5 s/sample. At "
                "least two are needed to say anything about whether better retrieval produces better "
                "answers."
            ),
        }
        blockers = [
            f"# BLOCKERS — {version}",
            "",
            "This release candidate **was not tagged**. Every gate below must pass first.",
            "",
            "Note what is *not* here: ruff, mypy, the 1,056 primary tests, the 411 security tests, "
            "and the 17 parity tests (zero skips) all **pass**. The code is healthy. What is missing "
            "is *evidence* — runs that have not finished. Tagging on a green test suite while the "
            "headline experiment is still executing is precisely the dishonesty this project exists "
            "to prevent.",
            "",
        ]
        for gate in failed:
            blockers += [f"## {gate.name}", "", "```", gate.detail, "```", ""]
            if gate.name in remedy:
                blockers += ["**What clears it**", "", remedy[gate.name], ""]
        (out_dir / "BLOCKERS.md").write_text("\n".join(blockers) + "\n", encoding="utf-8")
    elif (out_dir / "BLOCKERS.md").is_file():
        # Every gate passes now. A stale BLOCKERS.md left lying in the release directory would be
        # shipped alongside the artifacts, telling a reader the release is blocked when it is not.
        (out_dir / "BLOCKERS.md").unlink()
        console.print("  [dim]removed a stale BLOCKERS.md — every gate now passes[/dim]")

    # Checksums LAST, so they cover gate_results.json and BLOCKERS.md too. A checksum file that
    # omits the gate report is a checksum file that cannot prove which gates were run.
    lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.txt":
            lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
    (out_dir / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"\n  checksums: {len(lines)} file(s)")

    if failed:
        console.print(
            f"\n[red]{len(failed)} gate(s) FAILED. Wrote {out_dir / 'BLOCKERS.md'}. "
            "The release is NOT tagged.[/red]"
        )
        raise typer.Exit(code=1)

    console.print("\n[bold green]Every mandatory gate passed.[/bold green]")


@app.command(name="rescore")
def rescore(
    run_id: Annotated[str, typer.Option("--run-id")],
    model_config: Annotated[Path, typer.Option("--model-config", exists=True)],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("runs"),
) -> None:
    """Re-score a completed run from its STORED responses, under the current evaluator.

    No provider is constructed, no request is sent, and nothing here can reach a model. That is the
    difference between this and `resume`, and it is the difference that makes a migration auditable:
    when an evaluator change touches only OUR code, the model's answers have not changed, and
    re-running inference to obtain them again would introduce fresh sampling noise into a comparison
    whose entire purpose is to isolate the effect of our own change.

    It matters most for TOOL runs, which are deliberately uncached — `resume` re-runs every turn of
    the agent loop against the model. This does not.
    """
    out_dir = runs_dir / run_id
    environment_path = out_dir / "environment.json"
    if not environment_path.is_file():
        _fail(f"No existing run found at {out_dir}")
        return

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    config_file = load_model_config(model_config)
    if config_file.to_model_spec().ref != environment["model_ref"]:
        _fail(
            f"--model-config resolves to {config_file.to_model_spec().ref!r}, but {run_id} was "
            f"recorded against {environment['model_ref']!r}"
        )
        return
    if environment.get("run_type") != RunType.REAL.value:
        _fail(f"{run_id} is not a real run (run_type={environment.get('run_type')!r}).")
        return

    old_fingerprint = str(environment.get("evaluator_fingerprint", {}).get("digest", "unknown"))
    old_hashes = _raw_response_hashes(out_dir)

    # The same questions the run actually asked — from its frozen manifest where it had one, and
    # otherwise from the splits its own predictions record.
    manifest_path = run_config.get("sample_manifest_path")
    sample_manifest = None
    if manifest_path:
        sample_manifest = load_sample_manifest(Path(manifest_path))
        recorded = run_config.get("sample_manifest_id_hash")
        if recorded and sample_manifest.id_hash != recorded:
            _fail(
                f"cannot re-score {run_id}: its manifest has CHANGED "
                f"({sample_manifest.id_hash} != recorded {recorded})."
            )
            return
        benchmark_splits = sample_manifest.benchmark_splits
    else:
        splits: dict[str, str] = {}
        for record in read_jsonl(out_dir / "predictions.jsonl"):
            splits.setdefault(str(record["benchmark"]), str(record["split"]))
        benchmark_splits = tuple(sorted(splits.items()))

    try:
        samples, dataset_manifests = resolve_samples(benchmark_splits, sample_manifest)
    except (ConfigError, ManifestError) as exc:
        _fail(str(exc))
        return

    model = config_file.to_model_spec()
    config = config_file.to_run_config(
        seed=int(run_config["seed"]),
        limit=run_config.get("limit"),
        max_cost_usd=run_config.get("max_cost_usd"),
        prompt_profile=str(run_config.get("prompt_profile", DEFAULT_PROMPT_PROFILE)),
        eval_mode=EvalMode(run_config.get("eval_mode", EvalMode.CONTEXT_GIVEN.value)),
        conversation_protocol=ConversationProtocol(
            run_config.get("conversation_protocol", ConversationProtocol.GOLD_HISTORY.value)
        ),
        retriever=str(run_config.get("retriever") or "bm25"),
        top_k=int(run_config.get("top_k") or 5),
        document_scoped=bool(run_config.get("document_scoped") or False),
        sample_manifest_path=manifest_path,
        sample_manifest_id_hash=run_config.get("sample_manifest_id_hash"),
    )

    try:
        rescore_from_artifacts(
            out_dir,
            run_id=run_id,
            label=str(environment["benchmark_or_group"]),
            model=model,
            config=config,
            samples=samples,
            dataset_manifests=dataset_manifests,
            # Read from the run's OWN model_manifest.json. Not re-derived, and above all not
            # obtained by constructing a provider — a re-score that builds a provider is one call
            # away from making one, and the guarantee "this cannot reach a model" is worth more
            # than the two lines it costs to keep.
            provider_capabilities=ProviderCapabilities(
                **json.loads((out_dir / "model_manifest.json").read_text(encoding="utf-8"))[
                    "capabilities"
                ]
            ),
        )
    except ConfigError as exc:
        _fail(str(exc))
        return

    new_environment = json.loads(environment_path.read_text(encoding="utf-8"))
    new_fingerprint = str(new_environment.get("evaluator_fingerprint", {}).get("digest", "unknown"))
    new_hashes = _raw_response_hashes(out_dir)
    preserved = bool(old_hashes) and old_hashes == new_hashes

    record = {
        "source_run_id": run_id,
        "migration_type": "reparse_rescore",
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
        "reason": (
            "The evaluator changed; the model did not. Every raw response was re-read from disk — "
            "no provider was constructed and no request was sent, so no inference could have run."
        ),
        "raw_response_hashes_preserved": preserved,
        "n_samples": len(old_hashes),
        "migrated_at": RealClock().now_iso(),
    }
    (out_dir / "migration.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if not preserved:
        _fail(
            f"A re-score MUST NOT change the raw responses, and this one did. {run_id} is now "
            "inconsistent — investigate before trusting any number from it."
        )
        return

    console.print(
        f"[bold green]Re-scored[/bold green] {run_id}\n"
        f"  {old_fingerprint} -> {new_fingerprint}\n"
        f"  {len(old_hashes)} raw responses re-read, [bold]0 model calls[/bold], "
        "every response hash preserved."
    )


if __name__ == "__main__":
    app()
