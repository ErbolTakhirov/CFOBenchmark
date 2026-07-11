"""The mock must never be mistakable for a model result.

The mock provider is handed the gold answers. A mock run that scores 100 % says exactly nothing
about any model — so these tests pin down the four barriers that keep it out of anywhere a reader
might mistake it for one:

1. it does not run at all unless explicitly asked for (``--allow-mock``);
2. the run is stamped ``run_type=mock_test`` and ``eligible_for_leaderboard=false``;
3. every report it produces is watermarked before the first number;
4. the leaderboard refuses to rank it.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from financebench.cli import app
from financebench.schemas.common import RunType
from financebench.storage.artifacts import MOCK_WATERMARK_TITLE

runner = CliRunner()

MOCK_CONFIG = Path("configs/models/mock.yaml").resolve()


def _eval_args(out: Path, *extra: str) -> list[str]:
    return [
        "eval",
        "--group",
        "smoke",
        "--model-config",
        str(MOCK_CONFIG),
        "--output-dir",
        str(out),
        *extra,
    ]


def test_mock_eval_is_refused_without_allow_mock(tmp_path: Path) -> None:
    """The gate itself. Without the flag, no run happens at all."""
    result = runner.invoke(app, _eval_args(tmp_path / "runs"))

    assert result.exit_code != 0
    assert "--allow-mock" in result.output
    # Nothing was written — a refused run leaves no artifacts a reader could stumble across.
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_mock_eval_with_allow_mock_is_stamped_and_barred(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    result = runner.invoke(app, _eval_args(runs, "--allow-mock"))
    assert result.exit_code == 0, result.output
    assert "MOCK — NOT A MODEL RESULT" in result.output

    (run_dir,) = list(runs.iterdir())
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["run_type"] == RunType.MOCK_TEST.value
    assert environment["eligible_for_leaderboard"] is False


def test_mock_reports_are_watermarked_before_any_number(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert runner.invoke(app, _eval_args(runs, "--allow-mock")).exit_code == 0
    (run_dir,) = list(runs.iterdir())

    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    for document in (summary, report):
        assert MOCK_WATERMARK_TITLE in document
        # The watermark precedes the metrics table — a reader who stops after one screen still
        # knows this is not a model result.
        assert document.index(MOCK_WATERMARK_TITLE) < document.lower().index("metric")


def test_leaderboard_refuses_to_rank_a_mock_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    assert runner.invoke(app, _eval_args(runs, "--allow-mock")).exit_code == 0

    result = runner.invoke(app, ["leaderboard", "--runs-dir", str(runs), "--output", str(reports)])
    assert result.exit_code == 0, result.output

    ranked = json.loads((reports / "leaderboard.json").read_text(encoding="utf-8"))
    excluded = json.loads((reports / "leaderboard_excluded.json").read_text(encoding="utf-8"))

    assert ranked == [], "a mock run must never appear on the leaderboard"
    assert len(excluded) == 1
    assert excluded[0]["run_type"] == RunType.MOCK_TEST.value
    assert "excluded" in result.output.lower()

    # …and it isn't hiding in the human-facing renderings either.
    assert "mock" not in (reports / "leaderboard.md").read_text(encoding="utf-8").lower()
    assert "mock" not in (reports / "leaderboard.csv").read_text(encoding="utf-8").lower()
