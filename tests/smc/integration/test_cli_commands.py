"""Integration tests for CLI commands via typer.testing.CliRunner.

Tests the ``smc ingest`` and ``smc detect`` commands end-to-end.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smc.cli.main import app
from smc.data.schemas import Timeframe

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_csv_fixture(
    csv_dir: Path,
    instrument: str,
    timeframe: Timeframe,
    num_bars: int = 200,
) -> None:
    """Create a CSV fixture with enough bars for detection."""
    tf_dir = csv_dir / str(timeframe)
    tf_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tf_dir / f"{instrument}.csv"

    lines = [
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
    ]

    base_price = 2050.0
    for i in range(num_bars):
        day = 2 + (i // 24)
        month_day = 2 + ((day - 2) % 27)
        hour = i % 24

        trend = 0.05 * i
        cycle = 8.0 * math.sin(2 * math.pi * i / 40)
        noise = 2.0 * math.sin(i * 7.3)

        close_price = base_price + trend + cycle + noise
        bar_range = abs(3.0 * math.sin(i * 1.7)) + 1.0
        open_price = close_price - 0.3 * bar_range * (1 if i % 3 == 0 else -1)
        high_price = max(open_price, close_price) + abs(bar_range * 0.4)
        low_price = min(open_price, close_price) - abs(bar_range * 0.4)
        volume = 800.0 + 500.0 * abs(math.sin(i * 2.1))
        spread = 2.5 + abs(math.sin(i)) * 1.5

        date_str = f"2024.01.{month_day:02d}"
        time_str = f"{hour:02d}:00"

        lines.append(
            f"{date_str}\t{time_str}\t{open_price:.2f}\t{high_price:.2f}\t"
            f"{low_price:.2f}\t{close_price:.2f}\t{volume:.1f}\t0\t{spread:.1f}"
        )

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def csv_fixture(tmp_path: Path) -> Path:
    """Create a temporary CSV fixture directory."""
    csv_dir = tmp_path / "csv"
    _create_csv_fixture(csv_dir, "XAUUSD", Timeframe.H1)
    return csv_dir


@pytest.fixture()
def ingested_lake(tmp_path: Path, csv_fixture: Path) -> Path:
    """Ingest CSV fixture data and return the data_dir."""
    from smc.data.ingest import ingest_csv_files

    data_dir = tmp_path / "lake"
    ingest_csv_files(
        csv_dir=csv_fixture,
        instrument="XAUUSD",
        timeframe=Timeframe.H1,
        data_dir=data_dir,
    )
    return data_dir


# ---------------------------------------------------------------------------
# Tests: smc ingest
# ---------------------------------------------------------------------------


class TestCLIIngest:
    """Tests for the ``smc ingest`` CLI command."""

    def test_ingest_success(self, csv_fixture: Path, tmp_path: Path) -> None:
        """Verify ``smc ingest`` completes successfully with valid CSV data."""
        data_dir = tmp_path / "lake"
        result = runner.invoke(
            app,
            [
                "ingest",
                "--csv-dir", str(csv_fixture),
                "--instrument", "XAUUSD",
                "--timeframe", "H1",
            ],
            env={"SMC_DATA_DIR": str(data_dir)},
        )
        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "Ingest" in result.output

    def test_ingest_invalid_timeframe(self, csv_fixture: Path) -> None:
        """Verify that an invalid timeframe produces an error."""
        result = runner.invoke(
            app,
            [
                "ingest",
                "--csv-dir", str(csv_fixture),
                "--timeframe", "INVALID",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid timeframe" in result.output

    def test_ingest_missing_csv_dir(self, tmp_path: Path) -> None:
        """Verify that a missing CSV dir produces an error."""
        fake_dir = tmp_path / "does_not_exist"
        result = runner.invoke(
            app,
            [
                "ingest",
                "--csv-dir", str(fake_dir),
            ],
        )
        # typer validates --csv-dir exists before entering the command body
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Tests: smc detect
# ---------------------------------------------------------------------------


class TestCLIDetect:
    """Tests for the ``smc detect`` CLI command."""

    def test_detect_success(self, ingested_lake: Path) -> None:
        """Verify ``smc detect`` completes successfully after ingest."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--instrument", "XAUUSD",
                "--timeframe", "H1",
            ],
            env={"SMC_DATA_DIR": str(ingested_lake)},
        )
        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "Detection Summary" in result.output

    def test_detect_invalid_timeframe(self) -> None:
        """Verify that an invalid timeframe produces an error."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--timeframe", "INVALID",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid timeframe" in result.output

    def test_detect_empty_lake(self, tmp_path: Path) -> None:
        """Verify that detection on an empty lake produces an error."""
        empty_lake = tmp_path / "empty_lake"
        empty_lake.mkdir(parents=True)
        result = runner.invoke(
            app,
            [
                "detect",
                "--instrument", "XAUUSD",
                "--timeframe", "H1",
            ],
            env={"SMC_DATA_DIR": str(empty_lake)},
        )
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "no data" in result.output.lower()


# ---------------------------------------------------------------------------
# Tests: smc backtest
# ---------------------------------------------------------------------------


class TestCLIBacktest:
    """Tests for the ``smc backtest`` CLI command."""

    def test_backtest_empty_lake(self, tmp_path: Path) -> None:
        """Verify that backtest on an empty lake produces an error."""
        empty_lake = tmp_path / "empty_lake"
        empty_lake.mkdir(parents=True)
        result = runner.invoke(
            app,
            [
                "backtest",
                "--instrument", "XAUUSD",
            ],
            env={"SMC_DATA_DIR": str(empty_lake)},
        )
        # Should fail because no data for walk-forward windows
        assert result.exit_code == 1

    def test_backtest_renders_header(self, tmp_path: Path) -> None:
        """Verify that backtest displays the configuration panel."""
        empty_lake = tmp_path / "empty_lake"
        empty_lake.mkdir(parents=True)
        result = runner.invoke(
            app,
            [
                "backtest",
                "--instrument", "XAUUSD",
                "--train-months", "6",
                "--test-months", "2",
            ],
            env={"SMC_DATA_DIR": str(empty_lake)},
        )
        # Header panel should still be printed even if backtest fails
        assert "Walk-Forward Backtest" in result.output


# ---------------------------------------------------------------------------
# Tests: smc version & smc lake-info
# ---------------------------------------------------------------------------


class TestCLIMisc:
    """Tests for other CLI commands."""

    def test_version(self) -> None:
        """Verify ``smc version`` prints the version string."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "AI-SMC" in result.output

    def test_lake_info_empty(self, tmp_path: Path) -> None:
        """Verify ``smc lake-info`` handles an empty lake gracefully."""
        result = runner.invoke(
            app,
            ["lake-info"],
            env={"SMC_DATA_DIR": str(tmp_path)},
        )
        assert result.exit_code == 0

    def test_health(self, tmp_path: Path) -> None:
        """Verify ``smc health`` runs without crashing."""
        result = runner.invoke(
            app,
            ["health"],
            env={"SMC_DATA_DIR": str(tmp_path)},
        )
        assert result.exit_code == 0
        assert "System Health" in result.output
