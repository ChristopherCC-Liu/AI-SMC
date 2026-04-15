"""Integration tests for the SMC detect pipeline.

Tests the full chain: ingest CSV data -> detect_patterns() -> SMCSnapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.data.ingest import ingest_csv_files
from smc.data.schemas import Timeframe
from smc.smc_core.pipeline import detect_patterns
from smc.smc_core.types import SMCSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_large_csv_fixture(
    csv_dir: Path,
    instrument: str,
    timeframe: Timeframe,
    num_bars: int = 300,
) -> Path:
    """Create a CSV fixture with enough bars for SMC detection.

    SMC detection needs at least ~50 bars for reliable swing/OB detection.
    We generate 300 bars with a gentle uptrend + some pullbacks to produce
    detectable patterns.
    """
    import math

    tf_dir = csv_dir / str(timeframe)
    tf_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tf_dir / f"{instrument}.csv"

    lines = [
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
    ]

    base_price = 2050.0
    price = base_price

    for i in range(num_bars):
        # Calculate date and hour from bar index
        day = 2 + (i // 24)  # Start Jan 2, wrap days
        hour = i % 24

        # Skip weekends: day of month 7,8,14,15,21,22,28,29 are "weekends"
        # (simplified — just skip some days to avoid gaps)
        month_day = 2 + (i // 24)
        if month_day > 28:
            month_day = 2 + ((month_day - 2) % 27)

        # Price dynamics: uptrend with sine wave pullbacks
        trend = 0.05 * i  # gentle uptrend
        cycle = 8.0 * math.sin(2 * math.pi * i / 40)  # 40-bar cycle
        noise = 2.0 * math.sin(i * 7.3)  # pseudo-random noise

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
    return csv_path


def _ingest_fixture(tmp_path: Path) -> Path:
    """Create and ingest a CSV fixture, returning the data_dir."""
    csv_dir = tmp_path / "csv"
    data_dir = tmp_path / "lake"
    _create_large_csv_fixture(csv_dir, "XAUUSD", Timeframe.H1)
    ingest_csv_files(
        csv_dir=csv_dir,
        instrument="XAUUSD",
        timeframe=Timeframe.H1,
        data_dir=data_dir,
    )
    return data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectPipeline:
    """Integration tests for detect_patterns."""

    def test_detect_returns_snapshot(self, tmp_path: Path) -> None:
        """Verify that detect_patterns returns a valid SMCSnapshot."""
        data_dir = _ingest_fixture(tmp_path)

        snapshot = detect_patterns(
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        assert isinstance(snapshot, SMCSnapshot)
        assert snapshot.timeframe == Timeframe.H1

    def test_detect_snapshot_has_patterns(self, tmp_path: Path) -> None:
        """Verify that the snapshot contains detected patterns."""
        data_dir = _ingest_fixture(tmp_path)

        snapshot = detect_patterns(
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        # With 300 bars of trending data, we should detect at least some patterns
        total_patterns = (
            len(snapshot.swing_points)
            + len(snapshot.order_blocks)
            + len(snapshot.fvgs)
            + len(snapshot.structure_breaks)
            + len(snapshot.liquidity_levels)
        )
        assert total_patterns > 0, "Expected at least some patterns from 300 bars"

    def test_detect_snapshot_has_trend(self, tmp_path: Path) -> None:
        """Verify that the snapshot includes a valid trend direction."""
        data_dir = _ingest_fixture(tmp_path)

        snapshot = detect_patterns(
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        assert snapshot.trend_direction in ("bullish", "bearish", "ranging")

    def test_detect_with_specific_date(self, tmp_path: Path) -> None:
        """Verify detection with a specific date parameter."""
        data_dir = _ingest_fixture(tmp_path)

        snapshot = detect_patterns(
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            date=datetime(2024, 1, 10, tzinfo=timezone.utc),
            data_dir=data_dir,
        )

        assert isinstance(snapshot, SMCSnapshot)

    def test_detect_snapshot_is_frozen(self, tmp_path: Path) -> None:
        """Verify that the SMCSnapshot is immutable."""
        data_dir = _ingest_fixture(tmp_path)

        snapshot = detect_patterns(
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        with pytest.raises(Exception):  # ValidationError from frozen model
            snapshot.trend_direction = "bullish"  # type: ignore[misc]

    def test_detect_empty_lake_raises(self, tmp_path: Path) -> None:
        """Verify that detect_patterns raises for an empty lake."""
        data_dir = tmp_path / "empty_lake"
        data_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="No data"):
            detect_patterns(
                instrument="XAUUSD",
                timeframe=Timeframe.H1,
                data_dir=data_dir,
            )
