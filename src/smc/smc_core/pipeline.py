"""Convenience pipeline functions for the SMC detection layer.

Provides ``detect_patterns()`` — a one-call function that chains
:class:`ForexDataLake` query with :class:`SMCDetector` detection.

Usage::

    from datetime import datetime, timezone
    from pathlib import Path
    from smc.data.schemas import Timeframe
    from smc.smc_core.pipeline import detect_patterns

    snapshot = detect_patterns(
        instrument="XAUUSD",
        timeframe=Timeframe.H1,
        date=datetime(2024, 6, 15, tzinfo=timezone.utc),
        data_dir=Path("data/lake"),
    )
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.smc_core.types import SMCSnapshot

# Number of bars to look back for detection context
_DEFAULT_LOOKBACK_BARS = 500

# Approximate bar duration per timeframe for lookback calculation
_BAR_DURATION: dict[Timeframe, timedelta] = {
    Timeframe.D1: timedelta(days=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M1: timedelta(minutes=1),
}


def detect_patterns(
    *,
    instrument: str,
    timeframe: Timeframe,
    date: datetime | None = None,
    data_dir: Path,
    lookback_bars: int = _DEFAULT_LOOKBACK_BARS,
) -> SMCSnapshot:
    """Query the data lake and run SMC detection in one call.

    Pipeline steps:
    1. Query :class:`ForexDataLake` for recent bars up to *date*.
    2. Run :class:`SMCDetector.detect` on the returned DataFrame.
    3. Return the frozen :class:`SMCSnapshot`.

    Args:
        instrument: Symbol name, e.g. ``"XAUUSD"``.
        timeframe: Timeframe to detect on.
        date: Upper bound for the query window.  Defaults to the latest
            available data in the lake.
        data_dir: Root of the data lake.
        lookback_bars: Number of bars to look back from *date* for
            detection context.  Defaults to 500.

    Returns:
        Frozen :class:`SMCSnapshot` with all detected patterns.

    Raises:
        ValueError: if no data is found in the lake for the given parameters.
    """
    lake = ForexDataLake(data_dir)

    # Determine the end timestamp
    if date is not None:
        if date.tzinfo is None:
            end = date.replace(tzinfo=timezone.utc)
        else:
            end = date.astimezone(timezone.utc)
    else:
        # Use the latest available data
        available = lake.available_range(instrument, timeframe)
        if available is None:
            raise ValueError(
                f"No data in lake for {instrument}/{timeframe}. "
                f"Run 'smc ingest' first."
            )
        end = available[1] + timedelta(seconds=1)  # inclusive of last bar

    # Calculate start based on lookback
    bar_duration = _BAR_DURATION.get(timeframe, timedelta(hours=1))
    start = end - (bar_duration * lookback_bars)

    df = lake.query(instrument, timeframe, start=start, end=end)

    if df.is_empty():
        raise ValueError(
            f"No data found for {instrument}/{timeframe} "
            f"in range [{start.isoformat()}, {end.isoformat()})."
        )

    detector = SMCDetector()
    return detector.detect(df, timeframe)


__all__ = ["detect_patterns"]
