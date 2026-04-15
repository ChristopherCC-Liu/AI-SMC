"""Swing high/low detection for XAUUSD SMC analysis.

Wraps ``smartmoneyconcepts.smc.swing_highs_lows`` and converts the result
into immutable ``SwingPoint`` instances.  All intermediate operations use
immutable patterns — the library call requires Pandas but the public API
accepts and returns Polars / frozen data structures only.

XAUUSD point convention: 1 point = $0.01 (5-digit pricing). See constants.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from smartmoneyconcepts.smc import smc as SMC  # type: ignore[import-untyped]

from smc.smc_core._utils import to_aware_utc
from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import SwingPoint

__all__ = [
    "detect_swings",
    "filter_significant_swings",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TS_COLUMN = "ts"
_HIGH_COLUMN = "high"
_LOW_COLUMN = "low"


def _polars_to_ohlc_pandas(df: pl.DataFrame):  # type: ignore[return]
    """Convert a Polars OHLCV frame to the Pandas format expected by the SMC lib.

    The library requires lowercase column names: open, high, low, close.
    If the frame has a ``volume`` column it is forwarded; otherwise a zero
    column is synthesised so the library does not error on missing keys.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required OHLC columns: {sorted(missing)}")

    select_cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        select_cols.append("volume")

    pandas_df = df.select(select_cols).to_pandas()

    if "volume" not in pandas_df.columns:
        pandas_df["volume"] = 0.0

    return pandas_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_swings(
    df: pl.DataFrame,
    *,
    swing_length: int = 10,
) -> tuple[SwingPoint, ...]:
    """Detect swing highs and lows in *df* using the smartmoneyconcepts library.

    Parameters
    ----------
    df:
        Polars OHLCV DataFrame.  Must contain columns: ``ts``, ``open``,
        ``high``, ``low``, ``close``.  The ``ts`` column must be
        ``Datetime`` or ``Date`` dtype.
    swing_length:
        Number of candles on each side of a bar required to confirm a swing.
        Passed directly to ``SMC.swing_highs_lows``.

    Returns
    -------
    tuple[SwingPoint, ...]
        Immutable tuple of confirmed swing points ordered by timestamp.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    if len(df) == 0:
        return ()

    ohlc_pd = _polars_to_ohlc_pandas(df)
    shl_df = SMC.swing_highs_lows(ohlc_pd, swing_length=swing_length)

    # shl_df has same index as ohlc_pd with columns: HighLow, Level
    # HighLow == 1 → swing high, -1 → swing low, NaN → not a swing
    ts_series = df[_TS_COLUMN]

    swings: list[SwingPoint] = []
    for i in range(len(shl_df)):
        hl_val = shl_df["HighLow"].iloc[i]
        if hl_val != hl_val:  # NaN check (float NaN)
            continue
        hl_int = int(hl_val)
        if hl_int not in (1, -1):
            continue

        level = float(shl_df["Level"].iloc[i])
        ts_raw = ts_series[i]

        # Normalise to timezone-aware datetime
        if hasattr(ts_raw, "to_pydatetime"):
            ts_dt: datetime = ts_raw.to_pydatetime()
        elif isinstance(ts_raw, datetime):
            ts_dt = ts_raw
        else:
            # Polars may return a date-like scalar
            ts_dt = datetime(ts_raw.year, ts_raw.month, ts_raw.day, tzinfo=timezone.utc)

        ts_dt = to_aware_utc(ts_dt)
        swing_type: str = "high" if hl_int == 1 else "low"

        swings.append(
            SwingPoint(
                ts=ts_dt,
                price=level,
                swing_type=swing_type,  # type: ignore[arg-type]
                strength=swing_length,
            )
        )

    return tuple(swings)


def filter_significant_swings(
    swings: tuple[SwingPoint, ...],
    *,
    min_distance_points: float = 50.0,
) -> tuple[SwingPoint, ...]:
    """Remove swing points that are too close together in price.

    For XAUUSD, 1 point = $0.01, so ``min_distance_points=50`` translates to
    $0.50.  When two consecutive swings of the same type are closer than
    ``min_distance_points``, the weaker one (lower ``strength``) is dropped.
    If both have equal strength the later one is kept to reflect the more
    recent market structure.

    Parameters
    ----------
    swings:
        Input swing points (typically from ``detect_swings``).
    min_distance_points:
        Minimum price distance in points.  Values below this threshold are
        considered noise and dropped.  Defaults to 50 points ($0.50 on
        XAUUSD).

    Returns
    -------
    tuple[SwingPoint, ...]
        Filtered, immutable tuple of swing points.
    """
    if len(swings) < 2:
        return swings

    min_price_distance: float = min_distance_points * XAUUSD_POINT_SIZE

    # Work on a list; build a new filtered list without mutating inputs.
    candidates: list[SwingPoint] = list(swings)
    changed = True

    while changed:
        changed = False
        survivors: list[SwingPoint] = []
        i = 0
        while i < len(candidates):
            if i + 1 < len(candidates):
                a = candidates[i]
                b = candidates[i + 1]
                if a.swing_type == b.swing_type:
                    distance = abs(a.price - b.price)
                    if distance < min_price_distance:
                        # Keep the stronger / more-recent one
                        if a.strength > b.strength:
                            survivors.append(a)
                        else:
                            survivors.append(b)
                        i += 2
                        changed = True
                        continue
            survivors.append(candidates[i])
            i += 1
        candidates = survivors

    return tuple(candidates)
