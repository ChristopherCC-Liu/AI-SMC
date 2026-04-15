"""Fair Value Gap (FVG) detection and fill-status tracking.

Wraps ``smartmoneyconcepts.smc.fvg`` and converts results into immutable
``FairValueGap`` instances.  Fill percentage is re-evaluated each time
new bars arrive via ``update_fill_status``.

XAUUSD gap fill semantics
--------------------------
- **Bullish FVG** (gap up): the gap spans ``[FVGBottom, FVGTop]`` where
  ``FVGBottom = high[i-1]`` and ``FVGTop = low[i+1]``.  Price fills the
  gap by trading *downward* into it.  Fill % = how far price has penetrated
  from the top downward (i.e. ``(fvg.high - bar_low) / gap_size`` clamped
  to the gap boundaries).
- **Bearish FVG** (gap down): the gap spans ``[FVGBottom, FVGTop]`` where
  ``FVGBottom = high[i+1]`` and ``FVGTop = low[i-1]``.  Price fills the
  gap by trading *upward* into it.  Fill % = how far price has penetrated
  from the top downward.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from smartmoneyconcepts.smc import smc as SMC  # type: ignore[import-untyped]

from smc.data.schemas import Timeframe
from smc.smc_core._utils import ts_from_polars
from smc.smc_core.swing import _polars_to_ohlc_pandas
from smc.smc_core.types import FairValueGap

__all__ = [
    "detect_fvgs",
    "update_fill_status",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TS_COLUMN = "ts"


def _fill_pct_for_fvg(fvg: FairValueGap, bar_high: float, bar_low: float) -> float:
    """Compute the cumulative fill percentage after a single bar touches the gap.

    Parameters
    ----------
    fvg:
        The FVG being evaluated.
    bar_high, bar_low:
        OHLC high and low for the bar under inspection.

    Returns
    -------
    float
        Fill percentage in [0.0, 1.0].  Decreasing returns (once gap is
        100 % filled it stays at 1.0).
    """
    gap_size = fvg.high - fvg.low
    if gap_size <= 0:
        return 1.0

    if fvg.fvg_type == "bullish":
        # Gap filled from bottom up: price low dips into the gap
        if bar_low >= fvg.high:
            # Price hasn't entered the gap yet
            return fvg.filled_pct
        penetration = fvg.high - max(bar_low, fvg.low)
        new_pct = penetration / gap_size
    else:
        # bearish — gap filled from top down: price high climbs into gap
        if bar_high <= fvg.low:
            return fvg.filled_pct
        penetration = min(bar_high, fvg.high) - fvg.low
        new_pct = penetration / gap_size

    return min(1.0, max(fvg.filled_pct, new_pct))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_fvgs(
    df: pl.DataFrame,
    *,
    join_consecutive: bool = True,
    timeframe: Timeframe = Timeframe.H1,
) -> tuple[FairValueGap, ...]:
    """Detect Fair Value Gaps in *df* using the smartmoneyconcepts library.

    Parameters
    ----------
    df:
        Polars OHLCV DataFrame with columns: ``ts``, ``open``, ``high``,
        ``low``, ``close``.
    join_consecutive:
        When ``True`` (default), adjacent FVGs of the same type are merged
        into a single wider gap.  Forwarded directly to ``SMC.fvg``.
    timeframe:
        Timeframe label embedded in each returned ``FairValueGap``.

    Returns
    -------
    tuple[FairValueGap, ...]
        Immutable tuple of detected FVGs ordered chronologically.

    Raises
    ------
    ValueError
        If required OHLC columns are absent.
    """
    if len(df) == 0:
        return ()

    ohlc_pd = _polars_to_ohlc_pandas(df)
    fvg_df = SMC.fvg(ohlc_pd, join_consecutive=join_consecutive)

    # fvg_df columns: FVG (1=bullish, -1=bearish, NaN), Top, Bottom, MitigatedIndex
    ts_series = df[_TS_COLUMN]
    fvgs: list[FairValueGap] = []

    for i in range(len(fvg_df)):
        fvg_val = fvg_df["FVG"].iloc[i]
        if fvg_val != fvg_val:  # NaN check
            continue
        fvg_int = int(fvg_val)
        if fvg_int not in (1, -1):
            continue

        top = float(fvg_df["Top"].iloc[i])
        bottom = float(fvg_df["Bottom"].iloc[i])
        if top != top or bottom != bottom:  # NaN guard
            continue

        fvg_type: str = "bullish" if fvg_int == 1 else "bearish"
        ts = ts_from_polars(ts_series, i)

        # Check initial mitigation from library
        mit_idx_raw = fvg_df["MitigatedIndex"].iloc[i]
        initially_filled = bool(
            mit_idx_raw == mit_idx_raw and int(mit_idx_raw) > 0  # not NaN and > 0
        )

        fvgs.append(
            FairValueGap(
                ts=ts,
                high=top,
                low=bottom,
                fvg_type=fvg_type,  # type: ignore[arg-type]
                timeframe=timeframe,
                filled_pct=1.0 if initially_filled else 0.0,
                fully_filled=initially_filled,
            )
        )

    return tuple(fvgs)


def update_fill_status(
    fvgs: tuple[FairValueGap, ...],
    current_bars: pl.DataFrame,
) -> tuple[FairValueGap, ...]:
    """Re-evaluate fill percentage for each FVG against *current_bars*.

    Already fully-filled FVGs (``fully_filled=True``) are left unchanged.
    For each unfilled / partially-filled FVG the function scans
    *current_bars* bar-by-bar, accumulating fill penetration.

    Parameters
    ----------
    fvgs:
        Current tuple of fair value gaps.
    current_bars:
        Polars DataFrame with at minimum ``ts``, ``high``, ``low`` columns.

    Returns
    -------
    tuple[FairValueGap, ...]
        New immutable tuple with updated ``filled_pct`` and ``fully_filled``
        fields.  Original ``FairValueGap`` objects are never mutated.
    """
    if len(fvgs) == 0 or len(current_bars) == 0:
        return fvgs

    ts_series = current_bars[_TS_COLUMN]
    high_vals = current_bars["high"].to_list()
    low_vals = current_bars["low"].to_list()

    updated: list[FairValueGap] = []
    for fvg in fvgs:
        if fvg.fully_filled:
            updated.append(fvg)
            continue

        current_fvg = fvg
        for j in range(len(current_bars)):
            bar_ts = ts_from_polars(ts_series, j)
            # Only consider bars after the FVG formed
            if bar_ts <= current_fvg.ts:
                continue

            new_pct = _fill_pct_for_fvg(current_fvg, high_vals[j], low_vals[j])
            if new_pct != current_fvg.filled_pct:
                current_fvg = current_fvg.model_copy(
                    update={
                        "filled_pct": new_pct,
                        "fully_filled": new_pct >= 1.0,
                    }
                )
            if current_fvg.fully_filled:
                break

        updated.append(current_fvg)

    return tuple(updated)
