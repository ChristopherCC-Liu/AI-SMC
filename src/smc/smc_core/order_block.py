"""Order Block detection and mitigation tracking for the AI-SMC system.

Wraps ``smartmoneyconcepts.smc.ob`` and converts results into immutable
``OrderBlock`` instances.  Mitigation is tracked by comparing live price
action against the OB zone boundaries.

XAUUSD: Bullish OB is mitigated when a close or low trades below its low.
        Bearish OB is mitigated when a close or high trades above its high.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from smartmoneyconcepts.smc import smc as SMC  # type: ignore[import-untyped]

from smc.data.schemas import Timeframe
from smc.smc_core._utils import ts_from_polars
from smc.smc_core.swing import _polars_to_ohlc_pandas
from smc.smc_core.types import OrderBlock

__all__ = [
    "detect_order_blocks",
    "update_mitigation",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TS_COLUMN = "ts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_order_blocks(
    df: pl.DataFrame,
    *,
    swing_length: int = 10,
    timeframe: Timeframe = Timeframe.H1,
) -> tuple[OrderBlock, ...]:
    """Detect bullish and bearish order blocks in *df*.

    The underlying library requires a prior call to ``swing_highs_lows`` so
    both calls are made internally.  Results are returned as an immutable
    tuple of frozen ``OrderBlock`` instances.

    Parameters
    ----------
    df:
        Polars OHLCV DataFrame with columns: ``ts``, ``open``, ``high``,
        ``low``, ``close``, optionally ``volume``.
    swing_length:
        Swing detection window forwarded to ``SMC.swing_highs_lows``.
    timeframe:
        Timeframe label to embed in each ``OrderBlock``.

    Returns
    -------
    tuple[OrderBlock, ...]
        Immutable tuple of order blocks, ordered chronologically by
        ``ts_start``.

    Raises
    ------
    ValueError
        If required OHLC columns are missing.
    """
    if len(df) == 0:
        return ()

    ohlc_pd = _polars_to_ohlc_pandas(df)
    shl_df = SMC.swing_highs_lows(ohlc_pd, swing_length=swing_length)
    ob_df = SMC.ob(ohlc_pd, shl_df)

    # ob_df columns: OB (1=bullish, -1=bearish, 0=none), Top, Bottom,
    #                OBVolume, MitigatedIndex, Percentage
    ts_series = df[_TS_COLUMN]
    obs: list[OrderBlock] = []

    for i in range(len(ob_df)):
        ob_val = ob_df["OB"].iloc[i]
        # OB column contains 0, 1, -1 as integers.  Guard against NaN first.
        if ob_val != ob_val:  # NaN check
            continue
        if int(ob_val) == 0:
            continue
        top = float(ob_df["Top"].iloc[i])
        bottom = float(ob_df["Bottom"].iloc[i])
        if top == 0.0 and bottom == 0.0:
            continue

        ob_type: str = "bullish" if int(ob_val) == 1 else "bearish"
        ts_start = ts_from_polars(ts_series, i)

        # ts_end: use the next bar's timestamp if available, else ts_start
        ts_end = ts_from_polars(ts_series, min(i + 1, len(ts_series) - 1))

        # Mitigation: MitigatedIndex > 0 means it was mitigated at that index
        mitigated_index = ob_df["MitigatedIndex"].iloc[i]
        mitigated = bool(mitigated_index > 0)
        mitigated_at: datetime | None = None
        if mitigated:
            mit_idx = int(mitigated_index)
            if mit_idx < len(ts_series):
                mitigated_at = ts_from_polars(ts_series, mit_idx)

        obs.append(
            OrderBlock(
                ts_start=ts_start,
                ts_end=ts_end,
                high=top,
                low=bottom,
                ob_type=ob_type,  # type: ignore[arg-type]
                timeframe=timeframe,
                mitigated=mitigated,
                mitigated_at=mitigated_at,
            )
        )

    return tuple(obs)


def update_mitigation(
    obs: tuple[OrderBlock, ...],
    current_bars: pl.DataFrame,
) -> tuple[OrderBlock, ...]:
    """Return a new tuple with mitigation flags updated against *current_bars*.

    Already-mitigated order blocks are left unchanged.  For each unmitigated
    OB the function scans *current_bars* in chronological order and marks
    the first bar that breaches the zone.

    Mitigation rules for XAUUSD
    ---------------------------
    - **Bullish OB**: mitigated when ``low < ob.low`` (price trades below
      the order block's lower boundary).
    - **Bearish OB**: mitigated when ``high > ob.high`` (price trades above
      the order block's upper boundary).

    Parameters
    ----------
    obs:
        Current tuple of order blocks.
    current_bars:
        Polars DataFrame with at minimum ``ts``, ``high``, ``low`` columns.

    Returns
    -------
    tuple[OrderBlock, ...]
        New immutable tuple with updated mitigation state.  Original objects
        are never mutated.
    """
    if len(obs) == 0 or len(current_bars) == 0:
        return obs

    ts_series = current_bars[_TS_COLUMN]
    high_vals = current_bars["high"].to_list()
    low_vals = current_bars["low"].to_list()

    updated: list[OrderBlock] = []
    for ob in obs:
        if ob.mitigated:
            updated.append(ob)
            continue

        mitigated = False
        mitigated_at: datetime | None = None

        for j in range(len(current_bars)):
            bar_high = high_vals[j]
            bar_low = low_vals[j]
            bar_ts = ts_from_polars(ts_series, j)

            # Skip bars that predate or coincide with the OB formation
            if bar_ts <= ob.ts_start:
                continue

            if ob.ob_type == "bullish" and bar_low < ob.low:
                mitigated = True
                mitigated_at = bar_ts
                break
            if ob.ob_type == "bearish" and bar_high > ob.high:
                mitigated = True
                mitigated_at = bar_ts
                break

        if mitigated:
            # Pydantic frozen models require model_copy to produce a new instance
            updated.append(ob.model_copy(update={"mitigated": True, "mitigated_at": mitigated_at}))
        else:
            updated.append(ob)

    return tuple(updated)
