"""Market structure detection: Break of Structure (BOS) and Change of Character (CHoCH).

Wraps ``smartmoneyconcepts.smc.bos_choch`` and converts output into
immutable ``StructureBreak`` instances.  A ``current_trend`` helper derives
the prevailing market bias from recent structure breaks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import polars as pl

from smartmoneyconcepts.smc import smc as SMC  # type: ignore[import-untyped]

from smc.data.schemas import Timeframe
from smc.smc_core._utils import ts_from_polars
from smc.smc_core.swing import _polars_to_ohlc_pandas
from smc.smc_core.types import StructureBreak

__all__ = [
    "detect_structure",
    "current_trend",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TS_COLUMN = "ts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_structure(
    df: pl.DataFrame,
    *,
    swing_length: int = 10,
    timeframe: Timeframe = Timeframe.H1,
) -> tuple[StructureBreak, ...]:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH) events.

    The library only returns *confirmed* breaks (the candle that actually
    broke through the level is known), so every ``StructureBreak`` has a
    valid ``ts`` that corresponds to the candle at which the break occurred.

    Parameters
    ----------
    df:
        Polars OHLCV DataFrame with columns: ``ts``, ``open``, ``high``,
        ``low``, ``close``.
    swing_length:
        Window passed to the underlying ``SMC.swing_highs_lows`` call.
    timeframe:
        Timeframe label embedded in each ``StructureBreak``.

    Returns
    -------
    tuple[StructureBreak, ...]
        Immutable tuple of confirmed structure breaks, ordered by timestamp.

    Raises
    ------
    ValueError
        If required OHLC columns are absent.
    """
    if len(df) == 0:
        return ()

    ohlc_pd = _polars_to_ohlc_pandas(df)
    shl_df = SMC.swing_highs_lows(ohlc_pd, swing_length=swing_length)
    bc_df = SMC.bos_choch(ohlc_pd, shl_df)

    # bc_df columns: BOS, CHOCH, Level, BrokenIndex
    # BOS / CHOCH: 1=bullish, -1=bearish, NaN=none
    # Level: price at which the structure level was set
    # BrokenIndex: bar index where the break was confirmed
    ts_series = df[_TS_COLUMN]
    breaks: list[StructureBreak] = []

    for i in range(len(bc_df)):
        bos_val = bc_df["BOS"].iloc[i]
        choch_val = bc_df["CHOCH"].iloc[i]

        # Determine which (if any) break type applies
        is_bos = bos_val == bos_val and int(bos_val) != 0  # not NaN and non-zero
        is_choch = choch_val == choch_val and int(choch_val) != 0

        if not is_bos and not is_choch:
            continue

        break_type: str = "bos" if is_bos else "choch"
        raw_direction = int(bos_val) if is_bos else int(choch_val)
        direction: str = "bullish" if raw_direction == 1 else "bearish"

        level = float(bc_df["Level"].iloc[i])
        if level != level:  # NaN guard
            continue

        # Use BrokenIndex (the confirming candle) as the event timestamp
        broken_idx_raw = bc_df["BrokenIndex"].iloc[i]
        if broken_idx_raw == broken_idx_raw and int(broken_idx_raw) > 0:
            broken_idx = int(broken_idx_raw)
            if broken_idx < len(ts_series):
                ts = ts_from_polars(ts_series, broken_idx)
            else:
                ts = ts_from_polars(ts_series, i)
        else:
            ts = ts_from_polars(ts_series, i)

        breaks.append(
            StructureBreak(
                ts=ts,
                price=level,
                break_type=break_type,  # type: ignore[arg-type]
                direction=direction,  # type: ignore[arg-type]
                timeframe=timeframe,
            )
        )

    # Sort by timestamp to ensure chronological order
    return tuple(sorted(breaks, key=lambda b: b.ts))


def current_trend(
    breaks: tuple[StructureBreak, ...],
) -> Literal["bullish", "bearish", "ranging"]:
    """Derive the current market trend from recent structure breaks.

    Logic
    -----
    1. If there are no confirmed breaks → **ranging**.
    2. If the most recent break is a CHoCH → a reversal just occurred; return
       the CHoCH direction.
    3. If the last 2 or more confirmed breaks are BOS in the same direction →
       **trending** in that direction.
    4. Mixed or single BOS → return the direction of the single most-recent
       break; if the recent breaks alternate direction → **ranging**.

    Parameters
    ----------
    breaks:
        Tuple of structure breaks ordered chronologically (oldest first).
        Typically the output of ``detect_structure``.

    Returns
    -------
    Literal["bullish", "bearish", "ranging"]
    """
    if not breaks:
        return "ranging"

    # Work from the most recent break backwards
    recent = list(reversed(breaks))

    # Rule 2: most recent is CHoCH → immediate reversal signal
    if recent[0].break_type == "choch":
        return recent[0].direction  # type: ignore[return-value]

    # Gather only BOS breaks from the most recent
    bos_recent = [b for b in recent if b.break_type == "bos"]
    if not bos_recent:
        return "ranging"

    # Rule 3: last 2+ BOS are the same direction → confirmed trend
    if len(bos_recent) >= 2:
        last_two_directions = {b.direction for b in bos_recent[:2]}
        if len(last_two_directions) == 1:
            return bos_recent[0].direction  # type: ignore[return-value]
        # Mixed directions → ranging
        return "ranging"

    # Single BOS — return its direction
    return bos_recent[0].direction  # type: ignore[return-value]
