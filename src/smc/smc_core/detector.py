"""SMC Detector — orchestrates all sub-detectors for single and multi-timeframe analysis.

``SMCDetector`` is the primary entry point for the SMC detection layer.
It composes swing, order-block, FVG, structure, and liquidity detectors
into a single ``detect`` call that returns a frozen ``SMCSnapshot``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from smc.data.schemas import Timeframe
from smc.smc_core.fvg import detect_fvgs, update_fill_status
from smc.smc_core.liquidity import detect_liquidity_levels, detect_liquidity_sweep
from smc.smc_core.order_block import detect_order_blocks, update_mitigation
from smc.smc_core.structure import current_trend, detect_structure
from smc.smc_core.swing import detect_swings, filter_significant_swings
from smc.smc_core.types import SMCSnapshot

__all__ = ["SMCDetector"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS_COLUMN = "ts"
_DEFAULT_SWING_LENGTH = 10
_DEFAULT_MIN_SWING_POINTS = 50.0
_DEFAULT_LIQUIDITY_TOLERANCE_POINTS = 5.0


def _latest_ts(df: pl.DataFrame) -> datetime:
    """Return the most recent ``ts`` value in *df* as a timezone-aware datetime."""
    ts_raw = df[_TS_COLUMN].max()
    if ts_raw is None:
        return datetime.now(tz=timezone.utc)
    if hasattr(ts_raw, "to_pydatetime"):
        dt = ts_raw.to_pydatetime()
    elif isinstance(ts_raw, datetime):
        dt = ts_raw
    else:
        dt = datetime(ts_raw.year, ts_raw.month, ts_raw.day, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SMCDetector:
    """Orchestrates SMC sub-detectors and produces immutable ``SMCSnapshot`` objects.

    Parameters
    ----------
    swing_length:
        Number of candles on each side of a bar used to confirm swing
        highs/lows.  Forwarded to every sub-detector that needs it.
        Defaults to 10.
    min_swing_points:
        Minimum price distance (in points) for the swing noise filter.
        Defaults to 50 points ($0.50 on XAUUSD).
    liquidity_tolerance_points:
        Tolerance (in points) for clustering equal-high/low liquidity levels.
        Defaults to 5 points ($0.05 on XAUUSD).

    Examples
    --------
    >>> detector = SMCDetector()
    >>> snapshot = detector.detect(h1_df, Timeframe.H1)
    >>> multi = detector.detect_multi_tf({Timeframe.H4: h4_df, Timeframe.H1: h1_df})
    """

    def __init__(
        self,
        *,
        swing_length: int = _DEFAULT_SWING_LENGTH,
        min_swing_points: float = _DEFAULT_MIN_SWING_POINTS,
        liquidity_tolerance_points: float = _DEFAULT_LIQUIDITY_TOLERANCE_POINTS,
        swing_length_map: dict[Timeframe, int] | None = None,
    ) -> None:
        if swing_length < 1:
            raise ValueError(f"swing_length must be >= 1, got {swing_length}")
        if min_swing_points < 0:
            raise ValueError(f"min_swing_points must be >= 0, got {min_swing_points}")
        if liquidity_tolerance_points < 0:
            raise ValueError(
                f"liquidity_tolerance_points must be >= 0, got {liquidity_tolerance_points}"
            )
        if swing_length_map is not None:
            for tf, sl in swing_length_map.items():
                if sl < 1:
                    raise ValueError(
                        f"swing_length_map[{tf}] must be >= 1, got {sl}"
                    )

        self._swing_length = swing_length
        self._min_swing_points = min_swing_points
        self._liquidity_tolerance_points = liquidity_tolerance_points
        self._swing_length_map: dict[Timeframe, int] = (
            dict(swing_length_map) if swing_length_map is not None else {}
        )

    # ------------------------------------------------------------------
    # Properties (read-only)
    # ------------------------------------------------------------------

    @property
    def swing_length(self) -> int:
        """Swing confirmation window (bars on each side)."""
        return self._swing_length

    @property
    def min_swing_points(self) -> float:
        """Minimum swing separation in points."""
        return self._min_swing_points

    @property
    def liquidity_tolerance_points(self) -> float:
        """Equal-high/low clustering tolerance in points."""
        return self._liquidity_tolerance_points

    @property
    def swing_length_map(self) -> dict[Timeframe, int]:
        """Per-timeframe swing_length overrides (empty dict if none)."""
        return dict(self._swing_length_map)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, df: pl.DataFrame, timeframe: Timeframe) -> SMCSnapshot:
        """Run all sub-detectors on *df* and return a frozen ``SMCSnapshot``.

        This method is the canonical way to get a complete SMC picture for a
        single timeframe.  It:

        1. Detects and filters swing highs/lows.
        2. Detects order blocks and updates mitigation against all bars.
        3. Detects fair value gaps and updates fill status against all bars.
        4. Detects BOS/CHoCH structure breaks.
        5. Derives trend direction from structure breaks.
        6. Detects liquidity levels and evaluates sweeps against the last bar.

        Parameters
        ----------
        df:
            Polars OHLCV DataFrame.  Must contain ``ts``, ``open``, ``high``,
            ``low``, ``close``.  ``volume`` is optional.
        timeframe:
            Timeframe label to embed in all pattern objects and the snapshot.

        Returns
        -------
        SMCSnapshot
            Frozen snapshot of all detected SMC patterns at the latest ``ts``
            in *df*.

        Raises
        ------
        ValueError
            If *df* is empty or missing required columns.
        """
        if len(df) == 0:
            raise ValueError("Cannot detect SMC patterns on an empty DataFrame.")

        # Resolve per-TF swing_length (fall back to global default)
        effective_sl = self._swing_length_map.get(timeframe, self._swing_length)

        # 1. Swing points
        raw_swings = detect_swings(df, swing_length=effective_sl)
        swing_points = filter_significant_swings(
            raw_swings, min_distance_points=self._min_swing_points
        )

        # 2. Order blocks (detect then run mitigation over the full bar history)
        raw_obs = detect_order_blocks(
            df, swing_length=effective_sl, timeframe=timeframe
        )
        order_blocks = update_mitigation(raw_obs, df)

        # 3. Fair value gaps (detect then update fill status over all bars)
        raw_fvgs = detect_fvgs(df, join_consecutive=True, timeframe=timeframe)
        fvgs = update_fill_status(raw_fvgs, df)

        # 4. Structure breaks
        structure_breaks = detect_structure(
            df, swing_length=effective_sl, timeframe=timeframe
        )

        # 5. Trend direction
        trend_direction = current_trend(structure_breaks)

        # 6. Liquidity levels — detect from swings, then check last bar for sweeps
        liquidity_levels = detect_liquidity_levels(
            df, swing_points, tolerance_points=self._liquidity_tolerance_points
        )
        if len(df) > 0:
            last_bar = df[-1]
            last_high = float(last_bar["high"][0])
            last_low = float(last_bar["low"][0])
            last_ts = _latest_ts(df)
            liquidity_levels = detect_liquidity_sweep(
                liquidity_levels, last_high, last_low, last_ts
            )

        snapshot_ts = _latest_ts(df)

        return SMCSnapshot(
            ts=snapshot_ts,
            timeframe=timeframe,
            swing_points=swing_points,
            order_blocks=order_blocks,
            fvgs=fvgs,
            structure_breaks=structure_breaks,
            liquidity_levels=liquidity_levels,
            trend_direction=trend_direction,
        )

    def detect_multi_tf(
        self, data: dict[Timeframe, pl.DataFrame]
    ) -> dict[Timeframe, SMCSnapshot]:
        """Run ``detect`` across all provided timeframes.

        Parameters
        ----------
        data:
            Mapping of ``Timeframe`` → Polars OHLCV DataFrame.  Each
            DataFrame is processed independently.

        Returns
        -------
        dict[Timeframe, SMCSnapshot]
            New dict mapping each timeframe to its frozen ``SMCSnapshot``.
            The dict is constructed from scratch; no prior state is mutated.

        Raises
        ------
        ValueError
            If any constituent DataFrame is empty or malformed.
        """
        return {tf: self.detect(df, tf) for tf, df in data.items()}
