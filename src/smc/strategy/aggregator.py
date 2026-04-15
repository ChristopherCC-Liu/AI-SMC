"""Multi-timeframe strategy aggregator — the orchestration layer.

``MultiTimeframeAggregator`` ties together the full strategy pipeline:
1. Detect SMC patterns on all timeframes (D1, H4, H1, M15)
2. Compute HTF bias from D1 + H4
3. Scan H1 zones aligned with bias
4. Check M15 entries inside each zone
5. Score confluence for each setup
6. Sort by confluence score, filter by threshold

This is the single entry point for generating trade setups from raw OHLCV data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.smc_core.types import SMCSnapshot
from smc.strategy.confluence import (
    TRADEABLE_THRESHOLD,
    effective_threshold,
    score_confluence,
)
from smc.strategy.entry_trigger import check_entry
from smc.strategy.htf_bias import compute_htf_bias
from smc.strategy.regime import classify_regime
from smc.strategy.types import TradeSetup
from smc.strategy.zone_scanner import scan_zones

__all__ = ["MultiTimeframeAggregator"]


class MultiTimeframeAggregator:
    """Orchestrates the full multi-timeframe SMC strategy pipeline.

    Parameters
    ----------
    detector:
        An ``SMCDetector`` instance used to detect SMC patterns on each
        timeframe.  Callers control swing_length and other detector
        parameters via the detector's constructor.
    swing_length:
        Swing confirmation window passed to the detector.  Defaults to 10.
        Note: if a pre-configured detector is passed, this parameter is
        ignored — it's provided for convenience when constructing both.

    Examples
    --------
    >>> detector = SMCDetector(swing_length=10)
    >>> agg = MultiTimeframeAggregator(detector=detector)
    >>> setups = agg.generate_setups(data={...}, current_price=2350.0)
    """

    # Default per-TF swing_length values — D1 fastest (5), H4 medium (7),
    # H1/M15 standard (10).  Faster HTF swing detection catches trend
    # changes sooner without adding noise on lower timeframes.
    _DEFAULT_SWING_LENGTH_MAP: dict[Timeframe, int] = {
        Timeframe.D1: 5,
        Timeframe.H4: 7,
        Timeframe.H1: 10,
        Timeframe.M15: 10,
    }

    # Zone cooldown duration after a losing trade (24 hours)
    _ZONE_COOLDOWN_HOURS = 24

    def __init__(
        self,
        detector: SMCDetector,
        swing_length: int = 10,
    ) -> None:
        self._detector = detector
        self._zone_cooldowns: dict[tuple[float, float, str], datetime] = {}
        # If the detector was not created with a swing_length_map,
        # inject the default one so all aggregator pipelines benefit.
        if not detector.swing_length_map:
            self._detector = SMCDetector(
                swing_length=detector.swing_length,
                min_swing_points=detector.min_swing_points,
                liquidity_tolerance_points=detector.liquidity_tolerance_points,
                swing_length_map=self._DEFAULT_SWING_LENGTH_MAP,
            )

    @property
    def detector(self) -> SMCDetector:
        """The underlying SMC detector."""
        return self._detector

    def record_zone_loss(
        self,
        zone_high: float,
        zone_low: float,
        direction: str,
        loss_time: datetime,
    ) -> None:
        """Record a losing trade for zone cooldown tracking.

        After a zone produces a losing trade, it enters a 24-hour cooldown
        period during which no new trades will be generated from the same zone.
        """
        key = (round(zone_high, 2), round(zone_low, 2), direction)
        cooldown_until = loss_time + timedelta(hours=self._ZONE_COOLDOWN_HOURS)
        self._zone_cooldowns[key] = cooldown_until

    def clear_cooldowns(self) -> None:
        """Clear all zone cooldowns. Called between walk-forward windows."""
        self._zone_cooldowns.clear()

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
    ) -> tuple[TradeSetup, ...]:
        """Run the full strategy pipeline and return scored trade setups.

        Parameters
        ----------
        data:
            Mapping of Timeframe to Polars OHLCV DataFrame.  At minimum,
            D1, H4, H1, and M15 must be present for the full pipeline.
            Missing timeframes will be handled gracefully with reduced
            capability.
        current_price:
            The current market price for XAUUSD.

        Returns
        -------
        tuple[TradeSetup, ...]
            Trade setups sorted by confluence score descending.
            Only includes setups that meet the tradeable threshold (>= 0.45).
        """
        # Step 1: Detect SMC patterns on all available timeframes
        snapshots = self._detect_all(data)

        # Step 2: Compute HTF bias (tiered — accepts None for missing TFs)
        d1_snap = snapshots.get(Timeframe.D1)
        h4_snap = snapshots.get(Timeframe.H4)

        bias = compute_htf_bias(d1_snap, h4_snap)

        if bias.direction == "neutral":
            return ()

        # Step 2b: Regime filter — suppress Tier 2/3 in ranging markets
        regime = classify_regime(data.get(Timeframe.D1))
        is_tier1 = bias.rationale.startswith("Tier 1:")
        if regime == "ranging" and not is_tier1:
            return ()

        # Step 3: Scan H1 zones
        h1_snap = snapshots.get(Timeframe.H1)
        if h1_snap is None:
            return ()

        zones = scan_zones(h1_snap, bias)
        if not zones:
            return ()

        # Step 4: Check M15 entries for each zone
        m15_snap = snapshots.get(Timeframe.M15)
        if m15_snap is None:
            return ()

        # Sprint 4: Compute H1 ATR(14) for adaptive SL buffer
        h1_atr = self._compute_h1_atr(data.get(Timeframe.H1))

        # Determine the effective confluence threshold based on bias tier
        min_confluence = effective_threshold(bias.rationale)

        now = datetime.now(tz=timezone.utc)
        setups: list[TradeSetup] = []

        for zone in zones:
            # Zone cooldown: skip zones that recently produced a loss
            zone_key = (round(zone.zone_high, 2), round(zone.zone_low, 2), zone.direction)
            if zone_key in self._zone_cooldowns:
                cooldown_until = self._zone_cooldowns[zone_key]
                if now < cooldown_until:
                    continue

            entry = check_entry(m15_snap, zone, current_price, h1_atr=h1_atr)
            if entry is None:
                continue

            # Step 5: Score confluence with tier-gated threshold
            conf_score = score_confluence(bias, zone, entry)

            if conf_score < min_confluence:
                continue

            setups.append(
                TradeSetup(
                    entry_signal=entry,
                    bias=bias,
                    zone=zone,
                    confluence_score=conf_score,
                    generated_at=now,
                )
            )

        # Step 6: Sort by confluence score descending
        sorted_setups = sorted(setups, key=lambda s: s.confluence_score, reverse=True)
        return tuple(sorted_setups)

    @staticmethod
    def _compute_h1_atr(h1_df: pl.DataFrame | None) -> float:
        """Compute H1 ATR(14) in points from an H1 OHLCV DataFrame.

        Returns 0.0 if insufficient data, which causes the entry trigger
        to fall back to the minimum SL buffer floor.
        """
        atr_period = 14
        if h1_df is None or len(h1_df) < atr_period + 1:
            return 0.0

        high = h1_df["high"].to_list()
        low = h1_df["low"].to_list()
        close = h1_df["close"].to_list()

        tr_values: list[float] = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr_values.append(max(hl, hc, lc))

        if len(tr_values) < atr_period:
            return 0.0

        atr_price = sum(tr_values[-atr_period:]) / atr_period
        # Convert from price units to points (1 point = $0.01)
        from smc.smc_core.constants import XAUUSD_POINT_SIZE
        return atr_price / XAUUSD_POINT_SIZE

    def _detect_all(
        self,
        data: dict[Timeframe, pl.DataFrame],
    ) -> dict[Timeframe, SMCSnapshot]:
        """Run detection on all provided timeframes.

        Skips empty DataFrames gracefully rather than raising.
        """
        snapshots: dict[Timeframe, SMCSnapshot] = {}
        for tf, df in data.items():
            if len(df) == 0:
                continue
            snapshots[tf] = self._detector.detect(df, tf)
        return snapshots
