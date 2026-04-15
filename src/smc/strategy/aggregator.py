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

from datetime import datetime, timezone

import polars as pl

from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.smc_core.types import SMCSnapshot
from smc.strategy.confluence import TRADEABLE_THRESHOLD, score_confluence
from smc.strategy.entry_trigger import check_entry
from smc.strategy.htf_bias import compute_htf_bias
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

    def __init__(
        self,
        detector: SMCDetector,
        swing_length: int = 10,
    ) -> None:
        self._detector = detector

    @property
    def detector(self) -> SMCDetector:
        """The underlying SMC detector."""
        return self._detector

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
            Only includes setups that meet the tradeable threshold (>= 0.6).
        """
        # Step 1: Detect SMC patterns on all available timeframes
        snapshots = self._detect_all(data)

        # Step 2: Compute HTF bias (requires D1 and H4)
        d1_snap = snapshots.get(Timeframe.D1)
        h4_snap = snapshots.get(Timeframe.H4)

        if d1_snap is None or h4_snap is None:
            return ()

        bias = compute_htf_bias(d1_snap, h4_snap)

        if bias.direction == "neutral":
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

        now = datetime.now(tz=timezone.utc)
        setups: list[TradeSetup] = []

        for zone in zones:
            entry = check_entry(m15_snap, zone, current_price)
            if entry is None:
                continue

            # Step 5: Score confluence
            conf_score = score_confluence(bias, zone, entry)

            if conf_score < TRADEABLE_THRESHOLD:
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
