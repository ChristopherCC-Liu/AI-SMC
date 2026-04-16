"""v1.1 aggregator — v1 strict pipeline + fvg_fill SHORT direction inversion.

3D scan finding: ``fvg_fill_in_zone SHORT`` has 33.4% WR across 82 trades
in 5 gate versions (STABLE_BAD). Inverting to LONG in non-trending regimes
captures the 66.6% theoretical inverted WR.

Inherits ALL v1 behavior unchanged:
- D1 + H4 bias (``compute_htf_bias``)
- 0.45 confluence threshold
- 24h zone cooldown
- Max 3 zones (from ``RegimeParams.max_concurrent``)
- Regime filter
- ``check_entry`` (v1) for all triggers

ONLY change: post-process v1 output to flip ``fvg_fill_in_zone`` SHORT
setups to LONG when the D1 regime is NOT trending.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from smc.data.schemas import Timeframe
from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.regime import classify_regime
from smc.strategy.types import EntrySignal, TradeSetup

__all__ = ["AggregatorV1_1"]


class AggregatorV1_1(MultiTimeframeAggregator):
    """v1.1: v1 strict pipeline + fvg_fill SHORT direction inversion.

    3D scan finding: fvg_fill_in_zone SHORT has 33.4% WR across 82 trades
    in 5 gate versions (STABLE_BAD). Inverting to LONG in non-trending
    regimes captures the 66.6% theoretical inverted WR.

    The inversion is a pure post-processing step:
    1. Run v1 pipeline EXACTLY (via super())
    2. Classify D1 regime
    3. If regime != "trending", flip any fvg_fill_in_zone SHORT setups to LONG

    No new trades are added or removed. The setup count is identical to v1.
    Only direction, SL, and TP are mirrored on qualifying setups.
    """

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None = None,
    ) -> tuple[TradeSetup, ...]:
        """Run v1 pipeline + post-process fvg_fill SHORT inversion.

        Parameters
        ----------
        data:
            Mapping of Timeframe to Polars OHLCV DataFrame.
        current_price:
            The current market price for XAUUSD.
        bar_ts:
            Current bar timestamp for regime cache lookup.

        Returns
        -------
        tuple[TradeSetup, ...]
            Trade setups sorted by confluence score descending.
            Identical count to v1 — only direction may differ.
        """
        # Step 1: Run v1 pipeline EXACTLY
        v1_setups = super().generate_setups(data, current_price, bar_ts=bar_ts)

        if not v1_setups:
            return v1_setups

        # Step 2: Classify D1 regime for inversion decision
        regime = classify_regime(data.get(Timeframe.D1))

        # Step 3: In trending regime, no inversion — return v1 as-is
        if regime == "trending":
            return v1_setups

        # Step 4: Post-process — flip fvg_fill_in_zone SHORT setups to LONG
        processed = tuple(
            self._flip_setup(setup)
            if (
                setup.entry_signal.trigger_type == "fvg_fill_in_zone"
                and setup.entry_signal.direction == "short"
            )
            else setup
            for setup in v1_setups
        )

        return processed

    @staticmethod
    def _flip_setup(setup: TradeSetup) -> TradeSetup:
        """Flip a SHORT fvg_fill setup to LONG: mirror SL/TP distances.

        For a SHORT setup with:
          entry=2350, SL=2355 (above), TP1=2340 (below), TP2=2330 (below)
        Flipped LONG:
          entry=2350, SL=2345 (below), TP1=2360 (above), TP2=2370 (above)

        SL/TP distances are preserved symmetrically. RR ratio stays the same.
        """
        entry = setup.entry_signal

        sl_distance = abs(entry.stop_loss - entry.entry_price)
        tp1_distance = abs(entry.entry_price - entry.take_profit_1)
        tp2_distance = abs(entry.entry_price - entry.take_profit_2)

        flipped_entry = EntrySignal(
            entry_price=entry.entry_price,
            direction="long",
            stop_loss=round(entry.entry_price - sl_distance, 2),
            take_profit_1=round(entry.entry_price + tp1_distance, 2),
            take_profit_2=round(entry.entry_price + tp2_distance, 2),
            risk_points=entry.risk_points,
            reward_points=entry.reward_points,
            rr_ratio=entry.rr_ratio,
            trigger_type=entry.trigger_type,
            grade=entry.grade,
        )

        # Flip the zone direction too (short -> long) so downstream consumers
        # see a consistent long setup
        flipped_zone = setup.zone.model_copy(update={"direction": "long"})

        return setup.model_copy(
            update={
                "entry_signal": flipped_entry,
                "zone": flipped_zone,
            }
        )
