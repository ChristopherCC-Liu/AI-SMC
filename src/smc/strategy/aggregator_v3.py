"""Hybrid v3 aggregator — v1 strict pipeline + fvg_sweep_continuation bolt-on.

Sprint 8 finding: v1 (PF 1.66) is the strongest base. v2 FAILED because it
relaxed everything. But ``fvg_sweep_continuation`` (50% WR, PF 3.8, +$178)
is a proven signal from v2 worth incorporating.

Hybrid v3 = v1 strict base + fvg_sweep_continuation. Nothing else changes.

Inherits ALL v1 behavior:
- D1 + H4 bias (``compute_htf_bias``)
- 0.45 confluence threshold
- 24h zone cooldown
- Max 3 zones (from ``RegimeParams.max_concurrent``)
- Regime filter
- ``check_entry`` (v1) for normal triggers

ONLY adds ``fvg_sweep_continuation`` as an additional trigger type checked
AFTER the v1 pipeline completes, on zones that v1 didn't already match.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from smc.data.schemas import Timeframe
from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.detector import SMCDetector
from smc.smc_core.types import SMCSnapshot
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.confluence import score_confluence
from smc.strategy.entry_trigger import (
    _compute_sl,
    _find_next_liquidity_level,
    _grade_entry,
    _SL_ATR_MULTIPLIER,
    _TP2_RR_RATIO,
)
from smc.strategy.entry_v2 import _find_fvg_sweep_continuation
from smc.strategy.htf_bias import compute_htf_bias
from smc.strategy.types import EntrySignal, TradeSetup, TradeZone
from smc.strategy.zone_scanner import scan_zones

__all__ = ["AggregatorV3"]

# Default ob_breakout flag — disabled, not enough data (3 trades in Sprint 8)
_OB_BREAKOUT_DEFAULT = False


class AggregatorV3(MultiTimeframeAggregator):
    """Hybrid v3: v1 strict pipeline + fvg_sweep_continuation bolt-on.

    Inherits ALL v1 behavior. Only adds ``fvg_sweep_continuation`` as an
    additional trigger type after normal v1 signals are checked.

    Parameters
    ----------
    detector:
        An ``SMCDetector`` instance used to detect SMC patterns on each
        timeframe.
    swing_length:
        Swing confirmation window passed to the detector.
    enable_ob_breakout:
        Enable ob_breakout trigger (default False — only 3 trades in data).
    enable_fvg_sweep:
        Enable fvg_sweep_continuation trigger (default True — proven signal).
    ai_regime_enabled:
        Enable AI regime classification for parameter routing.
    regime_cache:
        Optional regime cache for backtest mode.
    """

    def __init__(
        self,
        detector: SMCDetector,
        swing_length: int = 10,
        *,
        enable_ob_breakout: bool = _OB_BREAKOUT_DEFAULT,
        enable_fvg_sweep: bool = True,
        enable_ob_test_trigger: bool = False,
        ai_regime_enabled: bool = False,
        regime_cache: object | None = None,
    ) -> None:
        super().__init__(
            detector=detector,
            swing_length=swing_length,
            enable_ob_test_trigger=enable_ob_test_trigger,
            ai_regime_enabled=ai_regime_enabled,
            regime_cache=regime_cache,
        )
        self._enable_ob_breakout = enable_ob_breakout
        self._enable_fvg_sweep = enable_fvg_sweep

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None = None,
    ) -> tuple[TradeSetup, ...]:
        """Run v1 pipeline + fvg_sweep bolt-on.

        Steps 1-6: Run v1 pipeline EXACTLY (via super()).
        Step 7: ADDITIONALLY check fvg_sweep_continuation on zones
                that v1 didn't already produce a setup for.
        Combine, sort by confluence, cap at max_concurrent.

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
        """
        # Step 1-6: Run v1 pipeline EXACTLY
        v1_setups = super().generate_setups(data, current_price, bar_ts=bar_ts)

        if not self._enable_fvg_sweep:
            return v1_setups

        # Step 7: Check fvg_sweep_continuation on remaining zones
        fvg_sweep_setups = self._check_fvg_sweeps(
            data, current_price, bar_ts, v1_setups,
        )

        if not fvg_sweep_setups:
            return v1_setups

        # Combine: v1 setups + fvg_sweep setups
        all_setups = list(v1_setups) + fvg_sweep_setups

        # Sort by confluence score descending
        sorted_setups = sorted(
            all_setups, key=lambda s: s.confluence_score, reverse=True,
        )

        # Cap at max_concurrent — use v1's regime params for the cap
        max_concurrent = self._resolve_max_concurrent(data, bar_ts)
        return tuple(sorted_setups[:max_concurrent])

    def _check_fvg_sweeps(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None,
        v1_setups: tuple[TradeSetup, ...],
    ) -> list[TradeSetup]:
        """Check fvg_sweep_continuation on zones not already matched by v1.

        Re-runs the bias/zone detection from v1 but only looks for the
        fvg_sweep trigger on zones that v1 didn't produce setups for.
        """
        # Re-derive bias and zones (same as v1 pipeline steps 1-3)
        snapshots = self._detect_all(data)
        d1_snap = snapshots.get(Timeframe.D1)
        h4_snap = snapshots.get(Timeframe.H4)

        bias = compute_htf_bias(d1_snap, h4_snap, d1_df=data.get(Timeframe.D1))
        if bias.direction == "neutral":
            return []

        h1_snap = snapshots.get(Timeframe.H1)
        if h1_snap is None:
            return []

        zones = scan_zones(h1_snap, bias)
        if not zones:
            return []

        m15_snap = snapshots.get(Timeframe.M15)
        if m15_snap is None:
            return []

        # Build set of zone keys already covered by v1 setups
        v1_zone_keys: set[tuple[float, float, str]] = set()
        for setup in v1_setups:
            z = setup.zone
            v1_zone_keys.add((round(z.zone_high, 2), round(z.zone_low, 2), z.direction))

        # Also skip cooldown/active zones (same as v1)
        now = datetime.now(tz=timezone.utc)
        h1_atr = self._compute_h1_atr(data.get(Timeframe.H1))

        # Import regime params for confluence floor and SL/TP params
        from smc.ai.regime_classifier import classify_regime_ai
        from smc.strategy.confluence import effective_threshold

        ai_assessment = classify_regime_ai(
            d1_df=data.get(Timeframe.D1),
            h4_df=data.get(Timeframe.H4),
            ai_enabled=self._ai_regime_enabled,
            cache=self._regime_cache,
            cache_ts=bar_ts,
        )
        regime_params = ai_assessment.param_preset

        tier_floor = effective_threshold(bias.rationale)
        min_confluence = max(tier_floor, regime_params.confluence_floor)

        setups: list[TradeSetup] = []

        for zone in zones:
            zone_key = (round(zone.zone_high, 2), round(zone.zone_low, 2), zone.direction)

            # Skip zones already matched by v1
            if zone_key in v1_zone_keys:
                continue

            # Zone cooldown check
            if zone_key in self._zone_cooldowns:
                cooldown_until = self._zone_cooldowns[zone_key]
                if now < cooldown_until:
                    continue

            # Zone anti-clustering
            if zone_key in self._active_zones:
                continue

            # Check fvg_sweep_continuation
            if not _find_fvg_sweep_continuation(m15_snap, zone, current_price):
                continue

            # Trigger type filter — fvg_sweep must be in regime-permitted triggers
            if "fvg_sweep_continuation" not in regime_params.allowed_triggers:
                continue

            # Build EntrySignal for fvg_sweep using v1 SL/TP logic
            entry = self._build_fvg_sweep_entry(
                zone, current_price, m15_snap, h1_atr,
                sl_atr_multiplier=regime_params.sl_atr_multiplier,
                tp1_rr=regime_params.tp1_rr,
            )
            if entry is None:
                continue

            # Score confluence using v1 scorer
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

        return setups

    @staticmethod
    def _build_fvg_sweep_entry(
        zone: TradeZone,
        current_price: float,
        m15_snap: SMCSnapshot,
        h1_atr: float,
        *,
        sl_atr_multiplier: float = _SL_ATR_MULTIPLIER,
        tp1_rr: float = 2.5,
    ) -> EntrySignal | None:
        """Build an EntrySignal for fvg_sweep_continuation using v1 SL/TP logic.

        FVG sweep continuation trades in the SWEEP direction:
        - Bullish FVG fully filled (swept down) -> SHORT continuation
        - Bearish FVG fully filled (swept up) -> LONG continuation

        The direction is derived from the zone's direction since zone_scanner
        already aligns zones with bias.
        """
        from smc.instruments import get_instrument_config
        _cfg = get_instrument_config("XAUUSD")
        entry_price = current_price
        stop_loss = _compute_sl(zone, h1_atr, _cfg, price=current_price)
        risk_points = abs(entry_price - stop_loss) / _cfg.point_size

        if risk_points == 0:
            return None

        # TP1 at configured RR ratio
        reward_1 = risk_points * tp1_rr
        if zone.direction == "long":
            tp1 = entry_price + reward_1 * _cfg.point_size
        else:
            tp1 = entry_price - reward_1 * _cfg.point_size

        # TP2 at next liquidity level or fallback to 1:4 RR
        tp2_liq = _find_next_liquidity_level(m15_snap, zone, current_price)
        if tp2_liq is not None:
            tp2 = tp2_liq
        else:
            reward_2_points = risk_points * _TP2_RR_RATIO
            if zone.direction == "long":
                tp2 = entry_price + reward_2_points * _cfg.point_size
            else:
                tp2 = entry_price - reward_2_points * _cfg.point_size

        rr_ratio = reward_1 / risk_points if risk_points > 0 else 0.0
        grade = _grade_entry("fvg_sweep_continuation", zone, rr_ratio)

        return EntrySignal(
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            risk_points=round(risk_points, 1),
            reward_points=round(reward_1, 1),
            rr_ratio=round(rr_ratio, 2),
            trigger_type="fvg_sweep_continuation",
            direction=zone.direction,
            grade=grade,  # type: ignore[arg-type]
        )

    def _resolve_max_concurrent(
        self,
        data: dict[Timeframe, pl.DataFrame],
        bar_ts: datetime | None,
    ) -> int:
        """Resolve max_concurrent from regime params (same as v1)."""
        from smc.ai.regime_classifier import classify_regime_ai

        ai_assessment = classify_regime_ai(
            d1_df=data.get(Timeframe.D1),
            h4_df=data.get(Timeframe.H4),
            ai_enabled=self._ai_regime_enabled,
            cache=self._regime_cache,
            cache_ts=bar_ts,
        )
        return ai_assessment.param_preset.max_concurrent
