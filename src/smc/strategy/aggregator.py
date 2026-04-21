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

from smc.ai.models import AIRegimeAssessment, RegimeParams
from smc.ai.param_router import route as route_regime_params
from smc.ai.regime_classifier import classify_regime_ai, extract_regime_context
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
from smc.strategy.sl_fitness_judge import judge_sl_fitness
from smc.strategy.types import TradeSetup
from smc.strategy.zone_scanner import scan_zones

__all__ = ["MultiTimeframeAggregator"]


class MultiTimeframeAggregator:
    """Orchestrates the full multi-timeframe SMC strategy pipeline.

    .. versionchanged:: Sprint 5
       Added zone anti-clustering via ``_active_zones`` set.
       Max 1 entry per zone at a time (``_MAX_ENTRIES_PER_ZONE = 1``).

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

    # Sprint 5: Anti-clustering — max concurrent entries per unique zone.
    # Sentinel constant for v6 runner verification.
    _MAX_ENTRIES_PER_ZONE = 1

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
        *,
        enable_ob_test_trigger: bool = False,
        ai_regime_enabled: bool = False,
        regime_cache: "RegimeCacheLookup | None" = None,
        sl_fitness_enabled: bool = False,
        sl_fitness_min_sl_atr_ratio: float = 0.5,
        sl_fitness_max_sl_atr_ratio: float = 2.5,
        sl_fitness_low_vol_percentile: float = 0.3,
        sl_fitness_transition_conf_floor: float = 0.6,
        sl_fitness_counter_trend_ai_conf: float = 0.6,
        synthetic_zones_enabled: bool = False,
        synthetic_zones_min_historical: int = 2,
    ) -> None:
        self._detector = detector
        self._enable_ob_test_trigger = enable_ob_test_trigger
        self._ai_regime_enabled = ai_regime_enabled
        self._regime_cache = regime_cache
        self._zone_cooldowns: dict[tuple[float, float, str], datetime] = {}
        self._active_zones: set[tuple[float, float, str]] = set()
        # Round 4 Alt-B W2: optional macro overlay bias ∈ [-0.3, +0.3].
        # Set by live_demo.py via set_macro_bias() at the start of each cycle.
        # Default 0.0 = no-op; backward-compatible with all existing callers.
        self._macro_bias: float = 0.0
        # Round 5 A-track Task #7: SL fitness judge (shadow mode by default).
        self._sl_fitness_enabled = sl_fitness_enabled
        self._sl_fitness_min_sl_atr_ratio = sl_fitness_min_sl_atr_ratio
        self._sl_fitness_max_sl_atr_ratio = sl_fitness_max_sl_atr_ratio
        self._sl_fitness_low_vol_percentile = sl_fitness_low_vol_percentile
        self._sl_fitness_transition_conf_floor = sl_fitness_transition_conf_floor
        self._sl_fitness_counter_trend_ai_conf = sl_fitness_counter_trend_ai_conf
        # Round 5 A-track Task #9: ATH synthetic zones (augment, not replace).
        self._synthetic_zones_enabled = synthetic_zones_enabled
        self._synthetic_zones_min_historical = synthetic_zones_min_historical
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

    def mark_zone_active(
        self,
        zone_high: float,
        zone_low: float,
        direction: str,
    ) -> None:
        """Record that a trade is now active from this zone.

        Prevents additional entries in the same zone while one is open
        (Sprint 5 anti-clustering fix).
        """
        key = (round(zone_high, 2), round(zone_low, 2), direction)
        self._active_zones.add(key)

    def clear_zone_active(
        self,
        zone_high: float,
        zone_low: float,
        direction: str,
    ) -> None:
        """Mark a zone's trade as resolved (SL or TP hit)."""
        key = (round(zone_high, 2), round(zone_low, 2), direction)
        self._active_zones.discard(key)

    def clear_active_zones(self) -> None:
        """Clear all active zone tracking. Called between walk-forward windows."""
        self._active_zones.clear()

    def set_macro_bias(self, macro_bias: float) -> None:
        """Set the macro overlay bias for the next ``generate_setups`` call.

        Called by ``live_demo.py`` at the start of each cycle (when
        ``macro_enabled=True``) to inject the macro signal from
        ``MacroLayer.compute_macro_bias``.  The value is forwarded to
        ``score_confluence`` as the ``macro_bias`` argument.

        Parameters
        ----------
        macro_bias:
            Aggregated macro bias ∈ [-0.3, +0.3].  0.0 means no overlay
            (backward-compatible default).
        """
        self._macro_bias = float(macro_bias)

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None = None,
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
        bar_ts:
            Current bar timestamp — used for regime cache lookup in
            backtest mode.  When None, the regime classifier computes
            from the data directly.

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

        # Round 4.6-S-diag (USER CATCH: 系统不开单): measure-first instrumentation.
        # Records per-call diagnostic so live_demo / dashboard can surface why
        # aggregator returned 0 setups (htf_bias neutral? regime filter? direction filter?).
        self._last_setup_diagnostic = {
            "htf_bias_direction": bias.direction,
            "htf_bias_confidence": round(bias.confidence, 3),
            "htf_bias_rationale": bias.rationale,
            "stage_reject": None,
            "final_count": 0,
        }

        if bias.direction == "neutral":
            self._last_setup_diagnostic["stage_reject"] = "htf_bias_neutral"
            return ()

        # Step 2b: Regime classification — cache → AI → ATR fallback
        ai_assessment = classify_regime_ai(
            d1_df=data.get(Timeframe.D1),
            h4_df=data.get(Timeframe.H4),
            ai_enabled=self._ai_regime_enabled,
            cache=self._regime_cache,
            cache_ts=bar_ts,
        )
        regime_params = ai_assessment.param_preset
        # Round 5 A-track Task #8: expose ai_assessment so live_demo can
        # read the AI regime (e.g. TREND_UP / ATH_BREAKOUT) and derive
        # trail params for the strategy_server /signal response.
        self._last_ai_assessment = ai_assessment

        # Round 5 A-track Task #7: shadow-mode SL fitness judge needs a
        # RegimeContext snapshot alongside the assessment.  extract_regime_context
        # is pure + ~1ms — the duplicate work is cheap enough that we prefer it
        # over plumbing ctx through the classifier's cache/fallback paths.
        # Only computed when the judge is flagged ON (no overhead when off).
        regime_ctx = (
            extract_regime_context(
                d1_df=data.get(Timeframe.D1),
                h4_df=data.get(Timeframe.H4),
            )
            if self._sl_fitness_enabled
            else None
        )

        # Legacy ranging gate: when NOT using AI regime (Sprint 5 compat),
        # suppress Tier 2/3 in ranging markets.  When AI regime is active,
        # the direction filter + allowed_triggers handle this more precisely.
        if not self._ai_regime_enabled and not self._regime_cache:
            regime = classify_regime(data.get(Timeframe.D1))
            is_tier1 = bias.rationale.startswith("Tier 1:")
            if regime == "ranging" and not is_tier1:
                self._last_setup_diagnostic["stage_reject"] = "legacy_ranging_gate"
                return ()

        # Sprint 6: Direction filter — block counter-trend trades
        bias_dir_map = {"bullish": "long", "bearish": "short"}
        trade_direction = bias_dir_map.get(bias.direction)
        if trade_direction and trade_direction not in regime_params.allowed_directions:
            self._last_setup_diagnostic["stage_reject"] = f"direction_filter_{trade_direction}_not_in_{regime_params.allowed_directions}"
            return ()

        # Step 3: Scan H1 zones
        h1_snap = snapshots.get(Timeframe.H1)
        if h1_snap is None:
            self._last_setup_diagnostic["stage_reject"] = "h1_snapshot_missing"
            return ()

        zones = scan_zones(h1_snap, bias)

        # Round 5 A-track Task #9: synthetic zone augmentation for ATH regimes.
        # When historical zones are scarce (< min_historical) AND price sits
        # in the top 5% of the 52-week range, fall back to synthesized
        # anchors (VWAP bands / session H/L / round numbers / prev-week H/L).
        # This addresses the W14+W15 2024 drought (6 months of zero setups
        # during XAU's ATH rally) without displacing real OB/FVG zones.
        if self._synthetic_zones_enabled and len(zones) < self._synthetic_zones_min_historical:
            try:
                from smc.smc_core.synthetic_zones import build_synthetic_zones
                d1_df = data.get(Timeframe.D1)
                # 52w range — pull from the last 252 D1 bars (1 trading year).
                if d1_df is not None and len(d1_df) >= 10:
                    n_52w = min(len(d1_df), 252)
                    recent_52w = d1_df[-n_52w:]
                    p_52w_high = float(recent_52w["high"].max() or 0.0)
                    p_52w_low = float(recent_52w["low"].min() or 0.0)
                else:
                    p_52w_high = 0.0
                    p_52w_low = 0.0

                synthetic = build_synthetic_zones(
                    m15_df=data.get(Timeframe.M15),
                    h1_df=data.get(Timeframe.H1),
                    current_price=current_price,
                    price_52w_high=p_52w_high,
                    price_52w_low=p_52w_low,
                    now=bar_ts,
                )
                # Filter to bias-aligned zones so we don't resurrect
                # counter-trend synthetic anchors.
                bias_dir = "long" if bias.direction == "bullish" else "short"
                synthetic_aligned = tuple(z for z in synthetic if z.direction == bias_dir)
                if synthetic_aligned:
                    zones = tuple(list(zones) + list(synthetic_aligned))
                    self._last_setup_diagnostic["synthetic_zones_added"] = len(synthetic_aligned)
                    try:
                        from smc.monitor.structured_log import info as _log_info
                        _log_info(
                            "synthetic_zones_augment",
                            added=len(synthetic_aligned),
                            historical=len(zones) - len(synthetic_aligned),
                            price=round(current_price, 2),
                            p_52w_high=p_52w_high,
                            p_52w_low=p_52w_low,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    from smc.monitor.structured_log import warn as _log_warn
                    _log_warn(
                        "synthetic_zones_error",
                        error_type=type(exc).__name__,
                        error_msg=str(exc)[:200],
                    )
                except Exception:
                    pass

        if not zones:
            self._last_setup_diagnostic["stage_reject"] = "no_h1_zones"
            return ()

        # Step 4: Check M15 entries for each zone
        m15_snap = snapshots.get(Timeframe.M15)
        if m15_snap is None:
            self._last_setup_diagnostic["stage_reject"] = "m15_snapshot_missing"
            return ()
        self._last_setup_diagnostic["h1_zones_count"] = len(zones)

        # Sprint 4: Compute H1 ATR(14) for adaptive SL buffer
        h1_atr = self._compute_h1_atr(data.get(Timeframe.H1))

        # Sprint 6: Confluence floor = max(tier floor, regime floor)
        tier_floor = effective_threshold(bias.rationale)
        min_confluence = max(tier_floor, regime_params.confluence_floor)

        now = datetime.now(tz=timezone.utc)
        setups: list[TradeSetup] = []

        # Sprint 5: Intra-call zone dedup
        zones_used_this_call: set[tuple[float, float, str]] = set()

        # Round 4.6-T: per-zone reject counter (measure-first, 指向具体 gate)
        zone_rejects = {
            "cooldown": 0, "active_zones": 0, "intra_call_dedup": 0,
            "entry_none": 0, "trigger_filter": 0, "confluence_low": 0,
        }
        # Round 4.6-T-v2 (USER "解决到开仓"): record zone coordinates + price distance
        # 以辨析 entry_none 是 "price 距 zone 太远" 还是 "trigger 没形成".
        zone_details: list[dict] = []

        for zone in zones:
            zone_key = (round(zone.zone_high, 2), round(zone.zone_low, 2), zone.direction)
            zone_details.append({
                "high": round(zone.zone_high, 2),
                "low": round(zone.zone_low, 2),
                "direction": zone.direction,
                "dist_from_price": round(
                    min(abs(current_price - zone.zone_high), abs(current_price - zone.zone_low)),
                    2,
                ),
                "in_expanded_zone": bool(
                    (zone.zone_low - (zone.zone_high - zone.zone_low) * 0.25)
                    <= current_price
                    <= (zone.zone_high + (zone.zone_high - zone.zone_low) * 0.25)
                ),
            })

            # Zone cooldown: skip zones that recently produced a loss
            if zone_key in self._zone_cooldowns:
                cooldown_until = self._zone_cooldowns[zone_key]
                if now < cooldown_until:
                    zone_rejects["cooldown"] += 1
                    continue

            # Sprint 5: Zone anti-clustering
            if zone_key in self._active_zones:
                zone_rejects["active_zones"] += 1
                continue
            if zone_key in zones_used_this_call:
                zone_rejects["intra_call_dedup"] += 1
                continue

            # Sprint 6: Pass regime-aware SL/TP params to entry trigger
            entry = check_entry(
                m15_snap, zone, current_price, h1_atr=h1_atr,
                enable_ob_test="ob_test_rejection" in regime_params.allowed_triggers,
                sl_atr_multiplier=regime_params.sl_atr_multiplier,
                tp1_rr=regime_params.tp1_rr,
            )
            if entry is None:
                zone_rejects["entry_none"] += 1
                continue

            # Sprint 6: Trigger type filter — only regime-permitted triggers
            if entry.trigger_type not in regime_params.allowed_triggers:
                zone_rejects["trigger_filter"] += 1
                continue

            # Step 5: Score confluence (+ optional macro overlay bias)
            conf_score = score_confluence(bias, zone, entry, macro_bias=self._macro_bias)

            if conf_score < min_confluence:
                zone_rejects["confluence_low"] += 1
                continue

            # Round 5 A-track Task #7: SL fitness judge — SHADOW MODE.
            # When enabled, evaluate the 7-rule fitness check and emit
            # sl_fitness_shadow_veto{accepted, rule_id, regime, direction,
            # rr_planned, conf} telemetry.  Does NOT block the setup in
            # shadow mode — we measure first, enforce later (see design
            # doc §6 "Option 2 shadow mode").
            if self._sl_fitness_enabled and regime_ctx is not None:
                try:
                    verdict = judge_sl_fitness(
                        entry=entry,
                        regime_assessment=ai_assessment,
                        regime_ctx=regime_ctx,
                        confluence_score=conf_score,
                        h1_atr_points=h1_atr,
                        d1_atr_pct=regime_ctx.d1_atr_pct,
                        min_sl_atr_ratio=self._sl_fitness_min_sl_atr_ratio,
                        max_sl_atr_ratio=self._sl_fitness_max_sl_atr_ratio,
                        low_vol_percentile=self._sl_fitness_low_vol_percentile,
                        transition_conf_floor=self._sl_fitness_transition_conf_floor,
                        counter_trend_ai_conf=self._sl_fitness_counter_trend_ai_conf,
                    )
                    # Emit telemetry regardless of accept/veto so ops can
                    # measure veto rate + false-positive rate over 10 days.
                    try:
                        from smc.monitor.structured_log import info as _log_info
                        _log_info(
                            "sl_fitness_shadow_veto",
                            accepted=verdict.accept,
                            rule_id=verdict.rule_id,
                            reason=verdict.reason,
                            regime=ai_assessment.regime,
                            direction=entry.direction,
                            rr_planned=entry.rr_ratio,
                            confluence=round(conf_score, 3),
                            ai_conf=round(ai_assessment.confidence, 3),
                            risk_points=entry.risk_points,
                            d1_atr_pct=regime_ctx.d1_atr_pct,
                        )
                    except Exception:
                        pass  # telemetry never breaks the pipeline
                except Exception as exc:
                    # Never let the judge break the live loop — log and continue
                    # as if the judge had accepted.
                    try:
                        from smc.monitor.structured_log import warn as _log_warn
                        _log_warn(
                            "sl_fitness_judge_error",
                            error_type=type(exc).__name__,
                            error_msg=str(exc)[:200],
                        )
                    except Exception:
                        pass

            setups.append(
                TradeSetup(
                    entry_signal=entry,
                    bias=bias,
                    zone=zone,
                    confluence_score=conf_score,
                    generated_at=now,
                )
            )
            zones_used_this_call.add(zone_key)

        # Step 6: Sort by confluence score descending, cap by regime max_concurrent
        sorted_setups = sorted(setups, key=lambda s: s.confluence_score, reverse=True)
        capped = sorted_setups[:regime_params.max_concurrent]
        self._last_setup_diagnostic["zone_rejects"] = zone_rejects
        self._last_setup_diagnostic["zone_details"] = zone_details
        self._last_setup_diagnostic["current_price"] = round(current_price, 2)
        self._last_setup_diagnostic["min_confluence"] = round(min_confluence, 3)
        if not capped:
            self._last_setup_diagnostic["stage_reject"] = (
                f"all_entries_failed (zones={len(zones)}, min_conf={min_confluence:.2f})"
            )
        self._last_setup_diagnostic["final_count"] = len(capped)
        return tuple(capped)

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
