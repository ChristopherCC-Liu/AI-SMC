"""V2 multi-timeframe strategy aggregator — simplified cascade with AI direction.

``AggregatorV2`` replaces the v1 ``MultiTimeframeAggregator`` pipeline with:
1. AI direction (from ``DirectionEngine``) replaces ``compute_htf_bias``
2. Simplified cascade: H4 → H1 → M15 (no D1 detection)
3. Neutral gate: very loose (only block confidence < 0.2)
4. Neutral AI direction scans BOTH long and short for inverted signals
5. ``check_entry_v2`` with 6 triggers (normal + inverted)
6. ``score_confluence_v2`` with AI direction weight + mode bonus
7. Zone cooldown 8h (was 24h), max 2 entries per zone (was 1)
8. Max concurrent setups = 5

V1 ``aggregator.py`` is preserved untouched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from smc.ai.direction_engine import DirectionEngine
from smc.ai.models import AIDirection
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.smc_core.types import SMCSnapshot
from smc.strategy.confluence_v2 import TRADEABLE_THRESHOLD_V2, score_confluence_v2
from smc.strategy.entry_v2 import check_entry_v2
from smc.strategy.regime import classify_regime
from smc.strategy.types import TradeSetupV2, TradeZone
from smc.strategy.zone_scanner_v2 import scan_zones_v2

__all__ = ["AggregatorV2"]

# ---------------------------------------------------------------------------
# Constants (v2 — relaxed)
# ---------------------------------------------------------------------------

_ZONE_COOLDOWN_HOURS = 8
_MAX_ENTRIES_PER_ZONE = 2
_MAX_CONCURRENT = 5
_NEUTRAL_CONFIDENCE_GATE = 0.2

# V2 only detects on H4, H1, M15 (no D1)
_V2_TIMEFRAMES = (Timeframe.H4, Timeframe.H1, Timeframe.M15)

# Per-TF swing_length defaults (no D1)
_V2_SWING_LENGTH_MAP: dict[Timeframe, int] = {
    Timeframe.H4: 7,
    Timeframe.H1: 10,
    Timeframe.M15: 10,
}


class AggregatorV2:
    """V2 strategy pipeline orchestrator with AI direction filtering.

    Parameters
    ----------
    detector:
        An ``SMCDetector`` instance used to detect SMC patterns.
    direction_engine:
        ``DirectionEngine`` instance for AI directional assessment.
    enable_inverted:
        Enable inverted entry signals (ob_breakout, choch_continuation).
    enable_fvg_sweep:
        Enable fvg_sweep_continuation signal.
    ai_regime_enabled:
        Enable AI regime classification for parameter routing.
    regime_cache:
        Optional regime cache for backtest mode.
    """

    _ZONE_COOLDOWN_HOURS = _ZONE_COOLDOWN_HOURS
    _MAX_ENTRIES_PER_ZONE = _MAX_ENTRIES_PER_ZONE

    def __init__(
        self,
        detector: SMCDetector,
        direction_engine: DirectionEngine,
        *,
        enable_inverted: bool = True,
        enable_fvg_sweep: bool = True,
        ai_regime_enabled: bool = False,
        regime_cache: object | None = None,
    ) -> None:
        self._detector = detector
        self._direction_engine = direction_engine
        self._enable_inverted = enable_inverted
        self._enable_fvg_sweep = enable_fvg_sweep
        self._ai_regime_enabled = ai_regime_enabled
        self._regime_cache = regime_cache
        self._zone_cooldowns: dict[tuple[float, float, str], datetime] = {}
        self._active_zones: dict[tuple[float, float, str], int] = {}

        # Inject v2 swing_length_map if detector doesn't have one
        if not detector.swing_length_map:
            self._detector = SMCDetector(
                swing_length=detector.swing_length,
                min_swing_points=detector.min_swing_points,
                liquidity_tolerance_points=detector.liquidity_tolerance_points,
                swing_length_map=_V2_SWING_LENGTH_MAP,
            )

    @property
    def detector(self) -> SMCDetector:
        """The underlying SMC detector."""
        return self._detector

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def record_zone_loss(
        self,
        zone_high: float,
        zone_low: float,
        direction: str,
        loss_time: datetime,
    ) -> None:
        """Record a losing trade for zone cooldown (8h in v2, was 24h)."""
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
        """Record a trade active in this zone (v2: up to 2 per zone)."""
        key = (round(zone_high, 2), round(zone_low, 2), direction)
        self._active_zones[key] = self._active_zones.get(key, 0) + 1

    def clear_zone_active(
        self,
        zone_high: float,
        zone_low: float,
        direction: str,
    ) -> None:
        """Mark a zone trade as resolved (SL or TP hit)."""
        key = (round(zone_high, 2), round(zone_low, 2), direction)
        count = self._active_zones.get(key, 0)
        if count <= 1:
            self._active_zones.pop(key, None)
        else:
            self._active_zones[key] = count - 1

    def clear_active_zones(self) -> None:
        """Clear all active zone tracking. Called between walk-forward windows."""
        self._active_zones.clear()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None = None,
    ) -> tuple[TradeSetupV2, ...]:
        """Run the v2 strategy pipeline and return scored trade setups.

        V2 pipeline steps:
        1. AI direction (from DirectionEngine.get_direction())
        2. Neutral gate: very loose (only block confidence < 0.2)
        3. Detect H4 + H1 + M15 (NO D1)
        4. Zone scan with MAX_ZONES=5
        5. check_entry_v2 (6 triggers, normal + inverted)
        6. score_confluence_v2
        7. max_concurrent=5 cap

        When AI direction is neutral:
        - Scan BOTH long and short zones for inverted signals only
        - Normal signals are suppressed

        Parameters
        ----------
        data:
            Mapping of Timeframe to Polars OHLCV DataFrame.
        current_price:
            Current XAUUSD price.
        bar_ts:
            Current bar timestamp for cache lookups (backtest mode).

        Returns
        -------
        tuple[TradeSetupV2, ...]
            Trade setups sorted by confluence score descending.
        """
        # Step 1: AI direction
        ai_dir = self._direction_engine.get_direction(
            h4_df=data.get(Timeframe.H4),
            bar_ts=bar_ts,
        )

        # Step 2: Neutral gate — very loose, only block very low confidence
        if ai_dir.confidence < _NEUTRAL_CONFIDENCE_GATE:
            return ()

        is_neutral = ai_dir.direction == "neutral"

        # Step 3: Detect SMC patterns on H4 + H1 + M15 (no D1)
        snapshots = self._detect_v2(data)

        h1_snap = snapshots.get(Timeframe.H1)
        if h1_snap is None:
            return ()

        m15_snap = snapshots.get(Timeframe.M15)
        if m15_snap is None:
            return ()

        # Compute H1 ATR(14) for adaptive SL buffer
        h1_atr = self._compute_h1_atr(data.get(Timeframe.H1))

        # Classify regime for entry_v2 gating
        regime = classify_regime(data.get(Timeframe.H4))

        now = datetime.now(tz=timezone.utc)

        # Step 4+5+6: Zone scan → entry check → confluence scoring
        if is_neutral:
            # Neutral: scan BOTH directions for inverted signals
            setups = self._scan_both_directions(
                h1_snap, m15_snap, current_price, h1_atr, regime,
                ai_dir, now,
            )
        else:
            # Directional: scan aligned direction
            setups = self._scan_directional(
                h1_snap, m15_snap, current_price, h1_atr, regime,
                ai_dir, now,
            )

        # Step 7: Sort by confluence score descending, cap at MAX_CONCURRENT
        sorted_setups = sorted(
            setups, key=lambda s: s.confluence_score, reverse=True,
        )
        return tuple(sorted_setups[:_MAX_CONCURRENT])

    # ------------------------------------------------------------------
    # Directional scan (bullish or bearish AI direction)
    # ------------------------------------------------------------------

    def _scan_directional(
        self,
        h1_snap: SMCSnapshot,
        m15_snap: SMCSnapshot,
        current_price: float,
        h1_atr: float,
        regime: str,
        ai_dir: AIDirection,
        now: datetime,
    ) -> list[TradeSetupV2]:
        """Scan zones in the AI direction for both normal and inverted entries."""
        zones = scan_zones_v2(h1_snap, ai_dir.direction)
        if not zones:
            return []

        return self._process_zones(
            zones, m15_snap, current_price, h1_atr, regime,
            ai_dir, now, allow_normal=True,
        )

    # ------------------------------------------------------------------
    # Neutral scan (both directions, inverted only)
    # ------------------------------------------------------------------

    def _scan_both_directions(
        self,
        h1_snap: SMCSnapshot,
        m15_snap: SMCSnapshot,
        current_price: float,
        h1_atr: float,
        regime: str,
        ai_dir: AIDirection,
        now: datetime,
    ) -> list[TradeSetupV2]:
        """Scan both long and short zones for inverted signals only."""
        setups: list[TradeSetupV2] = []

        for bias_dir in ("bullish", "bearish"):
            zones = scan_zones_v2(h1_snap, bias_dir)
            if not zones:
                continue

            direction_setups = self._process_zones(
                zones, m15_snap, current_price, h1_atr, regime,
                ai_dir, now, allow_normal=False,
            )
            setups.extend(direction_setups)

        return setups

    # ------------------------------------------------------------------
    # Zone processing (shared logic)
    # ------------------------------------------------------------------

    def _process_zones(
        self,
        zones: tuple[TradeZone, ...],
        m15_snap: SMCSnapshot,
        current_price: float,
        h1_atr: float,
        regime: str,
        ai_dir: AIDirection,
        now: datetime,
        *,
        allow_normal: bool,
    ) -> list[TradeSetupV2]:
        """Process zones through entry check and confluence scoring."""
        setups: list[TradeSetupV2] = []
        zones_used_this_call: dict[tuple[float, float, str], int] = {}

        for zone in zones:
            zone_key = (
                round(zone.zone_high, 2),
                round(zone.zone_low, 2),
                zone.direction,
            )

            # Zone cooldown check
            if zone_key in self._zone_cooldowns:
                cooldown_until = self._zone_cooldowns[zone_key]
                if now < cooldown_until:
                    continue

            # Zone active count check (v2: max 2 per zone)
            active_count = self._active_zones.get(zone_key, 0)
            call_count = zones_used_this_call.get(zone_key, 0)
            if active_count + call_count >= self._MAX_ENTRIES_PER_ZONE:
                continue

            # Check entry
            entry = check_entry_v2(
                m15_snap,
                zone,
                current_price,
                h1_atr=h1_atr,
                regime=regime,  # type: ignore[arg-type]
                enable_inverted=self._enable_inverted,
                enable_fvg_sweep=self._enable_fvg_sweep,
            )
            if entry is None:
                continue

            # Filter: in neutral/inverted-only mode, skip normal entries
            if not allow_normal and entry.entry_mode == "normal":
                continue

            # Score confluence
            conf_score = score_confluence_v2(
                ai_dir.confidence, zone, entry,
            )

            if conf_score < TRADEABLE_THRESHOLD_V2:
                continue

            setups.append(
                TradeSetupV2(
                    entry_signal=entry,
                    ai_direction=ai_dir.direction,
                    ai_confidence=ai_dir.confidence,
                    zone=zone,
                    confluence_score=conf_score,
                    entry_mode=entry.entry_mode,
                    generated_at=now,
                )
            )
            zones_used_this_call[zone_key] = call_count + 1

        return setups

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_v2(
        self,
        data: dict[Timeframe, pl.DataFrame],
    ) -> dict[Timeframe, SMCSnapshot]:
        """Run detection on H4, H1, M15 only (no D1)."""
        snapshots: dict[Timeframe, SMCSnapshot] = {}
        for tf in _V2_TIMEFRAMES:
            df = data.get(tf)
            if df is None or len(df) == 0:
                continue
            snapshots[tf] = self._detector.detect(df, tf)
        return snapshots

    @staticmethod
    def _compute_h1_atr(h1_df: pl.DataFrame | None) -> float:
        """Compute H1 ATR(14) in points from an H1 OHLCV DataFrame.

        Returns 0.0 if insufficient data.
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
        from smc.smc_core.constants import XAUUSD_POINT_SIZE

        return atr_price / XAUUSD_POINT_SIZE
