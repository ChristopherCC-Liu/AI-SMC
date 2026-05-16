"""Regime Engine — unified market regime detection (sidecar).

Reads OHLC + optional macro inputs (DXY, US10Y, VIX) and emits a
:class:`RegimeSnapshot` describing the current regime, confidence,
and transition probabilities. The detector is rule-based on rolling
volatility, ATR percentile, and gap detection across a 20/60-bar
double window — no HMM library dependency.

Public surface:
  * :class:`MarketRegime`
  * :class:`RegimeSnapshot`
  * :class:`RegimeDetector`
  * :func:`regime_adaptive_weights`

Isolation: this module does NOT import ``rule_engine`` (still
red-line). Tier-1 unsealed modules (``decision_server`` /
``phase_d_walk_forward``) are also not imported here — the regime
engine is upstream of the candidate generator and stays pure.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence


__all__ = [
    "MarketRegime",
    "RegimeSnapshot",
    "RegimeDetector",
    "regime_adaptive_weights",
    "GATE_BASE_WEIGHTS",
]


class MarketRegime(str, Enum):
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    EXTREME = "EXTREME"
    CRISIS = "CRISIS"


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: MarketRegime
    confidence: float
    short_window_vol: float
    long_window_vol: float
    atr_percentile: float
    max_gap_pct: float
    transition_probs: tuple[tuple[MarketRegime, float], ...]
    n_bars_observed: int
    macro_context: Mapping[str, float]
    blocking_conditions: tuple[str, ...]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


GATE_BASE_WEIGHTS: Mapping[str, float] = {
    "G1": 1.0, "G2": 1.0, "G3": 1.0, "G4": 1.0,
    "G5": 1.0, "G6": 1.0, "G7": 1.0, "G8": 1.0,
}


# Multipliers per regime — applied to GATE_BASE_WEIGHTS.
_SAFETY_GATES = ("G1", "G2", "G3")
_AGGRESSIVE_GATES = ("G7", "G8")

_REGIME_WEIGHT_DELTAS: Mapping[MarketRegime, Mapping[str, float]] = {
    MarketRegime.LOW_VOL: {
        # Slight relaxation across the board; a touch more weight on
        # G7/G8 to encourage opportunity detection in calm tape.
        "G7": 1.10, "G8": 1.10,
    },
    MarketRegime.NORMAL: {},
    MarketRegime.HIGH_VOL: {
        "G1": 1.20, "G2": 1.20, "G3": 1.20,
        "G7": 0.90, "G8": 0.90,
    },
    MarketRegime.EXTREME: {
        "G1": 1.50, "G2": 1.50, "G3": 1.50,
        "G7": 0.70, "G8": 0.70,
    },
    MarketRegime.CRISIS: {
        "G1": 1.50, "G2": 1.50, "G3": 1.50,
        "G7": 0.70, "G8": 0.70,
    },
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _ohlc_to_log_returns(bars: Sequence[Mapping[str, float]]) -> list[float]:
    out: list[float] = []
    prev_close: float | None = None
    for b in bars:
        try:
            close = float(b.get("close", b.get("c", 0.0)))
        except (TypeError, ValueError):
            continue
        if close <= 0.0:
            prev_close = None
            continue
        if prev_close is not None and prev_close > 0.0:
            out.append(math.log(close / prev_close))
        prev_close = close
    return out


def _ohlc_to_atr_series(
    bars: Sequence[Mapping[str, float]],
) -> list[float]:
    """Per-bar true range (high - low fallback to abs(close - prev_close))."""
    out: list[float] = []
    prev_close: float | None = None
    for b in bars:
        try:
            high = float(b.get("high", b.get("h", 0.0)))
            low = float(b.get("low", b.get("l", 0.0)))
            close = float(b.get("close", b.get("c", 0.0)))
        except (TypeError, ValueError):
            continue
        if high <= 0.0 or low <= 0.0 or close <= 0.0:
            prev_close = close if close > 0.0 else prev_close
            continue
        tr = high - low
        if prev_close is not None:
            tr = max(tr, abs(high - prev_close), abs(low - prev_close))
        out.append(tr)
        prev_close = close
    return out


def _max_gap_pct(bars: Sequence[Mapping[str, float]]) -> float:
    """Largest absolute open-vs-prev-close gap, in pct of prev close."""
    worst = 0.0
    prev_close: float | None = None
    for b in bars:
        try:
            o = float(b.get("open", b.get("o", 0.0)))
            c = float(b.get("close", b.get("c", 0.0)))
        except (TypeError, ValueError):
            continue
        if prev_close is not None and prev_close > 0.0 and o > 0.0:
            worst = max(worst, abs(o - prev_close) / prev_close)
        if c > 0.0:
            prev_close = c
    return worst


# Decision thresholds — applied to short-window log-return stddev.
# Calibrated against XAUUSD daily/4h: typical NORMAL ~0.6%, HIGH_VOL
# ~1.2%, EXTREME ~2.0%, CRISIS ~3.0%+.
_VOL_THRESHOLD_LOW = 0.003
_VOL_THRESHOLD_NORMAL = 0.009
_VOL_THRESHOLD_HIGH = 0.018
_VOL_THRESHOLD_EXTREME = 0.030

# Gap escalation thresholds (fraction of price).
_GAP_ESCALATE_HIGH = 0.005   # 0.5% gap → at least HIGH_VOL
_GAP_ESCALATE_EXTREME = 0.010  # 1% gap → at least EXTREME

# ATR percentile escalation — long-window ATR p90 reading raises the
# floor regime by one notch.
_ATR_PCTILE_ESCALATE = 0.90


@dataclass(frozen=True)
class _DetectorConfig:
    short_window: int = 20
    long_window: int = 60
    min_bars_for_full_detection: int = 20


class RegimeDetector:
    """Stateless detector. Call :meth:`detect` with the latest bars."""

    def __init__(
        self,
        *,
        short_window: int = 20,
        long_window: int = 60,
    ) -> None:
        if short_window < 2:
            raise ValueError("short_window must be >= 2")
        if long_window < short_window:
            raise ValueError("long_window must be >= short_window")
        self._cfg = _DetectorConfig(
            short_window=short_window,
            long_window=long_window,
            min_bars_for_full_detection=short_window,
        )

    def detect(
        self,
        *,
        bars: Iterable[Mapping[str, float]],
        macro: Mapping[str, float] | None = None,
    ) -> RegimeSnapshot:
        bar_list = list(bars)
        macro_ctx: dict[str, float] = {
            k: float(v) for k, v in (macro or {}).items()
        }

        if len(bar_list) < self._cfg.min_bars_for_full_detection:
            return RegimeSnapshot(
                regime=MarketRegime.NORMAL,
                confidence=0.0,
                short_window_vol=0.0,
                long_window_vol=0.0,
                atr_percentile=0.0,
                max_gap_pct=0.0,
                transition_probs=(),
                n_bars_observed=len(bar_list),
                macro_context=macro_ctx,
                blocking_conditions=(
                    f"insufficient_bars:n={len(bar_list)}<"
                    f"{self._cfg.min_bars_for_full_detection}",
                ),
            )

        short_bars = bar_list[-self._cfg.short_window:]
        long_bars = bar_list[-self._cfg.long_window:]

        short_returns = _ohlc_to_log_returns(short_bars)
        long_returns = _ohlc_to_log_returns(long_bars)
        short_vol = _stddev(short_returns)
        long_vol = _stddev(long_returns)

        atr_series = _ohlc_to_atr_series(long_bars)
        latest_atr = atr_series[-1] if atr_series else 0.0
        atr_pctile = (
            sum(1 for a in atr_series if a <= latest_atr) / len(atr_series)
            if atr_series else 0.0
        )
        # ATR "spike" detector — latest ATR has to be at least 1.8×
        # the median to count as a meaningful escalation signal,
        # otherwise we'd flag every quiet tape where the last bar
        # happens to be the locally largest.
        atr_median = (
            sorted(atr_series)[len(atr_series) // 2] if atr_series else 0.0
        )
        atr_spike = atr_median > 0.0 and latest_atr >= atr_median * 1.8

        gap = _max_gap_pct(short_bars)

        regime = self._classify(
            short_vol=short_vol, long_vol=long_vol,
            atr_pctile=atr_pctile, atr_spike=atr_spike,
            gap=gap, macro=macro_ctx,
        )
        confidence = self._confidence(
            short_vol=short_vol, long_vol=long_vol, regime=regime, gap=gap,
        )
        transitions = self._transition_probs(
            current=regime, short_vol=short_vol, long_vol=long_vol,
        )

        return RegimeSnapshot(
            regime=regime,
            confidence=confidence,
            short_window_vol=round(short_vol, 6),
            long_window_vol=round(long_vol, 6),
            atr_percentile=round(atr_pctile, 4),
            max_gap_pct=round(gap, 6),
            transition_probs=transitions,
            n_bars_observed=len(bar_list),
            macro_context=macro_ctx,
            blocking_conditions=(),
        )

    def _classify(
        self,
        *,
        short_vol: float,
        long_vol: float,
        atr_pctile: float,
        atr_spike: bool,
        gap: float,
        macro: Mapping[str, float],
    ) -> MarketRegime:
        if short_vol >= _VOL_THRESHOLD_EXTREME or gap >= _GAP_ESCALATE_EXTREME * 2:
            base = MarketRegime.CRISIS
        elif short_vol >= _VOL_THRESHOLD_HIGH or gap >= _GAP_ESCALATE_EXTREME:
            base = MarketRegime.EXTREME
        elif short_vol >= _VOL_THRESHOLD_NORMAL or gap >= _GAP_ESCALATE_HIGH:
            base = MarketRegime.HIGH_VOL
        elif short_vol <= _VOL_THRESHOLD_LOW:
            base = MarketRegime.LOW_VOL
        else:
            base = MarketRegime.NORMAL

        # ATR-percentile escalator only fires when the base classification
        # is in the "calm" half — it's there to catch sneaky moves that
        # don't yet show in short-window stddev. Once we've already
        # classified HIGH_VOL or worse, the stddev / gap signals are
        # already escalating us; double-counting via ATR would
        # over-flag. Also requires a meaningful spike (latest ATR
        # ≥ 1.8× median), not just a percentile rank.
        if base in (MarketRegime.LOW_VOL, MarketRegime.NORMAL) and \
           atr_pctile >= _ATR_PCTILE_ESCALATE and atr_spike:
            base = _escalate_regime(base)

        # VIX > 30 → at least HIGH_VOL; VIX > 40 → at least EXTREME.
        vix = macro.get("VIX", 0.0)
        if vix >= 40.0:
            base = max(base, MarketRegime.EXTREME, key=_regime_rank)
        elif vix >= 30.0:
            base = max(base, MarketRegime.HIGH_VOL, key=_regime_rank)

        return base

    def _confidence(
        self,
        *,
        short_vol: float,
        long_vol: float,
        regime: MarketRegime,
        gap: float,
    ) -> float:
        if regime == MarketRegime.NORMAL:
            mid = (_VOL_THRESHOLD_LOW + _VOL_THRESHOLD_NORMAL) / 2
            spread = (_VOL_THRESHOLD_NORMAL - _VOL_THRESHOLD_LOW) / 2
            if spread <= 0:
                return 0.5
            return max(
                0.0, min(1.0, 1.0 - abs(short_vol - mid) / spread)
            )
        # Distance past the threshold relative to the next threshold's
        # gap → higher distance = higher confidence, capped at 1.0.
        distance = max(short_vol, gap * 2.0)
        if regime == MarketRegime.LOW_VOL:
            return max(
                0.0, min(1.0, (_VOL_THRESHOLD_LOW - distance) / _VOL_THRESHOLD_LOW)
            )
        return max(0.0, min(1.0, distance / _VOL_THRESHOLD_EXTREME))

    def _transition_probs(
        self,
        *,
        current: MarketRegime,
        short_vol: float,
        long_vol: float,
    ) -> tuple[tuple[MarketRegime, float], ...]:
        """Crude transition estimate: if short_vol > long_vol, lean
        toward escalation; otherwise lean toward de-escalation."""
        ratio = (short_vol / long_vol) if long_vol > 0 else 1.0
        out: dict[MarketRegime, float] = {current: 0.6}
        nbr_up = _escalate_regime(current)
        nbr_down = _de_escalate_regime(current)
        if ratio > 1.1:
            out[nbr_up] = out.get(nbr_up, 0.0) + 0.30
            out[nbr_down] = out.get(nbr_down, 0.0) + 0.10
        elif ratio < 0.9:
            out[nbr_down] = out.get(nbr_down, 0.0) + 0.30
            out[nbr_up] = out.get(nbr_up, 0.0) + 0.10
        else:
            out[nbr_up] = out.get(nbr_up, 0.0) + 0.20
            out[nbr_down] = out.get(nbr_down, 0.0) + 0.20
        # Re-add current if escalate/de-escalate collapse to current.
        out.setdefault(current, 0.6)
        # Normalise.
        total = sum(out.values())
        if total <= 0:
            return ((current, 1.0),)
        return tuple(
            sorted(
                ((r, round(p / total, 4)) for r, p in out.items()),
                key=lambda kv: -kv[1],
            )
        )


_REGIME_RANK: Mapping[MarketRegime, int] = {
    MarketRegime.LOW_VOL: 0,
    MarketRegime.NORMAL: 1,
    MarketRegime.HIGH_VOL: 2,
    MarketRegime.EXTREME: 3,
    MarketRegime.CRISIS: 4,
}


def _regime_rank(r: MarketRegime) -> int:
    return _REGIME_RANK[r]


def _escalate_regime(r: MarketRegime) -> MarketRegime:
    order = list(_REGIME_RANK.keys())
    idx = order.index(r)
    return order[min(len(order) - 1, idx + 1)]


def _de_escalate_regime(r: MarketRegime) -> MarketRegime:
    order = list(_REGIME_RANK.keys())
    idx = order.index(r)
    return order[max(0, idx - 1)]


# ---------------------------------------------------------------------------
# Adaptive weights
# ---------------------------------------------------------------------------


def regime_adaptive_weights(
    *,
    regime: MarketRegime,
    base_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return a fresh weights dict adjusted for ``regime``.

    Multipliers are applied to ``base_weights`` (defaults to
    :data:`GATE_BASE_WEIGHTS`). Unknown gate ids in the base map are
    passed through unchanged.
    """
    base = dict(base_weights) if base_weights is not None else dict(
        GATE_BASE_WEIGHTS
    )
    deltas = _REGIME_WEIGHT_DELTAS.get(regime, {})
    out: dict[str, float] = {}
    for gate_id, w in base.items():
        mult = deltas.get(gate_id, 1.0)
        out[gate_id] = round(float(w) * mult, 6)
    return out
