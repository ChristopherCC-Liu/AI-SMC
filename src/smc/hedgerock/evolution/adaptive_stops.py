"""Adaptive volatility stop-loss advisory (sidecar — report-only).

Computes Parkinson + Garman-Klass volatility estimators against
recent OHLC bars, classifies the current volatility regime, and
returns a :class:`StopRecommendation` describing:

  * the suggested ATR multiplier in the band [1.2, 3.5];
  * a position-size scale per regime (COMPRESSED 1.2× → EXTREME 0.3×);
  * a human-readable ``reasoning`` string for the report.

This module never writes anywhere and never modifies live stops.
The recommendation is advisory and is surfaced exclusively in the
sidecar recommendation report.

Public surface:
  * :class:`VolatilityRegime`
  * :class:`StopRecommendation`
  * :func:`compute_stop_recommendation`
  * :func:`parkinson_volatility`
  * :func:`garman_klass_volatility`

Isolation: no imports of ``rule_engine`` or the Tier-1 unsealed
prod modules.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence


__all__ = [
    "ATR_MULT_LO",
    "ATR_MULT_HI",
    "POSITION_SCALE_BY_REGIME",
    "StopRecommendation",
    "VolatilityRegime",
    "atr_multiplier_for_ratio",
    "compute_stop_recommendation",
    "garman_klass_volatility",
    "parkinson_volatility",
]


class VolatilityRegime(str, Enum):
    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    EXTREME = "EXTREME"


# ATR multiplier clamp band per spec.
ATR_MULT_LO: float = 1.2
ATR_MULT_HI: float = 3.5


POSITION_SCALE_BY_REGIME: Mapping[VolatilityRegime, float] = {
    VolatilityRegime.COMPRESSED: 1.2,
    VolatilityRegime.NORMAL: 1.0,
    VolatilityRegime.ELEVATED: 0.6,
    VolatilityRegime.EXTREME: 0.3,
}


# σ_ratio thresholds for regime classification.
_RATIO_COMPRESSED = 0.7
_RATIO_NORMAL_HI = 1.3
_RATIO_ELEVATED_HI = 2.0


# Window sizes per spec.
_HV_WINDOW = 30
_MA_WINDOW = 60
_MIN_BARS_FOR_RATIO = _HV_WINDOW + _MA_WINDOW  # 90


@dataclass(frozen=True)
class StopRecommendation:
    vol_regime: VolatilityRegime
    atr_multiplier: float
    position_scale: float
    parkinson_vol: float
    garman_klass_vol: float
    realized_vol_30: float
    realized_vol_ma_60: float
    sigma_ratio: float
    reasoning: str
    n_bars_observed: int
    blocking_conditions: tuple[str, ...]
    advisory_only: bool = True
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------


def parkinson_volatility(
    bars: Sequence[Mapping[str, float]],
) -> float:
    """Parkinson (high-low) variance estimator.

        σ²_P = (1 / (4 n ln 2)) · Σ (ln(H/L))²

    Returns the per-bar volatility (square root of variance). Bars
    with non-positive H/L are skipped.
    """
    samples: list[float] = []
    for b in bars:
        try:
            h = float(b.get("high", b.get("h", 0.0)))
            l = float(b.get("low", b.get("l", 0.0)))
        except (TypeError, ValueError):
            continue
        if h <= 0.0 or l <= 0.0 or h < l:
            continue
        if h == l:
            samples.append(0.0)
            continue
        x = math.log(h / l)
        samples.append(x * x)
    if not samples:
        return 0.0
    var = sum(samples) / (4.0 * len(samples) * math.log(2.0))
    return math.sqrt(max(0.0, var))


_GK_OC_COEFF = 2.0 * math.log(2.0) - 1.0


def garman_klass_volatility(
    bars: Sequence[Mapping[str, float]],
) -> float:
    """Garman-Klass variance estimator.

        σ²_GK = (1 / n) · Σ [0.5 (ln(H/L))² − (2 ln 2 − 1) (ln(C/O))²]

    Per-bar value clamped at 0 to keep downstream sqrt safe.
    """
    samples: list[float] = []
    for b in bars:
        try:
            o = float(b.get("open", b.get("o", 0.0)))
            h = float(b.get("high", b.get("h", 0.0)))
            l = float(b.get("low", b.get("l", 0.0)))
            c = float(b.get("close", b.get("c", 0.0)))
        except (TypeError, ValueError):
            continue
        if min(o, h, l, c) <= 0.0 or h < l:
            continue
        if h == l and o == c:
            samples.append(0.0)
            continue
        hl = math.log(h / l) if h > l else 0.0
        oc = math.log(c / o) if (o > 0.0 and c > 0.0) else 0.0
        per_bar = 0.5 * hl * hl - _GK_OC_COEFF * oc * oc
        samples.append(max(0.0, per_bar))
    if not samples:
        return 0.0
    var = sum(samples) / len(samples)
    return math.sqrt(max(0.0, var))


def _realized_volatility(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return float(statistics.pstdev(returns))


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


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _classify_regime(sigma_ratio: float) -> VolatilityRegime:
    if sigma_ratio < _RATIO_COMPRESSED:
        return VolatilityRegime.COMPRESSED
    if sigma_ratio <= _RATIO_NORMAL_HI:
        return VolatilityRegime.NORMAL
    if sigma_ratio <= _RATIO_ELEVATED_HI:
        return VolatilityRegime.ELEVATED
    return VolatilityRegime.EXTREME


def atr_multiplier_for_ratio(sigma_ratio: float) -> float:
    """Public clamped ATR-multiplier formula. Exposed so tests can
    pin the clamping behaviour at the boundary."""
    raw = 2.0 + 0.8 * (float(sigma_ratio) - 1.0)
    return round(max(ATR_MULT_LO, min(ATR_MULT_HI, raw)), 4)


_atr_multiplier = atr_multiplier_for_ratio


def compute_stop_recommendation(
    *,
    bars: Iterable[Mapping[str, float]],
) -> StopRecommendation:
    """Compute an advisory :class:`StopRecommendation` from OHLC bars.

    Requires ≥ 90 bars (30 for the rolling HV plus 60 for its MA).
    With fewer bars, returns a NORMAL recommendation with neutral
    multipliers and an ``insufficient_bars`` blocker.
    """
    bar_list = list(bars)
    parkinson = parkinson_volatility(bar_list)
    gk = garman_klass_volatility(bar_list)

    if len(bar_list) < _MIN_BARS_FOR_RATIO:
        return StopRecommendation(
            vol_regime=VolatilityRegime.NORMAL,
            atr_multiplier=2.0,
            position_scale=POSITION_SCALE_BY_REGIME[VolatilityRegime.NORMAL],
            parkinson_vol=round(parkinson, 6),
            garman_klass_vol=round(gk, 6),
            realized_vol_30=0.0,
            realized_vol_ma_60=0.0,
            sigma_ratio=1.0,
            reasoning=(
                "insufficient_bars; advisory falls back to NORMAL "
                "(ATR ×2.0, position scale 1.0)"
            ),
            n_bars_observed=len(bar_list),
            blocking_conditions=(
                f"insufficient_bars:n={len(bar_list)}<{_MIN_BARS_FOR_RATIO}",
            ),
        )

    returns = _ohlc_to_log_returns(bar_list)
    if len(returns) < _HV_WINDOW + _MA_WINDOW:
        return StopRecommendation(
            vol_regime=VolatilityRegime.NORMAL,
            atr_multiplier=2.0,
            position_scale=POSITION_SCALE_BY_REGIME[VolatilityRegime.NORMAL],
            parkinson_vol=round(parkinson, 6),
            garman_klass_vol=round(gk, 6),
            realized_vol_30=0.0, realized_vol_ma_60=0.0, sigma_ratio=1.0,
            reasoning=(
                "insufficient_returns_after_filter; advisory falls "
                "back to NORMAL"
            ),
            n_bars_observed=len(bar_list),
            blocking_conditions=("insufficient_returns_after_filter",),
        )

    # HV(30) at the latest bar.
    hv30_now = _realized_volatility(returns[-_HV_WINDOW:])
    # MA of HV(30) over the last 60 windows. We slide the 30-bar HV
    # across the most recent 60 positions.
    ma_window: list[float] = []
    n = len(returns)
    for end in range(n - _MA_WINDOW + 1, n + 1):
        if end - _HV_WINDOW < 0:
            continue
        ma_window.append(
            _realized_volatility(returns[end - _HV_WINDOW: end])
        )
    if not ma_window:
        ma_hv = 0.0
    else:
        ma_hv = sum(ma_window) / len(ma_window)

    sigma_ratio = (hv30_now / ma_hv) if ma_hv > 0.0 else 1.0
    regime = _classify_regime(sigma_ratio)
    atr_mult = _atr_multiplier(sigma_ratio)
    pos_scale = POSITION_SCALE_BY_REGIME[regime]

    reasoning = (
        f"σ_ratio={sigma_ratio:.3f} (HV30={hv30_now:.5f} / "
        f"MA60(HV30)={ma_hv:.5f}); regime={regime.value}; "
        f"ATR ×{atr_mult:.2f}; position scale ×{pos_scale:.2f}; "
        f"Parkinson={parkinson:.5f}, Garman-Klass={gk:.5f}"
    )

    return StopRecommendation(
        vol_regime=regime,
        atr_multiplier=atr_mult,
        position_scale=pos_scale,
        parkinson_vol=round(parkinson, 6),
        garman_klass_vol=round(gk, 6),
        realized_vol_30=round(hv30_now, 6),
        realized_vol_ma_60=round(ma_hv, 6),
        sigma_ratio=round(sigma_ratio, 4),
        reasoning=reasoning,
        n_bars_observed=len(bar_list),
        blocking_conditions=(),
    )
