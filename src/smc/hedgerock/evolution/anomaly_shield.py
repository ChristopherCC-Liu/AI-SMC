"""Anomaly Shield — market-anomaly self-protection layer (sidecar).

Watches volatility / gap signals and emits an :class:`AnomalyState`
that the candidate generator and recommendation CLI consult before
issuing recommendations. The shield can:

  * tighten the confidence threshold and lengthen cooldowns at
    ELEVATED;
  * freeze new candidate recommendations at CRITICAL;
  * fully suspend the loop at LOCKDOWN.

Recovery is a graded step-down: each ``recovery_step_minutes``
window of "no fresh anomaly" lowers the level by one notch until
NORMAL.

Public surface:
  * :class:`AnomalyLevel`
  * :class:`AnomalyState`
  * :class:`AnomalyDetector`
  * :class:`ShieldAction`
  * :func:`shield_action`

Isolation: this module does NOT import ``rule_engine`` or the
Tier-1 unsealed prod modules. It runs purely on metric inputs that
the caller supplies.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence


__all__ = [
    "AnomalyDetector",
    "AnomalyLevel",
    "AnomalyState",
    "ShieldAction",
    "RECOVERY_STEP_MINUTES",
    "shield_action",
]


class AnomalyLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"
    LOCKDOWN = "LOCKDOWN"


_LEVEL_RANK: Mapping[AnomalyLevel, int] = {
    AnomalyLevel.NORMAL: 0,
    AnomalyLevel.ELEVATED: 1,
    AnomalyLevel.CRITICAL: 2,
    AnomalyLevel.LOCKDOWN: 3,
}


# Recovery step — each step lowers the level by one notch. 30 min
# matches the spec; the detector's ``next_recovery_at`` is computed
# from the last anomaly observation timestamp.
RECOVERY_STEP_MINUTES: int = 30


@dataclass(frozen=True)
class AnomalyState:
    level: AnomalyLevel
    triggers: tuple[str, ...]
    short_window_vol: float
    historical_vol_p90: float
    historical_vol_p95: float
    historical_vol_p99: float
    max_gap_pct: float
    n_bars_observed: int
    last_anomaly_at: str | None
    next_recovery_at: str | None
    blocking_conditions: tuple[str, ...]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class ShieldAction:
    level: AnomalyLevel
    confidence_threshold_multiplier: float
    cooldown_extension_minutes: int
    new_candidates_allowed: bool
    queue_frozen: bool
    full_lockdown: bool
    banner: str
    detail_reasons: tuple[str, ...]


# Spec-driven thresholds — gap fractions of price.
_GAP_ELEVATED = 0.0025
_GAP_CRITICAL = 0.005
_GAP_LOCKDOWN = 0.010


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


def _short_vol(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return float(statistics.pstdev(returns))


def _ohlc_to_log_returns(bars: Sequence[Mapping[str, float]]) -> list[float]:
    import math
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


def _max_gap_pct(bars: Sequence[Mapping[str, float]]) -> float:
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


class AnomalyDetector:
    """Stateless detector — caller supplies "now" and the previous
    state for graded recovery."""

    def __init__(
        self,
        *,
        short_window: int = 20,
        history_min_bars: int = 30,
        recovery_step_minutes: int = RECOVERY_STEP_MINUTES,
    ) -> None:
        if short_window < 2:
            raise ValueError("short_window must be >= 2")
        if history_min_bars < short_window:
            raise ValueError("history_min_bars must be >= short_window")
        self._short_window = short_window
        self._history_min_bars = history_min_bars
        self._recovery_step = recovery_step_minutes

    def detect(
        self,
        *,
        bars: Iterable[Mapping[str, float]],
        previous_state: AnomalyState | None = None,
        now: datetime | None = None,
    ) -> AnomalyState:
        bar_list = list(bars)
        ts = now or datetime.now(timezone.utc)

        if len(bar_list) < self._history_min_bars:
            return AnomalyState(
                level=AnomalyLevel.NORMAL,
                triggers=(),
                short_window_vol=0.0,
                historical_vol_p90=0.0,
                historical_vol_p95=0.0,
                historical_vol_p99=0.0,
                max_gap_pct=0.0,
                n_bars_observed=len(bar_list),
                last_anomaly_at=(
                    previous_state.last_anomaly_at if previous_state else None
                ),
                next_recovery_at=None,
                blocking_conditions=(
                    f"insufficient_bars:n={len(bar_list)}<"
                    f"{self._history_min_bars}",
                ),
            )

        short_returns = _ohlc_to_log_returns(bar_list[-self._short_window:])
        # Historical baseline = bars that came BEFORE the short window,
        # so the comparison is "current burst vs prior calm" rather
        # than "current vs current".
        history_only_bars = bar_list[: -self._short_window]
        history_returns = _ohlc_to_log_returns(history_only_bars)
        short_vol = _short_vol(short_returns)
        gap = _max_gap_pct(bar_list[-self._short_window:])

        baseline_window = 5
        history_vols: list[float] = []
        if len(history_returns) >= baseline_window * 2:
            for i in range(baseline_window, len(history_returns)):
                history_vols.append(
                    _short_vol(history_returns[i - baseline_window:i])
                )
        p90 = _percentile(history_vols, 0.90)
        p95 = _percentile(history_vols, 0.95)
        p99 = _percentile(history_vols, 0.99)

        triggers: list[str] = []
        if gap >= _GAP_LOCKDOWN:
            triggers.append(f"gap_lockdown:{gap:.4f}>={_GAP_LOCKDOWN}")
        elif gap >= _GAP_CRITICAL:
            triggers.append(f"gap_critical:{gap:.4f}>={_GAP_CRITICAL}")
        elif gap >= _GAP_ELEVATED:
            triggers.append(f"gap_elevated:{gap:.4f}>={_GAP_ELEVATED}")

        # Vol triggers combine a percentile threshold with a
        # multiplicative ratio over p90 — this avoids "every burst
        # over p99 triggers LOCKDOWN" on calm history where the
        # percentile spread is sub-bps.
        if p90 > 0.0:
            ratio = short_vol / p90
            if ratio >= 4.0 and short_vol >= 0.015:
                triggers.append(
                    f"vol_p99:{short_vol:.5f}>={p99:.5f} (ratio={ratio:.2f})"
                )
            elif ratio >= 2.5 and short_vol >= 0.005:
                triggers.append(
                    f"vol_p95:{short_vol:.5f}>={p95:.5f} (ratio={ratio:.2f})"
                )
            elif ratio >= 1.5 and short_vol >= p90:
                triggers.append(
                    f"vol_p90:{short_vol:.5f}>={p90:.5f} (ratio={ratio:.2f})"
                )

        # Compose the trigger-based level.
        triggered_level = AnomalyLevel.NORMAL
        for t in triggers:
            if t.startswith("vol_p99") or t.startswith("gap_lockdown"):
                triggered_level = max(
                    triggered_level, AnomalyLevel.LOCKDOWN, key=_level_rank,
                )
            elif t.startswith("vol_p95") or t.startswith("gap_critical"):
                triggered_level = max(
                    triggered_level, AnomalyLevel.CRITICAL, key=_level_rank,
                )
            elif t.startswith("vol_p90") or t.startswith("gap_elevated"):
                triggered_level = max(
                    triggered_level, AnomalyLevel.ELEVATED, key=_level_rank,
                )

        # Apply graded recovery from the previous state.
        graded_level = self._apply_recovery(
            triggered_level=triggered_level,
            previous_state=previous_state, now=ts,
        )
        last_anomaly_at = ts.isoformat() if triggers else (
            previous_state.last_anomaly_at if previous_state else None
        )
        next_recovery_at: str | None = None
        if graded_level != AnomalyLevel.NORMAL and last_anomaly_at:
            try:
                last_dt = datetime.fromisoformat(last_anomaly_at)
            except ValueError:
                last_dt = ts
            next_recovery_at = (
                last_dt + timedelta(minutes=self._recovery_step)
            ).isoformat()

        return AnomalyState(
            level=graded_level,
            triggers=tuple(triggers),
            short_window_vol=round(short_vol, 6),
            historical_vol_p90=round(p90, 6),
            historical_vol_p95=round(p95, 6),
            historical_vol_p99=round(p99, 6),
            max_gap_pct=round(gap, 6),
            n_bars_observed=len(bar_list),
            last_anomaly_at=last_anomaly_at,
            next_recovery_at=next_recovery_at,
            blocking_conditions=(),
        )

    def _apply_recovery(
        self,
        *,
        triggered_level: AnomalyLevel,
        previous_state: AnomalyState | None,
        now: datetime,
    ) -> AnomalyLevel:
        # No previous state → take the trigger directly.
        if previous_state is None:
            return triggered_level
        # Trigger is at or above the previous level → escalate.
        if _level_rank(triggered_level) >= _level_rank(previous_state.level):
            return triggered_level
        # Trigger is below previous → graded recovery: only step down
        # one notch per ``recovery_step_minutes`` of quiet time.
        if previous_state.last_anomaly_at is None:
            return triggered_level
        try:
            last_dt = datetime.fromisoformat(previous_state.last_anomaly_at)
        except ValueError:
            return triggered_level
        elapsed = (now - last_dt).total_seconds() / 60.0
        steps_allowed = int(elapsed // self._recovery_step)
        prev_rank = _level_rank(previous_state.level)
        new_rank = max(_level_rank(triggered_level), prev_rank - steps_allowed)
        return _level_from_rank(new_rank)


def _level_rank(level: AnomalyLevel) -> int:
    return _LEVEL_RANK[level]


def _level_from_rank(rank: int) -> AnomalyLevel:
    inverse = {v: k for k, v in _LEVEL_RANK.items()}
    return inverse.get(max(0, min(3, rank)), AnomalyLevel.NORMAL)


# ---------------------------------------------------------------------------
# Shield action
# ---------------------------------------------------------------------------


def shield_action(state: AnomalyState) -> ShieldAction:
    """Translate an :class:`AnomalyState` into the side-effect
    semantics the candidate generator + CLI need."""
    level = state.level
    detail = tuple(state.triggers)

    if level == AnomalyLevel.LOCKDOWN:
        return ShieldAction(
            level=level,
            confidence_threshold_multiplier=1.30,
            cooldown_extension_minutes=120,
            new_candidates_allowed=False,
            queue_frozen=True,
            full_lockdown=True,
            banner=(
                "MARKET ANOMALY — ALL RECOMMENDATIONS SUSPENDED "
                "(LOCKDOWN)"
            ),
            detail_reasons=detail,
        )
    if level == AnomalyLevel.CRITICAL:
        return ShieldAction(
            level=level,
            confidence_threshold_multiplier=1.20,
            cooldown_extension_minutes=60,
            new_candidates_allowed=False,
            queue_frozen=True,
            full_lockdown=False,
            banner=(
                "MARKET ANOMALY — NEW RECOMMENDATIONS PAUSED "
                "(CRITICAL)"
            ),
            detail_reasons=detail,
        )
    if level == AnomalyLevel.ELEVATED:
        return ShieldAction(
            level=level,
            confidence_threshold_multiplier=1.10,
            cooldown_extension_minutes=30,
            new_candidates_allowed=True,
            queue_frozen=False,
            full_lockdown=False,
            banner="ELEVATED ANOMALY — TIGHTENING CONFIDENCE FLOORS",
            detail_reasons=detail,
        )
    return ShieldAction(
        level=level,
        confidence_threshold_multiplier=1.0,
        cooldown_extension_minutes=0,
        new_candidates_allowed=True,
        queue_frozen=False,
        full_lockdown=False,
        banner="NORMAL — NO ANOMALY ACTIVE",
        detail_reasons=detail,
    )
