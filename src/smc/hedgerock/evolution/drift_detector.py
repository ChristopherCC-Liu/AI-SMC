"""Drift detector — pure-function diagnostic for HedgeRock evolution layer.

Read-only. Never mutates inputs. No file I/O, no DB, no network.
Only stdlib (datetime, math, statistics) — no extra dependencies.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

# ---------------------------------------------------------------------------
# Frozen result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftScore:
    score: float
    severity: str
    metric: str
    detail: str


@dataclass(frozen=True)
class FreshnessScore:
    age_days: float | None
    severity: str
    detail: str


@dataclass(frozen=True)
class StabilityScore:
    score: float
    severity: str
    diverged_keys: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class QualityScore:
    score: float
    severity: str
    detail: str


@dataclass(frozen=True)
class DriftReport:
    overall_severity: str
    regime_baseline: DriftScore
    evidence_freshness: FreshnessScore
    parameter_stability: StabilityScore
    recommendation_quality: QualityScore
    generated_at: str


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_CANONICAL_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3}


def _drift_severity(score: float) -> str:
    if score < 0.1:
        return "none"
    if score < 0.3:
        return "low"
    if score < 0.6:
        return "moderate"
    return "high"


def _stability_severity(score: float) -> str:
    if score < 0.05:
        return "stable"
    if score < 0.20:
        return "drifting"
    return "unstable"


def _freshness_severity(age_days: float | None) -> str:
    if age_days is None:
        return "stale"
    if age_days < 7:
        return "fresh"
    if age_days < 30:
        return "aging"
    return "stale"


def _quality_severity(score: float) -> str:
    if score >= 0.8:
        return "good"
    if score >= 0.5:
        return "watching"
    return "degraded"


_FRESHNESS_TO_CANONICAL = {"fresh": "none", "aging": "moderate", "stale": "high"}
_STABILITY_TO_CANONICAL = {
    "stable": "none",
    "drifting": "moderate",
    "unstable": "high",
}
_QUALITY_TO_CANONICAL = {"good": "none", "watching": "moderate", "degraded": "high"}


def _canonical(level: str) -> str:
    """Map any severity label to canonical 'none|low|moderate|high'."""
    if level in _CANONICAL_RANK:
        return level
    if level in _FRESHNESS_TO_CANONICAL:
        return _FRESHNESS_TO_CANONICAL[level]
    if level in _STABILITY_TO_CANONICAL:
        return _STABILITY_TO_CANONICAL[level]
    if level in _QUALITY_TO_CANONICAL:
        return _QUALITY_TO_CANONICAL[level]
    return "high"  # unknown → conservative


def _worst(*levels: str) -> str:
    canonical = [_canonical(level) for level in levels]
    return max(canonical, key=lambda lev: _CANONICAL_RANK[lev])


# ---------------------------------------------------------------------------
# Internal numeric helpers
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _bar_returns(bars: Sequence[Mapping[str, float]]) -> list[float]:
    """Compute simple returns from a series of bar mappings.

    Looks for a 'close' field; if absent, falls back to ('open' or 'price').
    Returns an empty list if fewer than 2 usable bars exist.
    """
    closes: list[float] = []
    for bar in bars:
        for key in ("close", "price", "open"):
            if key in bar:
                try:
                    closes.append(float(bar[key]))
                except (TypeError, ValueError):
                    pass
                break
    rets: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev == 0:
            continue
        rets.append((curr - prev) / prev)
    return rets


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(v) for v in values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _parse_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            text = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Pure-function diagnostic. Read-only — never modifies inputs."""

    @staticmethod
    def check_regime_baseline_decay(
        bars_recent: Sequence[Mapping[str, float]],
        bars_baseline: Sequence[Mapping[str, float]],
    ) -> DriftScore:
        recent_returns = _bar_returns(bars_recent)
        baseline_returns = _bar_returns(bars_baseline)

        if not recent_returns or not baseline_returns:
            return DriftScore(
                score=1.0,
                severity="high",
                metric="regime_baseline",
                detail="empty window",
            )

        recent_mean_abs = _mean_abs(recent_returns)
        baseline_mean_abs = _mean_abs(baseline_returns)
        recent_std = _stdev(recent_returns)
        baseline_std = _stdev(baseline_returns)

        eps = 1e-9
        mean_delta = abs(recent_mean_abs - baseline_mean_abs) / max(
            abs(baseline_mean_abs), eps
        )
        std_delta = abs(recent_std - baseline_std) / max(abs(baseline_std), eps)
        # Combine the two signals — take the larger (more conservative).
        raw = max(mean_delta, std_delta)
        score = _clamp01(raw)
        severity = _drift_severity(score)
        detail = (
            f"mean_abs_ret recent={recent_mean_abs:.6f} "
            f"baseline={baseline_mean_abs:.6f} | "
            f"std recent={recent_std:.6f} baseline={baseline_std:.6f}"
        )
        return DriftScore(
            score=score,
            severity=severity,
            metric="regime_baseline",
            detail=detail,
        )

    @staticmethod
    def check_evidence_freshness(
        evidence_registry: Mapping[str, str | datetime | None],
        *,
        now: datetime | None = None,
    ) -> FreshnessScore:
        reference = now if now is not None else datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)

        if not evidence_registry:
            return FreshnessScore(
                age_days=None,
                severity="stale",
                detail="empty evidence registry",
            )

        worst_age: float | None = None
        worst_key = ""
        has_none_entry = False

        for key, value in evidence_registry.items():
            parsed = _parse_dt(value)
            if parsed is None:
                has_none_entry = True
                worst_key = key
                worst_age = None
                break  # None always wins → stale
            age = (reference - parsed).total_seconds() / 86400.0
            if age < 0:
                age = 0.0
            if worst_age is None or age > worst_age:
                worst_age = age
                worst_key = key

        if has_none_entry:
            return FreshnessScore(
                age_days=None,
                severity="stale",
                detail=f"missing/unparseable timestamp for '{worst_key}'",
            )

        severity = _freshness_severity(worst_age)
        detail = f"oldest entry '{worst_key}' is {worst_age:.2f} days old"
        return FreshnessScore(age_days=worst_age, severity=severity, detail=detail)

    @staticmethod
    def check_parameter_stability(
        current_params: Mapping[str, float],
        baseline_params: Mapping[str, float],
    ) -> StabilityScore:
        shared_keys = sorted(set(current_params) & set(baseline_params))
        if not shared_keys:
            return StabilityScore(
                score=0.0,
                severity="stable",
                diverged_keys=(),
                detail="no shared parameters",
            )

        worst = 0.0
        diverged: list[str] = []
        per_key: list[tuple[str, float]] = []
        for key in shared_keys:
            try:
                cur = float(current_params[key])
                base = float(baseline_params[key])
            except (TypeError, ValueError):
                continue
            denom = max(abs(base), 1e-6)
            rel = abs(cur - base) / denom
            per_key.append((key, rel))
            if rel > 0.05:
                diverged.append(key)
            if rel > worst:
                worst = rel

        score = _clamp01(worst)
        severity = _stability_severity(score)
        detail = "; ".join(f"{k}:{v:.4f}" for k, v in per_key)
        return StabilityScore(
            score=score,
            severity=severity,
            diverged_keys=tuple(diverged),
            detail=detail,
        )

    @staticmethod
    def check_recommendation_quality(
        recent_outcomes: Sequence[Mapping[str, float]],
        baseline_outcomes: Sequence[Mapping[str, float]],
    ) -> QualityScore:
        if not recent_outcomes or not baseline_outcomes:
            return QualityScore(
                score=0.0,
                severity="degraded",
                detail="empty outcomes window",
            )

        def _winrate(outcomes: Sequence[Mapping[str, float]]) -> float:
            wins = 0
            for o in outcomes:
                try:
                    if float(o.get("win", 0.0)) > 0:
                        wins += 1
                except (TypeError, ValueError):
                    pass
            return wins / len(outcomes)

        def _mean_pnl(outcomes: Sequence[Mapping[str, float]]) -> float:
            vals: list[float] = []
            for o in outcomes:
                try:
                    vals.append(float(o.get("pnl", 0.0)))
                except (TypeError, ValueError):
                    pass
            return sum(vals) / len(vals) if vals else 0.0

        wr_recent = _winrate(recent_outcomes)
        wr_baseline = _winrate(baseline_outcomes)
        pnl_recent = _mean_pnl(recent_outcomes)
        pnl_baseline = _mean_pnl(baseline_outcomes)

        eps = 1e-9
        # Win-rate ratio: recent / baseline, saturates at 1.0 (parity = full credit).
        wr_ratio = max(0.0, min(wr_recent / max(wr_baseline, eps), 1.0))
        # PnL ratio: if recent meets-or-beats baseline pnl, full credit; else
        # scale linearly toward zero. Handles both positive and negative
        # baselines without sign-flip artifacts.
        if pnl_recent >= pnl_baseline:
            pnl_ratio = 1.0
        elif pnl_baseline > 0:
            pnl_ratio = max(0.0, pnl_recent / pnl_baseline)
        else:
            # Baseline non-positive and recent worse than baseline → degraded.
            pnl_ratio = 0.0

        score = _clamp01(0.5 * wr_ratio + 0.5 * pnl_ratio)
        severity = _quality_severity(score)
        detail = (
            f"winrate recent={wr_recent:.3f} baseline={wr_baseline:.3f} | "
            f"mean_pnl recent={pnl_recent:.4f} baseline={pnl_baseline:.4f}"
        )
        return QualityScore(score=score, severity=severity, detail=detail)

    def overall_assessment(
        self,
        *,
        bars_recent: Sequence[Mapping[str, float]] | None = None,
        bars_baseline: Sequence[Mapping[str, float]] | None = None,
        evidence_registry: Mapping[str, str | datetime | None] | None = None,
        current_params: Mapping[str, float] | None = None,
        baseline_params: Mapping[str, float] | None = None,
        recent_outcomes: Sequence[Mapping[str, float]] | None = None,
        baseline_outcomes: Sequence[Mapping[str, float]] | None = None,
        now: datetime | None = None,
    ) -> DriftReport:
        regime = self.check_regime_baseline_decay(
            bars_recent or (), bars_baseline or ()
        )
        freshness = self.check_evidence_freshness(
            evidence_registry or {}, now=now
        )
        stability = self.check_parameter_stability(
            current_params or {}, baseline_params or {}
        )
        quality = self.check_recommendation_quality(
            recent_outcomes or (), baseline_outcomes or ()
        )

        overall = _worst(
            regime.severity,
            freshness.severity,
            stability.severity,
            quality.severity,
        )

        ts_now = now if now is not None else datetime.now(timezone.utc)
        if ts_now.tzinfo is None:
            ts_now = ts_now.replace(tzinfo=timezone.utc)

        return DriftReport(
            overall_severity=overall,
            regime_baseline=regime,
            evidence_freshness=freshness,
            parameter_stability=stability,
            recommendation_quality=quality,
            generated_at=ts_now.isoformat(),
        )


__all__ = [
    "DriftScore",
    "FreshnessScore",
    "StabilityScore",
    "QualityScore",
    "DriftReport",
    "DriftDetector",
]
