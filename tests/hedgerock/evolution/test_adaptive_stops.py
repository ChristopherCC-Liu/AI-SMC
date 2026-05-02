"""Tests for the adaptive stop-loss advisory sidecar."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from smc.hedgerock.evolution.adaptive_stops import (
    ATR_MULT_HI,
    ATR_MULT_LO,
    POSITION_SCALE_BY_REGIME,
    StopRecommendation,
    VolatilityRegime,
    atr_multiplier_for_ratio,
    compute_stop_recommendation,
    garman_klass_volatility,
    parkinson_volatility,
)


_REPO = Path(__file__).resolve().parents[3]


def _bars_with_realized_vol(
    n: int, *, recent_sigma: float, history_sigma: float,
    base: float = 2000.0,
) -> list[dict]:
    """Build OHLC bars with a controllable contrast between the
    recent 30-bar window's realised vol and the prior 60 bars'."""
    out: list[dict] = []
    price = base
    pattern = (1.0, -1.0, 1.0, -1.0)
    # First (n-30) bars: history_sigma. Last 30: recent_sigma.
    for i in range(n):
        sigma = history_sigma if i < n - 30 else recent_sigma
        step = sigma * pattern[i % len(pattern)]
        new = price * math.exp(step)
        out.append({
            "open": price, "close": new,
            "high": max(price, new) * 1.001,
            "low": min(price, new) * 0.999,
        })
        price = new
    return out


# ---------------------------------------------------------------------------
# Parkinson + Garman-Klass estimators
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parkinson_volatility_zero_for_flat_bars() -> None:
    bars = [
        {"open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0}
        for _ in range(20)
    ]
    assert parkinson_volatility(bars) == 0.0


@pytest.mark.unit
def test_parkinson_volatility_known_constant_value() -> None:
    """For H/L = e^σ / e^(-σ) = e^(2σ), per-bar (ln(H/L))² = 4σ².
    σ²_P = (1/(4n·ln2)) · n · 4σ² = σ²/ln2 → σ_P = σ/√ln2.
    """
    sigma = 0.05
    h = math.exp(sigma)
    l = math.exp(-sigma)
    bars = [{"open": 1.0, "close": 1.0, "high": h, "low": l}
            for _ in range(50)]
    expected = sigma / math.sqrt(math.log(2.0))
    assert parkinson_volatility(bars) == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_parkinson_skips_invalid_bars() -> None:
    bars = [
        {"open": 1.0, "close": 1.0, "high": 1.05, "low": 0.95},
        {"open": -1.0, "close": -1.0, "high": -1.0, "low": -1.0},  # invalid
        {"open": 1.0, "close": 1.0, "high": 1.05, "low": 0.95},
    ]
    # Should not crash, should return finite positive value.
    out = parkinson_volatility(bars)
    assert out > 0.0
    assert math.isfinite(out)


@pytest.mark.unit
def test_garman_klass_volatility_zero_for_flat_bars() -> None:
    bars = [
        {"open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0}
        for _ in range(20)
    ]
    assert garman_klass_volatility(bars) == 0.0


@pytest.mark.unit
def test_garman_klass_finite_on_realistic_bars() -> None:
    bars = [
        {"open": 1.0, "close": 1.0 + 0.01 * (i % 2),
         "high": 1.02, "low": 0.98}
        for i in range(30)
    ]
    out = garman_klass_volatility(bars)
    assert out > 0.0
    assert math.isfinite(out)


@pytest.mark.unit
def test_garman_klass_handles_invalid_inputs() -> None:
    bars = [
        {"open": 1.0, "close": 1.0, "high": 1.02, "low": 0.98},
        {"open": "x", "close": None, "high": -1, "low": 0},
        {"open": 1.0, "close": 1.0, "high": 1.02, "low": 0.98},
    ]
    out = garman_klass_volatility(bars)
    assert out > 0.0


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_compressed_when_recent_vol_below_history() -> None:
    bars = _bars_with_realized_vol(
        n=120, recent_sigma=0.001, history_sigma=0.005,
    )
    rec = compute_stop_recommendation(bars=bars)
    assert rec.vol_regime == VolatilityRegime.COMPRESSED
    assert rec.position_scale == 1.2
    # COMPRESSED → ATR multiplier in lower half of band.
    assert rec.atr_multiplier < 2.0


@pytest.mark.unit
def test_recommendation_normal_when_recent_matches_history() -> None:
    bars = _bars_with_realized_vol(
        n=120, recent_sigma=0.005, history_sigma=0.005,
    )
    rec = compute_stop_recommendation(bars=bars)
    assert rec.vol_regime == VolatilityRegime.NORMAL
    assert rec.position_scale == 1.0


@pytest.mark.unit
def test_recommendation_elevated_when_recent_15x_history() -> None:
    bars = _bars_with_realized_vol(
        n=120, recent_sigma=0.0075, history_sigma=0.005,
    )
    rec = compute_stop_recommendation(bars=bars)
    assert rec.vol_regime == VolatilityRegime.ELEVATED
    assert rec.position_scale == 0.6


@pytest.mark.unit
def test_recommendation_extreme_when_recent_3x_history() -> None:
    bars = _bars_with_realized_vol(
        n=120, recent_sigma=0.020, history_sigma=0.005,
    )
    rec = compute_stop_recommendation(bars=bars)
    assert rec.vol_regime == VolatilityRegime.EXTREME
    assert rec.position_scale == 0.3
    # EXTREME → ATR multiplier in upper half of band.
    assert rec.atr_multiplier > 2.0


# ---------------------------------------------------------------------------
# Multiplier formula + clamping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_atr_multiplier_clamped_at_low_end_via_formula() -> None:
    """Direct formula test — sigma_ratio of 0 hits the lower clamp."""
    assert atr_multiplier_for_ratio(0.0) == pytest.approx(ATR_MULT_LO)
    assert atr_multiplier_for_ratio(-1.0) == pytest.approx(ATR_MULT_LO)


@pytest.mark.unit
def test_atr_multiplier_clamped_at_high_end_via_formula() -> None:
    """Direct formula test — large sigma_ratio hits the upper clamp."""
    assert atr_multiplier_for_ratio(10.0) == pytest.approx(ATR_MULT_HI)
    assert atr_multiplier_for_ratio(100.0) == pytest.approx(ATR_MULT_HI)


@pytest.mark.unit
def test_atr_multiplier_within_band_on_realized_inputs() -> None:
    for recent, hist in [
        (0.0001, 0.01), (0.001, 0.005), (0.005, 0.005),
        (0.01, 0.005), (0.05, 0.005),
    ]:
        rec = compute_stop_recommendation(
            bars=_bars_with_realized_vol(
                n=120, recent_sigma=recent, history_sigma=hist,
            ),
        )
        assert ATR_MULT_LO <= rec.atr_multiplier <= ATR_MULT_HI


@pytest.mark.unit
def test_atr_multiplier_increases_with_sigma_ratio() -> None:
    rec_low = compute_stop_recommendation(
        bars=_bars_with_realized_vol(
            n=120, recent_sigma=0.005, history_sigma=0.005,
        )
    )
    rec_high = compute_stop_recommendation(
        bars=_bars_with_realized_vol(
            n=120, recent_sigma=0.012, history_sigma=0.005,
        )
    )
    assert rec_high.atr_multiplier > rec_low.atr_multiplier


# ---------------------------------------------------------------------------
# Position scale lookup
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_position_scale_table_matches_spec() -> None:
    assert POSITION_SCALE_BY_REGIME == {
        VolatilityRegime.COMPRESSED: 1.2,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.ELEVATED: 0.6,
        VolatilityRegime.EXTREME: 0.3,
    }


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_bars_returns_normal_with_blocker() -> None:
    rec = compute_stop_recommendation(
        bars=_bars_with_realized_vol(
            n=30, recent_sigma=0.005, history_sigma=0.005,
        ),
    )
    assert rec.vol_regime == VolatilityRegime.NORMAL
    assert rec.atr_multiplier == 2.0
    assert any("insufficient" in b for b in rec.blocking_conditions)


@pytest.mark.unit
def test_empty_bars_handled_gracefully() -> None:
    rec = compute_stop_recommendation(bars=[])
    assert rec.vol_regime == VolatilityRegime.NORMAL
    assert rec.n_bars_observed == 0


@pytest.mark.unit
def test_malformed_bars_do_not_crash() -> None:
    bars = _bars_with_realized_vol(
        n=120, recent_sigma=0.005, history_sigma=0.005,
    )
    bars[5] = {"open": "?", "close": None, "high": -1, "low": 0}
    bars[10] = {}
    rec = compute_stop_recommendation(bars=bars)
    assert isinstance(rec, StopRecommendation)


# ---------------------------------------------------------------------------
# Reasoning + advisory contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reasoning_text_contains_keys() -> None:
    rec = compute_stop_recommendation(
        bars=_bars_with_realized_vol(
            n=120, recent_sigma=0.005, history_sigma=0.005,
        ),
    )
    for keyword in ("σ_ratio", "regime=", "ATR", "position scale",
                    "Parkinson", "Garman-Klass"):
        assert keyword in rec.reasoning


@pytest.mark.unit
def test_recommendation_is_advisory_only() -> None:
    rec = compute_stop_recommendation(
        bars=_bars_with_realized_vol(
            n=120, recent_sigma=0.005, history_sigma=0.005,
        ),
    )
    assert rec.advisory_only is True


@pytest.mark.unit
def test_recommendation_is_frozen() -> None:
    rec = compute_stop_recommendation(bars=[])
    with pytest.raises(Exception):
        rec.atr_multiplier = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source-level isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adaptive_stops_is_isolated_from_runtime() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "adaptive_stops.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "import smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
        "from smc.hedgerock import decision_server",
        "from smc.hedgerock import phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src
