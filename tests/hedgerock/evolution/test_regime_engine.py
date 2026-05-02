"""Tests for the Regime Engine sidecar."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from smc.hedgerock.evolution.regime_engine import (
    GATE_BASE_WEIGHTS,
    MarketRegime,
    RegimeDetector,
    RegimeSnapshot,
    regime_adaptive_weights,
)


_REPO = Path(__file__).resolve().parents[3]


def _bars(n: int, sigma: float, *, base: float = 2000.0) -> list[dict]:
    """Generate n synthetic OHLC bars with a target log-return stddev.

    Mixes magnitudes (0.5×, 1.0×, 1.5× sigma) so the ATR series has
    real variance — keeps the ATR-percentile escalator from firing
    spuriously on perfectly stable amplitudes. Deterministic.
    """
    out: list[dict] = []
    price = base
    pattern = (0.5, -1.0, 1.5, -0.5, 1.0, -1.5)
    for i in range(n):
        step = sigma * pattern[i % len(pattern)]
        new = price * math.exp(step)
        high = max(price, new) * (1.0 + abs(step) * 0.1)
        low = min(price, new) * (1.0 - abs(step) * 0.1)
        out.append({
            "open": price, "close": new, "high": high, "low": low,
        })
        price = new
    return out


# ---------------------------------------------------------------------------
# Classification across volatility tiers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_low_vol_regime_detected_for_quiet_tape() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.001))
    assert snap.regime == MarketRegime.LOW_VOL
    assert snap.n_bars_observed == 60
    assert snap.blocking_conditions == ()


@pytest.mark.unit
def test_normal_regime_detected_for_typical_xauusd() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.006))
    assert snap.regime == MarketRegime.NORMAL


@pytest.mark.unit
def test_high_vol_regime_detected() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.012))
    assert snap.regime == MarketRegime.HIGH_VOL


@pytest.mark.unit
def test_extreme_regime_detected() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.022))
    assert snap.regime == MarketRegime.EXTREME


@pytest.mark.unit
def test_crisis_regime_detected_under_huge_vol() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.040))
    assert snap.regime == MarketRegime.CRISIS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_bars_returns_normal_with_blocker() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(5, sigma=0.05))
    assert snap.regime == MarketRegime.NORMAL
    assert snap.confidence == 0.0
    assert any("insufficient_bars" in b for b in snap.blocking_conditions)


@pytest.mark.unit
def test_empty_bars_handled_gracefully() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=[])
    assert snap.regime == MarketRegime.NORMAL
    assert snap.n_bars_observed == 0


@pytest.mark.unit
def test_malformed_bars_do_not_crash() -> None:
    det = RegimeDetector()
    bars = _bars(60, sigma=0.006)
    bars[5] = {"open": "bogus", "close": None, "high": -1, "low": 0}
    bars[10] = {}
    snap = det.detect(bars=bars)
    assert isinstance(snap, RegimeSnapshot)


@pytest.mark.unit
def test_gap_detection_escalates_regime() -> None:
    det = RegimeDetector()
    bars = _bars(60, sigma=0.004)
    # Inject a 1.5% gap in the last short-window position.
    bars[-2]["close"] = 2000.0
    bars[-1]["open"] = 2030.0  # +1.5% gap
    bars[-1]["close"] = 2030.0
    bars[-1]["high"] = 2032.0
    bars[-1]["low"] = 2025.0
    snap = det.detect(bars=bars)
    assert snap.regime in {MarketRegime.EXTREME, MarketRegime.CRISIS}
    assert snap.max_gap_pct >= 0.014


# ---------------------------------------------------------------------------
# Macro context
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_high_vix_escalates_to_high_vol_minimum() -> None:
    det = RegimeDetector()
    snap = det.detect(
        bars=_bars(60, sigma=0.001),  # would be LOW_VOL
        macro={"VIX": 32.0},
    )
    assert snap.regime in {MarketRegime.HIGH_VOL, MarketRegime.EXTREME,
                           MarketRegime.CRISIS}


@pytest.mark.unit
def test_extreme_vix_pushes_to_extreme_minimum() -> None:
    det = RegimeDetector()
    snap = det.detect(
        bars=_bars(60, sigma=0.006),
        macro={"VIX": 45.0},
    )
    assert snap.regime in {MarketRegime.EXTREME, MarketRegime.CRISIS}


@pytest.mark.unit
def test_macro_context_round_trip() -> None:
    det = RegimeDetector()
    snap = det.detect(
        bars=_bars(60, sigma=0.006),
        macro={"VIX": 18.5, "DXY": 104.2, "US10Y": 4.31},
    )
    assert snap.macro_context == {"VIX": 18.5, "DXY": 104.2, "US10Y": 4.31}


# ---------------------------------------------------------------------------
# Snapshot is frozen + transition probs sum to ~1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_is_frozen() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.006))
    with pytest.raises(Exception):
        snap.regime = MarketRegime.CRISIS  # type: ignore[misc]


@pytest.mark.unit
def test_transition_probabilities_sum_to_one() -> None:
    det = RegimeDetector()
    snap = det.detect(bars=_bars(60, sigma=0.006))
    total = sum(p for _, p in snap.transition_probs)
    assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Adaptive weights
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extreme_regime_boosts_safety_gates_50_pct() -> None:
    weights = regime_adaptive_weights(regime=MarketRegime.EXTREME)
    for g in ("G1", "G2", "G3"):
        assert weights[g] == pytest.approx(1.50)


@pytest.mark.unit
def test_extreme_regime_dampens_aggressive_gates_30_pct() -> None:
    weights = regime_adaptive_weights(regime=MarketRegime.EXTREME)
    for g in ("G7", "G8"):
        assert weights[g] == pytest.approx(0.70)


@pytest.mark.unit
def test_crisis_regime_matches_extreme_pattern() -> None:
    extreme = regime_adaptive_weights(regime=MarketRegime.EXTREME)
    crisis = regime_adaptive_weights(regime=MarketRegime.CRISIS)
    for k in ("G1", "G2", "G3", "G7", "G8"):
        assert extreme[k] == crisis[k]


@pytest.mark.unit
def test_normal_regime_leaves_weights_unchanged() -> None:
    weights = regime_adaptive_weights(regime=MarketRegime.NORMAL)
    for g, w in GATE_BASE_WEIGHTS.items():
        assert weights[g] == pytest.approx(w)


@pytest.mark.unit
def test_low_vol_regime_relaxes_aggressive_gates_slightly() -> None:
    weights = regime_adaptive_weights(regime=MarketRegime.LOW_VOL)
    assert weights["G7"] > 1.0
    assert weights["G8"] > 1.0
    assert weights["G1"] == pytest.approx(1.0)


@pytest.mark.unit
def test_high_vol_regime_intermediate_between_normal_and_extreme() -> None:
    h = regime_adaptive_weights(regime=MarketRegime.HIGH_VOL)
    e = regime_adaptive_weights(regime=MarketRegime.EXTREME)
    n = regime_adaptive_weights(regime=MarketRegime.NORMAL)
    assert n["G1"] < h["G1"] < e["G1"]
    assert e["G7"] < h["G7"] < n["G7"]


@pytest.mark.unit
def test_adaptive_weights_returns_fresh_dict() -> None:
    a = regime_adaptive_weights(regime=MarketRegime.NORMAL)
    a["G1"] = 999.0
    b = regime_adaptive_weights(regime=MarketRegime.NORMAL)
    assert b["G1"] != 999.0


@pytest.mark.unit
def test_adaptive_weights_accepts_custom_base_weights() -> None:
    custom = {"G1": 2.0, "G2": 2.0, "G3": 2.0,
              "G4": 2.0, "G5": 2.0, "G6": 2.0,
              "G7": 2.0, "G8": 2.0}
    weights = regime_adaptive_weights(
        regime=MarketRegime.EXTREME, base_weights=custom,
    )
    assert weights["G1"] == pytest.approx(3.0)  # 2.0 * 1.5
    assert weights["G7"] == pytest.approx(1.4)  # 2.0 * 0.7


# ---------------------------------------------------------------------------
# Source-level isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_regime_engine_does_not_import_rule_engine_or_unsealed_modules(
) -> None:
    """Regime engine sits upstream of the candidate generator and
    does NOT need the unsealed prod modules. Keep its surface tight."""
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "regime_engine.py"
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
        assert f not in src, f"regime_engine imports forbidden: {f!r}"
