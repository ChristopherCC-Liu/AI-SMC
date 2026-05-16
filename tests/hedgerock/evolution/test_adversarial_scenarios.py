"""Tests for the adversarial-scenarios sidecar."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from smc.hedgerock.evolution.adversarial_scenarios import (
    BUILTIN_SCENARIOS,
    OHLCBar,
    SCENARIO_CATEGORIES,
    SEVERITY_RANGE,
    StressScenario,
    generate_synthetic_stress,
    iter_builtin_scenarios,
    scenario_to_window_history,
)


_REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Builtin scenarios
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_builtin_scenarios_contains_six_named_entries() -> None:
    expected = {
        "covid_crash_2020_03",
        "russia_ukraine_2022_02",
        "svb_crisis_2023_03",
        "mideast_2023_10",
        "yen_intervention_2024_04",
        "fed_pivot_2023_12",
    }
    assert set(BUILTIN_SCENARIOS.keys()) == expected


@pytest.mark.unit
def test_builtin_scenarios_data_integrity() -> None:
    """Every builtin scenario has consistent fields + non-empty bars."""
    for sid, scen in BUILTIN_SCENARIOS.items():
        assert scen.scenario_id == sid
        assert scen.name
        assert scen.description
        assert scen.category in SCENARIO_CATEGORIES
        assert SEVERITY_RANGE[0] <= scen.severity <= SEVERITY_RANGE[1]
        assert len(scen.ohlc_bars) == scen.duration_bars >= 8
        assert scen.max_drawdown_pct >= 0.0
        assert scen.source == "historical"
        # OHLC sanity per bar.
        for b in scen.ohlc_bars:
            assert b.high >= max(b.open, b.close)
            assert b.low <= min(b.open, b.close)
            assert b.high > 0 and b.low > 0


@pytest.mark.unit
def test_builtin_scenarios_max_drawdown_matches_close_history() -> None:
    """The recorded ``max_drawdown_pct`` must equal the peak-to-trough
    drawdown computed from the bars' closes (within rounding)."""
    for scen in BUILTIN_SCENARIOS.values():
        closes = [b.close for b in scen.ohlc_bars]
        peak = closes[0]
        worst = 0.0
        for c in closes:
            peak = max(peak, c)
            if peak > 0:
                worst = max(worst, (peak - c) / peak * 100.0)
        assert worst == pytest.approx(scen.max_drawdown_pct, abs=1e-3)


@pytest.mark.unit
def test_iter_builtin_scenarios_round_trip() -> None:
    scenarios = iter_builtin_scenarios()
    assert tuple(BUILTIN_SCENARIOS.values()) == scenarios


@pytest.mark.unit
def test_builtin_categories_cover_required_groups() -> None:
    """The user-spec required at least liquidity / geopolitical /
    flash_crash / central_bank coverage. Verify the builtin pack hits
    each."""
    cats = {s.category for s in BUILTIN_SCENARIOS.values()}
    for required in ("liquidity", "geopolitical", "flash_crash",
                     "central_bank"):
        assert required in cats, (
            f"missing required category: {required}; got {cats}"
        )


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthetic_up_scenario_has_positive_first_half_drift() -> None:
    s = generate_synthetic_stress(
        scenario_id="syn_up", base_vol=0.01,
        shock_multiplier=2.0, duration_bars=20, direction="up",
    )
    half = s.duration_bars // 2
    closes = [b.close for b in s.ohlc_bars]
    # First-half end should be ABOVE start.
    assert closes[half - 1] > closes[0]


@pytest.mark.unit
def test_synthetic_down_scenario_has_negative_first_half_drift() -> None:
    s = generate_synthetic_stress(
        scenario_id="syn_down", base_vol=0.01,
        shock_multiplier=2.0, duration_bars=20, direction="down",
    )
    half = s.duration_bars // 2
    closes = [b.close for b in s.ohlc_bars]
    assert closes[half - 1] < closes[0]


@pytest.mark.unit
def test_synthetic_recovery_leg_pulls_back() -> None:
    """Second half pulls back HALF the shock distance."""
    s = generate_synthetic_stress(
        scenario_id="syn_pull", base_vol=0.01,
        shock_multiplier=2.0, duration_bars=20, direction="down",
    )
    closes = [b.close for b in s.ohlc_bars]
    half = s.duration_bars // 2
    # Total recovery direction is opposite the shock.
    assert closes[-1] > closes[half - 1]
    # But not a full mean-reversion to start.
    assert closes[-1] < closes[0]


@pytest.mark.unit
def test_synthetic_severity_validated() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_stress(
            scenario_id="syn", base_vol=0.01, shock_multiplier=2.0,
            duration_bars=20, direction="up", severity=99,
        )


@pytest.mark.unit
def test_synthetic_short_duration_rejected() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_stress(
            scenario_id="syn", base_vol=0.01, shock_multiplier=2.0,
            duration_bars=2, direction="up",
        )


@pytest.mark.unit
def test_synthetic_invalid_direction_rejected() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_stress(
            scenario_id="syn", base_vol=0.01, shock_multiplier=2.0,
            duration_bars=20, direction="sideways",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_synthetic_negative_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_stress(
            scenario_id="syn", duration_bars=20, direction="up",
            base_vol=-1, shock_multiplier=2.0,
        )
    with pytest.raises(ValueError):
        generate_synthetic_stress(
            scenario_id="syn", duration_bars=20, direction="up",
            base_vol=0.01, shock_multiplier=-1,
        )


@pytest.mark.unit
def test_synthetic_is_deterministic() -> None:
    a = generate_synthetic_stress(
        scenario_id="syn_a", base_vol=0.012, shock_multiplier=1.8,
        duration_bars=24, direction="down",
    )
    b = generate_synthetic_stress(
        scenario_id="syn_a", base_vol=0.012, shock_multiplier=1.8,
        duration_bars=24, direction="down",
    )
    assert a.ohlc_bars == b.ohlc_bars


@pytest.mark.unit
def test_synthetic_marks_source_synthetic() -> None:
    s = generate_synthetic_stress(
        scenario_id="syn_source", base_vol=0.01, shock_multiplier=1.5,
        duration_bars=20, direction="up",
    )
    assert s.source == "synthetic"


# ---------------------------------------------------------------------------
# Frozen / immutability + JSON serialisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stress_scenario_is_frozen() -> None:
    scen = next(iter(BUILTIN_SCENARIOS.values()))
    with pytest.raises(Exception):
        scen.name = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_ohlc_bar_is_frozen() -> None:
    bar = BUILTIN_SCENARIOS["covid_crash_2020_03"].ohlc_bars[0]
    with pytest.raises(Exception):
        bar.close = 0.0  # type: ignore[misc]


@pytest.mark.unit
def test_scenario_round_trips_through_json() -> None:
    scen = BUILTIN_SCENARIOS["svb_crisis_2023_03"]
    blob = json.dumps(asdict(scen))
    restored = json.loads(blob)
    assert restored["scenario_id"] == scen.scenario_id
    assert restored["category"] == scen.category
    assert restored["max_drawdown_pct"] == scen.max_drawdown_pct
    assert len(restored["ohlc_bars"]) == scen.duration_bars


# ---------------------------------------------------------------------------
# scenario_to_window_history adapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scenario_to_window_history_emits_window_dicts() -> None:
    scen = BUILTIN_SCENARIOS["covid_crash_2020_03"]
    history = scenario_to_window_history(scen, window_bars=4)
    assert history
    for w in history:
        assert set(w.keys()) >= {"window_id", "pnl_pp", "dd_pp",
                                  "n_signals", "regime_bucket"}
        assert w["window_id"].startswith("covid_crash_2020_03_w")
        assert w["regime_bucket"] == "liquidity"
        assert w["dd_pp"] >= 0.0


@pytest.mark.unit
def test_scenario_to_window_history_rejects_zero_window_bars() -> None:
    with pytest.raises(ValueError):
        scenario_to_window_history(
            BUILTIN_SCENARIOS["fed_pivot_2023_12"], window_bars=0,
        )


# ---------------------------------------------------------------------------
# Source-level isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_scenarios_does_not_import_rule_engine() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "adversarial_scenarios.py"
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
