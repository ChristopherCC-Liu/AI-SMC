"""Ticket 3 Step 2 — Class C envelope mirror golden-fixture drift tests.

This test file is one of TWO places (the other being the
fixture-generator script) where importing production
``derive_envelope_params`` is permitted (per Ticket 3 plan §R3
boundary B). Sidecar runtime modules MUST NOT do this.

The strategy:
  1. Build ≥ 20 representative ``(now, regime, confidence,
     ea_state, prev_envelope_cooldown_until, overlay_params)``
     fixture cases covering every branch of derive_envelope_params:
        - ea_state_stale
        - crisis
        - news
        - dd_severe
        - spread_anomalous
        - consec_losses_cooldown
        - trend / breakout / unknown
        - no_ea_state
        - incomplete_ea_state (missing dd / spread)
        - range + sub_floor confidence (overlay-aware)
        - range + step-down by DD / consec / recent_pnl
        - range + step-up aggressive
        - range + step-up blocked (each blocker bucket)
        - range + normal
        - cooldown carryover from prev_envelope
  2. Call BOTH production ``derive_envelope_params`` AND mirror
     ``mirror_derive_envelope_params``.
  3. Assert the field-by-field results match.

Field comparison rules:
  - All MirroredDynamicParams fields must equal the corresponding
    DynamicParams fields by value.
  - cooldown_until tz-aware datetimes compare directly.
  - reason strings must match exactly (the mirror reproduces
    production's reason format word-for-word).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from smc.hedgerock.evolution.replay_envelope_mirror import (
    MirroredDynamicParams,
    mirror_derive_envelope_params,
)


_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _build_market_state(
    *,
    regime: str,
    confidence: float,
    regime_reason: str = "fixture",
    ea_state_stale: bool = False,
    ea_state_age_seconds: float | None = 5.0,
    dd_pct: float | None = 0.0,
    consec: int | None = 0,
    spread_pts: int | None = 20,
    recent_pnl: float | None = 0.0,
    recent_n: int | None = 10,
    ea_state_present: bool = True,
):
    """Build a real production MarketState for derive_envelope_params,
    AND parallel kwargs for mirror_derive_envelope_params, so the
    two are evaluated against the *same* inputs."""
    from smc.hedgerock.decision_server import MarketFeatures, EAState
    from smc.hedgerock.market_state import aggregate_market_state
    from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2

    ea: EAState | None = None
    if ea_state_present:
        ea = EAState(
            equity=10000.0, balance=10000.0,
            dd_pct=dd_pct,
            free_margin=10000.0,
            margin_level=999.0,
            open_lots=0.0, open_positions=0, floating_pnl=0.0,
            spread_pts=spread_pts,
            consec_losses=consec,
            recent_closed_pnl=recent_pnl,
            recent_sample_count=recent_n,
        )

    assessment = RegimeAssessmentV2(
        regime=regime,  # type: ignore[arg-type]
        confidence=confidence,
        reason=regime_reason,
        rule_votes=(),
    )
    state = aggregate_market_state(
        symbol="XAUUSD", now=_NOW,
        features=MarketFeatures(
            volatility_rank=0.5, hh_count=3, ll_count=3,
            h4_trend_bars=1, regime="CONSOLIDATION",
        ),
        regime_assessment=assessment,
        ea_state=ea,
        ea_state_recorded_at=(
            _NOW - timedelta(seconds=ea_state_age_seconds)
            if ea_state_age_seconds is not None and ea is not None
            else None
        ),
        freshness_seconds=60,  # so any age > 60s → stale
    )

    mirror_kwargs = dict(
        now=_NOW,
        regime=regime,
        confidence=confidence,
        regime_reason=regime_reason,
        ea_state_stale=state.ea_state_stale,
        ea_state_age_seconds=state.ea_state_age_seconds,
        dd_pct=dd_pct if ea is not None else None,
        consec_losses=consec if ea is not None else None,
        spread_pts=spread_pts if ea is not None else None,
        recent_closed_pnl=recent_pnl if ea is not None else None,
        recent_sample_count=recent_n if ea is not None else None,
        ea_state_present=ea is not None,
        prev_envelope_cooldown_until=None,
        overlay_params=None,
    )
    return state, mirror_kwargs


def _compare(prod_params, mirror_params: MirroredDynamicParams, label: str):
    """Compare every field of production DynamicParams against
    MirroredDynamicParams. The mirror reuses production reason
    strings; if they differ that's a drift signal too."""
    prod_dict = {k: v for k, v in asdict(prod_params).items()}
    mirror_dict = {k: v for k, v in asdict(mirror_params).items()}
    for key in prod_dict:
        assert key in mirror_dict, f"[{label}] mirror missing field {key}"
        assert prod_dict[key] == mirror_dict[key], (
            f"[{label}] field {key!r}: prod={prod_dict[key]!r} vs "
            f"mirror={mirror_dict[key]!r}"
        )


# ---------------------------------------------------------------------------
# Fixture generator: enumerate ≥ 20 representative cases
# ---------------------------------------------------------------------------


def _fixture_cases():
    """Yield (label, mkstate_kwargs, mirror_kwargs_overrides) tuples."""
    cases: list[tuple[str, dict, dict]] = []

    # 1. EA state stale
    cases.append(("ea_state_stale", dict(
        regime="range", confidence=0.85, ea_state_age_seconds=300,
    ), {}))

    # 2. Crisis
    cases.append(("crisis", dict(
        regime="crisis", confidence=0.9,
    ), {}))

    # 3. News
    cases.append(("news", dict(
        regime="news", confidence=0.85,
    ), {}))

    # 4. DD severe
    cases.append(("dd_severe", dict(
        regime="range", confidence=0.85, dd_pct=0.06,
    ), {}))

    # 5. Spread anomalous
    cases.append(("spread_anomalous", dict(
        regime="range", confidence=0.85, spread_pts=120,
    ), {}))

    # 6. Consecutive losses cooldown
    cases.append(("consec_losses_cooldown", dict(
        regime="range", confidence=0.85, consec=5,
    ), {}))

    # 7-9. Trend / breakout / unknown
    for r in ("trend_up", "trend_down", "breakout"):
        cases.append((f"regime_{r}", dict(
            regime=r, confidence=0.85,
        ), {}))

    # 10. No EA state
    cases.append(("no_ea_state", dict(
        regime="range", confidence=0.85, ea_state_present=False,
    ), {}))

    # 11. Incomplete EA state — dd_pct missing
    cases.append(("incomplete_ea_state_no_dd", dict(
        regime="range", confidence=0.85, dd_pct=None,
    ), {}))

    # 12. Incomplete EA state — spread missing
    cases.append(("incomplete_ea_state_no_spread", dict(
        regime="range", confidence=0.85, spread_pts=None,
    ), {}))

    # 13. Range sub-floor confidence
    cases.append(("range_sub_floor_confidence", dict(
        regime="range", confidence=0.50,
    ), {}))

    # 14. Range step-down by DD
    cases.append(("range_step_down_dd", dict(
        regime="range", confidence=0.85, dd_pct=0.03,
    ), {}))

    # 15. Range step-down by consec losses
    cases.append(("range_step_down_consec", dict(
        regime="range", confidence=0.85, consec=3,
    ), {}))

    # 16. Range step-down by recent_pnl
    cases.append(("range_step_down_recent_pnl", dict(
        regime="range", confidence=0.85, recent_pnl=-100.0, recent_n=10,
    ), {}))

    # 17. Range step-up aggressive (conf ≥ 0.80, healthy history)
    cases.append(("range_step_up_aggressive", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
        consec=0, recent_pnl=50.0, recent_n=10,
    ), {}))

    # 18. Range step-up blocked: history_unavailable
    cases.append(("range_step_up_blocked_history_unavailable", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
        consec=0, recent_pnl=None, recent_n=None,
    ), {}))

    # 19. Range step-up blocked: insufficient sample
    cases.append(("range_step_up_blocked_insufficient_sample", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
        consec=0, recent_pnl=10.0, recent_n=2,
    ), {}))

    # 20. Range step-up blocked: recent_pnl_blocks
    cases.append(("range_step_up_blocked_recent_pnl_negative", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
        consec=0, recent_pnl=-5.0, recent_n=10,
    ), {}))

    # 21. Range step-up blocked: history_incomplete (missing recent_pnl)
    cases.append(("range_step_up_blocked_missing_recent_pnl", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
        consec=0, recent_pnl=None, recent_n=10,
    ), {}))

    # 22. Range normal (conf in [0.55, 0.80))
    cases.append(("range_normal", dict(
        regime="range", confidence=0.65, dd_pct=0.0,
    ), {}))

    # 23. Cooldown carryover: prev cooldown extends past now
    cases.append(("cooldown_carryover_active", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
    ), {"prev_envelope_cooldown_until": _NOW + timedelta(minutes=10)}))

    # 24. Cooldown carryover: prev cooldown expired
    cases.append(("cooldown_carryover_expired", dict(
        regime="range", confidence=0.85, dd_pct=0.0,
    ), {"prev_envelope_cooldown_until": _NOW - timedelta(minutes=10)}))

    # 25. Overlay-aware: c1 lowers OBSERVE_FLOOR to 0.50, conf=0.52
    # → without overlay would be observe; with overlay → hedgerock.
    cases.append(("overlay_c1_lower_observe_floor_to_0.50", dict(
        regime="range", confidence=0.52, dd_pct=0.0,
    ), {"overlay_params": {
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": 0.50,
    }}))

    return cases


@pytest.mark.unit
def test_at_least_20_fixture_cases() -> None:
    cases = _fixture_cases()
    assert len(cases) >= 20, f"need ≥ 20 fixture cases, have {len(cases)}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    _fixture_cases(),
    ids=lambda c: c[0],
)
def test_mirror_matches_production_on_fixture(case: tuple[str, dict, dict]) -> None:
    """For every fixture case, mirror's output must equal production's
    output field-by-field. Production is allowed to be imported in
    THIS test file only."""
    label, mkstate_kwargs, overrides = case

    # Build production market state.
    state, mirror_kwargs = _build_market_state(**mkstate_kwargs)

    # Apply overrides specific to mirror (e.g. overlay_params,
    # prev_envelope_cooldown_until).
    mirror_kwargs.update(overrides)

    # If the case provides prev_envelope_cooldown_until, build a
    # matching SignalEnvelope for production.
    prev_envelope = None
    prev_cd = overrides.get("prev_envelope_cooldown_until")
    if prev_cd is not None:
        # Production _apply_cooldown_carryover only reads .cooldown_until.
        # Build a duck-typed stub; no need to import the SignalEnvelope class.
        class _Stub:
            def __init__(self, cd):
                self.cooldown_until = cd
        prev_envelope = _Stub(prev_cd)

    # Apply overlay to production by monkeypatching the rule_engine
    # module-level constants TEMPORARILY for this test only. We
    # restore after; this is test-side setattr, not runtime
    # setattr — and only happens in test_replay_envelope_mirror.py.
    overlay = mirror_kwargs.get("overlay_params") or {}
    import importlib
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    saved: dict[str, object] = {}
    try:
        for k, v in overlay.items():
            attr = k.split(".")[-1]
            saved[attr] = getattr(rule_engine, attr)
            setattr(rule_engine, attr, v)
        prod_params = rule_engine.derive_envelope_params(state, prev_envelope)
    finally:
        for attr, v in saved.items():
            setattr(rule_engine, attr, v)

    mirror_params = mirror_derive_envelope_params(**mirror_kwargs)
    _compare(prod_params, mirror_params, label)


# ---------------------------------------------------------------------------
# Mirror version is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mirror_c_version_is_deterministic() -> None:
    from smc.hedgerock.evolution.replay_envelope_mirror import (
        compute_mirror_c_version, MIRROR_C_VERSION,
    )
    assert MIRROR_C_VERSION == compute_mirror_c_version()
    assert isinstance(MIRROR_C_VERSION, str)
    assert len(MIRROR_C_VERSION) == 64


@pytest.mark.unit
def test_is_target_supported_includes_class_a_targets() -> None:
    from smc.hedgerock.evolution.replay_envelope_mirror import is_target_supported
    assert is_target_supported(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
    ) is True
    assert is_target_supported(
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"
    ) is True


@pytest.mark.unit
def test_is_target_supported_rejects_unknown() -> None:
    from smc.hedgerock.evolution.replay_envelope_mirror import is_target_supported
    assert is_target_supported("smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE") is False
    assert is_target_supported(
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE"
    ) is False
