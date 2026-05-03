"""Tests for hedgerock.schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from smc.hedgerock.schemas import (
    EXIT_DIRECTIVES,
    NEWS_DIRECTIONS,
    NEWS_INTENSITIES,
    SCHEMA_VERSION,
    SUPPORTED_TIMEFRAMES,
    SignalEnvelope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 4, 26, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def minimal_payload(now_utc: datetime) -> dict:
    # v2.0.0: regime is the lowercase RegimeV2 enum (range/trend_up/...).
    return {
        "symbol": "XAUUSD",
        "generated_at": now_utc,
        "active_timeframe": "H1",
        "active_strategy_id": "xauusd_h1_trend",
        "regime": "trend_up",
    }


# ---------------------------------------------------------------------------
# Required-field happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minimal_envelope_constructs(minimal_payload: dict) -> None:
    env = SignalEnvelope(**minimal_payload)
    assert env.symbol == "XAUUSD"
    assert env.active_timeframe == "H1"
    assert env.regime == "trend_up"
    assert env.schema_version == SCHEMA_VERSION
    # Defaults
    assert env.exit_directive == "none"
    assert env.grid_multiplier == 1.0
    assert env.lot_factor == 1.0
    assert env.news_intensity == "none"
    assert env.news_direction is None
    assert env.transition_lock_until_ts is None
    assert env.cooldown_until is None
    assert env.prev_regime is None
    assert env.confidence == 0.5
    # v2.0.0 defaults: safest possible — observe mode, hedgerock off.
    assert env.mode == "observe"
    assert env.hedgerock_enabled is False
    assert env.risk_tier == "observe"
    assert env.reason == ""


@pytest.mark.unit
def test_envelope_is_frozen(minimal_payload: dict) -> None:
    env = SignalEnvelope(**minimal_payload)
    with pytest.raises(ValidationError):
        env.symbol = "BTCUSD"  # type: ignore[misc]


@pytest.mark.unit
def test_envelope_round_trips_through_json(minimal_payload: dict) -> None:
    env = SignalEnvelope(**minimal_payload)
    blob = env.model_dump_json()
    revived = SignalEnvelope.model_validate_json(blob)
    assert revived == env


# ---------------------------------------------------------------------------
# Validation: required fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing",
    ["symbol", "generated_at", "active_timeframe", "active_strategy_id", "regime"],
)
def test_missing_required_field_rejected(minimal_payload: dict, missing: str) -> None:
    payload = {k: v for k, v in minimal_payload.items() if k != missing}
    with pytest.raises(ValidationError):
        SignalEnvelope(**payload)


@pytest.mark.unit
def test_empty_symbol_rejected(minimal_payload: dict) -> None:
    minimal_payload["symbol"] = ""
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


@pytest.mark.unit
def test_empty_strategy_id_rejected(minimal_payload: dict) -> None:
    minimal_payload["active_strategy_id"] = ""
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


# ---------------------------------------------------------------------------
# Validation: enums
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("tf", SUPPORTED_TIMEFRAMES)
def test_all_supported_timeframes_accepted(minimal_payload: dict, tf: str) -> None:
    minimal_payload["active_timeframe"] = tf
    env = SignalEnvelope(**minimal_payload)
    assert env.active_timeframe == tf


@pytest.mark.unit
@pytest.mark.parametrize("bad_tf", ["M1", "M30", "D1", "W1", "h1", "H 1", ""])
def test_unsupported_timeframe_rejected(minimal_payload: dict, bad_tf: str) -> None:
    minimal_payload["active_timeframe"] = bad_tf
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


@pytest.mark.unit
@pytest.mark.parametrize("directive", EXIT_DIRECTIVES)
def test_all_exit_directives_accepted(minimal_payload: dict, directive: str) -> None:
    minimal_payload["exit_directive"] = directive
    env = SignalEnvelope(**minimal_payload)
    assert env.exit_directive == directive


@pytest.mark.unit
def test_unknown_exit_directive_rejected(minimal_payload: dict) -> None:
    minimal_payload["exit_directive"] = "panic_sell"
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


@pytest.mark.unit
@pytest.mark.parametrize("intensity", NEWS_INTENSITIES)
def test_all_news_intensities_accepted(minimal_payload: dict, intensity: str) -> None:
    minimal_payload["news_intensity"] = intensity
    env = SignalEnvelope(**minimal_payload)
    assert env.news_intensity == intensity


@pytest.mark.unit
@pytest.mark.parametrize("direction", NEWS_DIRECTIONS)
def test_all_news_directions_accepted(minimal_payload: dict, direction: str) -> None:
    minimal_payload["news_direction"] = direction
    env = SignalEnvelope(**minimal_payload)
    assert env.news_direction == direction


# ---------------------------------------------------------------------------
# Validation: numeric ranges
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0.0, -0.1, 10.01, 100.0])
def test_grid_multiplier_out_of_range_rejected(minimal_payload: dict, bad: float) -> None:
    minimal_payload["grid_multiplier"] = bad
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


@pytest.mark.unit
@pytest.mark.parametrize("good", [0.1, 0.5, 1.0, 1.5, 5.0, 10.0])
def test_grid_multiplier_in_range_accepted(minimal_payload: dict, good: float) -> None:
    minimal_payload["grid_multiplier"] = good
    env = SignalEnvelope(**minimal_payload)
    assert env.grid_multiplier == good


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.5, 5.01, 100.0])
def test_lot_factor_out_of_range_rejected(minimal_payload: dict, bad: float) -> None:
    # v2.0.0: lot_factor=0.0 IS valid (used by mode=observe / mode=halt to
    # zero out new entries). Only negative or >5.0 are rejected.
    minimal_payload["lot_factor"] = bad
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


@pytest.mark.unit
def test_lot_factor_zero_accepted_for_safe_mode(minimal_payload: dict) -> None:
    """v2.0.0: lot_factor=0 is the canonical observe/halt sizing."""
    minimal_payload["lot_factor"] = 0.0
    env = SignalEnvelope(**minimal_payload)
    assert env.lot_factor == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -100.0])
def test_confidence_out_of_range_rejected(minimal_payload: dict, bad: float) -> None:
    minimal_payload["confidence"] = bad
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


@pytest.mark.unit
@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_confidence_in_range_accepted(minimal_payload: dict, good: float) -> None:
    minimal_payload["confidence"] = good
    env = SignalEnvelope(**minimal_payload)
    assert env.confidence == good


# ---------------------------------------------------------------------------
# Extra-field rejection (forbid keeps EA producers honest)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extra_field_rejected(minimal_payload: dict) -> None:
    minimal_payload["mystery_field"] = 42
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)


# ---------------------------------------------------------------------------
# Transition lock + news scenario
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_envelope_with_lock_and_news(minimal_payload: dict, now_utc: datetime) -> None:
    minimal_payload.update(
        {
            "prev_regime": "range",
            "regime": "trend_up",
            "transition_lock_until_ts": now_utc,
            "cooldown_until": now_utc,
            "exit_directive": "urgent_take_profit",
            "grid_multiplier": 1.2,
            "lot_factor": 0.8,
            "news_intensity": "high",
            "news_direction": "with",
            "news_event_name": "NFP",
            "confidence": 0.85,
            # v2.0.0 dynamic params
            "mode": "hedgerock",
            "hedgerock_enabled": True,
            "risk_tier": "normal",
            "max_next_lot": 0.10,
            "takeprofit_points": 800,
            "stoploss_points": 5000,
            "recovery_multiplier": 1.5,
            "max_orders_buy": 3,
            "max_orders_sell": 3,
            "reason": "trend_up regime, normal risk, low spread",
        }
    )
    env = SignalEnvelope(**minimal_payload)
    assert env.prev_regime == "range"
    assert env.regime == "trend_up"
    assert env.transition_lock_until_ts == now_utc
    assert env.cooldown_until == now_utc
    assert env.exit_directive == "urgent_take_profit"
    assert env.grid_multiplier == pytest.approx(1.2)
    assert env.lot_factor == pytest.approx(0.8)
    assert env.news_intensity == "high"
    assert env.news_direction == "with"
    assert env.news_event_name == "NFP"
    assert env.confidence == pytest.approx(0.85)
    assert env.mode == "hedgerock"
    assert env.hedgerock_enabled is True
    assert env.risk_tier == "normal"
    assert env.max_next_lot == pytest.approx(0.10)
    assert env.takeprofit_points == 800
    assert env.stoploss_points == 5000
    assert env.recovery_multiplier == pytest.approx(1.5)
    assert env.max_orders_buy == 3
    assert env.max_orders_sell == 3
    assert "trend_up" in env.reason


@pytest.mark.unit
def test_news_event_name_max_length(minimal_payload: dict) -> None:
    minimal_payload["news_event_name"] = "X" * 201
    with pytest.raises(ValidationError):
        SignalEnvelope(**minimal_payload)
