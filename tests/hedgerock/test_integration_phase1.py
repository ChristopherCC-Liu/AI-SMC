"""End-to-end integration tests for HedgeRock Phase 1.

These tests exercise the full pipeline (mock provider → router →
transition_lock → envelope → cache writer) without spinning a real
HTTP server (we use FastAPI's TestClient for that, which is in-process).

Marked `@pytest.mark.integration` to allow `pytest -m unit` to skip them.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.cache_writer import write_envelope
from smc.hedgerock.decision_server import PrevRegimeStore, create_app
from smc.hedgerock.mock_provider import (
    ScriptedMockProvider,
    StaticMockProvider,
    default_static_features,
)
from smc.hedgerock.schemas import SignalEnvelope


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_static_provider_default() -> None:
    p = StaticMockProvider()
    f1 = p.get_features("XAUUSD")
    f2 = p.get_features("XAUUSD")
    assert f1 == f2
    assert f1.regime == "TREND_UP"


@pytest.mark.unit
def test_scripted_provider_cycles() -> None:
    seq = [
        default_static_features("TREND_UP"),
        default_static_features("CONSOLIDATION"),
        default_static_features("TREND_DOWN"),
    ]
    p = ScriptedMockProvider(seq)
    assert p.get_features("XAUUSD").regime == "TREND_UP"
    assert p.get_features("XAUUSD").regime == "CONSOLIDATION"
    assert p.get_features("XAUUSD").regime == "TREND_DOWN"
    # Wraps around
    assert p.get_features("XAUUSD").regime == "TREND_UP"


@pytest.mark.unit
def test_scripted_provider_empty_rejected() -> None:
    with pytest.raises(ValueError):
        ScriptedMockProvider([])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_first_call_no_lock() -> None:
    provider = StaticMockProvider(default_static_features("TREND_UP"))
    store = PrevRegimeStore()
    app = create_app(provider, store)
    client = TestClient(app)

    body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    env = SignalEnvelope.model_validate(body)

    assert env.regime == "trend_up"
    assert env.prev_regime is None
    assert env.transition_lock_until_ts is None
    assert env.active_timeframe == "H1"  # mid-vol + trend
    assert env.confidence > 0


@pytest.mark.integration
def test_full_pipeline_extreme_regime_change_locks() -> None:
    provider = ScriptedMockProvider(
        [
            default_static_features("TREND_UP"),
            default_static_features("TREND_DOWN"),
        ]
    )
    store = PrevRegimeStore()
    app = create_app(provider, store)
    client = TestClient(app)

    client.get("/signal", params={"symbol": "XAUUSD"})
    body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    env = SignalEnvelope.model_validate(body)

    assert env.prev_regime == "trend_up"
    assert env.regime == "trend_down"
    assert env.transition_lock_until_ts is not None
    delta = (env.transition_lock_until_ts - env.generated_at).total_seconds()
    assert delta == pytest.approx(7200.0, abs=2.0)


@pytest.mark.integration
def test_full_pipeline_writes_cache(tmp_path: Path) -> None:
    """Pipeline → envelope → cache_writer → on-disk JSON the EA can read."""
    provider = StaticMockProvider(default_static_features("CONSOLIDATION"))
    app = create_app(provider, PrevRegimeStore())
    client = TestClient(app)

    body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    envelope = SignalEnvelope.model_validate(body)

    target = tmp_path / "RegimeCache.json"
    write_envelope(envelope, target)
    assert target.exists()

    # The EA reads the file as JSON; confirm the shape.
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["symbol"] == "XAUUSD"
    assert raw["regime"] == "range"
    assert "active_timeframe" in raw
    assert "schema_version" in raw
    assert "exit_directive" in raw


@pytest.mark.integration
def test_full_pipeline_transition_to_safe_regime_short_lock() -> None:
    """TREND_UP → TRANSITION should produce a 900s (15min) lock — not 7200s."""
    provider = ScriptedMockProvider(
        [
            default_static_features("TREND_UP"),
            default_static_features("TRANSITION"),
        ]
    )
    app = create_app(provider, PrevRegimeStore())
    client = TestClient(app)

    client.get("/signal", params={"symbol": "XAUUSD"})
    body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    env = SignalEnvelope.model_validate(body)

    assert env.transition_lock_until_ts is not None
    delta = (env.transition_lock_until_ts - env.generated_at).total_seconds()
    assert delta == pytest.approx(900.0, abs=2.0)


@pytest.mark.integration
def test_strategy_id_includes_regime() -> None:
    """The .set lookup key must encode regime so the EA picks the right preset."""
    provider = StaticMockProvider(default_static_features("ATH_BREAKOUT"))
    app = create_app(provider, PrevRegimeStore())
    client = TestClient(app)
    body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    # Phase C-hotfix #3: active_strategy_id uses the v2 regime enum
    # (lowercase "breakout"), not the legacy MarketRegimeAI literal.
    assert "breakout" in body["active_strategy_id"]


@pytest.mark.integration
def test_repeated_signals_do_not_leak_lock() -> None:
    """Same regime polled multiple times must keep transition_lock cleared."""
    provider = StaticMockProvider(default_static_features("TREND_UP"))
    app = create_app(provider, PrevRegimeStore())
    client = TestClient(app)
    for _ in range(5):
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        assert body["transition_lock_until_ts"] is None


@pytest.mark.integration
def test_smoke_script_passes() -> None:
    """Smoke script `smoke_hedgerock_phase1.py` runs to a green exit code.

    Imported and called inline — does not spawn a subprocess.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "smoke_hedgerock_phase1",
        Path(__file__).parent.parent.parent
        / "scripts"
        / "smoke_hedgerock_phase1.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main()
    assert rc == 0, "smoke script returned non-zero"
