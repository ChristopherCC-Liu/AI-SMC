"""audit-r4 v5 Option B: strategy_server /signal unified signals array.

The /signal endpoint now returns:

    {
      "symbol": "XAUUSD",
      "ts": "...", "fresh": true,
      "signals": [
        {"leg": "",       "magic": 19760418, "action": "BUY",  ...},
        {"leg": "_macro", "magic": 19760428, "action": "HOLD", ...}
      ],
      // backward-compat flat fields (mirror of signals[0] — control leg)
      "action": "BUY", ...
    }

These tests lock the array response contract and per-leg state isolation:
- Both legs appear in the array even when one is missing state.
- Each leg carries its own magic.
- signal_id is unique per leg.
- Backward-compat flat fields mirror the control leg.
- /status surfaces per-leg health.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """Import strategy_server with DATA_ROOT pointing at tmp_path."""
    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import strategy_server  # noqa: WPS433 — test-only import
    finally:
        sys.path.pop(0)

    # Make sure SMC_MACRO_MAGIC env doesn't leak between tests
    monkeypatch.delenv("SMC_MACRO_MAGIC", raising=False)
    monkeypatch.setattr(strategy_server, "DATA_ROOT", tmp_path)
    return TestClient(strategy_server.app)


def _write_state(tmp_path: Path, symbol: str, suffix: str, state: dict) -> None:
    sym_dir = tmp_path / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    (sym_dir / f"live_state{suffix}.json").write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# signals[] array shape
# ---------------------------------------------------------------------------

class TestSignalsArrayShape:
    def test_signals_array_present(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0, "sl": 2348.0, "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.7,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert "signals" in body
        assert isinstance(body["signals"], list)
        # control + treatment (both legs present in array, even without treatment state)
        assert len(body["signals"]) == 2

    def test_legs_are_ordered_control_first(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 1,
            "action": "HOLD",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["signals"][0]["leg"] == ""
        assert body["signals"][1]["leg"] == "_macro"

    def test_control_leg_carries_instrument_magic(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 1,
            "action": "HOLD",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        assert control["magic"] == 19760418  # XAU

    def test_treatment_leg_carries_macro_magic(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 1,
            "action": "HOLD",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        treatment = body["signals"][1]
        assert treatment["magic"] == 19760428

    def test_magic_values_differ_between_legs(self, client, tmp_path):
        """Broker reconcile split requires the two legs to carry distinct magics."""
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 1,
            "action": "HOLD",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        m1 = body["signals"][0]["magic"]
        m2 = body["signals"][1]["magic"]
        assert m1 != m2
        assert m1 > 0 and m2 > 0


# ---------------------------------------------------------------------------
# Per-leg independence — one leg updating doesn't affect the other
# ---------------------------------------------------------------------------

class TestPerLegIndependence:
    def test_only_control_state_present(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0, "sl": 2348.0, "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.7,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        treatment = body["signals"][1]
        assert control["action"] == "BUY"
        assert treatment["action"] == "HOLD"
        assert treatment["fresh"] is False

    def test_only_treatment_state_present(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "_macro", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 99,
            "action": "RANGE SELL",
            "trading_mode": "ranging",
            "best_setup": {
                "direction": "short",
                "entry": 2360.0, "sl": 2362.0, "tp1": 2356.0,
                "position_size_lots": 0.15,
                "confluence": 0.6,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        treatment = body["signals"][1]
        assert control["action"] == "HOLD"
        assert control["fresh"] is False
        assert treatment["action"] == "RANGE SELL"
        assert treatment["magic"] == 19760428

    def test_both_legs_present_different_actions(self, client, tmp_path):
        """Simulate control saying BUY, treatment saying RANGE SELL simultaneously."""
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0, "sl": 2348.0, "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.7,
            },
        })
        _write_state(tmp_path, "XAUUSD", "_macro", {
            "timestamp": "2026-04-18T10:00:05+00:00",
            "cycle": 42,
            "action": "RANGE SELL",
            "trading_mode": "ranging",
            "best_setup": {
                "direction": "short",
                "entry": 2360.0, "sl": 2362.0, "tp1": 2356.0,
                "position_size_lots": 0.15,
                "confluence": 0.6,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["signals"][0]["action"] == "BUY"
        assert body["signals"][0]["magic"] == 19760418
        assert body["signals"][1]["action"] == "RANGE SELL"
        assert body["signals"][1]["magic"] == 19760428

    def test_signal_ids_unique_per_leg(self, client, tmp_path):
        """Same cycle number must not produce identical signal_id across legs."""
        state = {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0, "sl": 2348.0, "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.7,
            },
        }
        _write_state(tmp_path, "XAUUSD", "", state)
        _write_state(tmp_path, "XAUUSD", "_macro", state)
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        id0 = body["signals"][0]["signal_id"]
        id1 = body["signals"][1]["signal_id"]
        assert id0 != id1
        # Each carries its leg identifier so the EA can dedupe per-leg.
        assert "_macro" in id1
        assert "_macro" not in id0


# ---------------------------------------------------------------------------
# Backward-compat flat fields (legacy EAs still work during rollout)
# ---------------------------------------------------------------------------

class TestBackwardCompatFlatFields:
    def test_flat_fields_mirror_control_leg(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0, "sl": 2348.0, "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.7,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["action"] == "BUY"
        assert body["direction"] == "long"
        assert body["entry"] == 2350.0
        assert body["sl"] == 2348.0
        assert body["tp"] == 2354.0
        assert body["trading_mode"] == "trending"

    def test_flat_fields_track_control_on_hold_treatment_active(self, client, tmp_path):
        """If control says HOLD but treatment says BUY, flat fields still mirror control."""
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "HOLD",
        })
        _write_state(tmp_path, "XAUUSD", "_macro", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0, "sl": 2348.0, "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.7,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        # Flat fields = control leg (HOLD)
        assert body["action"] == "HOLD"
        # But treatment in signals[] carries the BUY
        assert body["signals"][1]["action"] == "BUY"


# ---------------------------------------------------------------------------
# Freshness envelope — any fresh leg keeps envelope fresh
# ---------------------------------------------------------------------------

class TestEnvelopeFreshness:
    def test_envelope_fresh_when_any_leg_fresh(self, client, tmp_path):
        from datetime import datetime, timezone
        fresh_ts = datetime.now(tz=timezone.utc).isoformat()
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": fresh_ts,
            "cycle": 42,
            "action": "HOLD",
        })
        # No treatment state — still expect envelope fresh because control is fresh.
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["fresh"] is True

    def test_envelope_stale_when_all_legs_stale(self, client, tmp_path):
        # No state files at all → both legs absent → envelope not fresh.
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["fresh"] is False


# ---------------------------------------------------------------------------
# /status per-leg health
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    def test_status_lists_both_legs_per_symbol(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 1,
            "action": "HOLD",
        })
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        legs = body["legs"]
        # 2 symbols × 2 legs = 4 entries
        assert len(legs) >= 4
        xau_entries = [e for e in legs if e["symbol"] == "XAUUSD"]
        assert len(xau_entries) == 2
        leg_ids = sorted(e["leg"] for e in xau_entries)
        assert leg_ids == ["", "_macro"]

    def test_status_exposes_magic_per_leg(self, client, tmp_path):
        resp = client.get("/status")
        body = resp.json()
        xau_control = [e for e in body["legs"] if e["symbol"] == "XAUUSD" and e["leg"] == ""][0]
        xau_treat = [e for e in body["legs"] if e["symbol"] == "XAUUSD" and e["leg"] == "_macro"][0]
        assert xau_control["magic"] == 19760418
        assert xau_treat["magic"] == 19760428


# ---------------------------------------------------------------------------
# Macro magic env override
# ---------------------------------------------------------------------------

class TestMacroMagicEnvOverride:
    def test_smc_macro_magic_env_flows_to_signal(self, client, tmp_path, monkeypatch):
        """SMC_MACRO_MAGIC env overrides the treatment leg magic."""
        monkeypatch.setenv("SMC_MACRO_MAGIC", "30001000")
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 1,
            "action": "HOLD",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        treatment = body["signals"][1]
        assert treatment["magic"] == 30001000


# ---------------------------------------------------------------------------
# Round 5 A-track Task #8 — regime-dynamic trailing SL fields
# ---------------------------------------------------------------------------


class TestRegimeDynamicTrailing:
    """/signal forwards trail_activate_r / trail_distance_r / regime_label
    from live_state.best_setup so the EA can apply per-ticket trailing."""

    def _fresh_ts(self) -> str:
        """Use a recent timestamp so fresh=true in the response."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def test_trail_fields_forward_from_state(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": self._fresh_ts(),
            "cycle": 99,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 4000.0, "sl": 3980.0, "tp1": 4050.0,
                "position_size_lots": 0.3,
                "confluence": 0.75,
                "trail_activate_r": 0.3,
                "trail_distance_r": 0.5,
                "regime_label": "TREND_UP",
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        assert control["trail_activate_r"] == 0.3
        assert control["trail_distance_r"] == 0.5
        assert control["regime_label"] == "TREND_UP"

    def test_trail_fields_null_when_missing(self, client, tmp_path):
        """HOLD signal without best_setup → trail fields are None/null."""
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": self._fresh_ts(),
            "cycle": 5,
            "action": "HOLD",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        assert control["trail_activate_r"] is None
        assert control["trail_distance_r"] is None
        assert control["regime_label"] is None

    def test_ath_breakout_trail_values(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": self._fresh_ts(),
            "cycle": 10,
            "action": "BUY",
            "best_setup": {
                "direction": "long",
                "entry": 4000.0, "sl": 3960.0, "tp1": 4080.0,
                "position_size_lots": 0.2,
                "trail_activate_r": 0.8,
                "trail_distance_r": 0.7,
                "regime_label": "ATH_BREAKOUT",
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        assert control["trail_activate_r"] == 0.8
        assert control["trail_distance_r"] == 0.7
        assert control["regime_label"] == "ATH_BREAKOUT"

    def test_consolidation_tight_scalp_trail(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", "", {
            "timestamp": self._fresh_ts(),
            "cycle": 11,
            "action": "SELL",
            "best_setup": {
                "direction": "short",
                "entry": 4000.0, "sl": 4020.0, "tp1": 3950.0,
                "position_size_lots": 0.2,
                "trail_activate_r": 0.5,
                "trail_distance_r": 0.3,
                "regime_label": "CONSOLIDATION",
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        control = body["signals"][0]
        assert control["trail_activate_r"] == 0.5
        assert control["trail_distance_r"] == 0.3
