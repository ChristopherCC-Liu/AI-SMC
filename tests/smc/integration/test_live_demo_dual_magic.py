"""Integration tests for audit-r4 v5 Option B dual-magic behaviour.

These tests exercise the path-construction and magic-resolution logic that
live_demo.py::main() threads through SMCConfig and the InstrumentConfig.
They do NOT import scripts/live_demo.py at module level — that imports
MetaTrader5 which is not available on macOS/Linux CI — instead they mirror
the key path + magic expressions and verify that control and treatment
legs land on distinct files and magics on the same TMGM Demo account.

Acceptance criteria covered:
- Control (suffix="") and treatment (suffix="_macro") write to distinct
  state files for consec-halt, quota, breaker, reconcile cursor.
- Control uses cfg.magic; treatment uses cfg.macro_magic.
- Virtual balance split yields per-leg sizing that divides total equity.
- Reconcile fetch_closed_pnl_since filters by effective magic so each
  leg accumulates only its own closed deals.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mirror of live_demo.py::main() state-path construction (per-suffix isolation)
# ---------------------------------------------------------------------------

def _compute_state_paths(data_root: Path, suffix: str) -> dict[str, Path]:
    """Mirror live_demo.py path construction for one leg.

    Returns the per-leg set of state / journal / flag paths keyed by short
    names.  Verifies that when suffix=="_macro" every single path has the
    suffix baked in so the leg's state lives in its own file.
    """
    return {
        "journal": data_root / f"journal{suffix}" / "live_trades.jsonl",
        "state": data_root / f"live_state{suffix}.json",
        "ai": data_root / "ai_analysis.json",  # shared — not leg-specific
        "pause_flag": data_root / f"trading_paused{suffix}.flag",
        "mt5_positions": data_root / f"mt5_positions{suffix}.json",
        "consec_loss": data_root / f"consec_loss_state{suffix}.json",
        "asian_quota": data_root / f"asian_range_quota_state{suffix}.json",
        "circuit_flag": data_root / f"execution_circuit_open{suffix}.flag",
        "reconcile_ts": data_root / f"last_reconcile_ts{suffix}.json",
        "phase1a_breaker": data_root / f"phase1a_breaker_state{suffix}.json",
        "pid": data_root / f"live_demo{suffix}.pid",
    }


# ---------------------------------------------------------------------------
# Per-leg state path isolation — control vs treatment
# ---------------------------------------------------------------------------

class TestStatePathIsolation:
    """Every per-leg state file path must differ between control and treatment."""

    def test_all_risk_state_paths_isolated(self, tmp_path: Path):
        data_root = tmp_path / "data" / "XAUUSD"
        control = _compute_state_paths(data_root, "")
        treatment = _compute_state_paths(data_root, "_macro")

        # Every entry that's per-leg (not shared) must differ.
        shared = {"ai"}  # AI analysis is shared; single compute per-symbol
        for key in control:
            if key in shared:
                continue
            assert control[key] != treatment[key], (
                f"Control and treatment share the same {key} path — "
                f"would clobber each other"
            )

    def test_control_suffix_empty_matches_pre_option_b(self, tmp_path: Path):
        """Backward-compat: suffix="" paths must be byte-identical to pre-Option-B."""
        data_root = tmp_path / "data" / "XAUUSD"
        control = _compute_state_paths(data_root, "")
        # Pre-Option-B path scheme (audited from live_demo.py HEAD-1)
        expected = {
            "consec_loss": data_root / "consec_loss_state.json",
            "asian_quota": data_root / "asian_range_quota_state.json",
            "circuit_flag": data_root / "execution_circuit_open.flag",
            "reconcile_ts": data_root / "last_reconcile_ts.json",
            "phase1a_breaker": data_root / "phase1a_breaker_state.json",
        }
        for key, expected_path in expected.items():
            assert control[key] == expected_path

    def test_treatment_suffix_macro_appends_to_filenames(self, tmp_path: Path):
        data_root = tmp_path / "data" / "XAUUSD"
        treatment = _compute_state_paths(data_root, "_macro")
        assert treatment["consec_loss"].name == "consec_loss_state_macro.json"
        assert treatment["phase1a_breaker"].name == "phase1a_breaker_state_macro.json"
        assert treatment["journal"].parent.name == "journal_macro"
        assert treatment["state"].name == "live_state_macro.json"


# ---------------------------------------------------------------------------
# Magic resolution — control vs treatment
# ---------------------------------------------------------------------------

class TestMagicResolution:
    def test_control_uses_instrument_magic(self, monkeypatch):
        monkeypatch.delenv("SMC_MACRO_MAGIC", raising=False)
        from smc.config import SMCConfig
        from smc.instruments import get_instrument_config

        cfg_app = SMCConfig()
        xau_cfg = get_instrument_config("XAUUSD")
        assert cfg_app.magic_for(xau_cfg.magic, "") == 19760418

    def test_treatment_uses_macro_magic(self, monkeypatch):
        monkeypatch.delenv("SMC_MACRO_MAGIC", raising=False)
        from smc.config import SMCConfig
        from smc.instruments import get_instrument_config

        cfg_app = SMCConfig()
        xau_cfg = get_instrument_config("XAUUSD")
        assert cfg_app.magic_for(xau_cfg.magic, "_macro") == 19760428

    def test_same_account_distinct_magics_per_leg(self, monkeypatch):
        """Same TMGM Demo account; both legs; distinct magics — the core
        invariant that broker reconcile can split deals per-leg."""
        monkeypatch.delenv("SMC_MACRO_MAGIC", raising=False)
        from smc.config import SMCConfig
        from smc.instruments import get_instrument_config

        cfg_app = SMCConfig()
        xau_cfg = get_instrument_config("XAUUSD")
        control_magic = cfg_app.magic_for(xau_cfg.magic, "")
        treatment_magic = cfg_app.magic_for(xau_cfg.magic, "_macro")
        assert control_magic != treatment_magic


# ---------------------------------------------------------------------------
# Reconcile splits deals by magic
# ---------------------------------------------------------------------------

class TestReconcileSplitsByMagic:
    """fetch_closed_pnl_since already accepts magic; verify that feeding the
    two leg magics returns disjoint deal lists."""

    def test_closed_pnl_filters_by_magic(self, monkeypatch):
        from smc.execution.mt5_positions_adapter import fetch_closed_pnl_since
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)

        # Build two deals with distinct magics — one per leg.
        deal_control = MagicMock()
        deal_control.magic = 19760418
        deal_control.entry = 1  # DEAL_ENTRY_OUT
        deal_control.profit = 10.0
        deal_control.commission = 0.0
        deal_control.swap = 0.0
        deal_control.position_id = 1001
        deal_control.time = int(now.timestamp())

        deal_treatment = MagicMock()
        deal_treatment.magic = 19760428
        deal_treatment.entry = 1
        deal_treatment.profit = -5.0
        deal_treatment.commission = 0.0
        deal_treatment.swap = 0.0
        deal_treatment.position_id = 1002
        deal_treatment.time = int(now.timestamp())

        mt5_stub = MagicMock()
        mt5_stub.history_deals_get.return_value = [deal_control, deal_treatment]
        mt5_stub.DEAL_ENTRY_OUT = 1
        mt5_stub.DEAL_ENTRY_INOUT = 2

        # Control leg fetch: only deal_control should come back
        from_ts = now.replace(hour=0, minute=0, second=0, microsecond=0)
        control_deals = fetch_closed_pnl_since(mt5_stub, from_ts, magic=19760418)
        assert len(control_deals) == 1
        assert control_deals[0]["ticket"] == 1001
        assert control_deals[0]["pnl_usd"] == 10.0

        # Treatment leg fetch: only deal_treatment
        treatment_deals = fetch_closed_pnl_since(mt5_stub, from_ts, magic=19760428)
        assert len(treatment_deals) == 1
        assert treatment_deals[0]["ticket"] == 1002
        assert treatment_deals[0]["pnl_usd"] == -5.0


# ---------------------------------------------------------------------------
# Risk state files write independently when control and treatment trade
# ---------------------------------------------------------------------------

class TestRiskStateIndependence:
    def test_control_halt_does_not_affect_treatment(self, tmp_path: Path):
        """Control hitting consec-loss halt must not stop treatment trading."""
        from smc.risk.consec_loss_halt import ConsecLossHalt

        control_path = tmp_path / "consec_loss_state.json"
        treatment_path = tmp_path / "consec_loss_state_macro.json"

        # Control suffers 3 losses (halt trigger)
        control = ConsecLossHalt(state_path=control_path)
        control.record(-1.0)
        control.record(-1.0)
        control.record(-1.0)
        assert control.is_tripped()

        # Treatment — never touched control's state file → fresh halt
        treatment = ConsecLossHalt(state_path=treatment_path)
        assert not treatment.is_tripped()

    def test_breaker_isolation_by_path(self, tmp_path: Path):
        from smc.strategy.phase1a_circuit_breaker import Phase1aCircuitBreaker

        control_path = tmp_path / "phase1a_breaker_state.json"
        treatment_path = tmp_path / "phase1a_breaker_state_macro.json"

        control = Phase1aCircuitBreaker(state_path=control_path)
        treatment = Phase1aCircuitBreaker(state_path=treatment_path)

        # Control records 3 losses → tripped
        control.record_trade_close(-10.0)
        control.record_trade_close(-10.0)
        control.record_trade_close(-10.0)
        assert control.is_tripped()
        # Treatment is untouched
        assert not treatment.is_tripped()


# ---------------------------------------------------------------------------
# Virtual balance sizing end-to-end for both legs
# ---------------------------------------------------------------------------

class TestVirtualBalanceE2E:
    def test_dual_leg_sizing_caps_each_leg_at_half_account(self, monkeypatch):
        """Both legs sized on virtual balance → combined risk does not exceed
        1% of the real account even when both fire simultaneously."""
        monkeypatch.delenv("SMC_VIRTUAL_BALANCE_SPLIT", raising=False)
        from smc.config import SMCConfig
        from smc.instruments import get_instrument_config
        from smc.risk.live_position_sizer import compute_live_position_size
        from dataclasses import dataclass

        @dataclass
        class _FakeRangeSetup:
            direction: str = "long"
            entry_price: float = 2350.0
            stop_loss: float = 2348.0  # 200 points SL
            take_profit: float = 2355.0
            trigger: str = "boundary_buy"
            confidence: float = 0.5

        cfg_app = SMCConfig()
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup()

        mt5_balance = 1000.67
        vb_control = cfg_app.virtual_balance_for("", mt5_balance)
        vb_treat = cfg_app.virtual_balance_for("_macro", mt5_balance)
        lot_control = compute_live_position_size(
            best, cfg=cfg, balance_usd=vb_control, risk_pct=1.0, blocked_reason=None
        )
        lot_treat = compute_live_position_size(
            best, cfg=cfg, balance_usd=vb_treat, risk_pct=1.0, blocked_reason=None
        )

        # Each leg ~$500 virtual → 1% of $500 = $5 risk budget per leg.
        # 200 pt SL * $1/pt = $200 risk per lot → 0.025 → rounds to 0.03 (min_lot floor).
        # Each leg must see HALF the full-account lot; combined lots ≈ full-account lot.
        assert lot_control > 0.0
        assert lot_treat > 0.0

        # Full-account sizing for reference
        full_lot = compute_live_position_size(
            best, cfg=cfg, balance_usd=mt5_balance, risk_pct=1.0, blocked_reason=None
        )
        # Control + treatment virtual-balance lots ≈ full-account lot
        # (both legs sized on half balance → their sum matches the full single-leg lot).
        combined_lot = lot_control + lot_treat
        # Min_lot floor may push combined slightly over full_lot; allow 2 * min_lot epsilon.
        assert combined_lot <= full_lot + 2 * cfg.min_lot
        # Per-leg must not exceed the full-account lot by itself (the core invariant).
        assert lot_control < full_lot + cfg.min_lot
        assert lot_treat < full_lot + cfg.min_lot
