"""Unit tests for spread_gate — Round 10 P3.2 hygiene.

Verifies the rolling 20-bar median baseline computation and the entry
gate that blocks new opens when current spread > 1.5x baseline.
"""
from __future__ import annotations

import pytest

from smc.risk.spread_gate import (
    DEFAULT_BASELINE_WINDOW,
    DEFAULT_THRESHOLD_MULTIPLIER,
    SpreadGateDecision,
    check_spread_gate,
    compute_spread_baseline,
)


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


class TestComputeBaseline:
    def test_baseline_median_window_20(self):
        """Exactly 20 samples → median is the average of indices 9 and 10
        (sorted), i.e. for [1..20] the median is (10+11)/2 = 10.5."""
        spreads = list(range(1, 21))  # 1, 2, ..., 20
        baseline = compute_spread_baseline(spreads, window=20)
        assert baseline == 10.5

    def test_baseline_median_odd_window(self):
        """5 samples → median is the middle value (index 2 when sorted)."""
        spreads = [3.0, 1.0, 5.0, 2.0, 4.0]
        baseline = compute_spread_baseline(spreads, window=5)
        assert baseline == 3.0

    def test_baseline_returns_none_when_too_few_samples(self):
        spreads = list(range(1, 19))  # 18 samples, window=20
        assert compute_spread_baseline(spreads, window=20) is None

    def test_baseline_uses_last_window_samples(self):
        """When more samples than window, only the most recent `window`
        are used (rolling-window semantics)."""
        # First 100 are large, last 20 are small — baseline should be small
        spreads = [100.0] * 100 + [2.0] * 20
        baseline = compute_spread_baseline(spreads, window=20)
        assert baseline == 2.0

    def test_baseline_filters_zero_spreads(self):
        """MT5 occasionally yields spread=0 on stale ticks. Filter them
        before computing the median to avoid biasing the threshold low."""
        spreads = [0.0] * 5 + [10.0] * 20
        baseline = compute_spread_baseline(spreads, window=20)
        assert baseline == 10.0

    def test_baseline_returns_none_when_all_zeros(self):
        spreads = [0.0] * 50
        assert compute_spread_baseline(spreads, window=20) is None

    def test_baseline_window_zero_raises(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            compute_spread_baseline([1.0, 2.0], window=0)

    def test_baseline_negative_spread_filtered(self):
        """Negative spreads are nonsense — filter as defence."""
        spreads = [-1.0] * 10 + [5.0] * 20
        baseline = compute_spread_baseline(spreads, window=20)
        assert baseline == 5.0


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------


class TestCheckSpreadGate:
    def test_under_threshold_passes(self):
        decision = check_spread_gate(
            current_spread=12.0, baseline=10.0, multiplier=1.5
        )
        assert decision.can_open is True
        assert decision.rejection_reason is None

    def test_at_threshold_blocks(self):
        """Exactly at 1.5x → block (>= semantics)."""
        decision = check_spread_gate(
            current_spread=15.0, baseline=10.0, multiplier=1.5
        )
        assert decision.can_open is False
        assert "spread" in (decision.rejection_reason or "").lower()

    def test_over_threshold_blocks_with_reason(self):
        decision = check_spread_gate(
            current_spread=20.0, baseline=10.0, multiplier=1.5
        )
        assert decision.can_open is False
        reason = decision.rejection_reason or ""
        assert "20" in reason or "2.0" in reason  # current
        assert "10" in reason  # baseline
        assert "1.5" in reason or "15" in reason  # threshold

    def test_baseline_none_passes_through(self):
        """Bootstrap path — no baseline yet, must not block."""
        decision = check_spread_gate(
            current_spread=999.0, baseline=None, multiplier=1.5
        )
        assert decision.can_open is True
        assert decision.baseline_median is None

    def test_custom_multiplier(self):
        """multiplier=2.0 → 25 should pass against baseline=10."""
        decision = check_spread_gate(
            current_spread=18.0, baseline=10.0, multiplier=2.0
        )
        assert decision.can_open is True

        decision_block = check_spread_gate(
            current_spread=22.0, baseline=10.0, multiplier=2.0
        )
        assert decision_block.can_open is False

    def test_decision_is_frozen(self):
        decision = check_spread_gate(
            current_spread=1.0, baseline=1.0, multiplier=1.5
        )
        with pytest.raises(Exception):  # frozen dataclass
            decision.can_open = False  # type: ignore[misc]

    def test_negative_baseline_treated_as_invalid(self):
        """Negative baseline shouldn't happen but defend anyway — pass."""
        decision = check_spread_gate(
            current_spread=10.0, baseline=-1.0, multiplier=1.5
        )
        assert decision.can_open is True

    def test_decision_reports_threshold_value(self):
        decision = check_spread_gate(
            current_spread=20.0, baseline=10.0, multiplier=1.5
        )
        assert decision.threshold_multiplier == 1.5
        assert decision.baseline_median == 10.0
        assert decision.current_spread == 20.0

    def test_default_multiplier_is_1_5(self):
        """1.5x is the spec-mandated default."""
        decision = check_spread_gate(current_spread=14.99, baseline=10.0)
        assert decision.can_open is True
        decision_block = check_spread_gate(current_spread=15.0, baseline=10.0)
        assert decision_block.can_open is False
        assert DEFAULT_THRESHOLD_MULTIPLIER == 1.5

    def test_default_window_is_20(self):
        """20-bar window is the spec-mandated default."""
        assert DEFAULT_BASELINE_WINDOW == 20


# ---------------------------------------------------------------------------
# SMCConfig flag (R10 P3.2 refinement #3 — staged rollout)
# ---------------------------------------------------------------------------


class TestConfigFlag:
    """spread_gate_enabled is a SMCConfig field (R10 P3.2 staged rollout).

    Defaults to False so the gate is OFF for control leg until Phase 3
    bake validates no false-blocks. Treatment leg flips it ON via env
    var ``SMC_SPREAD_GATE_ENABLED`` (lenient bool parser).
    """

    def test_default_is_false(self, monkeypatch):
        """Without env override, the flag defaults to False."""
        from smc.config import SMCConfig
        monkeypatch.delenv("SMC_SPREAD_GATE_ENABLED", raising=False)
        cfg = SMCConfig()
        assert cfg.spread_gate_enabled is False

    def test_env_var_true_lowercase(self, monkeypatch):
        from smc.config import SMCConfig
        monkeypatch.setenv("SMC_SPREAD_GATE_ENABLED", "true")
        cfg = SMCConfig()
        assert cfg.spread_gate_enabled is True

    def test_env_var_one(self, monkeypatch):
        """Lenient bool parser accepts '1' as truthy."""
        from smc.config import SMCConfig
        monkeypatch.setenv("SMC_SPREAD_GATE_ENABLED", "1")
        cfg = SMCConfig()
        assert cfg.spread_gate_enabled is True

    def test_env_var_with_whitespace(self, monkeypatch):
        """Windows .bat ships values with trailing whitespace/CR;
        the lenient parser strips before normalising."""
        from smc.config import SMCConfig
        monkeypatch.setenv("SMC_SPREAD_GATE_ENABLED", "true ")
        cfg = SMCConfig()
        assert cfg.spread_gate_enabled is True

    def test_env_var_false_explicit(self, monkeypatch):
        from smc.config import SMCConfig
        monkeypatch.setenv("SMC_SPREAD_GATE_ENABLED", "false")
        cfg = SMCConfig()
        assert cfg.spread_gate_enabled is False
