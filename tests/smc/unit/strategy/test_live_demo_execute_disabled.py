"""audit-r3 V1 (HIGH): Python order_send is permanently disabled in EA arch.

Context:
  Since audit-r1 Round 5 T5 refactor, the MQL5 EA (AISMCReceiver.mq5)
  polls strategy_server /signal and executes via MT5 CTrade.  If
  scripts/live_demo.py ALSO called mt5.order_send (the legacy path
  gated by SMC_MT5_EXECUTE=1), an account would receive two orders per
  cycle — Python's **and** EA's — doubling risk and corrupting the
  consec_halt / phase1a_breaker / daily_pnl reconcile math.

Fix (live_demo.py):
  `_mt5_execute = False` unconditionally at every cycle.  The env var
  `SMC_MT5_EXECUTE` is retained ONLY as a misconfig detector: setting
  it emits a deprecation warning so ops can catch the mistake, but
  does NOT flip the kill switch back on.

These tests mirror the _mt5_execute resolution contract as a pure
function so we don't need to import scripts/live_demo.py (MT5 import
at module level).  Same pattern as test_live_demo_gate.py.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Mirror of _mt5_execute resolution from scripts/live_demo.py
# ---------------------------------------------------------------------------

def _resolve_mt5_execute(env: dict[str, str], *, paper_mode: bool) -> tuple[bool, bool]:
    """Return (mt5_execute, deprecated_env_set).

    Mirrors the post-V1 logic:
      - mt5_execute is ALWAYS False (hard-disabled).
      - deprecated_env_set tracks whether the user set SMC_MT5_EXECUTE=1
        for emitting a warning.
    """
    mt5_execute = False  # hard-disabled
    deprecated = env.get("SMC_MT5_EXECUTE", "0") == "1"
    return (mt5_execute, deprecated)


# ---------------------------------------------------------------------------
# Invariant: mt5_execute MUST always be False regardless of inputs
# ---------------------------------------------------------------------------

class TestHardDisable:
    def test_default_is_false(self):
        mt5_execute, _ = _resolve_mt5_execute({}, paper_mode=False)
        assert mt5_execute is False

    def test_env_var_1_still_false(self):
        """Even with explicit SMC_MT5_EXECUTE=1, kill switch stays closed."""
        mt5_execute, _ = _resolve_mt5_execute(
            {"SMC_MT5_EXECUTE": "1"}, paper_mode=False
        )
        assert mt5_execute is False

    def test_env_var_true_string_still_false(self):
        """'true' is NOT the gate value '1' — still False."""
        mt5_execute, _ = _resolve_mt5_execute(
            {"SMC_MT5_EXECUTE": "true"}, paper_mode=False
        )
        assert mt5_execute is False

    def test_paper_mode_still_false(self):
        mt5_execute, _ = _resolve_mt5_execute({}, paper_mode=True)
        assert mt5_execute is False

    def test_paper_mode_with_env_var_still_false(self):
        """Paper + env var both set — mt5_execute stays False."""
        mt5_execute, _ = _resolve_mt5_execute(
            {"SMC_MT5_EXECUTE": "1"}, paper_mode=True
        )
        assert mt5_execute is False

    def test_is_always_false_invariant(self):
        """Parametric sweep across all 2**3 combos of env/paper — never True."""
        for env_val in ("0", "1", "true", "yes", ""):
            for paper in (True, False):
                env = {"SMC_MT5_EXECUTE": env_val} if env_val else {}
                mt5_execute, _ = _resolve_mt5_execute(env, paper_mode=paper)
                assert mt5_execute is False, (
                    f"regression: mt5_execute leaked True at env={env_val}, paper={paper}"
                )


# ---------------------------------------------------------------------------
# Deprecation warning detection
# ---------------------------------------------------------------------------

class TestDeprecationWarning:
    def test_env_var_unset_no_warning(self):
        _, deprecated = _resolve_mt5_execute({}, paper_mode=False)
        assert deprecated is False

    def test_env_var_zero_no_warning(self):
        _, deprecated = _resolve_mt5_execute(
            {"SMC_MT5_EXECUTE": "0"}, paper_mode=False
        )
        assert deprecated is False

    def test_env_var_one_triggers_warning(self):
        _, deprecated = _resolve_mt5_execute(
            {"SMC_MT5_EXECUTE": "1"}, paper_mode=False
        )
        assert deprecated is True

    def test_env_var_truthy_but_not_1_no_warning(self):
        """Only literal '1' triggers; other truthy strings don't (preserves
        original env-parsing contract — fail safe)."""
        for val in ("true", "yes", "on", "TRUE", "2"):
            _, deprecated = _resolve_mt5_execute(
                {"SMC_MT5_EXECUTE": val}, paper_mode=False
            )
            assert deprecated is False, f"val={val} unexpectedly triggered warning"


# ---------------------------------------------------------------------------
# send_with_retry code path still importable (dormant, not deleted)
# ---------------------------------------------------------------------------

class TestDormantCodePath:
    def test_send_with_retry_still_importable(self):
        """Future rollback from EA arch requires send_with_retry to exist."""
        from smc.execution import mt5_send

        assert hasattr(mt5_send, "send_with_retry"), (
            "send_with_retry removed — V1 fix said RETAIN for rollback path"
        )

    def test_send_with_retry_signature_preserved(self):
        """Signature-level check — we didn't delete or rename during V1."""
        from smc.execution.mt5_send import send_with_retry
        import inspect

        sig = inspect.signature(send_with_retry)
        params = list(sig.parameters.keys())
        # First param must accept mt5 module / mock (historically positional).
        # Second param must accept the request dict.
        assert len(params) >= 2, (
            f"send_with_retry signature degraded: params={params}"
        )


# ---------------------------------------------------------------------------
# Documentation invariant — V1 comment must state it is "permanently disabled"
# (catches accidental revert in code review)
# ---------------------------------------------------------------------------

class TestCommentInvariant:
    def test_v1_comment_present_in_live_demo(self):
        """If the V1 explanation comment is removed, regression test fires."""
        from pathlib import Path

        live_demo = Path(__file__).resolve().parents[4] / "scripts" / "live_demo.py"
        text = live_demo.read_text()
        # Key substrings that the V1 comment block must carry:
        assert "V1 (HIGH)" in text or "audit-r3 V1" in text
        assert "EA" in text and "order_send" in text
        # And the hard-disable statement:
        assert "_mt5_execute = False" in text
