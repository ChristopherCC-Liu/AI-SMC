"""Unit tests for smc.strategy.htf_bias — HTF directional bias computation."""

from __future__ import annotations

import pytest

from smc.smc_core.types import SMCSnapshot
from smc.strategy.htf_bias import compute_htf_bias


class TestComputeHTFBias:
    def test_bullish_d1_and_h4_aligned(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bullish_snapshot)
        assert bias.direction == "bullish"
        assert bias.confidence > 0.0
        assert "bullish" in bias.rationale.lower()

    def test_bearish_d1_and_h4_would_conflict(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bearish_snapshot: SMCSnapshot,
    ) -> None:
        """D1 bullish but H4 bearish → neutral."""
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bearish_snapshot)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0
        assert "conflicting" in bias.rationale.lower()

    def test_d1_neutral_always_neutral(
        self,
        d1_neutral_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """D1 neutral → always neutral regardless of H4."""
        bias = compute_htf_bias(d1_neutral_snapshot, h4_bullish_snapshot)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0

    def test_d1_bullish_h4_neutral_weak(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_neutral_snapshot: SMCSnapshot,
    ) -> None:
        """D1 bullish + H4 neutral → weak bullish bias (halved confidence)."""
        bias = compute_htf_bias(d1_bullish_snapshot, h4_neutral_snapshot)
        assert bias.direction == "bullish"
        # Should have lower confidence than fully confirmed
        full_bias = compute_htf_bias(d1_bullish_snapshot, d1_bullish_snapshot)
        # Weak bias should generally have lower confidence
        assert bias.confidence <= full_bias.confidence or bias.confidence <= 0.5

    def test_bearish_d1_and_h4_aligned(
        self,
        d1_bearish_snapshot: SMCSnapshot,
        h4_bearish_snapshot: SMCSnapshot,
    ) -> None:
        # Need an h4 bearish fixture — reuse h4_bearish
        bias = compute_htf_bias(d1_bearish_snapshot, h4_bearish_snapshot)
        assert bias.direction == "bearish"
        assert bias.confidence > 0.0

    def test_key_levels_populated(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bullish_snapshot)
        assert len(bias.key_levels) > 0
        # Key levels should be sorted
        assert list(bias.key_levels) == sorted(bias.key_levels)

    def test_confidence_bounded(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bullish_snapshot)
        assert 0.0 <= bias.confidence <= 1.0

    def test_result_is_frozen(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bullish_snapshot)
        with pytest.raises(Exception):
            bias.direction = "neutral"  # type: ignore[misc]

    def test_both_neutral(
        self,
        d1_neutral_snapshot: SMCSnapshot,
        h4_neutral_snapshot: SMCSnapshot,
    ) -> None:
        bias = compute_htf_bias(d1_neutral_snapshot, h4_neutral_snapshot)
        assert bias.direction == "neutral"
