"""Unit tests for smc.strategy.htf_bias — HTF directional bias computation."""

from __future__ import annotations

import pytest

from smc.smc_core.types import SMCSnapshot
from smc.strategy.htf_bias import compute_htf_bias


class TestComputeHTFBias:
    def test_bullish_d1_and_h4_aligned_tier1(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """D1 + H4 aligned → Tier 1 with confidence >= 0.7."""
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bullish_snapshot)
        assert bias.direction == "bullish"
        assert bias.confidence >= 0.7
        assert "tier 1" in bias.rationale.lower()

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

    def test_d1_neutral_h4_bullish_tier2(
        self,
        d1_neutral_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """D1 neutral + H4 bullish → Tier 2 H4-only bias (conf 0.4-0.7)."""
        bias = compute_htf_bias(d1_neutral_snapshot, h4_bullish_snapshot)
        assert bias.direction == "bullish"
        assert 0.4 <= bias.confidence <= 0.7
        assert "tier 2" in bias.rationale.lower()

    def test_d1_bullish_h4_neutral_tier3(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_neutral_snapshot: SMCSnapshot,
    ) -> None:
        """D1 bullish + H4 neutral → Tier 3 D1-only bias (conf 0.3-0.5)."""
        bias = compute_htf_bias(d1_bullish_snapshot, h4_neutral_snapshot)
        assert bias.direction == "bullish"
        assert 0.3 <= bias.confidence <= 0.5
        # Should have lower confidence than Tier 1 (fully confirmed)
        full_bias = compute_htf_bias(d1_bullish_snapshot, d1_bullish_snapshot)
        assert bias.confidence < full_bias.confidence

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

    def test_both_none_returns_neutral(self) -> None:
        """Both snapshots None → neutral."""
        bias = compute_htf_bias(None, None)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0

    def test_d1_none_h4_bullish_tier2(
        self,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """D1 None + H4 bullish → Tier 2 H4-only bias."""
        bias = compute_htf_bias(None, h4_bullish_snapshot)
        assert bias.direction == "bullish"
        assert 0.4 <= bias.confidence <= 0.7
        assert "tier 2" in bias.rationale.lower()

    def test_d1_bullish_h4_none_tier3(
        self,
        d1_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """D1 bullish + H4 None → Tier 3 D1-only bias."""
        bias = compute_htf_bias(d1_bullish_snapshot, None)
        assert bias.direction == "bullish"
        assert 0.3 <= bias.confidence <= 0.5
        assert "tier 3" in bias.rationale.lower()

    def test_tier1_higher_than_tier2_and_tier3(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """Tier 1 confidence > Tier 2 > Tier 3."""
        tier1 = compute_htf_bias(d1_bullish_snapshot, h4_bullish_snapshot)
        tier2 = compute_htf_bias(None, h4_bullish_snapshot)
        tier3 = compute_htf_bias(d1_bullish_snapshot, None)
        assert tier1.confidence > tier2.confidence
        assert tier2.confidence > tier3.confidence
