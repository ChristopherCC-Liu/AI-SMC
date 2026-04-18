"""audit-r2 ops #18: htf_bias_tier classification helper.

Used by live_demo journal + live_state.json to bucket HTF bias confidence
so Round 3 S2 can size the two-stage soft-multiplier decision from
distribution data.
"""
from __future__ import annotations

import pytest

from smc.strategy.htf_bias import htf_bias_tier


class TestTier1:
    def test_floor_0_7(self):
        assert htf_bias_tier(0.7) == "tier_1"

    def test_mid(self):
        assert htf_bias_tier(0.85) == "tier_1"

    def test_ceiling_1_0(self):
        assert htf_bias_tier(1.0) == "tier_1"


class TestTier2:
    def test_floor_0_4(self):
        assert htf_bias_tier(0.4) == "tier_2"

    def test_below_tier_1_boundary(self):
        # 0.699... must still be tier_2 (tier_1 requires >= 0.7)
        assert htf_bias_tier(0.699) == "tier_2"

    def test_middle_of_range(self):
        assert htf_bias_tier(0.55) == "tier_2"


class TestTier3:
    def test_floor_0_3(self):
        assert htf_bias_tier(0.3) == "tier_3"

    def test_just_below_tier_2(self):
        assert htf_bias_tier(0.399) == "tier_3"


class TestNeutral:
    def test_zero(self):
        """D1 vs H4 disagreement path returns conf 0.0 → neutral."""
        assert htf_bias_tier(0.0) == "neutral"

    def test_below_tier_3_floor(self):
        assert htf_bias_tier(0.29) == "neutral"

    def test_tiny_positive(self):
        assert htf_bias_tier(0.01) == "neutral"


class TestBoundaries:
    """Boundary values must be classified deterministically to avoid
    tier-drift as Round 3 S2 sizing policies rely on them."""

    @pytest.mark.parametrize(
        "conf,expected",
        [
            (0.7, "tier_1"),
            (0.6999, "tier_2"),
            (0.4, "tier_2"),
            (0.3999, "tier_3"),
            (0.3, "tier_3"),
            (0.2999, "neutral"),
        ],
    )
    def test_boundaries(self, conf: float, expected: str):
        assert htf_bias_tier(conf) == expected
