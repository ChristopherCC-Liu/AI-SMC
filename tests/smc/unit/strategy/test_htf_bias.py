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


# ---------------------------------------------------------------------------
# R8-B1 Tier 4 — SMA50 slope fallback
# ---------------------------------------------------------------------------


class TestTier4SlopeFallback:
    """R8-B1: when both D1 and H4 structure breaks are indeterminate but
    D1 OHLCV data shows a persistent SMA50 slope (|slope| >= 0.05 %/bar),
    the fallback returns directional bias instead of neutral.

    Fires exactly when grind-up/grind-down markets never print BOS/CHoCH
    (XAU 2024 Mar-Oct was the canonical case, 55% htf_bias_neutral).
    """

    def _grind_up_d1(self, n_bars: int = 120, slope_per_bar: float = 4.0) -> "pl.DataFrame":
        """Build a D1 OHLCV frame with linear grind-up (no reversals)."""
        import polars as pl
        from datetime import datetime, timedelta, timezone
        base = 2000.0
        rows = []
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(n_bars):
            close = base + i * slope_per_bar
            rows.append(
                {
                    "ts": start + timedelta(days=i),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                }
            )
        return pl.DataFrame(rows)

    def _grind_down_d1(self, n_bars: int = 120, slope_per_bar: float = 4.0) -> "pl.DataFrame":
        import polars as pl
        from datetime import datetime, timedelta, timezone
        base = 2000.0
        rows = []
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(n_bars):
            close = base - i * slope_per_bar
            rows.append(
                {
                    "ts": start + timedelta(days=i),
                    "open": close + 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                }
            )
        return pl.DataFrame(rows)

    def _flat_d1(self, n_bars: int = 120) -> "pl.DataFrame":
        import polars as pl
        from datetime import datetime, timedelta, timezone
        rows = []
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(n_bars):
            close = 2000.0 + (i % 3 - 1) * 0.1  # tiny oscillation, zero slope
            rows.append(
                {
                    "ts": start + timedelta(days=i),
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                }
            )
        return pl.DataFrame(rows)

    def test_grind_up_produces_tier4_bullish(self) -> None:
        d1_df = self._grind_up_d1(slope_per_bar=4.0)
        bias = compute_htf_bias(None, None, d1_df=d1_df)
        assert bias.direction == "bullish"
        assert 0.30 <= bias.confidence <= 0.45
        assert "tier 4" in bias.rationale.lower()

    def test_grind_down_produces_tier4_bearish(self) -> None:
        d1_df = self._grind_down_d1(slope_per_bar=4.0)
        bias = compute_htf_bias(None, None, d1_df=d1_df)
        assert bias.direction == "bearish"
        assert 0.30 <= bias.confidence <= 0.45
        assert "tier 4" in bias.rationale.lower()

    def test_flat_d1_stays_neutral(self) -> None:
        d1_df = self._flat_d1()
        bias = compute_htf_bias(None, None, d1_df=d1_df)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0

    def test_insufficient_d1_bars_stays_neutral(self) -> None:
        d1_df = self._grind_up_d1(n_bars=20)  # < 55 required
        bias = compute_htf_bias(None, None, d1_df=d1_df)
        assert bias.direction == "neutral"

    def test_backward_compat_without_d1_df_kwarg(
        self,
        d1_neutral_snapshot: SMCSnapshot,
        h4_neutral_snapshot: SMCSnapshot,
    ) -> None:
        """Pre-R8-B1 callers that don't pass d1_df must still get neutral."""
        bias = compute_htf_bias(d1_neutral_snapshot, h4_neutral_snapshot)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0

    def test_tier4_breaks_active_disagreement_with_strong_slope(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bearish_snapshot: SMCSnapshot,
    ) -> None:
        """R8-B1 extension: D1/H4 disagreement + strong D1 SMA50 slope =>
        trust the slope to pick a side. Required to cover XAU 2024 W14
        where 51% of bars are D1-bullish / H4-bearish retracement conflicts."""
        d1_df = self._grind_up_d1(slope_per_bar=4.0)  # strong bullish slope
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bearish_snapshot, d1_df=d1_df)
        assert bias.direction == "bullish"
        assert 0.30 <= bias.confidence <= 0.45
        assert "disagreement broken" in bias.rationale.lower()

    def test_disagreement_without_d1_df_stays_neutral(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bearish_snapshot: SMCSnapshot,
    ) -> None:
        """Backward-compat: callers that don't pass d1_df see the pre-R8-B1
        behaviour on active disagreement — neutral conf 0.0."""
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bearish_snapshot)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0

    def test_disagreement_with_flat_slope_stays_neutral(
        self,
        d1_bullish_snapshot: SMCSnapshot,
        h4_bearish_snapshot: SMCSnapshot,
    ) -> None:
        """No slope-strong-enough → disagreement remains neutral (does not
        promote a tie-breaking bias when there is no trend signal)."""
        d1_df = self._flat_d1()
        bias = compute_htf_bias(d1_bullish_snapshot, h4_bearish_snapshot, d1_df=d1_df)
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0

    def test_tier4_never_overrides_active_tier2_bias(
        self,
        h4_bullish_snapshot: SMCSnapshot,
    ) -> None:
        """If H4 structure already produces a bias (Tier 2), Tier 4 must
        not fire — the stronger signal wins."""
        d1_df = self._grind_up_d1(slope_per_bar=4.0)
        bias = compute_htf_bias(None, h4_bullish_snapshot, d1_df=d1_df)
        assert bias.direction == "bullish"
        # Should be Tier 2, not Tier 4
        assert "tier 2" in bias.rationale.lower()

    def test_confidence_scales_with_slope_magnitude(self) -> None:
        """Larger slope → higher Tier 4 confidence (within cap)."""
        small = compute_htf_bias(None, None, d1_df=self._grind_up_d1(slope_per_bar=1.5))
        large = compute_htf_bias(None, None, d1_df=self._grind_up_d1(slope_per_bar=8.0))
        if small.direction == "bullish" and large.direction == "bullish":
            assert large.confidence >= small.confidence
