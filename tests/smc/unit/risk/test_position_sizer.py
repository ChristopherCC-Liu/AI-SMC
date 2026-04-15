"""Unit tests for position sizing logic."""

from __future__ import annotations

import pytest

from smc.risk.position_sizer import compute_position_size
from smc.risk.types import PositionSize


# ---------------------------------------------------------------------------
# Core formula tests
# ---------------------------------------------------------------------------


class TestComputePositionSize:
    """Verify lot sizing formula: lots = risk_usd / (sl_points * point_value_per_lot)."""

    def test_standard_case_10k_1pct_300pt(self) -> None:
        """$10k balance, 1% risk, 300-point SL -> ~0.33 lots."""
        ps = compute_position_size(
            balance_usd=10_000.0,
            risk_pct=1.0,
            sl_distance_points=300.0,
        )
        # risk_usd = 10000 * 0.01 = 100
        # point_value = 10 / 10 = 1.0
        # raw_lots = 100 / (300 * 1.0) = 0.3333...
        # rounded = 0.33
        assert ps.lots == 0.33
        assert ps.risk_usd == 100.0
        assert ps.risk_points == 300.0
        assert isinstance(ps, PositionSize)

    def test_small_balance_500_1pct_200pt(self) -> None:
        """$500 balance, 1% risk, 200-point SL -> 0.03 lots."""
        ps = compute_position_size(
            balance_usd=500.0,
            risk_pct=1.0,
            sl_distance_points=200.0,
        )
        # risk_usd = 5.0, raw_lots = 5 / (200 * 1) = 0.025 -> rounded 0.03
        assert ps.lots == 0.03
        assert ps.risk_usd == 5.0

    def test_large_balance_100k_2pct_500pt(self) -> None:
        """$100k balance, 2% risk, 500-point SL -> clamped to max 1.0 lot."""
        ps = compute_position_size(
            balance_usd=100_000.0,
            risk_pct=2.0,
            sl_distance_points=500.0,
        )
        # risk_usd = 2000, raw_lots = 2000 / 500 = 4.0 -> clamped to 1.0
        assert ps.lots == 1.0
        assert ps.risk_usd == 2_000.0

    def test_tiny_sl_clamps_to_max(self) -> None:
        """Very tight SL produces large lot size that gets clamped to max."""
        ps = compute_position_size(
            balance_usd=10_000.0,
            risk_pct=1.0,
            sl_distance_points=10.0,
        )
        # raw_lots = 100 / (10 * 1) = 10.0 -> clamped to 1.0
        assert ps.lots == 1.0

    def test_wide_sl_clamps_to_min(self) -> None:
        """Very wide SL on small balance produces tiny lot size clamped to min."""
        ps = compute_position_size(
            balance_usd=100.0,
            risk_pct=0.5,
            sl_distance_points=1000.0,
        )
        # risk_usd = 0.5, raw_lots = 0.5 / 1000 = 0.0005 -> clamped to 0.01
        assert ps.lots == 0.01

    def test_custom_max_lot_size(self) -> None:
        """Custom max_lot_size parameter is respected."""
        ps = compute_position_size(
            balance_usd=50_000.0,
            risk_pct=2.0,
            sl_distance_points=100.0,
            max_lot_size=0.5,
        )
        # raw_lots = 1000 / 100 = 10.0 -> clamped to 0.5
        assert ps.lots == 0.5

    def test_custom_min_lot_size(self) -> None:
        """Custom min_lot_size parameter is respected."""
        ps = compute_position_size(
            balance_usd=100.0,
            risk_pct=0.5,
            sl_distance_points=1000.0,
            min_lot_size=0.05,
        )
        assert ps.lots == 0.05

    def test_margin_required(self) -> None:
        """Margin is computed as lots * margin_per_lot."""
        ps = compute_position_size(
            balance_usd=10_000.0,
            risk_pct=1.0,
            sl_distance_points=300.0,
            margin_per_lot=2_000.0,
        )
        assert ps.margin_required_usd == ps.lots * 2_000.0

    def test_custom_pip_value(self) -> None:
        """Non-default pip value changes lot calculation."""
        ps = compute_position_size(
            balance_usd=10_000.0,
            risk_pct=1.0,
            sl_distance_points=300.0,
            pip_value_per_lot=20.0,  # $20/pip -> $2/point
        )
        # raw_lots = 100 / (300 * 2.0) = 0.1667 -> 0.17
        assert ps.lots == 0.17

    def test_result_is_frozen(self) -> None:
        ps = compute_position_size(
            balance_usd=10_000.0, risk_pct=1.0, sl_distance_points=300.0
        )
        with pytest.raises(Exception):
            ps.lots = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestPositionSizerValidation:
    def test_zero_balance_raises(self) -> None:
        with pytest.raises(ValueError, match="balance_usd"):
            compute_position_size(balance_usd=0.0, risk_pct=1.0, sl_distance_points=300.0)

    def test_negative_balance_raises(self) -> None:
        with pytest.raises(ValueError, match="balance_usd"):
            compute_position_size(balance_usd=-1000.0, risk_pct=1.0, sl_distance_points=300.0)

    def test_zero_risk_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_pct"):
            compute_position_size(balance_usd=10_000.0, risk_pct=0.0, sl_distance_points=300.0)

    def test_negative_risk_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_pct"):
            compute_position_size(balance_usd=10_000.0, risk_pct=-1.0, sl_distance_points=300.0)

    def test_zero_sl_raises(self) -> None:
        with pytest.raises(ValueError, match="sl_distance_points"):
            compute_position_size(balance_usd=10_000.0, risk_pct=1.0, sl_distance_points=0.0)

    def test_negative_sl_raises(self) -> None:
        with pytest.raises(ValueError, match="sl_distance_points"):
            compute_position_size(balance_usd=10_000.0, risk_pct=1.0, sl_distance_points=-50.0)

    def test_zero_pip_value_raises(self) -> None:
        with pytest.raises(ValueError, match="pip_value_per_lot"):
            compute_position_size(
                balance_usd=10_000.0,
                risk_pct=1.0,
                sl_distance_points=300.0,
                pip_value_per_lot=0.0,
            )
