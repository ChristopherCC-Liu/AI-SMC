"""Unit tests for exposure and margin checks."""

from __future__ import annotations

from smc.risk.exposure import check_margin, check_max_exposure


# ---------------------------------------------------------------------------
# check_margin
# ---------------------------------------------------------------------------


class TestCheckMargin:
    def test_no_margin_used(self) -> None:
        """Zero margin used is always safe."""
        assert check_margin(balance=10_000.0, margin_used=0.0) is True

    def test_negative_margin_used(self) -> None:
        """Negative margin (shouldn't happen) treated as no margin."""
        assert check_margin(balance=10_000.0, margin_used=-100.0) is True

    def test_above_200pct(self) -> None:
        """$10k balance / $2k margin = 500% -> safe."""
        assert check_margin(balance=10_000.0, margin_used=2_000.0) is True

    def test_exactly_200pct(self) -> None:
        """$10k / $5k = 200% -> exactly at limit, should pass."""
        assert check_margin(balance=10_000.0, margin_used=5_000.0) is True

    def test_below_200pct(self) -> None:
        """$10k / $6k = 166.7% -> below 200% limit."""
        assert check_margin(balance=10_000.0, margin_used=6_000.0) is False

    def test_custom_threshold(self) -> None:
        """Custom margin_level_min = 150%. $10k / $6k = 166.7% -> passes."""
        assert check_margin(
            balance=10_000.0, margin_used=6_000.0, margin_level_min=150.0
        ) is True

    def test_zero_balance(self) -> None:
        """Zero balance with margin used -> unsafe."""
        assert check_margin(balance=0.0, margin_used=1_000.0) is False

    def test_negative_balance(self) -> None:
        """Negative balance -> unsafe."""
        assert check_margin(balance=-500.0, margin_used=1_000.0) is False


# ---------------------------------------------------------------------------
# check_max_exposure
# ---------------------------------------------------------------------------


class TestCheckMaxExposure:
    def test_no_open_positions(self) -> None:
        """No existing positions, adding 0.5 lots <= 1.0 max."""
        assert check_max_exposure(open_lots=0.0, new_lots=0.5) is True

    def test_within_limit(self) -> None:
        """0.3 + 0.5 = 0.8 <= 1.0 max."""
        assert check_max_exposure(open_lots=0.3, new_lots=0.5) is True

    def test_exactly_at_limit(self) -> None:
        """0.5 + 0.5 = 1.0 == 1.0 max -> should pass (<=)."""
        assert check_max_exposure(open_lots=0.5, new_lots=0.5) is True

    def test_exceeds_limit(self) -> None:
        """0.6 + 0.5 = 1.1 > 1.0 max."""
        assert check_max_exposure(open_lots=0.6, new_lots=0.5) is False

    def test_custom_max_lots(self) -> None:
        """Custom max of 2.0 lots."""
        assert check_max_exposure(open_lots=1.5, new_lots=0.4, max_total_lots=2.0) is True

    def test_custom_max_lots_exceeded(self) -> None:
        assert check_max_exposure(open_lots=1.5, new_lots=0.6, max_total_lots=2.0) is False

    def test_already_at_max(self) -> None:
        """Already at 1.0 lot, adding any more should fail."""
        assert check_max_exposure(open_lots=1.0, new_lots=0.01) is False
