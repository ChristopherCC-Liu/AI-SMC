"""Unit tests for smc.risk.margin_cap.check_margin_cap."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from smc.risk.margin_cap import MarginCheckResult, check_margin_cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1


def _make_mt5(
    *,
    equity: float = 1000.0,
    margin: float = 0.0,
    account_info_none: bool = False,
    calc_margin_none: bool = False,
    proposed_margin: float = 100.0,
) -> Mock:
    """Build a minimal MT5Protocol mock for margin_cap tests."""
    mt5 = Mock()

    if account_info_none:
        mt5.account_info.return_value = None
    else:
        acc = Mock()
        acc.equity = equity
        acc.margin = margin
        mt5.account_info.return_value = acc

    if calc_margin_none:
        mt5.order_calc_margin.return_value = None
    else:
        mt5.order_calc_margin.return_value = proposed_margin

    return mt5


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarginCapCanTrade:
    def test_within_cap_returns_can_trade_true(self) -> None:
        """margin_used=0, proposed=200, equity=1000 → 20% < 40% → can_trade."""
        mt5 = _make_mt5(equity=1000.0, margin=0.0, proposed_margin=200.0)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        assert result.can_trade is True
        assert result.reason == "ok"
        assert result.proposed_margin == pytest.approx(200.0)
        assert result.total_after == pytest.approx(200.0)
        assert result.current_equity == pytest.approx(1000.0)
        assert result.cap_ratio == pytest.approx(0.40)

    def test_exactly_at_cap_boundary_can_trade(self) -> None:
        """total_after / equity == 0.40 exactly → NOT exceeded (strict >), so can_trade."""
        mt5 = _make_mt5(equity=1000.0, margin=200.0, proposed_margin=200.0)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        # 400 / 1000 = 0.40, which is NOT > 0.40, so can_trade=True
        assert result.can_trade is True
        assert result.total_after == pytest.approx(400.0)


@pytest.mark.unit
class TestMarginCapExceeded:
    def test_cap_exceeded_returns_can_trade_false(self) -> None:
        """margin_used=300, proposed=200, equity=1000 → 50% > 40% → cap_exceeded."""
        mt5 = _make_mt5(equity=1000.0, margin=300.0, proposed_margin=200.0)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        assert result.can_trade is False
        assert "cap_exceeded" in result.reason
        assert "50.0%" in result.reason
        assert result.total_after == pytest.approx(500.0)
        assert result.current_margin_used == pytest.approx(300.0)

    def test_custom_max_pct_respected(self) -> None:
        """Custom max_pct=0.30 trips on 35% margin usage."""
        mt5 = _make_mt5(equity=1000.0, margin=200.0, proposed_margin=150.0)

        result = check_margin_cap(
            mt5,
            symbol="BTCUSD",
            action=ORDER_TYPE_SELL,
            volume=0.01,
            price=50000.0,
            max_pct=0.30,
        )

        # 350 / 1000 = 35% > 30%
        assert result.can_trade is False
        assert "cap_exceeded" in result.reason
        assert result.cap_ratio == pytest.approx(0.30)


@pytest.mark.unit
class TestMarginCapErrorPaths:
    def test_account_info_none_returns_can_trade_false(self) -> None:
        mt5 = _make_mt5(account_info_none=True)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        assert result.can_trade is False
        assert result.reason == "account_info_unavailable"
        assert result.current_equity == pytest.approx(0.0)
        assert result.proposed_margin == pytest.approx(0.0)

    def test_order_calc_margin_none_returns_can_trade_false(self) -> None:
        mt5 = _make_mt5(equity=1000.0, margin=50.0, calc_margin_none=True)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        assert result.can_trade is False
        assert result.reason == "order_calc_margin_failed"
        assert result.proposed_margin == pytest.approx(0.0)
        # total_after falls back to margin_used (no proposed added)
        assert result.total_after == pytest.approx(50.0)

    def test_equity_non_positive_returns_can_trade_false(self) -> None:
        mt5 = _make_mt5(equity=0.0, margin=0.0, proposed_margin=50.0)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        assert result.can_trade is False
        assert result.reason == "equity_non_positive"
        assert result.current_equity == pytest.approx(0.0)
        assert result.proposed_margin == pytest.approx(50.0)


@pytest.mark.unit
class TestMarginCapImmutability:
    def test_result_is_frozen_dataclass(self) -> None:
        """MarginCheckResult must be a frozen dataclass — no field mutation allowed."""
        mt5 = _make_mt5(equity=1000.0, margin=0.0, proposed_margin=100.0)

        result = check_margin_cap(
            mt5, symbol="XAUUSD", action=ORDER_TYPE_BUY, volume=0.01, price=2300.0
        )

        assert isinstance(result, MarginCheckResult)
        with pytest.raises((AttributeError, TypeError)):
            result.can_trade = False  # type: ignore[misc]
