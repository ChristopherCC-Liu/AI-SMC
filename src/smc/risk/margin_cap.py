"""Global account margin cap — gate on total open-position margin before new orders.

Prevents dual-symbol processes (XAU + BTC on same $1000 demo) from collectively
exceeding a configurable fraction of equity. Each live_demo instance checks this
before placing an order; if total margin (existing + proposed) exceeds the cap,
the order is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MarginCheckResult:
    can_trade: bool
    reason: str
    current_margin_used: float
    current_equity: float
    proposed_margin: float
    total_after: float
    cap_ratio: float


class MT5Protocol(Protocol):
    def account_info(self) -> Any: ...
    def order_calc_margin(self, action: int, symbol: str, volume: float, price: float) -> float | None: ...


def check_margin_cap(
    mt5_client: MT5Protocol,
    *,
    symbol: str,
    action: int,  # mt5.ORDER_TYPE_BUY or ORDER_TYPE_SELL
    volume: float,
    price: float,
    max_pct: float = 0.40,
) -> MarginCheckResult:
    """Return whether opening ``volume`` lots of ``symbol`` at ``price`` stays
    within ``max_pct`` of account equity after existing open positions.

    ``max_pct=0.40`` means total margin (used + proposed) may not exceed 40% of equity.
    """
    acc = mt5_client.account_info()
    if acc is None:
        return MarginCheckResult(
            can_trade=False,
            reason="account_info_unavailable",
            current_margin_used=0.0,
            current_equity=0.0,
            proposed_margin=0.0,
            total_after=0.0,
            cap_ratio=max_pct,
        )

    equity = float(getattr(acc, "equity", 0.0))
    margin_used = float(getattr(acc, "margin", 0.0))

    proposed = mt5_client.order_calc_margin(action, symbol, volume, price)
    if proposed is None:
        return MarginCheckResult(
            can_trade=False,
            reason="order_calc_margin_failed",
            current_margin_used=margin_used,
            current_equity=equity,
            proposed_margin=0.0,
            total_after=margin_used,
            cap_ratio=max_pct,
        )

    total_after = margin_used + float(proposed)

    if equity <= 0.0:
        return MarginCheckResult(
            can_trade=False,
            reason="equity_non_positive",
            current_margin_used=margin_used,
            current_equity=equity,
            proposed_margin=float(proposed),
            total_after=total_after,
            cap_ratio=max_pct,
        )

    ratio_after = total_after / equity
    if ratio_after > max_pct:
        return MarginCheckResult(
            can_trade=False,
            reason=f"cap_exceeded: {ratio_after:.1%} > {max_pct:.1%}",
            current_margin_used=margin_used,
            current_equity=equity,
            proposed_margin=float(proposed),
            total_after=total_after,
            cap_ratio=max_pct,
        )

    return MarginCheckResult(
        can_trade=True,
        reason="ok",
        current_margin_used=margin_used,
        current_equity=equity,
        proposed_margin=float(proposed),
        total_after=total_after,
        cap_ratio=max_pct,
    )
