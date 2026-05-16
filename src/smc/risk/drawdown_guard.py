"""Circuit-breaker that halts trading when drawdown limits are breached.

The guard is **stateless** — the caller is responsible for tracking
``peak_balance`` (high-water mark) and ``daily_pnl`` (today's realised P&L).
This keeps the guard pure and easy to test.

Two independent trip-wires:
1. **Daily loss** — if today's loss exceeds ``max_daily_loss_pct`` of balance,
   trading is halted for the rest of the day.
2. **Total drawdown** — if the account has drawn down more than
   ``max_drawdown_pct`` from its equity peak, all trading is halted until
   the operator manually resets the peak.
"""

from __future__ import annotations

from smc.risk.types import RiskBudget


class DrawdownGuard:
    """Stateless circuit breaker — caller tracks peak_balance and daily_pnl.

    Parameters
    ----------
    max_daily_loss_pct:
        Maximum intra-day loss as a percentage of balance (e.g. 3.0 = 3 %).
    max_drawdown_pct:
        Maximum drawdown from peak equity (e.g. 10.0 = 10 %).
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,
        max_drawdown_pct: float = 10.0,
    ) -> None:
        if max_daily_loss_pct <= 0:
            raise ValueError(f"max_daily_loss_pct must be positive, got {max_daily_loss_pct}")
        if max_drawdown_pct <= 0:
            raise ValueError(f"max_drawdown_pct must be positive, got {max_drawdown_pct}")

        self._max_daily_loss_pct = max_daily_loss_pct
        self._max_drawdown_pct = max_drawdown_pct

    def check_budget(
        self,
        balance: float,
        peak_balance: float,
        daily_pnl: float,
    ) -> RiskBudget:
        """Check whether the account is cleared to trade.

        Parameters
        ----------
        balance:
            Current account balance in USD.
        peak_balance:
            Historical equity high-water mark.
        daily_pnl:
            Today's realised P&L (negative = loss).

        Returns
        -------
        RiskBudget
            Snapshot indicating whether trading is permitted.
        """
        if balance <= 0:
            return RiskBudget(
                can_trade=False,
                available_risk_pct=0.0,
                used_risk_pct=100.0,
                daily_loss_pct=100.0,
                total_drawdown_pct=100.0,
                rejection_reason="Balance is zero or negative",
            )

        # --- Daily loss check ---
        # daily_pnl is negative when losing; compute loss as positive percentage
        daily_loss_pct = abs(min(daily_pnl, 0.0)) / balance * 100.0

        # --- Total drawdown check ---
        # peak_balance should be >= balance; clamp to avoid negative drawdown
        effective_peak = max(peak_balance, balance)
        total_drawdown_pct = (effective_peak - balance) / effective_peak * 100.0

        # --- Circuit breaker logic ---
        rejection_reason: str | None = None

        if total_drawdown_pct >= self._max_drawdown_pct:
            rejection_reason = (
                f"Max drawdown breached: {total_drawdown_pct:.1f}% >= "
                f"{self._max_drawdown_pct:.1f}% limit"
            )
        elif daily_loss_pct >= self._max_daily_loss_pct:
            rejection_reason = (
                f"Daily loss limit breached: {daily_loss_pct:.1f}% >= "
                f"{self._max_daily_loss_pct:.1f}% limit"
            )

        can_trade = rejection_reason is None

        # Available risk = how much more daily loss is allowed before the
        # daily circuit breaker trips (0.0 if already tripped).
        available_risk_pct = max(self._max_daily_loss_pct - daily_loss_pct, 0.0)
        used_risk_pct = min(daily_loss_pct, self._max_daily_loss_pct)

        # R10 P1.1: every halt tier in this guard blocks new opens but
        # leaves existing positions to their own SL/TP/trail. force_close
        # is reserved for a future emergency tier (ratio<=0.80).
        return RiskBudget(
            can_trade=can_trade,
            available_risk_pct=round(available_risk_pct, 4),
            used_risk_pct=round(used_risk_pct, 4),
            daily_loss_pct=round(daily_loss_pct, 4),
            total_drawdown_pct=round(total_drawdown_pct, 4),
            rejection_reason=rejection_reason,
            block_opens=not can_trade,
            force_close=False,
        )
