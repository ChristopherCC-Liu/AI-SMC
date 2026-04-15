"""Health monitoring for the AI-SMC live trading system.

Runs a battery of checks every cycle to ensure the system is operating
within safe parameters.  Any failed check triggers an alert via the
configured alerter.

Checks performed
----------------
1. MT5 connection alive (broker can return account info)
2. Data freshness (last bar timestamp < staleness threshold)
3. Margin level above minimum (> 200%)
4. No unreconciled positions (local state matches broker)
5. Daily loss limit not breached
"""

from __future__ import annotations

from datetime import datetime, timezone

from smc.monitor.types import HealthCheckResult, HealthStatus


class HealthMonitor:
    """Runs all health checks and returns an aggregate status.

    Parameters
    ----------
    max_data_age_seconds:
        Maximum acceptable age of the last bar in seconds.  Default 180
        (3 minutes) — generous for M15 data.
    min_margin_level_pct:
        Minimum acceptable margin level as a percentage.  Default 200.
    max_daily_loss_pct:
        Maximum daily loss as a percentage of starting balance.
        Default 3.0 (matches SMCConfig default).
    """

    def __init__(
        self,
        *,
        max_data_age_seconds: float = 180.0,
        min_margin_level_pct: float = 200.0,
        max_daily_loss_pct: float = 3.0,
    ) -> None:
        self._max_data_age_seconds = max_data_age_seconds
        self._min_margin_level_pct = min_margin_level_pct
        self._max_daily_loss_pct = max_daily_loss_pct

    def check_all(
        self,
        *,
        broker_connected: bool,
        last_bar_ts: datetime | None,
        margin_level_pct: float | None,
        unreconciled_count: int,
        daily_pnl_pct: float,
    ) -> HealthStatus:
        """Run all health checks and return aggregate status.

        Parameters
        ----------
        broker_connected:
            Whether the broker connection is alive.
        last_bar_ts:
            Timestamp of the most recent bar received.  None if no data yet.
        margin_level_pct:
            Current margin level as percentage.  None if no positions open.
        unreconciled_count:
            Number of positions that differ between local and broker state.
        daily_pnl_pct:
            Today's realised PnL as a percentage of starting balance.
            Negative values indicate loss.

        Returns
        -------
        HealthStatus
            Aggregate status with individual check results.
        """
        checks: list[HealthCheckResult] = [
            self._check_connection(broker_connected),
            self._check_data_freshness(last_bar_ts),
            self._check_margin(margin_level_pct),
            self._check_reconciliation(unreconciled_count),
            self._check_daily_loss(daily_pnl_pct),
        ]

        all_ok = all(c.passed for c in checks)
        now = datetime.now(tz=timezone.utc)

        return HealthStatus(
            all_ok=all_ok,
            checks=tuple(checks),
            checked_at=now,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_connection(self, broker_connected: bool) -> HealthCheckResult:
        if broker_connected:
            return HealthCheckResult(
                name="mt5_connection",
                passed=True,
                detail="Broker connection alive",
            )
        return HealthCheckResult(
            name="mt5_connection",
            passed=False,
            detail="Broker connection lost",
        )

    def _check_data_freshness(self, last_bar_ts: datetime | None) -> HealthCheckResult:
        if last_bar_ts is None:
            return HealthCheckResult(
                name="data_freshness",
                passed=False,
                detail="No bar data received yet",
            )

        now = datetime.now(tz=timezone.utc)
        # Ensure last_bar_ts is tz-aware
        if last_bar_ts.tzinfo is None:
            last_bar_ts = last_bar_ts.replace(tzinfo=timezone.utc)

        age_seconds = (now - last_bar_ts).total_seconds()
        if age_seconds <= self._max_data_age_seconds:
            return HealthCheckResult(
                name="data_freshness",
                passed=True,
                detail=f"Last bar {age_seconds:.0f}s ago (limit: {self._max_data_age_seconds:.0f}s)",
            )
        return HealthCheckResult(
            name="data_freshness",
            passed=False,
            detail=f"Data stale: last bar {age_seconds:.0f}s ago (limit: {self._max_data_age_seconds:.0f}s)",
        )

    def _check_margin(self, margin_level_pct: float | None) -> HealthCheckResult:
        if margin_level_pct is None:
            return HealthCheckResult(
                name="margin_level",
                passed=True,
                detail="No open positions — margin check skipped",
            )
        if margin_level_pct >= self._min_margin_level_pct:
            return HealthCheckResult(
                name="margin_level",
                passed=True,
                detail=f"Margin level {margin_level_pct:.1f}% (min: {self._min_margin_level_pct:.0f}%)",
            )
        return HealthCheckResult(
            name="margin_level",
            passed=False,
            detail=(
                f"Margin level {margin_level_pct:.1f}% below minimum "
                f"{self._min_margin_level_pct:.0f}%"
            ),
        )

    def _check_reconciliation(self, unreconciled_count: int) -> HealthCheckResult:
        if unreconciled_count == 0:
            return HealthCheckResult(
                name="reconciliation",
                passed=True,
                detail="All positions reconciled",
            )
        return HealthCheckResult(
            name="reconciliation",
            passed=False,
            detail=f"{unreconciled_count} unreconciled position(s)",
        )

    def _check_daily_loss(self, daily_pnl_pct: float) -> HealthCheckResult:
        # daily_pnl_pct is negative for losses
        loss_pct = abs(min(daily_pnl_pct, 0.0))
        if loss_pct < self._max_daily_loss_pct:
            return HealthCheckResult(
                name="daily_loss",
                passed=True,
                detail=f"Daily loss {loss_pct:.2f}% (limit: {self._max_daily_loss_pct:.1f}%)",
            )
        return HealthCheckResult(
            name="daily_loss",
            passed=False,
            detail=(
                f"Daily loss {loss_pct:.2f}% breaches limit "
                f"{self._max_daily_loss_pct:.1f}%"
            ),
        )


__all__ = ["HealthMonitor"]
