"""Tests for smc.monitor.health.HealthMonitor.

Covers all five health checks:
1. MT5 connection
2. Data freshness
3. Margin level
4. Position reconciliation
5. Daily loss limit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.monitor.health import HealthMonitor


@pytest.fixture()
def monitor() -> HealthMonitor:
    """Standard health monitor with default thresholds."""
    return HealthMonitor(
        max_data_age_seconds=180.0,
        min_margin_level_pct=200.0,
        max_daily_loss_pct=3.0,
    )


# ---------------------------------------------------------------------------
# All checks pass
# ---------------------------------------------------------------------------


class TestAllPassHealthy:
    """When everything is healthy, all_ok is True."""

    def test_all_ok_when_healthy(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now - timedelta(seconds=10),
            margin_level_pct=350.0,
            unreconciled_count=0,
            daily_pnl_pct=0.5,
        )
        assert status.all_ok is True
        assert len(status.checks) == 5
        assert all(c.passed for c in status.checks)

    def test_timestamp_is_utc(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        assert status.checked_at.tzinfo is not None


# ---------------------------------------------------------------------------
# MT5 Connection
# ---------------------------------------------------------------------------


class TestConnectionCheck:
    def test_connection_alive(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        conn = [c for c in status.checks if c.name == "mt5_connection"][0]
        assert conn.passed is True

    def test_connection_lost(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=False,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        conn = [c for c in status.checks if c.name == "mt5_connection"][0]
        assert conn.passed is False
        assert status.all_ok is False


# ---------------------------------------------------------------------------
# Data Freshness
# ---------------------------------------------------------------------------


class TestDataFreshness:
    def test_fresh_data(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now - timedelta(seconds=30),
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        freshness = [c for c in status.checks if c.name == "data_freshness"][0]
        assert freshness.passed is True

    def test_stale_data(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now - timedelta(seconds=300),
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        freshness = [c for c in status.checks if c.name == "data_freshness"][0]
        assert freshness.passed is False

    def test_no_data_yet(self, monitor: HealthMonitor) -> None:
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=None,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        freshness = [c for c in status.checks if c.name == "data_freshness"][0]
        assert freshness.passed is False


# ---------------------------------------------------------------------------
# Margin Level
# ---------------------------------------------------------------------------


class TestMarginLevel:
    def test_adequate_margin(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=500.0,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        margin = [c for c in status.checks if c.name == "margin_level"][0]
        assert margin.passed is True

    def test_low_margin(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=150.0,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        margin = [c for c in status.checks if c.name == "margin_level"][0]
        assert margin.passed is False

    def test_no_positions_skips_margin(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        margin = [c for c in status.checks if c.name == "margin_level"][0]
        assert margin.passed is True

    def test_exact_minimum_passes(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=200.0,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        margin = [c for c in status.checks if c.name == "margin_level"][0]
        assert margin.passed is True


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_all_reconciled(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=0.0,
        )
        recon = [c for c in status.checks if c.name == "reconciliation"][0]
        assert recon.passed is True

    def test_unreconciled_positions(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=2,
            daily_pnl_pct=0.0,
        )
        recon = [c for c in status.checks if c.name == "reconciliation"][0]
        assert recon.passed is False
        assert "2" in recon.detail


# ---------------------------------------------------------------------------
# Daily Loss
# ---------------------------------------------------------------------------


class TestDailyLoss:
    def test_within_limit(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=-2.0,
        )
        loss = [c for c in status.checks if c.name == "daily_loss"][0]
        assert loss.passed is True

    def test_limit_breached(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=-3.5,
        )
        loss = [c for c in status.checks if c.name == "daily_loss"][0]
        assert loss.passed is False

    def test_profitable_day_passes(self, monitor: HealthMonitor) -> None:
        now = datetime.now(tz=timezone.utc)
        status = monitor.check_all(
            broker_connected=True,
            last_bar_ts=now,
            margin_level_pct=None,
            unreconciled_count=0,
            daily_pnl_pct=5.0,
        )
        loss = [c for c in status.checks if c.name == "daily_loss"][0]
        assert loss.passed is True


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    def test_strict_monitor(self) -> None:
        strict = HealthMonitor(
            max_data_age_seconds=30.0,
            min_margin_level_pct=500.0,
            max_daily_loss_pct=1.0,
        )
        now = datetime.now(tz=timezone.utc)
        status = strict.check_all(
            broker_connected=True,
            last_bar_ts=now - timedelta(seconds=60),
            margin_level_pct=300.0,
            unreconciled_count=0,
            daily_pnl_pct=-1.5,
        )
        assert status.all_ok is False
        failed_names = {c.name for c in status.checks if not c.passed}
        assert "data_freshness" in failed_names
        assert "margin_level" in failed_names
        assert "daily_loss" in failed_names
