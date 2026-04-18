"""audit-r2 ops #4: reconcile cursor must survive restarts.

Prevents silent double-counting of closed deals into consec_halt /
phase1a_breaker / daily_pnl when the live_demo process restarts.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.monitor.reconcile_cursor import (
    load_reconcile_cursor,
    save_reconcile_cursor,
)


# ---------------------------------------------------------------------------
# Save + reload roundtrip (primary regression: crash → restart → no replay)
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_save_then_load_returns_same_ts(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        ts = datetime(2026, 4, 18, 10, 30, 0, tzinfo=timezone.utc)
        save_reconcile_cursor(cursor_path, ts)
        loaded = load_reconcile_cursor(cursor_path)
        assert loaded == ts

    def test_save_uses_atomic_rename(self, tmp_path: Path):
        """File should exist after save; no .tmp leftover on success."""
        cursor_path = tmp_path / "cursor.json"
        ts = datetime.now(timezone.utc)
        save_reconcile_cursor(cursor_path, ts)
        assert cursor_path.exists()
        # tmp file cleaned up by atomic_write_json
        assert not cursor_path.with_suffix(".json.tmp").exists()

    def test_save_overwrites_previous(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        ts1 = datetime(2026, 4, 18, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 4, 18, 14, 0, 0, tzinfo=timezone.utc)
        save_reconcile_cursor(cursor_path, ts1)
        save_reconcile_cursor(cursor_path, ts2)
        assert load_reconcile_cursor(cursor_path) == ts2


# ---------------------------------------------------------------------------
# Fallback paths (first-boot, missing, corrupt, impossible values)
# ---------------------------------------------------------------------------

class TestFallback:
    def test_missing_file_returns_now_minus_12h(self, tmp_path: Path):
        """First boot: no persisted cursor → 12h lookback."""
        cursor_path = tmp_path / "nonexistent.json"
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)

    def test_corrupt_json_returns_fallback(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text("{this is not json")
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)

    def test_missing_ts_field_returns_fallback(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text(json.dumps({"other_field": "hello"}))
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)

    def test_invalid_ts_string_returns_fallback(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text(json.dumps({"ts": "not-a-date"}))
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)

    def test_empty_file_returns_fallback(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text("")
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)


# ---------------------------------------------------------------------------
# Sanity bounds (never return future; cap very-old timestamps)
# ---------------------------------------------------------------------------

class TestSanityBounds:
    def test_future_ts_falls_back_to_default(self, tmp_path: Path):
        """Persisted ts > now = impossible (clock skew?) → don't hide future deals."""
        cursor_path = tmp_path / "cursor.json"
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        future_ts = now + timedelta(hours=2)
        save_reconcile_cursor(cursor_path, future_ts)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)

    def test_very_old_ts_capped_to_7_days(self, tmp_path: Path):
        """Week-long outage → cap scan to last 7 days, not months of history."""
        cursor_path = tmp_path / "cursor.json"
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        ancient_ts = now - timedelta(days=30)
        save_reconcile_cursor(cursor_path, ancient_ts)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(days=7)

    def test_recent_ts_within_window_preserved(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        recent_ts = now - timedelta(minutes=15)
        save_reconcile_cursor(cursor_path, recent_ts)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == recent_ts


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------

class TestTimezone:
    def test_naive_ts_on_save_normalised_to_utc(self, tmp_path: Path):
        """Defensive: if caller passes a naive datetime, coerce to UTC."""
        cursor_path = tmp_path / "cursor.json"
        naive = datetime(2026, 4, 18, 10, 30, 0)  # no tzinfo
        save_reconcile_cursor(cursor_path, naive)
        loaded = load_reconcile_cursor(cursor_path)
        assert loaded.tzinfo is not None
        assert loaded == naive.replace(tzinfo=timezone.utc)

    def test_naive_ts_on_disk_treated_as_utc(self, tmp_path: Path):
        cursor_path = tmp_path / "cursor.json"
        # Simulate a write by an older version that didn't normalise
        cursor_path.write_text(json.dumps({"ts": "2026-04-18T10:30:00"}))
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded.tzinfo is not None
        assert loaded == datetime(2026, 4, 18, 10, 30, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Regression scenario: simulate crash + restart
# ---------------------------------------------------------------------------

class TestCrashRestartScenario:
    def test_restart_does_not_replay_deals(self, tmp_path: Path):
        """
        Timeline:
          T+0   — process running, last reconcile at T+0, cursor saved
          T+1h  — process crashed (cursor still reflects T+0)
          T+2h  — process restarts → load_reconcile_cursor reads T+0
                   NOT "T+2h - 12h" which would replay 11h of history
        """
        cursor_path = tmp_path / "cursor.json"
        start = datetime(2026, 4, 18, 10, 0, 0, tzinfo=timezone.utc)
        save_reconcile_cursor(cursor_path, start)
        # Crash simulated by doing nothing.
        restart_time = start + timedelta(hours=2)
        loaded = load_reconcile_cursor(cursor_path, now=restart_time)
        # Must return the saved cursor, not restart_time - 12h
        assert loaded == start
        assert loaded != restart_time - timedelta(hours=12)

    def test_cold_start_still_scans_12h(self, tmp_path: Path):
        """First deploy: no cursor yet → original 12h lookback preserved."""
        cursor_path = tmp_path / "cursor.json"
        now = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
        loaded = load_reconcile_cursor(cursor_path, now=now)
        assert loaded == now - timedelta(hours=12)
