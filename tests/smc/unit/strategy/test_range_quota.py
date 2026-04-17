import json
import pytest
from datetime import date, datetime, timezone

from smc.strategy.range_quota import AsianRangeQuota


class TestAsianRangeQuota:
    def test_initial_not_exhausted(self):
        q = AsianRangeQuota()
        now = datetime(2026, 4, 17, 6, 30, tzinfo=timezone.utc)
        assert not q.is_exhausted_today(now)

    def test_record_and_exhaust_same_day(self, tmp_path):
        q = AsianRangeQuota()
        now = datetime(2026, 4, 17, 6, 30, tzinfo=timezone.utc)
        q2 = q.record_open(now, state_path=tmp_path / "quota.json")
        later_same_day = datetime(2026, 4, 17, 7, 45, tzinfo=timezone.utc)
        assert q2.is_exhausted_today(later_same_day)

    def test_next_day_resets(self, tmp_path):
        q = AsianRangeQuota()
        now = datetime(2026, 4, 17, 23, 59, tzinfo=timezone.utc)
        q2 = q.record_open(now, state_path=tmp_path / "quota.json")
        next_day = datetime(2026, 4, 18, 0, 1, tzinfo=timezone.utc)
        assert not q2.is_exhausted_today(next_day)

    def test_frozen_immutability(self):
        q = AsianRangeQuota()
        with pytest.raises(Exception):  # FrozenInstanceError
            q.last_open_date = date(2026, 4, 17)

    def test_record_returns_new_instance(self, tmp_path):
        q = AsianRangeQuota()
        q2 = q.record_open(
            datetime(2026, 4, 17, 6, 30, tzinfo=timezone.utc),
            state_path=tmp_path / "quota.json",
        )
        assert q is not q2
        assert q.last_open_date is None  # original unchanged

    def test_record_overwrites_previous_open(self, tmp_path):
        q = AsianRangeQuota(last_open_date=date(2026, 4, 16))
        q2 = q.record_open(
            datetime(2026, 4, 17, 6, 30, tzinfo=timezone.utc),
            state_path=tmp_path / "quota.json",
        )
        assert q2.last_open_date == date(2026, 4, 17)

    def test_boundary_utc_6_in_transition(self):
        # UTC 6:00 sharp → date is 2026-04-17 → same day exhaustion if already opened
        q = AsianRangeQuota(last_open_date=date(2026, 4, 17))
        at_6 = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)
        assert q.is_exhausted_today(at_6)


class TestAsianRangeQuotaPersistence:
    """Round 4.6-H2: save/load across process restarts."""

    def test_load_no_file_returns_fresh(self, tmp_path):
        q = AsianRangeQuota.load(state_path=tmp_path / "missing.json")
        assert q.last_open_date is None

    def test_record_persists_and_load_restores(self, tmp_path):
        path = tmp_path / "quota.json"
        q = AsianRangeQuota()
        q2 = q.record_open(
            datetime(2026, 4, 17, 6, 30, tzinfo=timezone.utc),
            state_path=path,
        )
        assert path.exists()
        raw = json.loads(path.read_text())
        assert raw["last_open_date"] == "2026-04-17"

        # Simulate process restart: fresh load from disk
        q3 = AsianRangeQuota.load(state_path=path)
        assert q3.last_open_date == date(2026, 4, 17)
        # And still exhausted for today's UTC date
        assert q3.is_exhausted_today(
            datetime(2026, 4, 17, 23, 0, tzinfo=timezone.utc)
        )

    def test_load_corrupt_file_returns_fresh(self, tmp_path):
        path = tmp_path / "quota.json"
        path.write_text("not json")
        q = AsianRangeQuota.load(state_path=path)
        assert q.last_open_date is None

    def test_load_missing_field_returns_fresh(self, tmp_path):
        path = tmp_path / "quota.json"
        path.write_text(json.dumps({"other": "key"}))
        q = AsianRangeQuota.load(state_path=path)
        assert q.last_open_date is None
