"""Tests for ``smc.hedgerock.evolution.auto_purifier``.

Covers: queue purge, calibration purge, candidate archive, the full
sweep, audit log shape, the forbidden-workspace guard, and the frozen
contract on the public dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.auto_purifier import (
    AutoPurifier,
    FullPurgeReport,
    PurgeAction,
    PurgeReport,
)


_NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _ts_days_ago(days: float) -> str:
    return _iso(_NOW - timedelta(days=days))


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.mark.unit
def test_queue_purge_archives_stale_keeps_fresh(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    queue_dir = ws / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / "shadow_test_queue.jsonl"

    rows = [
        {"id": "fresh-1", "enqueued_at": _ts_days_ago(1)},
        {"id": "fresh-2", "enqueued_at": _ts_days_ago(10)},
        {"id": "fresh-3", "enqueued_at": _ts_days_ago(30)},
        {"id": "stale-1", "enqueued_at": _ts_days_ago(72)},
        {"id": "stale-2", "enqueued_at": _ts_days_ago(180)},
    ]
    queue_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )

    report = AutoPurifier.purge_stale_queue_items(
        queue_path, max_age_days=60, now=_NOW
    )

    assert report.target == "queue"
    assert report.n_archived == 2
    assert report.n_kept == 3

    archive_dir = ws / "archive" / "queue"
    archived_ids = {p.stem for p in archive_dir.glob("*.json")}
    assert archived_ids == {"stale-1", "stale-2"}

    remaining = [
        json.loads(line) for line in queue_path.read_text().splitlines()
    ]
    remaining_ids = {r["id"] for r in remaining}
    assert remaining_ids == {"fresh-1", "fresh-2", "fresh-3"}


@pytest.mark.unit
def test_queue_purge_empty_file_returns_zero_report(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    queue_dir = ws / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / "shadow_test_queue.jsonl"
    queue_path.write_text("", encoding="utf-8")

    report = AutoPurifier.purge_stale_queue_items(
        queue_path, max_age_days=60, now=_NOW
    )

    assert report.n_archived == 0
    assert report.n_kept == 0
    assert report.actions == ()
    # Audit log should not have been written for an empty queue.
    audit_log = Path(report.audit_log_path)
    assert not audit_log.exists()


@pytest.mark.unit
def test_queue_purge_missing_file_is_graceful(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    queue_dir = ws / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / "shadow_test_queue.jsonl"
    # File deliberately not created.

    report = AutoPurifier.purge_stale_queue_items(
        queue_path, max_age_days=60, now=_NOW
    )

    assert report.n_archived == 0
    assert report.n_kept == 0
    assert report.actions == ()


@pytest.mark.unit
def test_calibrations_purge_expires_old_priors(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    cal_dir = ws / "calibrator"
    cal_dir.mkdir()
    cal_path = cal_dir / "state.json"
    cal_path.write_text(
        json.dumps(
            {
                "schema": "calibrator/v1",
                "saved_at": _iso(_NOW),
                "priors": [
                    {
                        "parameter_class": "fresh-A",
                        "last_updated_at": _ts_days_ago(5),
                    },
                    {
                        "parameter_class": "fresh-B",
                        "last_updated_at": _ts_days_ago(20),
                    },
                    {
                        "parameter_class": "stale-A",
                        "last_updated_at": _ts_days_ago(60),
                    },
                    {
                        "parameter_class": "stale-B",
                        "last_updated_at": _ts_days_ago(120),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = AutoPurifier.purge_expired_calibrations(
        cal_path, max_age_days=45, now=_NOW
    )

    assert report.n_archived == 2
    assert report.n_kept == 2

    new_doc = json.loads(cal_path.read_text())
    kept_ids = {p["parameter_class"] for p in new_doc["priors"]}
    assert kept_ids == {"fresh-A", "fresh-B"}

    archive_dir = ws / "archive" / "calibrations"
    archive_files = list(archive_dir.glob("state-*.json"))
    assert len(archive_files) == 1
    archived = json.loads(archive_files[0].read_text())
    archived_ids = {p["parameter_class"] for p in archived["priors"]}
    assert archived_ids == {"stale-A", "stale-B"}


@pytest.mark.unit
def test_calibrations_purge_idempotent(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    cal_dir = ws / "calibrator"
    cal_dir.mkdir()
    cal_path = cal_dir / "state.json"
    cal_path.write_text(
        json.dumps(
            {
                "schema": "calibrator/v1",
                "saved_at": _iso(_NOW),
                "priors": [
                    {
                        "parameter_class": "fresh",
                        "last_updated_at": _ts_days_ago(5),
                    },
                    {
                        "parameter_class": "stale",
                        "last_updated_at": _ts_days_ago(120),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    first = AutoPurifier.purge_expired_calibrations(
        cal_path, max_age_days=45, now=_NOW
    )
    second = AutoPurifier.purge_expired_calibrations(
        cal_path, max_age_days=45, now=_NOW
    )

    assert first.n_archived == 1
    assert second.n_archived == 0
    assert second.n_kept == 1


@pytest.mark.unit
def test_candidates_archive_writes_json_files(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    archive_root = ws / "archive"

    candidates = [
        {"id": "cand-fresh", "generated_at": _ts_days_ago(10)},
        {"id": "cand-stale-1", "generated_at": _ts_days_ago(120)},
        {"id": "cand-stale-2", "generated_at": _ts_days_ago(365)},
    ]

    report = AutoPurifier.archive_obsolete_candidates(
        candidates,
        max_age_days=90,
        archive_root=archive_root,
        now=_NOW,
    )

    assert report.n_archived == 2
    assert report.n_kept == 1

    archived_ids = {
        p.stem for p in (archive_root / "candidates").glob("*.json")
    }
    assert archived_ids == {"cand-stale-1", "cand-stale-2"}


@pytest.mark.unit
def test_candidates_archive_idempotent_no_duplicates(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    archive_root = ws / "archive"
    audit_log_path = ws / "audit" / "auto_purifier.jsonl"

    stale = [{"id": "cand-stale", "generated_at": _ts_days_ago(200)}]

    first = AutoPurifier.archive_obsolete_candidates(
        stale,
        max_age_days=90,
        archive_root=archive_root,
        audit_log_path=audit_log_path,
        now=_NOW,
    )
    second = AutoPurifier.archive_obsolete_candidates(
        stale,
        max_age_days=90,
        archive_root=archive_root,
        audit_log_path=audit_log_path,
        now=_NOW,
    )

    assert first.n_archived == 1
    assert second.n_archived == 0

    # Only one audit line, even after two runs.
    audit_lines = audit_log_path.read_text().strip().splitlines()
    assert len(audit_lines) == 1


@pytest.mark.unit
def test_run_full_purification_all_three_targets(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)

    # Queue
    queue_dir = ws / "queue"
    queue_dir.mkdir()
    (queue_dir / "shadow_test_queue.jsonl").write_text(
        json.dumps({"id": "q-stale", "enqueued_at": _ts_days_ago(120)}) + "\n"
        + json.dumps({"id": "q-fresh", "enqueued_at": _ts_days_ago(2)}) + "\n",
        encoding="utf-8",
    )

    # Calibrations
    cal_dir = ws / "calibrator"
    cal_dir.mkdir()
    (cal_dir / "state.json").write_text(
        json.dumps(
            {
                "schema": "calibrator/v1",
                "saved_at": _iso(_NOW),
                "priors": [
                    {
                        "parameter_class": "stale",
                        "last_updated_at": _ts_days_ago(120),
                    },
                    {
                        "parameter_class": "fresh",
                        "last_updated_at": _ts_days_ago(5),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Candidates
    cand_dir = ws / "candidates"
    cand_dir.mkdir()
    (cand_dir / "c-stale.json").write_text(
        json.dumps({"id": "c-stale", "generated_at": _ts_days_ago(200)}),
        encoding="utf-8",
    )
    (cand_dir / "c-fresh.json").write_text(
        json.dumps({"id": "c-fresh", "generated_at": _ts_days_ago(5)}),
        encoding="utf-8",
    )

    report = AutoPurifier.run_full_purification(ws, now=_NOW)

    assert isinstance(report, FullPurgeReport)
    assert report.queue is not None
    assert report.calibrations is not None
    assert report.candidates is not None
    assert report.queue.n_archived == 1
    assert report.calibrations.n_archived == 1
    assert report.candidates.n_archived == 1
    assert report.overall_archived == 3


@pytest.mark.unit
def test_run_full_purification_only_queue_present(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    queue_dir = ws / "queue"
    queue_dir.mkdir()
    (queue_dir / "shadow_test_queue.jsonl").write_text(
        json.dumps({"id": "q1", "enqueued_at": _ts_days_ago(1)}) + "\n",
        encoding="utf-8",
    )

    report = AutoPurifier.run_full_purification(ws, now=_NOW)

    assert report.queue is not None
    assert report.calibrations is None
    assert report.candidates is None


@pytest.mark.unit
def test_audit_log_lines_have_six_keys(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    queue_dir = ws / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / "shadow_test_queue.jsonl"
    queue_path.write_text(
        json.dumps({"id": "q-stale", "enqueued_at": _ts_days_ago(200)}) + "\n",
        encoding="utf-8",
    )

    report = AutoPurifier.purge_stale_queue_items(
        queue_path, max_age_days=60, now=_NOW
    )
    audit_log = Path(report.audit_log_path)
    assert audit_log.exists()

    lines = audit_log.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    expected_keys = {
        "timestamp",
        "operation",
        "item_id",
        "from",
        "to",
        "reason",
        "max_age_days",
    }
    assert expected_keys.issubset(record.keys())
    assert record["operation"] == "purge_stale_queue_items"
    assert record["item_id"] == "q-stale"
    assert record["max_age_days"] == 60


@pytest.mark.unit
def test_forbidden_workspace_under_approved_raises(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "approved" / "ws"
    bad.mkdir(parents=True)
    with pytest.raises(ValueError):
        AutoPurifier.run_full_purification(bad, now=_NOW)

    # Also enforced on the lower-level helpers.
    queue_path = bad / "queue" / "shadow_test_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        AutoPurifier.purge_stale_queue_items(queue_path, now=_NOW)


@pytest.mark.unit
def test_frozen_dataclasses_reject_mutation() -> None:
    action = PurgeAction(
        item_id="x", archived_path="/tmp/x.json", reason="age=100d > max=60d"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.item_id = "y"  # type: ignore[misc]

    report = PurgeReport(
        target="queue",
        actions=(action,),
        n_archived=1,
        n_kept=0,
        audit_log_path="/tmp/audit.jsonl",
        generated_at=_iso(_NOW),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.n_archived = 99  # type: ignore[misc]

    full = FullPurgeReport(
        queue=None,
        calibrations=None,
        candidates=None,
        workspace="/tmp/ws",
        overall_archived=0,
        generated_at=_iso(_NOW),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        full.overall_archived = 5  # type: ignore[misc]
