"""Unit tests for ``scripts/build_gate_funnel.py``.

R10 P1.2 — verifies multi-magic separation, per-hour buckets, multi-day
trend, and graceful handling of empty / malformed jsonl.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# The script is not part of the installed package; load it dynamically so we
# can test the helper functions without a full CLI invocation.
_SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "build_gate_funnel.py"
_spec = importlib.util.spec_from_file_location("build_gate_funnel", _SCRIPT_PATH)
assert _spec and _spec.loader, f"could not load {_SCRIPT_PATH}"
build_gate_funnel = importlib.util.module_from_spec(_spec)
sys.modules["build_gate_funnel"] = build_gate_funnel
_spec.loader.exec_module(build_gate_funnel)  # type: ignore[union-attr]

pytestmark = pytest.mark.unit


# ── fixture helpers ──────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _record(
    *,
    hour: int,
    magic: int,
    stage_reject: str | None,
    date: str = "2026-04-25",
    symbol: str = "XAUUSD",
    zone_rejects: dict[str, int] | None = None,
) -> dict:
    return {
        "bar_ts": f"{date}T{hour:02d}:00:00+00:00",
        "symbol": symbol,
        "magic": magic,
        "diagnostic": {
            "stage_reject": stage_reject,
            "final_count": 0 if stage_reject else 1,
            "zone_rejects": zone_rejects or {
                "cooldown": 0, "active_zones": 0, "intra_call_dedup": 0,
                "entry_none": 0, "trigger_filter": 0, "confluence_low": 0,
            },
        },
    }


# ── histogram correctness ────────────────────────────────────────────────────


def test_histogram_counts_match_input(tmp_path: Path) -> None:
    """Synthetic 24h jsonl produces a histogram that sums to the input total."""
    path = tmp_path / "aggregator_gates_2026-04-25.jsonl"
    records = (
        [_record(hour=0, magic=19760418, stage_reject="htf_bias_neutral")] * 5
        + [_record(hour=1, magic=19760418, stage_reject="no_h1_zones")] * 3
        + [_record(hour=2, magic=19760418, stage_reject=None)] * 2  # passed
    )
    _write_jsonl(path, records)

    report = build_gate_funnel.build_funnel_report([path])
    assert report["total_calls"] == 10
    assert report["passed_all_gates"] == 2
    assert report["by_stage_reject"]["htf_bias_neutral"] == 5
    assert report["by_stage_reject"]["no_h1_zones"] == 3
    assert report["by_stage_reject"]["passed_all_gates"] == 2
    # Sum of histogram == total
    assert sum(report["by_stage_reject"].values()) == report["total_calls"]


def test_per_hour_bucket_separates_correctly(tmp_path: Path) -> None:
    """UTC hour bucket contains per-hour totals + reject/passed split."""
    path = tmp_path / "aggregator_gates_2026-04-25.jsonl"
    records = [
        _record(hour=0, magic=19760418, stage_reject="htf_bias_neutral"),
        _record(hour=0, magic=19760418, stage_reject=None),
        _record(hour=14, magic=19760418, stage_reject="no_h1_zones"),
    ]
    _write_jsonl(path, records)
    report = build_gate_funnel.build_funnel_report([path])
    assert report["by_hour_utc"]["0"] == {"total": 2, "rejects": 1, "passed": 1}
    assert report["by_hour_utc"]["14"] == {"total": 1, "rejects": 1, "passed": 0}


# ── empty / malformed handling ───────────────────────────────────────────────


def test_empty_jsonl_returns_zero_totals(tmp_path: Path) -> None:
    """Empty jsonl file yields a zero-filled report, never raises."""
    path = tmp_path / "aggregator_gates_2026-04-25.jsonl"
    path.write_text("", encoding="utf-8")
    report = build_gate_funnel.build_funnel_report([path])
    assert report["total_calls"] == 0
    assert report["passed_all_gates"] == 0
    assert report["by_stage_reject"] == {}
    assert report["by_magic"] == {}
    assert report["by_date"] == {}


def test_no_files_returns_empty_report(tmp_path: Path) -> None:
    """No journal files in the window → empty report, exit 0."""
    report = build_gate_funnel.build_funnel_report([])
    assert report["total_calls"] == 0
    assert report["files"] == []


def test_malformed_lines_skipped_silently(tmp_path: Path) -> None:
    """Garbage lines do not crash the parser."""
    path = tmp_path / "aggregator_gates_2026-04-25.jsonl"
    good = json.dumps(
        _record(hour=0, magic=19760418, stage_reject="htf_bias_neutral")
    )
    path.write_text(f"{good}\n{{not json}}\n\n{good}\n", encoding="utf-8")
    report = build_gate_funnel.build_funnel_report([path])
    assert report["total_calls"] == 2  # 2 good, malformed skipped


# ── multi-magic separation ───────────────────────────────────────────────────


def test_multi_magic_separation_correct(tmp_path: Path) -> None:
    """50 control + 30 treatment → counts match per leg, sum to total."""
    path = tmp_path / "aggregator_gates_2026-04-25.jsonl"
    control_records = [
        _record(hour=h % 24, magic=19760418, stage_reject="htf_bias_neutral")
        for h in range(50)
    ]
    treatment_records = [
        _record(hour=h % 24, magic=19760428, stage_reject="no_h1_zones")
        for h in range(30)
    ]
    _write_jsonl(path, control_records + treatment_records)
    report = build_gate_funnel.build_funnel_report([path])
    assert report["total_calls"] == 80
    assert report["by_magic"]["19760418"]["total"] == 50
    assert report["by_magic"]["19760428"]["total"] == 30
    assert (
        report["by_magic"]["19760418"]["total"]
        + report["by_magic"]["19760428"]["total"]
        == report["total_calls"]
    )
    assert report["by_magic"]["19760418"]["by_stage_reject"]["htf_bias_neutral"] == 50
    assert report["by_magic"]["19760428"]["by_stage_reject"]["no_h1_zones"] == 30


# ── multi-day trend ──────────────────────────────────────────────────────────


def test_multi_day_trend_per_date_series(tmp_path: Path) -> None:
    """Three jsonl files spanning 3 UTC days → by_date contains all three."""
    files: list[Path] = []
    for i, date in enumerate(["2026-04-23", "2026-04-24", "2026-04-25"]):
        p = tmp_path / f"aggregator_gates_{date}.jsonl"
        recs = [
            _record(hour=10, magic=19760418, stage_reject="htf_bias_neutral", date=date)
        ] * (i + 1)  # 1, 2, 3
        _write_jsonl(p, recs)
        files.append(p)
    report = build_gate_funnel.build_funnel_report(files)
    assert report["total_calls"] == 1 + 2 + 3
    assert sorted(report["by_date"].keys()) == [
        "2026-04-23",
        "2026-04-24",
        "2026-04-25",
    ]
    assert report["by_date"]["2026-04-23"]["total"] == 1
    assert report["by_date"]["2026-04-24"]["total"] == 2
    assert report["by_date"]["2026-04-25"]["total"] == 3


# ── zone_rejects aggregate ───────────────────────────────────────────────────


def test_zone_rejects_aggregate_sums_across_records(tmp_path: Path) -> None:
    """Sub-reason counters add up correctly across many records."""
    path = tmp_path / "aggregator_gates_2026-04-25.jsonl"
    records = [
        _record(hour=0, magic=19760418, stage_reject=None,
                zone_rejects={"entry_none": 4, "confluence_low": 1, "cooldown": 0}),
        _record(hour=1, magic=19760418, stage_reject="no_h1_zones",
                zone_rejects={"entry_none": 0, "confluence_low": 0}),
        _record(hour=2, magic=19760418, stage_reject=None,
                zone_rejects={"entry_none": 2, "confluence_low": 3}),
    ]
    _write_jsonl(path, records)
    report = build_gate_funnel.build_funnel_report([path])
    assert report["zone_rejects_aggregate"]["entry_none"] == 6
    assert report["zone_rejects_aggregate"]["confluence_low"] == 4
    # Zero-value entries should NOT appear in the aggregate.
    assert "cooldown" not in report["zone_rejects_aggregate"]


# ── file discovery ──────────────────────────────────────────────────────────


def test_discover_journal_files_window(tmp_path: Path) -> None:
    """discover_journal_files returns only files in the requested window."""
    for date in ["2026-04-20", "2026-04-23", "2026-04-25"]:
        (tmp_path / f"aggregator_gates_{date}.jsonl").write_text("", encoding="utf-8")
    end = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    files = build_gate_funnel.discover_journal_files(tmp_path, days=3, end_date=end)
    # Window 04-23..04-25; 04-20 excluded.
    assert [p.name for p in files] == [
        "aggregator_gates_2026-04-23.jsonl",
        "aggregator_gates_2026-04-25.jsonl",
    ]


def test_discover_journal_files_rejects_zero_days(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_gate_funnel.discover_journal_files(tmp_path, days=0)


# ── CLI entrypoint smoke test ────────────────────────────────────────────────


def test_main_writes_report_file(tmp_path: Path) -> None:
    """Running the CLI main() with a populated dir produces a JSON report."""
    journal_dir = tmp_path / "diagnostics"
    journal_dir.mkdir()
    path = journal_dir / "aggregator_gates_2026-04-25.jsonl"
    _write_jsonl(
        path,
        [_record(hour=0, magic=19760418, stage_reject="htf_bias_neutral")] * 3,
    )
    out_path = tmp_path / "out.json"
    rc = build_gate_funnel.main(
        [
            "--days", "1",
            "--journal-dir", str(journal_dir),
            "--out", str(out_path),
            "--end-date", "2026-04-25",
        ]
    )
    assert rc == 0
    assert out_path.exists()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["total_calls"] == 3
    assert parsed["by_stage_reject"]["htf_bias_neutral"] == 3
