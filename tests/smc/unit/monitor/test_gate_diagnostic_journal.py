"""Unit tests for ``smc.monitor.gate_diagnostic_journal``.

R10 P1.2 — verify per-day jsonl persistence + UTC rotation + fail-closed IO
+ enabled-flag opt-out + escalation on repeated failures.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.monitor.gate_diagnostic_journal import (
    DEFAULT_JOURNAL_DIR,
    append_gate_diagnostic,
    journal_path_for,
    reset_failure_counter,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_failure_counter() -> None:
    """Each test starts with a clean per-day failure counter."""
    reset_failure_counter()
    yield
    reset_failure_counter()


# ── fixture helpers ──────────────────────────────────────────────────────────


def _sample_diagnostic() -> dict:
    """A representative ``_last_setup_diagnostic`` dict."""
    return {
        "htf_bias_direction": "bullish",
        "htf_bias_confidence": 0.65,
        "htf_bias_rationale": "Tier 1: D1+H4 BOS aligned",
        "stage_reject": "no_h1_zones",
        "final_count": 0,
        "h1_zones_count": 0,
        "min_confluence": 0.45,
        "current_price": 2350.12,
        "zone_rejects": {
            "cooldown": 0,
            "active_zones": 0,
            "intra_call_dedup": 0,
            "entry_none": 3,
            "trigger_filter": 1,
            "confluence_low": 2,
        },
        # Verbose field that should be stripped from the persisted record.
        "zone_details": [
            {"high": 2360.0, "low": 2340.0, "direction": "long",
             "dist_from_price": 5.2, "in_expanded_zone": True}
            for _ in range(20)
        ],
    }


# ── basic write + schema ─────────────────────────────────────────────────────


def test_append_writes_one_jsonl_line(tmp_path: Path) -> None:
    """append_gate_diagnostic writes exactly one parseable JSON line."""
    bar_ts = datetime(2026, 4, 25, 12, 34, 0, tzinfo=timezone.utc)

    out_path = append_gate_diagnostic(
        _sample_diagnostic(),
        bar_ts=bar_ts,
        symbol="XAUUSD",
        magic=19760418,
        journal_dir=tmp_path,
    )
    assert out_path is not None
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"expected 1 line, got {len(lines)}"

    record = json.loads(lines[0])
    assert record["bar_ts"] == "2026-04-25T12:34:00+00:00"
    assert record["symbol"] == "XAUUSD"
    assert record["magic"] == 19760418
    assert record["diagnostic"]["stage_reject"] == "no_h1_zones"
    assert record["diagnostic"]["final_count"] == 0
    # zone_rejects must survive
    assert record["diagnostic"]["zone_rejects"]["entry_none"] == 3


def test_append_strips_zone_details_to_keep_size_bounded(tmp_path: Path) -> None:
    """zone_details (verbose per-zone payload) must be excluded."""
    bar_ts = datetime(2026, 4, 25, 1, 0, tzinfo=timezone.utc)
    out_path = append_gate_diagnostic(
        _sample_diagnostic(),
        bar_ts=bar_ts,
        symbol="XAUUSD",
        magic=19760418,
        journal_dir=tmp_path,
    )
    assert out_path is not None
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert "zone_details" not in record["diagnostic"]
    # Aggregate zone_rejects must remain accessible.
    assert "zone_rejects" in record["diagnostic"]


def test_naive_bar_ts_treated_as_utc(tmp_path: Path) -> None:
    """A naive datetime should be treated as UTC, both for filename + isoformat."""
    bar_ts_naive = datetime(2026, 4, 25, 23, 59, 59)  # no tzinfo
    out_path = append_gate_diagnostic(
        _sample_diagnostic(),
        bar_ts=bar_ts_naive,
        symbol="XAUUSD",
        magic=19760418,
        journal_dir=tmp_path,
    )
    assert out_path is not None
    assert out_path.name == "aggregator_gates_2026-04-25.jsonl"
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["bar_ts"].endswith("+00:00")


def test_multiple_appends_accumulate_in_same_file(tmp_path: Path) -> None:
    """Two appends on the same UTC day must produce two lines in one file."""
    same_day_ts1 = datetime(2026, 4, 25, 0, 30, tzinfo=timezone.utc)
    same_day_ts2 = datetime(2026, 4, 25, 23, 30, tzinfo=timezone.utc)
    p1 = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=same_day_ts1, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    p2 = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=same_day_ts2, symbol="XAUUSD",
        magic=19760428, journal_dir=tmp_path,
    )
    assert p1 == p2  # same path
    lines = p1.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    magics = {json.loads(line)["magic"] for line in lines}
    assert magics == {19760418, 19760428}


# ── per-UTC-day rotation ─────────────────────────────────────────────────────


def test_per_day_rotation_at_utc_midnight(tmp_path: Path) -> None:
    """Two diff UTC days → two diff files."""
    day1 = datetime(2026, 4, 25, 23, 59, 59, tzinfo=timezone.utc)
    day2 = datetime(2026, 4, 26, 0, 0, 1, tzinfo=timezone.utc)
    p1 = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=day1, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    p2 = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=day2, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    assert p1 != p2
    assert p1.name == "aggregator_gates_2026-04-25.jsonl"
    assert p2.name == "aggregator_gates_2026-04-26.jsonl"


def test_journal_path_for_uses_default_dir() -> None:
    """journal_path_for falls back to DEFAULT_JOURNAL_DIR when no dir given."""
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    p = journal_path_for(bar_ts)
    assert p.parent == DEFAULT_JOURNAL_DIR
    assert p.name == "aggregator_gates_2026-04-25.jsonl"


# ── directory creation + fail-closed IO ──────────────────────────────────────


def test_append_creates_missing_dir(tmp_path: Path) -> None:
    """Helper must create ``journal_dir`` (and any intermediate parts)."""
    nested = tmp_path / "deep" / "nested" / "diagnostics"
    assert not nested.exists()
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    out = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=nested,
    )
    assert out is not None
    assert nested.exists()
    assert out.parent == nested


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_append_swallows_io_errors_with_exc_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A read-only journal_dir must NOT raise — helper logs WARN with exc_info."""
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    target = ro_dir / "aggregator_gates_2026-04-25.jsonl"
    target.write_text("", encoding="utf-8")
    target.chmod(stat.S_IRUSR)
    ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with caplog.at_level(logging.WARNING, logger="smc.monitor.gate_diagnostic_journal"):
            out = append_gate_diagnostic(
                _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
                magic=19760418, journal_dir=ro_dir,
            )
        assert out is None  # fail-closed
        # Helper must have logged at WARNING (1st failure on this day) with
        # an attached traceback so ops can root-cause it.
        records = [r for r in caplog.records if "gate_diag_write_failed" in r.getMessage()]
        assert len(records) == 1, f"expected 1 warning, got {len(records)}"
        assert records[0].levelno == logging.WARNING
        # exc_info=True attaches a traceback tuple via record.exc_info
        assert records[0].exc_info is not None
        assert records[0].exc_info[0] is not None  # type
    finally:
        ro_dir.chmod(stat.S_IRWXU)
        target.chmod(stat.S_IRWXU)


# ── enabled flag (backtest opt-out) ──────────────────────────────────────────


def test_disabled_flag_is_a_noop(tmp_path: Path) -> None:
    """``enabled=False`` must NOT touch the disk — backtest pollution guard."""
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    out = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path, enabled=False,
    )
    assert out is None
    # No file should have been created and no parent dir traversal happened.
    assert list(tmp_path.iterdir()) == []


def test_enabled_default_true_writes(tmp_path: Path) -> None:
    """Without explicit enabled kwarg the helper still writes (live-demo path)."""
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    out = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    assert out is not None
    assert out.exists()


# ── reproducible filename derived from bar_ts (not now()) ────────────────────


def test_filename_uses_bar_ts_not_wall_clock(tmp_path: Path) -> None:
    """Filename must come from bar_ts (replay reproducibility) — not datetime.now."""
    # Use a date in the past; the file must land on that date even though
    # now() is in 2026-04-25 (per the test environment).
    historical_bar = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
    out = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=historical_bar, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    assert out is not None
    assert out.name == "aggregator_gates_2024-01-15.jsonl"


# ── failure escalation (≥3 failures on the same UTC day) ─────────────────────


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_third_failure_escalates_to_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Three consecutive failures on the same UTC day → 3rd one logs at ERROR."""
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    target = ro_dir / "aggregator_gates_2026-04-25.jsonl"
    target.write_text("", encoding="utf-8")
    target.chmod(stat.S_IRUSR)
    ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with caplog.at_level(logging.WARNING, logger="smc.monitor.gate_diagnostic_journal"):
            for _ in range(3):
                append_gate_diagnostic(
                    _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
                    magic=19760418, journal_dir=ro_dir,
                )
        levels = [r.levelno for r in caplog.records if "gate_diag_write_failed" in r.getMessage()]
        assert levels == [
            logging.WARNING, logging.WARNING, logging.ERROR,
        ], f"unexpected escalation pattern: {levels}"
    finally:
        ro_dir.chmod(stat.S_IRWXU)
        target.chmod(stat.S_IRWXU)


def test_success_after_failure_resets_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful write clears the per-day failure counter so the next
    transient failure is a fresh WARNING (not still escalated).

    Uses pytest's ``monkeypatch`` fixture (not a manual try/finally) so the
    ``Path.open`` patch is rolled back even if pytest is interrupted by
    SIGINT / OOM / timeout. Manual try/finally would leak the patch
    across the rest of the session under those failure modes — caught by
    defense-impl-lead during cross-review of P1.2.
    """
    import smc.monitor.gate_diagnostic_journal as mod

    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    real_open = Path.open
    fail_count = {"n": 0}

    def flaky_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if fail_count["n"] < 2 and "a" in args:
            fail_count["n"] += 1
            raise OSError("simulated disk hiccup")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    for _ in range(2):
        assert append_gate_diagnostic(
            _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
            magic=19760418, journal_dir=tmp_path,
        ) is None
    # Counter has 2 — next attempt with the real open should succeed and reset it.
    assert mod._failure_counter.get(bar_ts.date()) == 2
    monkeypatch.setattr(Path, "open", real_open)
    assert append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    ) is not None
    assert mod._failure_counter.get(bar_ts.date()) is None
