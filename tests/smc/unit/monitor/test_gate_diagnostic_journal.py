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
    SCHEMA_VERSION,
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
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["bar_ts"] == "2026-04-25T12:34:00+00:00"
    assert record["symbol"] == "XAUUSD"
    assert record["magic"] == 19760418
    assert record["diagnostic"]["stage_reject"] == "no_h1_zones"
    assert record["diagnostic"]["final_count"] == 0
    # zone_rejects must survive
    assert record["diagnostic"]["zone_rejects"]["entry_none"] == 3
    # commit 0 backward-compat: when no config is supplied, no flags key.
    assert "flags" not in record


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


# ── R10 P4 calibration commit 0: schema_version=2 + flags ────────────────────


class _FakeConfig:
    """Stand-in for ``SMCConfig`` exposing the R10 flag fields.

    We deliberately use a duck-typed stub instead of importing SMCConfig:
    keeping the helper config-agnostic means it works against any object
    exposing the named fields, which is the same Protocol-style flexibility
    the calibration runner relies on.
    """

    def __init__(
        self,
        *,
        macro_enabled: bool = False,
        ai_regime_enabled: bool = False,
        ai_mode_router_enabled: bool = False,
        ai_regime_trust_threshold: float = 0.6,
        range_trend_filter_enabled: bool = False,
        range_ai_regime_gate_enabled: bool = False,
        range_require_regime_valid: bool = False,
        range_reversal_confirm_enabled: bool = False,
        range_ai_direction_entry_gate_enabled: bool = False,
        mode_router_trending_dominance_enabled: bool = False,
        spread_gate_enabled: bool = False,
        max_concurrent_per_symbol: int = 1,
        anti_stack_cooldown_minutes: int = 60,
    ) -> None:
        self.macro_enabled = macro_enabled
        self.ai_regime_enabled = ai_regime_enabled
        self.ai_mode_router_enabled = ai_mode_router_enabled
        self.ai_regime_trust_threshold = ai_regime_trust_threshold
        self.range_trend_filter_enabled = range_trend_filter_enabled
        self.range_ai_regime_gate_enabled = range_ai_regime_gate_enabled
        self.range_require_regime_valid = range_require_regime_valid
        self.range_reversal_confirm_enabled = range_reversal_confirm_enabled
        self.range_ai_direction_entry_gate_enabled = (
            range_ai_direction_entry_gate_enabled
        )
        self.mode_router_trending_dominance_enabled = (
            mode_router_trending_dominance_enabled
        )
        self.spread_gate_enabled = spread_gate_enabled
        self.max_concurrent_per_symbol = max_concurrent_per_symbol
        self.anti_stack_cooldown_minutes = anti_stack_cooldown_minutes


def test_schema_version_field_present_on_every_record(tmp_path: Path) -> None:
    """Every persisted record must carry ``schema_version`` for forward-compat.

    Calibration consumers key off this version to migrate gracefully — a
    record without it would have to be inferred as v1 by absence-of-field,
    which is brittle.  Make the version explicit in every record.
    """
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    out_path = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    assert out_path is not None
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["schema_version"] == 2


def test_flags_omitted_when_no_config_supplied(tmp_path: Path) -> None:
    """Backward-compat: callers that don't supply ``config`` produce records
    without a ``flags`` key.  Calibration's ``_extract_flags_from_record``
    helper treats absence as "use boot-time snapshot fallback."
    """
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    out_path = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path,
    )
    assert out_path is not None
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert "flags" not in record


def test_flags_present_when_config_supplied_treatment(tmp_path: Path) -> None:
    """When ``config`` is supplied, ``flags`` snapshots all R10 fields."""
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    treatment_cfg = _FakeConfig(
        macro_enabled=True,
        ai_regime_enabled=True,
        ai_mode_router_enabled=True,
        range_trend_filter_enabled=True,
        range_ai_regime_gate_enabled=True,
        range_require_regime_valid=True,
        spread_gate_enabled=True,
    )
    out_path = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760428, journal_dir=tmp_path, config=treatment_cfg,
    )
    assert out_path is not None
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert "flags" in record
    flags = record["flags"]
    assert flags["macro_enabled"] is True
    assert flags["ai_regime_enabled"] is True
    assert flags["ai_mode_router_enabled"] is True
    assert flags["range_trend_filter_enabled"] is True
    assert flags["range_ai_regime_gate_enabled"] is True
    assert flags["spread_gate_enabled"] is True
    # The non-default-True fields must serialize their actual values, not
    # be silently coerced.
    assert flags["max_concurrent_per_symbol"] == 1
    assert flags["anti_stack_cooldown_minutes"] == 60


def test_flags_present_when_config_supplied_control(tmp_path: Path) -> None:
    """Control-leg config (all flags False) produces a flags dict with False
    values — the absence of a field would be ambiguous (is it "False" or
    "missing"?), so we explicitly emit every known field's actual value.
    """
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    control_cfg = _FakeConfig()  # all defaults — all flags False
    out_path = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760418, journal_dir=tmp_path, config=control_cfg,
    )
    assert out_path is not None
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    flags = record["flags"]
    # Every R10 boolean flag is explicitly False (NOT missing).
    assert flags["macro_enabled"] is False
    assert flags["ai_regime_enabled"] is False
    assert flags["range_trend_filter_enabled"] is False
    assert flags["range_ai_regime_gate_enabled"] is False
    assert flags["spread_gate_enabled"] is False


def test_flags_handle_partial_config_gracefully(tmp_path: Path) -> None:
    """A config object missing some R10 fields still produces a partial
    snapshot — missing fields are silently omitted (calibration consumers
    tolerate partial records since they fall back on the boot-time snapshot
    for the same arm membership).
    """
    bar_ts = datetime(2026, 4, 25, tzinfo=timezone.utc)

    class PartialConfig:
        # Only exposes 2 of the 13 known fields; rest are missing.
        macro_enabled = True
        spread_gate_enabled = True

    out_path = append_gate_diagnostic(
        _sample_diagnostic(), bar_ts=bar_ts, symbol="XAUUSD",
        magic=19760428, journal_dir=tmp_path, config=PartialConfig(),
    )
    assert out_path is not None
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    flags = record["flags"]
    assert flags == {"macro_enabled": True, "spread_gate_enabled": True}
    # Fields not on the partial config are not invented out of thin air.
    assert "ai_regime_enabled" not in flags
    assert "range_trend_filter_enabled" not in flags


def test_serialize_flags_returns_empty_dict_for_none() -> None:
    """``_serialize_flags(None)`` is a defensive no-op returning an empty dict.

    This is the boundary between "caller wants flags but forgot the config"
    (would land here) and "caller deliberately omits flags" (skips the
    helper entirely).  We treat both as benign — the writer never crashes
    just because the call site forgot to wire SMCConfig in.
    """
    from smc.monitor.gate_diagnostic_journal import _serialize_flags

    assert _serialize_flags(None) == {}


def test_serialize_flags_pure_does_not_mutate_config() -> None:
    """``_serialize_flags`` must not mutate the supplied config — the snapshot
    is read-only.  This is a property test for the pure-function contract.
    """
    from smc.monitor.gate_diagnostic_journal import _serialize_flags

    cfg = _FakeConfig(macro_enabled=True, range_trend_filter_enabled=True)
    snap_before = vars(cfg).copy()
    _serialize_flags(cfg)
    snap_after = vars(cfg).copy()
    assert snap_before == snap_after
