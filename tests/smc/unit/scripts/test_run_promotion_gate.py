"""Unit tests for ``scripts.run_promotion_gate`` (R10 cal-2 runner).

Covers the full pipeline: journal load → arm classification → per-arm
PF/CI/histogram → verdict assembly → JSON output. Uses synthetic journal
entries so tests run without any baked production data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package, so import via path injection.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_promotion_gate as runner  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic journal-entry helpers
# ---------------------------------------------------------------------------


CONTROL_FLAGS_OFF: dict[str, object] = {
    "range_trend_filter_enabled": False,
    "range_ai_regime_gate_enabled": False,
    "spread_gate_enabled": False,
}
TREATMENT_FLAGS_ON: dict[str, object] = {
    "range_trend_filter_enabled": True,
    "range_ai_regime_gate_enabled": True,
    "spread_gate_enabled": True,
}


def _entry(
    *,
    trade_id: str,
    magic: int,
    pnl_usd: float,
    flags: dict[str, object] | None,
    loss_count: int | None = 0,
    timestamp: str = "2026-04-26T12:00:00+00:00",
) -> dict[str, object]:
    """Construct a synthetic journal entry mirroring live_demo's shape."""
    entry: dict[str, object] = {
        "trade_id": trade_id,
        "time": timestamp,
        "magic": magic,
        "pnl_usd": pnl_usd,
    }
    if flags is not None:
        entry["flags"] = flags
    if loss_count is not None:
        entry["loss_count_in_window_at_open"] = loss_count
    return entry


def _balanced_entries(n_per_arm: int) -> list[dict[str, object]]:
    """Build N control + N treatment entries with mixed pnl outcomes.

    ``n_per_arm >= 30`` is the min_n threshold for evaluate_promotion to
    consider a verdict valid; tests that exercise the verdict path use
    n_per_arm=40 to comfortably clear that gate.
    """
    entries: list[dict[str, object]] = []
    for i in range(n_per_arm):
        # Control: 60% wins of $10, 40% losses of $8 → PF ≈ 1.875
        pnl = 10.0 if i % 5 < 3 else -8.0
        entries.append(
            _entry(
                trade_id=f"c{i}",
                magic=runner.DEFAULT_CONTROL_MAGIC,
                pnl_usd=pnl,
                flags=CONTROL_FLAGS_OFF,
                loss_count=i % 7,  # spreads across 0-6 buckets
            )
        )
    for i in range(n_per_arm):
        # Treatment: 70% wins of $10, 30% losses of $5 → PF ≈ 4.67
        pnl = 10.0 if i % 10 < 7 else -5.0
        entries.append(
            _entry(
                trade_id=f"t{i}",
                magic=runner.DEFAULT_TREATMENT_MAGIC,
                pnl_usd=pnl,
                flags=TREATMENT_FLAGS_ON,
                loss_count=(i + 1) % 7,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# load_journal
# ---------------------------------------------------------------------------


def test_load_journal_parses_jsonl_into_dicts(tmp_path: Path) -> None:
    """Each non-blank line becomes one dict, in file order."""
    p = tmp_path / "j.jsonl"
    p.write_text(
        '{"trade_id": "a", "pnl_usd": 1.0}\n'
        '{"trade_id": "b", "pnl_usd": -2.0}\n'
    )
    rows = runner.load_journal(p)
    assert len(rows) == 2
    assert rows[0]["trade_id"] == "a"
    assert rows[1]["pnl_usd"] == -2.0


def test_load_journal_skips_blank_lines(tmp_path: Path) -> None:
    """Trailing newlines and blank lines mid-file are ignored."""
    p = tmp_path / "j.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n\n')
    rows = runner.load_journal(p)
    assert len(rows) == 2


def test_load_journal_raises_on_malformed_line(tmp_path: Path) -> None:
    """Malformed JSON surfaces as ValueError with line number — silent
    skip would bias the calibration verdict by an unknown amount.
    """
    p = tmp_path / "j.jsonl"
    p.write_text('{"good": 1}\n{this is not json\n')
    with pytest.raises(ValueError, match="line ?2|:2:"):
        runner.load_journal(p)


# ---------------------------------------------------------------------------
# build_default_arms
# ---------------------------------------------------------------------------


def test_build_default_arms_produces_two_arms() -> None:
    """Phase 1 always has exactly 2 arms: control and treatment."""
    arms = runner.build_default_arms(
        control_magic=19760418, treatment_magic=19760428
    )
    assert len(arms) == 2
    labels = {arm.arm_label for arm in arms}
    assert labels == {runner.ARM_CONTROL, runner.ARM_TREATMENT}


def test_build_default_arms_uses_strict_matching() -> None:
    """Phase 1 must be strict; subset is reserved for Phase 2 ablation."""
    arms = runner.build_default_arms(control_magic=1, treatment_magic=2)
    for arm in arms:
        assert arm.matching_mode == "strict"


def test_build_default_arms_pins_magics() -> None:
    """Each arm has its required_magic set to the supplied value, never None."""
    arms = runner.build_default_arms(control_magic=42, treatment_magic=99)
    by_label = {arm.arm_label: arm for arm in arms}
    assert by_label[runner.ARM_CONTROL].required_magic == 42
    assert by_label[runner.ARM_TREATMENT].required_magic == 99


# ---------------------------------------------------------------------------
# per_arm_metrics histogram
# ---------------------------------------------------------------------------


def test_per_arm_metrics_buckets_loss_counts_as_string_keys() -> None:
    """JSON-serialisable histogram must use string keys (bool 0/1 rule)."""
    from smc.eval.calibration_arms import _FlagSnapshot, TradeView

    snap = _FlagSnapshot(
        range_trend_filter_enabled=False,
        range_ai_regime_gate_enabled=False,
        spread_gate_enabled=False,
    )
    trades = [
        TradeView("a", "t", "control", snap, 1.0, 0),
        TradeView("b", "t", "control", snap, 1.0, 0),
        TradeView("c", "t", "control", snap, 1.0, 2),
    ]
    metrics = runner.per_arm_metrics(trades)
    hist = metrics["consec_loss_distribution"]
    assert hist == {"0": 2, "2": 1}
    # Keys must be str (JSON-serialisable) — int keys would dump as
    # surprises across Python int subclasses.
    for key in hist:
        assert isinstance(key, str)


def test_per_arm_metrics_handles_legacy_none_loss_count() -> None:
    """Pre-cal-0b records (loss_count=None) bucket under 'unknown' rather
    than being silently dropped — partial-data truth-telling.
    """
    from smc.eval.calibration_arms import _FlagSnapshot, TradeView

    snap = _FlagSnapshot(
        range_trend_filter_enabled=False,
        range_ai_regime_gate_enabled=False,
        spread_gate_enabled=False,
    )
    trades = [
        TradeView("a", "t", "control", snap, 1.0, None),
        TradeView("b", "t", "control", snap, 1.0, 0),
        TradeView("c", "t", "control", snap, 1.0, None),
    ]
    metrics = runner.per_arm_metrics(trades)
    assert metrics["consec_loss_distribution"] == {"unknown": 2, "0": 1}


def test_per_arm_metrics_empty_arm_returns_empty_histogram() -> None:
    """Zero trades → empty dict (NOT a missing key)."""
    metrics = runner.per_arm_metrics([])
    assert metrics["consec_loss_distribution"] == {}


# ---------------------------------------------------------------------------
# _serialize_float JSON-safe encoding
# ---------------------------------------------------------------------------


def test_serialize_float_passes_finite_through() -> None:
    """Normal floats survive untouched."""
    assert runner._serialize_float(1.5) == 1.5
    assert runner._serialize_float(0.0) == 0.0
    assert runner._serialize_float(-3.14) == -3.14


def test_serialize_float_converts_inf_to_string() -> None:
    """JSON has no native infinity; we emit the canonical string sentinel."""
    assert runner._serialize_float(float("inf")) == "inf"
    assert runner._serialize_float(float("-inf")) == "-inf"


def test_serialize_float_converts_nan_to_string() -> None:
    """NaN comparison via self-equality is the canonical Python idiom."""
    assert runner._serialize_float(float("nan")) == "nan"


# ---------------------------------------------------------------------------
# run_calibration end-to-end
# ---------------------------------------------------------------------------


def test_run_calibration_produces_full_payload_shape() -> None:
    """End-to-end: synthetic balanced journal yields all 4 top-level keys."""
    entries = _balanced_entries(n_per_arm=40)
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    assert set(payload.keys()) == {
        runner.ARM_CONTROL,
        runner.ARM_TREATMENT,
        "verdict",
        "diagnostics",
    }


def test_run_calibration_per_arm_record_has_expected_fields() -> None:
    """Each arm record must carry pf/ci_lower/ci_upper/n_trades/distribution."""
    entries = _balanced_entries(n_per_arm=40)
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    for arm in (runner.ARM_CONTROL, runner.ARM_TREATMENT):
        rec = payload[arm]
        assert set(rec.keys()) == {
            "pf",
            "ci_lower",
            "ci_upper",
            "n_trades",
            "consec_loss_distribution",
        }
        assert rec["n_trades"] == 40


def test_run_calibration_verdict_includes_promote_and_rationale() -> None:
    """Verdict block must surface the binary decision + reasoning string."""
    entries = _balanced_entries(n_per_arm=40)
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    verdict = payload["verdict"]
    assert isinstance(verdict["promote"], bool)
    assert isinstance(verdict["rationale"], str)
    assert verdict["rationale"]  # not empty


def test_run_calibration_diagnostics_buckets_are_zero_when_clean() -> None:
    """Clean control/treatment-only journal hits no diagnostic buckets."""
    entries = _balanced_entries(n_per_arm=40)
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    diag = payload["diagnostics"]
    # All 3 buckets present + all zero on a clean journal.
    assert diag == {
        "_control_with_unexpected_flag_on": 0,
        "_treatment_with_missing_flag": 0,
        "_unmatched": 0,
    }


def test_run_calibration_catches_control_with_flags_on() -> None:
    """A control-magic trade with treatment flags ON lands in the diagnostic."""
    entries = _balanced_entries(n_per_arm=30)
    # Inject 2 deployment-bug trades: control magic + treatment flags
    entries.append(
        _entry(
            trade_id="bug1",
            magic=runner.DEFAULT_CONTROL_MAGIC,
            pnl_usd=5.0,
            flags=TREATMENT_FLAGS_ON,
        )
    )
    entries.append(
        _entry(
            trade_id="bug2",
            magic=runner.DEFAULT_CONTROL_MAGIC,
            pnl_usd=-3.0,
            flags=TREATMENT_FLAGS_ON,
        )
    )
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    diag = payload["diagnostics"]
    assert diag["_control_with_unexpected_flag_on"] == 2
    assert diag["_treatment_with_missing_flag"] == 0
    # Control arm count must NOT include the two bugs (they go to diagnostic).
    assert payload[runner.ARM_CONTROL]["n_trades"] == 30


def test_run_calibration_catches_treatment_with_missing_flags() -> None:
    """Treatment magic + flags OFF lands in the symmetric diagnostic bucket."""
    entries = _balanced_entries(n_per_arm=30)
    entries.append(
        _entry(
            trade_id="bug1",
            magic=runner.DEFAULT_TREATMENT_MAGIC,
            pnl_usd=2.0,
            flags=CONTROL_FLAGS_OFF,
        )
    )
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    diag = payload["diagnostics"]
    assert diag["_treatment_with_missing_flag"] == 1


def test_run_calibration_catches_unknown_magic_in_unmatched() -> None:
    """A third magic (e.g., manual operator entry) lands in _unmatched."""
    entries = _balanced_entries(n_per_arm=30)
    entries.append(
        _entry(
            trade_id="manual",
            magic=99999999,  # neither control nor treatment magic
            pnl_usd=1.0,
            flags=None,  # manual entries usually have no flags dict
        )
    )
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    assert payload["diagnostics"]["_unmatched"] == 1


def test_run_calibration_handles_legacy_pre_cal_0_records() -> None:
    """Records lacking 'flags' dict (pre-cal-0) flow into diagnostics, not
    crash. They become _unmatched (no flag-snapshot to compare)."""
    entries = [
        _entry(
            trade_id=f"l{i}",
            magic=runner.DEFAULT_CONTROL_MAGIC,
            pnl_usd=1.0,
            flags=None,
            loss_count=None,
        )
        for i in range(10)
    ]
    payload = runner.run_calibration(entries, n_iter=100, rng_seed=0)
    # Legacy entries have no flags → no arm match → _unmatched.
    assert payload["diagnostics"]["_unmatched"] == 10
    assert payload[runner.ARM_CONTROL]["n_trades"] == 0
    assert payload[runner.ARM_TREATMENT]["n_trades"] == 0


def test_run_calibration_is_deterministic_for_fixed_rng_seed() -> None:
    """Same input + same seed → same output (bootstrap reproducibility)."""
    entries = _balanced_entries(n_per_arm=40)
    payload1 = runner.run_calibration(entries, n_iter=500, rng_seed=42)
    payload2 = runner.run_calibration(entries, n_iter=500, rng_seed=42)
    assert payload1 == payload2


def test_run_calibration_promote_false_when_n_below_min() -> None:
    """Below the 30-trade min_n, evaluate_promotion withholds promotion."""
    entries = _balanced_entries(n_per_arm=20)  # below default min_n=30
    payload = runner.run_calibration(entries, n_iter=500, rng_seed=0)
    assert payload["verdict"]["promote"] is False


def test_run_calibration_histogram_buckets_match_input_distribution() -> None:
    """The consec_loss_distribution must match the actual input frequency.

    Input has loss_count cycling 0..6 across 35 control trades, so each
    bucket should have either 5 or 6 occurrences (35/7 = 5).
    """
    entries = []
    for i in range(35):
        entries.append(
            _entry(
                trade_id=f"c{i}",
                magic=runner.DEFAULT_CONTROL_MAGIC,
                pnl_usd=1.0,
                flags=CONTROL_FLAGS_OFF,
                loss_count=i % 7,
            )
        )
    payload = runner.run_calibration(entries, n_iter=100, rng_seed=0)
    hist = payload[runner.ARM_CONTROL]["consec_loss_distribution"]
    # 35 / 7 = 5 occurrences per bucket
    for bucket in ("0", "1", "2", "3", "4", "5", "6"):
        assert hist.get(bucket) == 5, f"bucket {bucket}: {hist}"


# ---------------------------------------------------------------------------
# write_output (file IO)
# ---------------------------------------------------------------------------


def test_write_output_creates_directory_and_file(tmp_path: Path) -> None:
    """write_output mkdirs missing dirs + writes a parseable JSON file."""
    out_dir = tmp_path / "round10"
    out_path = runner.write_output(
        {"verdict": {"promote": True}},
        bake_window="2026-04-26",
        output_dir=out_dir,
    )
    assert out_path.exists()
    assert out_path.parent == out_dir
    assert out_path.name == "calibration_2026-04-26.json"
    parsed = json.loads(out_path.read_text())
    assert parsed["verdict"]["promote"] is True


def test_write_output_overwrites_existing_file(tmp_path: Path) -> None:
    """A re-run with same bake_window must overwrite, not append."""
    runner.write_output({"v": 1}, bake_window="bw", output_dir=tmp_path)
    runner.write_output({"v": 2}, bake_window="bw", output_dir=tmp_path)
    parsed = json.loads((tmp_path / "calibration_bw.json").read_text())
    assert parsed == {"v": 2}


# ---------------------------------------------------------------------------
# CLI main entry point
# ---------------------------------------------------------------------------


def test_main_returns_nonzero_when_journal_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing journal must surface as a non-zero exit code (ops alert)."""
    bogus = tmp_path / "does_not_exist.jsonl"
    rc = runner.main([
        "--journal", str(bogus),
        "--bake-window", "2026-04-26",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "journal not found" in err.lower()


def test_main_writes_output_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: journal exists, runner writes a JSON, exits 0."""
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "\n".join(json.dumps(e) for e in _balanced_entries(40)) + "\n"
    )
    out_dir = tmp_path / "round10"
    rc = runner.main([
        "--journal", str(journal),
        "--bake-window", "test-window",
        "--output-dir", str(out_dir),
        "--n-iter", "200",
    ])
    assert rc == 0
    out_file = out_dir / "calibration_test-window.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert "verdict" in payload
    out = capsys.readouterr().out
    assert "[CALIBRATION-VERDICT]" in out
