"""End-to-end calibration pipeline integration test (R10 cal-3).

Drives the full chain: synthetic journal → defense's classify_trades →
foundation's run_calibration → JSON output. Validates the contract
between cal-1 (classifier) and cal-2 (runner) holds at the seam.

Distinct from `tests/smc/unit/scripts/test_run_promotion_gate.py`, which
exercises cal-2 in isolation: that test focuses on cal-2's internal
shape; this test focuses on the cross-module contract — specifically
that ``ArmMembership`` instances flow through ``evaluate_promotion``
without explicit conversion (Protocol composability), and that the
diagnostic-bucket counts surfaced by the runner match the structural
invariants defense's classifier guarantees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package, so import via path injection.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_promotion_gate as runner  # noqa: E402

from smc.eval.calibration_arms import (  # noqa: E402
    ArmDefinition,
    TradeView,
    classify_trades,
)
from smc.eval.promotion_gate import evaluate_promotion  # noqa: E402

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture: synthetic journal that exercises every classification path
# ---------------------------------------------------------------------------


_CONTROL_FLAGS = {
    "range_trend_filter_enabled": False,
    "range_ai_regime_gate_enabled": False,
    "spread_gate_enabled": False,
}
_TREATMENT_FLAGS = {
    "range_trend_filter_enabled": True,
    "range_ai_regime_gate_enabled": True,
    "spread_gate_enabled": True,
}


def _entry(
    *, trade_id: str, magic: int, pnl_usd: float,
    flags: dict | None, loss_count: int | None = 0,
) -> dict:
    entry: dict = {
        "trade_id": trade_id,
        "time": "2026-04-26T12:00:00+00:00",
        "magic": magic,
        "pnl_usd": pnl_usd,
    }
    if flags is not None:
        entry["flags"] = flags
    if loss_count is not None:
        entry["loss_count_in_window_at_open"] = loss_count
    return entry


@pytest.fixture
def mixed_journal() -> list[dict]:
    """Build a journal that exercises every code path:
    - 35 control + 35 treatment (above min_n=30 for verdict)
    - 1 control_with_unexpected_flag_on (deployment bug)
    - 1 treatment_with_missing_flag (deployment bug)
    - 1 unknown-magic manual entry
    - 1 legacy pre-cal-0 record (no flags) → unmatched
    """
    entries: list[dict] = []
    for i in range(35):
        pnl = 12.0 if i % 5 < 3 else -7.0
        entries.append(_entry(
            trade_id=f"c{i}",
            magic=runner.DEFAULT_CONTROL_MAGIC,
            pnl_usd=pnl,
            flags=_CONTROL_FLAGS,
            loss_count=i % 7,
        ))
    for i in range(35):
        pnl = 14.0 if i % 10 < 7 else -4.0
        entries.append(_entry(
            trade_id=f"t{i}",
            magic=runner.DEFAULT_TREATMENT_MAGIC,
            pnl_usd=pnl,
            flags=_TREATMENT_FLAGS,
            loss_count=(i + 2) % 7,
        ))
    entries.append(_entry(
        trade_id="bug_ctrl",
        magic=runner.DEFAULT_CONTROL_MAGIC,
        pnl_usd=2.0,
        flags=_TREATMENT_FLAGS,
    ))
    entries.append(_entry(
        trade_id="bug_treat",
        magic=runner.DEFAULT_TREATMENT_MAGIC,
        pnl_usd=-3.0,
        flags=_CONTROL_FLAGS,
    ))
    entries.append(_entry(
        trade_id="manual",
        magic=99999999,
        pnl_usd=1.0,
        flags=None,
    ))
    entries.append(_entry(
        trade_id="legacy",
        magic=runner.DEFAULT_CONTROL_MAGIC,
        pnl_usd=0.5,
        flags=None,
        loss_count=None,
    ))
    return entries


# ---------------------------------------------------------------------------
# Contract test: ArmMembership IS-A TradeLike (Protocol composability)
# ---------------------------------------------------------------------------


def test_arm_membership_flows_into_evaluate_promotion_without_conversion(
    mixed_journal: list[dict],
) -> None:
    """Defense's TradeView satisfies the TradeLike Protocol via duck typing.

    The "call succeeded" is itself the composability proof — if any
    field were missing or mistyped, evaluate_promotion's per-trade
    pnl_usd access (or numpy boundary in the bootstrap loop) would
    raise. No explicit @runtime_checkable + isinstance() required.
    """
    arms = runner.build_default_arms(
        control_magic=runner.DEFAULT_CONTROL_MAGIC,
        treatment_magic=runner.DEFAULT_TREATMENT_MAGIC,
    )
    classified = classify_trades(
        mixed_journal,
        arm_definitions=arms,
        control_magic=runner.DEFAULT_CONTROL_MAGIC,
        treatment_magic=runner.DEFAULT_TREATMENT_MAGIC,
    )
    control = classified.arms[runner.ARM_CONTROL]
    treatment = classified.arms[runner.ARM_TREATMENT]

    # Both must materialize as Sequence[TradeView]; runner ships them
    # straight to evaluate_promotion. If TradeView's pnl_usd field were
    # renamed or missing, this call would raise.
    verdict = evaluate_promotion(
        baseline_trades=control,
        treatment_trades=treatment,
        n_iter=200,
        rng_seed=0,
    )

    assert verdict.n_baseline == len(control)
    assert verdict.n_treatment == len(treatment)
    assert isinstance(verdict.pf_baseline, float)
    assert isinstance(verdict.pf_treatment, float)


def test_classified_trade_views_carry_required_fields(
    mixed_journal: list[dict],
) -> None:
    """Each TradeView has the 6 fields cal-2 reads.

    Pins the contract that cal-1's TradeView dataclass exposes
    pnl_usd + loss_count_in_window_at_open + arm_label, the 3 fields
    cal-2 actually consumes during accumulator + verdict assembly.
    """
    arms = runner.build_default_arms(
        control_magic=runner.DEFAULT_CONTROL_MAGIC,
        treatment_magic=runner.DEFAULT_TREATMENT_MAGIC,
    )
    classified = classify_trades(
        mixed_journal,
        arm_definitions=arms,
        control_magic=runner.DEFAULT_CONTROL_MAGIC,
        treatment_magic=runner.DEFAULT_TREATMENT_MAGIC,
    )
    for trade in classified.arms[runner.ARM_CONTROL]:
        assert isinstance(trade, TradeView)
        assert hasattr(trade, "pnl_usd")
        assert hasattr(trade, "loss_count_in_window_at_open")
        assert hasattr(trade, "arm_label")
        assert trade.arm_label == runner.ARM_CONTROL


# ---------------------------------------------------------------------------
# End-to-end pipeline: full runner against the mixed journal
# ---------------------------------------------------------------------------


def test_full_pipeline_emits_complete_verdict_payload(
    mixed_journal: list[dict],
) -> None:
    """run_calibration on the mixed journal produces the complete output:
    both arms with PF/CI/n/histogram, verdict block, all 3 diagnostics.
    """
    payload = runner.run_calibration(
        mixed_journal,
        n_iter=500,
        rng_seed=0,
    )
    # Top-level keys
    assert set(payload.keys()) == {
        runner.ARM_CONTROL,
        runner.ARM_TREATMENT,
        "verdict",
        "diagnostics",
    }
    # Per-arm shape — 5 keys including the histogram
    for arm in (runner.ARM_CONTROL, runner.ARM_TREATMENT):
        assert set(payload[arm].keys()) == {
            "pf",
            "ci_lower",
            "ci_upper",
            "n_trades",
            "consec_loss_distribution",
        }
        assert payload[arm]["n_trades"] == 35
    # Diagnostics: deployment-bug bucket counts
    assert payload["diagnostics"] == {
        "_control_with_unexpected_flag_on": 1,
        "_treatment_with_missing_flag": 1,
        # manual + legacy both → unmatched (legacy has no flags)
        "_unmatched": 2,
    }


def test_end_to_end_writes_parseable_json(
    tmp_path: Path,
    mixed_journal: list[dict],
) -> None:
    """The on-disk JSON must be round-trippable + structurally identical
    to the in-memory payload (modulo float precision)."""
    payload = runner.run_calibration(mixed_journal, n_iter=500, rng_seed=0)
    out_path = runner.write_output(
        payload, bake_window="integration-test", output_dir=tmp_path,
    )
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    # Top-level must round-trip cleanly
    assert set(parsed.keys()) == set(payload.keys())
    # Verdict block must round-trip with same promote bool
    assert parsed["verdict"]["promote"] == payload["verdict"]["promote"]
    # Diagnostic counts integer round-trip
    assert parsed["diagnostics"] == payload["diagnostics"]


def test_pipeline_via_cli_main(tmp_path: Path, mixed_journal: list[dict]) -> None:
    """The CLI entry point exercises the full chain: write a journal file,
    invoke main(), assert the output JSON matches the in-memory payload.

    This is the "as ops would run it" smoke test — if the CLI changed
    a default that drifts from the python-API call signature, this
    catches it before VPS deployment.
    """
    journal_path = tmp_path / "j.jsonl"
    journal_path.write_text(
        "\n".join(json.dumps(e) for e in mixed_journal) + "\n"
    )
    out_dir = tmp_path / "round10"
    rc = runner.main([
        "--journal", str(journal_path),
        "--bake-window", "integration",
        "--output-dir", str(out_dir),
        "--n-iter", "300",
    ])
    assert rc == 0
    out_file = out_dir / "calibration_integration.json"
    assert out_file.exists()
    parsed = json.loads(out_file.read_text())
    # CLI-driven payload mirrors the python-API run_calibration shape
    assert set(parsed.keys()) == {"control", "treatment", "verdict", "diagnostics"}
    assert parsed["diagnostics"]["_control_with_unexpected_flag_on"] == 1
    assert parsed["diagnostics"]["_treatment_with_missing_flag"] == 1


# ---------------------------------------------------------------------------
# Histogram contract: cal-0b field semantics flow through unchanged
# ---------------------------------------------------------------------------


def test_histogram_bucket_keys_match_input_loss_count_values(
    mixed_journal: list[dict],
) -> None:
    """Each loss_count cycling 0..6 across 35 trades → 5 occurrences each.

    The cal-0b → cal-1 → cal-2 chain must preserve the loss_count_in_window
    integer through the journal-write → classifier-extraction → runner-
    histogram pipeline. If anyone in the chain accidentally swaps the
    accessor (e.g., to consec_losses), the bucket counts diverge.
    """
    payload = runner.run_calibration(mixed_journal, n_iter=200, rng_seed=0)
    # Control loss_count was i % 7 across i in 0..34 → 5 each for buckets 0..6
    control_hist = payload[runner.ARM_CONTROL]["consec_loss_distribution"]
    for bucket in ("0", "1", "2", "3", "4", "5", "6"):
        assert control_hist.get(bucket) == 5, (
            f"control bucket {bucket}: {control_hist}"
        )
    # Treatment loss_count was (i+2) % 7 — same distribution shape
    treatment_hist = payload[runner.ARM_TREATMENT]["consec_loss_distribution"]
    for bucket in ("0", "1", "2", "3", "4", "5", "6"):
        assert treatment_hist.get(bucket) == 5, (
            f"treatment bucket {bucket}: {treatment_hist}"
        )


# ---------------------------------------------------------------------------
# Producer-real fixture: catches cal-1 ↔ SMCConfig schema drift at design time
# ---------------------------------------------------------------------------


def test_classify_trades_against_real_smcconfig_schema() -> None:
    """Pin the cal-1 ↔ SMCConfig contract by instantiating the real producer.

    Drift this catches: if a future cal-1 maintainer adds a field to
    ``FeatureFlagSnapshot`` that doesn't exist on ``SMCConfig``, or if a
    future SMCConfig evolution renames a field that cal-1's Protocol still
    expects, this test fails loud at design time. No production breakage.

    The test is the producer-real defense against the silent-zero failure
    mode that motivated cal-1-tweak / cal-2-tweak: with synthesized fixture
    flags the pipeline tested green but produced 0-trade arms on real bake.
    Constructing ``SMCConfig()`` directly + running ``_serialize_flags(cfg)``
    means the fixture mirrors what cal-0 actually emits in production.

    See ``feedback_bidirectional_grep_at_design.md`` (memory file) for the
    pattern + the R10 incident that motivated it.
    """
    from smc.config import SMCConfig
    from smc.monitor.gate_diagnostic_journal import _serialize_flags

    # Real producer instantiation — NOT a synthesized dict. If SMCConfig
    # ever loses a field cal-1's Protocol references, _serialize_flags
    # output omits it, _flags_match strict-mode rejects, and this test
    # fails by surfacing 0 trades in arms["control"].
    cfg = SMCConfig()
    flags_dict = _serialize_flags(cfg)

    arms = runner.build_default_arms(
        control_magic=runner.DEFAULT_CONTROL_MAGIC,
        treatment_magic=runner.DEFAULT_TREATMENT_MAGIC,
    )

    # Build 35 control entries (above min_n=30) using the real flags
    # snapshot. SMCConfig() defaults are control-leg semantics (all R10
    # flags False), so trades with control magic + real-default flags
    # must populate the control arm cleanly.
    entries = [
        {
            "trade_id": f"real_ctrl_{i}",
            "time": "2026-04-26T12:00:00+00:00",
            "magic": runner.DEFAULT_CONTROL_MAGIC,
            "pnl_usd": 1.0 if i % 2 == 0 else -0.5,
            "flags": flags_dict,
            "loss_count_in_window_at_open": 0,
        }
        for i in range(35)
    ]

    classified = classify_trades(
        entries,
        arm_definitions=arms,
        control_magic=runner.DEFAULT_CONTROL_MAGIC,
        treatment_magic=runner.DEFAULT_TREATMENT_MAGIC,
    )

    # The 35 entries built from the real SMCConfig snapshot MUST land in
    # the control arm. If they fall to _unmatched, the Protocol diverged
    # from SMCConfig — the same failure mode this commit fixes.
    assert len(classified.arms[runner.ARM_CONTROL]) == 35, (
        f"control arm should have 35 trades from real SMCConfig snapshot, "
        f"got {len(classified.arms[runner.ARM_CONTROL])}. "
        f"Diagnostic counts: "
        f"unmatched={len(classified.diagnostics.get('_unmatched', ()))}, "
        f"control_unexpected="
        f"{len(classified.diagnostics.get('_control_with_unexpected_flag_on', ()))}, "
        f"treatment_missing="
        f"{len(classified.diagnostics.get('_treatment_with_missing_flag', ()))}. "
        f"flags emitted by SMCConfig: {sorted(flags_dict.keys())}"
    )
    # Diagnostic buckets must all be empty for a clean control-only journal.
    assert len(classified.diagnostics.get("_unmatched", ())) == 0
    assert (
        len(classified.diagnostics.get("_control_with_unexpected_flag_on", ()))
        == 0
    )
    assert (
        len(classified.diagnostics.get("_treatment_with_missing_flag", ())) == 0
    )


def test_serialize_flags_output_matches_protocol_field_set() -> None:
    """Verify ``_serialize_flags(SMCConfig())`` emits at minimum the fields
    cal-1's ``FeatureFlagSnapshot`` Protocol declares.

    Mechanically enforces the bidirectional-grep discipline: if a future
    cal-1 evolution adds a field to the Protocol without a corresponding
    SMCConfig field, this test fails loud. Catches the drift class at the
    serializer-Protocol seam, before any classifier integration runs.
    """
    from smc.config import SMCConfig
    from smc.eval.calibration_arms import FeatureFlagSnapshot
    from smc.monitor.gate_diagnostic_journal import _serialize_flags

    cfg = SMCConfig()
    flags_dict = _serialize_flags(cfg)

    # Every Protocol-required field must be emitted by the serializer.
    # Use Protocol.__annotations__ so the test auto-discovers field set
    # without manual maintenance — when cal-1 changes the Protocol, this
    # test re-runs against the new field set automatically.
    protocol_fields = set(FeatureFlagSnapshot.__annotations__.keys())
    serializer_fields = set(flags_dict.keys())

    missing = protocol_fields - serializer_fields
    assert not missing, (
        f"Protocol fields not emitted by _serialize_flags: {sorted(missing)}. "
        f"Either add them to SMCConfig + _FLAG_FIELDS, or remove from "
        f"FeatureFlagSnapshot Protocol."
    )
