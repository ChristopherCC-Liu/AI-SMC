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
    "persistent_dd_breaker_enabled": False,
    "consec_loss_window_size": 3,
}
_TREATMENT_FLAGS = {
    "range_trend_filter_enabled": True,
    "range_ai_regime_gate_enabled": True,
    "spread_gate_enabled": True,
    "persistent_dd_breaker_enabled": True,
    "consec_loss_window_size": 6,
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
