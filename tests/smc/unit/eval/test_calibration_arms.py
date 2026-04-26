"""Unit tests for ``smc.eval.calibration_arms`` (R10 cal-1).

Covers:
- Protocol composability: ``TradeView`` IS-A ``TradeLike`` via duck typing
  (instances flow into ``evaluate_promotion`` without DTO conversion).
- Strict matching: under-specified arm definition fails closed.
- Subset matching: Phase 2 ablation arms allow unspecified flags to vary.
- Three diagnostic buckets: control_unexpected / treatment_missing / unmatched.
- Backward-compat: legacy pre-cal-0/cal-0b entries (no flags dict, no
  loss_count_in_window_at_open) classify cleanly via fallback_snapshot.
- Field extraction: ``pnl_usd`` top-level + ``loss_count_in_window_at_open``
  with ``None`` default for missing.
- Journal order preserved per arm (downstream computes rolling stats).
"""
from __future__ import annotations

import pytest

from smc.eval.calibration_arms import (
    DIAGNOSTIC_CONTROL_UNEXPECTED,
    DIAGNOSTIC_TREATMENT_MISSING,
    DIAGNOSTIC_UNMATCHED,
    ArmDefinition,
    FeatureFlagSnapshot,
    TradeView,
    classify_trades,
)
from smc.eval.promotion_gate import TradeLike, evaluate_promotion

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CONTROL_MAGIC = 19760418
_TREATMENT_MAGIC = 19760428


def _all_off_flags() -> dict[str, bool | int]:
    return {
        "range_trend_filter_enabled": False,
        "range_ai_regime_gate_enabled": False,
        "spread_gate_enabled": False,
        "persistent_dd_breaker_enabled": False,
        "consec_loss_window_size": 3,
    }


def _all_on_flags() -> dict[str, bool | int]:
    return {
        "range_trend_filter_enabled": True,
        "range_ai_regime_gate_enabled": True,
        "spread_gate_enabled": True,
        "persistent_dd_breaker_enabled": True,
        "consec_loss_window_size": 6,
    }


def _journal_entry(
    *,
    time: str = "2026-04-25T12:00:00+00:00",
    magic: int = _CONTROL_MAGIC,
    pnl_usd: float = 10.0,
    flags: dict[str, bool | int] | None = None,
    loss_count: int | None = 0,
    trade_id: str | None = None,
) -> dict[str, object]:
    """Build a synthetic journal entry mirroring the live_demo.py shape."""
    entry: dict[str, object] = {
        "time": time,
        "magic": magic,
        "pnl_usd": pnl_usd,
    }
    if flags is not None:
        entry["flags"] = flags
    if loss_count is not None:
        entry["loss_count_in_window_at_open"] = loss_count
    if trade_id is not None:
        entry["trade_id"] = trade_id
    return entry


def _control_arm() -> ArmDefinition:
    return ArmDefinition(
        arm_label="control",
        required_flags=_all_off_flags(),
        required_magic=_CONTROL_MAGIC,
    )


def _treatment_arm() -> ArmDefinition:
    return ArmDefinition(
        arm_label="treatment_full",
        required_flags=_all_on_flags(),
        required_magic=_TREATMENT_MAGIC,
    )


# ---------------------------------------------------------------------------
# Basic classification — strict mode happy path
# ---------------------------------------------------------------------------


class TestStrictClassificationHappyPath:
    """Strict matching: control + treatment trades land in the right arms."""

    def test_control_trade_with_all_flags_off_matches_control_arm(self):
        entry = _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags())
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["control"]) == 1
        assert len(result.arms["treatment_full"]) == 0
        assert result.arms["control"][0].arm_label == "control"

    def test_treatment_trade_with_all_flags_on_matches_treatment_arm(self):
        entry = _journal_entry(magic=_TREATMENT_MAGIC, flags=_all_on_flags())
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["treatment_full"]) == 1
        assert len(result.arms["control"]) == 0

    def test_journal_order_preserved_per_arm(self):
        entries = [
            _journal_entry(time=f"2026-04-25T12:0{i}:00+00:00",
                           magic=_CONTROL_MAGIC, flags=_all_off_flags(),
                           pnl_usd=float(i))
            for i in range(5)
        ]
        result = classify_trades(
            entries,
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        pnls = [t.pnl_usd for t in result.arms["control"]]
        assert pnls == [0.0, 1.0, 2.0, 3.0, 4.0], "arm trades must preserve input order"

    def test_under_specified_strict_arm_fails_closed(self):
        """Strict mode requires ALL FeatureFlagSnapshot fields in required_flags.

        If only 1 field is named, the classifier refuses to match — better
        to fail closed (diagnostic bucket) than to silently match arms
        with under-specified definitions.
        """
        partial_arm = ArmDefinition(
            arm_label="partial",
            required_flags={"spread_gate_enabled": True},  # only 1 of 5
            required_magic=_TREATMENT_MAGIC,
        )
        entry = _journal_entry(magic=_TREATMENT_MAGIC, flags=_all_on_flags())
        result = classify_trades(
            [entry],
            arm_definitions=[partial_arm],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        # Treatment magic + all flags ON, but partial_arm under-specified
        # → falls through to diagnostic. With treatment_magic + all flags
        # on (so all-flags-on holds), treatment_with_missing_flag is FALSE,
        # so this lands in unmatched.
        assert len(result.arms["partial"]) == 0
        assert len(result.diagnostics[DIAGNOSTIC_UNMATCHED]) == 1


# ---------------------------------------------------------------------------
# Subset matching — Phase 2 ablation arms
# ---------------------------------------------------------------------------


class TestSubsetMatching:
    """Subset mode: only named flags compared; unnamed allowed to vary."""

    def test_subset_arm_matches_when_only_named_flags_align(self):
        """Ablation: 'spread_gate_only' arm cares only about spread_gate=True."""
        spread_only = ArmDefinition(
            arm_label="spread_gate_only",
            required_flags={"spread_gate_enabled": True},
            required_magic=_TREATMENT_MAGIC,
            matching_mode="subset",
        )
        # Mixed flags: spread_gate=True, others vary
        flags = _all_off_flags()
        flags["spread_gate_enabled"] = True
        flags["consec_loss_window_size"] = 4  # also varies
        entry = _journal_entry(magic=_TREATMENT_MAGIC, flags=flags)
        result = classify_trades(
            [entry],
            arm_definitions=[spread_only],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["spread_gate_only"]) == 1

    def test_subset_arm_rejects_when_named_flag_mismatches(self):
        spread_only = ArmDefinition(
            arm_label="spread_gate_only",
            required_flags={"spread_gate_enabled": True},
            required_magic=_TREATMENT_MAGIC,
            matching_mode="subset",
        )
        flags = _all_on_flags()
        flags["spread_gate_enabled"] = False  # the named flag is OFF
        entry = _journal_entry(magic=_TREATMENT_MAGIC, flags=flags)
        result = classify_trades(
            [entry],
            arm_definitions=[spread_only],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["spread_gate_only"]) == 0


# ---------------------------------------------------------------------------
# Diagnostic buckets — all three deployment-bug paths
# ---------------------------------------------------------------------------


class TestDiagnosticBuckets:
    """Three symmetric buckets for deployment bugs."""

    def test_control_with_unexpected_flag_lands_in_control_diagnostic(self):
        """Control magic but at least one flag accidentally ON."""
        flags = _all_off_flags()
        flags["spread_gate_enabled"] = True  # the deployment mistake
        entry = _journal_entry(magic=_CONTROL_MAGIC, flags=flags)
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["control"]) == 0
        assert len(result.diagnostics[DIAGNOSTIC_CONTROL_UNEXPECTED]) == 1

    def test_treatment_with_missing_flag_lands_in_treatment_diagnostic(self):
        """Treatment magic but at least one expected flag is OFF."""
        flags = _all_on_flags()
        flags["spread_gate_enabled"] = False  # the deployment mistake
        entry = _journal_entry(magic=_TREATMENT_MAGIC, flags=flags)
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["treatment_full"]) == 0
        assert len(result.diagnostics[DIAGNOSTIC_TREATMENT_MISSING]) == 1

    def test_third_magic_lands_in_unmatched_diagnostic(self):
        """Unknown magic (e.g., manual operator entry) → unmatched bucket."""
        entry = _journal_entry(magic=99999999, flags=_all_off_flags())
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.diagnostics[DIAGNOSTIC_UNMATCHED]) == 1
        assert len(result.diagnostics[DIAGNOSTIC_CONTROL_UNEXPECTED]) == 0
        assert len(result.diagnostics[DIAGNOSTIC_TREATMENT_MISSING]) == 0

    def test_three_buckets_always_present_in_output(self):
        """Even with no diagnostic-bucket trades, all 3 bucket keys exist."""
        result = classify_trades(
            [],
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert DIAGNOSTIC_CONTROL_UNEXPECTED in result.diagnostics
        assert DIAGNOSTIC_TREATMENT_MISSING in result.diagnostics
        assert DIAGNOSTIC_UNMATCHED in result.diagnostics


# ---------------------------------------------------------------------------
# Backward compat — legacy pre-cal-0 / pre-cal-0b records
# ---------------------------------------------------------------------------


class TestBackwardCompatLegacyRecords:
    """Calibration v1 must run against any mix of pre/post journals."""

    def test_legacy_record_no_flags_dict_uses_fallback_snapshot(self):
        """Pre-cal-0 entry has no `flags` key → fallback_snapshot kicks in."""
        from smc.eval.calibration_arms import _FlagSnapshot
        fallback = _FlagSnapshot(
            range_trend_filter_enabled=False,
            range_ai_regime_gate_enabled=False,
            spread_gate_enabled=False,
            persistent_dd_breaker_enabled=False,
            consec_loss_window_size=3,
        )
        legacy_entry = _journal_entry(magic=_CONTROL_MAGIC, flags=None)
        result = classify_trades(
            [legacy_entry],
            arm_definitions=[_control_arm()],
            fallback_snapshot=fallback,
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        # With control magic + fallback all-off, must match control arm.
        assert len(result.arms["control"]) == 1

    def test_legacy_record_no_loss_count_field_yields_none(self):
        """Pre-cal-0b entry has no `loss_count_in_window_at_open` → None."""
        legacy_entry = _journal_entry(
            magic=_CONTROL_MAGIC, flags=_all_off_flags(), loss_count=None,
        )
        result = classify_trades(
            [legacy_entry],
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert result.arms["control"][0].loss_count_in_window_at_open is None

    def test_post_cal_0b_record_loss_count_preserved_as_int(self):
        entry = _journal_entry(
            magic=_CONTROL_MAGIC, flags=_all_off_flags(), loss_count=4,
        )
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert result.arms["control"][0].loss_count_in_window_at_open == 4

    def test_legacy_record_no_fallback_falls_through_to_diagnostic(self):
        """No `flags` AND no fallback_snapshot → unmatched bucket."""
        legacy_entry = _journal_entry(magic=_CONTROL_MAGIC, flags=None)
        result = classify_trades(
            [legacy_entry],
            arm_definitions=[_control_arm()],
            fallback_snapshot=None,
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        # No snapshot to compare → cannot match arm. Magic == control AND
        # snapshot is None → diagnostic logic skips the "any flag on"
        # check (snapshot is None) and falls through to unmatched.
        assert len(result.arms["control"]) == 0
        assert len(result.diagnostics[DIAGNOSTIC_UNMATCHED]) == 1


# ---------------------------------------------------------------------------
# Field extraction — pnl_usd robustness
# ---------------------------------------------------------------------------


class TestFieldExtraction:
    """``pnl_usd`` top-level extraction + missing-field defensive cast."""

    def test_pnl_usd_top_level_int_cast_to_float(self):
        entry = _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags(),
                               pnl_usd=42)  # int, not float
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert result.arms["control"][0].pnl_usd == 42.0
        assert isinstance(result.arms["control"][0].pnl_usd, float)

    def test_pnl_usd_missing_defaults_to_zero(self):
        entry = _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags())
        del entry["pnl_usd"]
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert result.arms["control"][0].pnl_usd == 0.0


# ---------------------------------------------------------------------------
# Protocol composability — TradeView IS-A TradeLike via duck typing
# ---------------------------------------------------------------------------


class TestProtocolComposability:
    """The whole point of cal-1: TradeView flows into evaluate_promotion."""

    def test_trade_view_satisfies_tradelike_protocol_via_pnl_usd(self):
        """Structural subtyping: TradeView has pnl_usd: float, so IS-A TradeLike.

        Python's typing.Protocol uses structural subtyping (PEP 544) — no
        explicit inheritance required. This test locks in the contract:
        instances of TradeView CAN be passed where TradeLike is expected,
        and the no-conversion design holds.
        """
        from smc.eval.calibration_arms import _FlagSnapshot
        view = TradeView(
            trade_id="t1",
            timestamp_utc="2026-04-25T12:00:00+00:00",
            arm_label="control",
            flag_snapshot=_FlagSnapshot(
                range_trend_filter_enabled=False,
                range_ai_regime_gate_enabled=False,
                spread_gate_enabled=False,
                persistent_dd_breaker_enabled=False,
                consec_loss_window_size=3,
            ),
            pnl_usd=15.5,
            loss_count_in_window_at_open=0,
        )
        # Static type assertion via assignment — if Protocol shape were
        # wrong, mypy/pyright would flag this. At runtime this is just
        # an attribute access.
        as_tradelike: TradeLike = view  # noqa: F841
        assert view.pnl_usd == 15.5

    def test_classified_trades_flow_into_evaluate_promotion_no_unwrap(self):
        """Integration-style: classify → evaluate_promotion in one pipeline.

        This is the END-TO-END proof of the Protocol composability win:
        no list-comprehension reshape, no DTO conversion, just hand the
        classified arms straight to the gate evaluator.
        """
        # 30 control + 30 treatment trades to hit min_n=30 in promotion gate.
        entries = []
        for i in range(30):
            entries.append(_journal_entry(
                time=f"2026-04-25T12:{i:02d}:00+00:00",
                magic=_CONTROL_MAGIC,
                flags=_all_off_flags(),
                pnl_usd=1.0,  # control: small wins
                trade_id=f"c{i}",
            ))
        for i in range(30):
            entries.append(_journal_entry(
                time=f"2026-04-25T13:{i:02d}:00+00:00",
                magic=_TREATMENT_MAGIC,
                flags=_all_on_flags(),
                pnl_usd=2.0,  # treatment: bigger wins (PF improvement)
                trade_id=f"t{i}",
            ))
        result = classify_trades(
            entries,
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        baseline = result.arms["control"]
        treatment = result.arms["treatment_full"]
        assert len(baseline) == 30
        assert len(treatment) == 30
        # The composability proof: no [t.pnl_usd for t in ...] reshape,
        # just direct pass-through. If TradeView didn't satisfy TradeLike
        # this call would TypeError on the .pnl_usd attribute access.
        verdict = evaluate_promotion(
            baseline, treatment,
            min_n=30, n_iter=200, rng_seed=0,
        )
        assert verdict.n_baseline == 30
        assert verdict.n_treatment == 30


# ---------------------------------------------------------------------------
# Empty-input + edge cases
# ---------------------------------------------------------------------------


class TestEmptyAndEdgeCases:

    def test_empty_journal_yields_empty_arms_and_diagnostics(self):
        result = classify_trades(
            [],
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert all(len(t) == 0 for t in result.arms.values())
        assert all(len(t) == 0 for t in result.diagnostics.values())

    def test_no_arm_definitions_yields_all_unmatched(self):
        entry = _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags())
        result = classify_trades(
            [entry],
            arm_definitions=[],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        # No arms defined → trade lands in diagnostic. With magic==control
        # and all-off flags, no _any_flag_on so → unmatched.
        assert len(result.diagnostics[DIAGNOSTIC_UNMATCHED]) == 1

    def test_first_match_wins_when_two_arms_overlap(self):
        """Order matters: the first arm whose flags+magic match wins."""
        treat_arm = _treatment_arm()  # arm_label = "treatment_full"
        treat_arm_dup = ArmDefinition(
            arm_label="treatment_dup",
            required_flags=_all_on_flags(),
            required_magic=_TREATMENT_MAGIC,
        )
        entry = _journal_entry(magic=_TREATMENT_MAGIC, flags=_all_on_flags())
        # Order: treatment_full first → it wins.
        result = classify_trades(
            [entry],
            arm_definitions=[treat_arm, treat_arm_dup],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["treatment_full"]) == 1
        assert len(result.arms["treatment_dup"]) == 0

    def test_required_magic_none_skips_magic_check(self):
        """Arm with required_magic=None matches any magic on flag agreement."""
        any_magic_arm = ArmDefinition(
            arm_label="any",
            required_flags=_all_off_flags(),
            required_magic=None,
        )
        entry = _journal_entry(magic=99999999, flags=_all_off_flags())
        result = classify_trades(
            [entry],
            arm_definitions=[any_magic_arm],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["any"]) == 1

    def test_invalid_magic_string_falls_through_to_unmatched(self):
        """Non-int magic field → magic_int=None → unmatched."""
        entry = _journal_entry(magic="not-a-number", flags=_all_off_flags())  # type: ignore[arg-type]
        result = classify_trades(
            [entry],
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        # Magic can't be parsed → no arm matches → unmatched bucket.
        assert len(result.arms["control"]) == 0
        assert len(result.diagnostics[DIAGNOSTIC_UNMATCHED]) == 1


# ---------------------------------------------------------------------------
# Multi-trade mixed-arm classification
# ---------------------------------------------------------------------------


class TestMixedJournalClassification:
    """Realistic multi-arm journal with a mix of clean and diagnostic trades."""

    def test_mixed_journal_distributes_across_arms_and_diagnostics(self):
        entries = [
            # 3 clean control
            _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags(), trade_id="c1"),
            _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags(), trade_id="c2"),
            _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags(), trade_id="c3"),
            # 2 clean treatment
            _journal_entry(magic=_TREATMENT_MAGIC, flags=_all_on_flags(), trade_id="t1"),
            _journal_entry(magic=_TREATMENT_MAGIC, flags=_all_on_flags(), trade_id="t2"),
            # 1 control with deployment bug (spread_gate ON)
            _journal_entry(
                magic=_CONTROL_MAGIC,
                flags={**_all_off_flags(), "spread_gate_enabled": True},
                trade_id="bug1",
            ),
            # 1 treatment with deployment bug (one flag OFF)
            _journal_entry(
                magic=_TREATMENT_MAGIC,
                flags={**_all_on_flags(), "spread_gate_enabled": False},
                trade_id="bug2",
            ),
            # 1 manual operator entry on a third magic
            _journal_entry(magic=99999999, flags=_all_off_flags(), trade_id="manual"),
        ]
        result = classify_trades(
            entries,
            arm_definitions=[_control_arm(), _treatment_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        assert len(result.arms["control"]) == 3
        assert len(result.arms["treatment_full"]) == 2
        assert len(result.diagnostics[DIAGNOSTIC_CONTROL_UNEXPECTED]) == 1
        assert len(result.diagnostics[DIAGNOSTIC_TREATMENT_MISSING]) == 1
        assert len(result.diagnostics[DIAGNOSTIC_UNMATCHED]) == 1

    def test_classification_total_count_equals_input_count(self):
        """Every input trade must land somewhere — no silent drops."""
        entries = [
            _journal_entry(magic=_CONTROL_MAGIC, flags=_all_off_flags(), trade_id=f"e{i}")
            for i in range(7)
        ]
        result = classify_trades(
            entries,
            arm_definitions=[_control_arm()],
            control_magic=_CONTROL_MAGIC,
            treatment_magic=_TREATMENT_MAGIC,
        )
        total = sum(len(t) for t in result.arms.values()) + sum(
            len(t) for t in result.diagnostics.values()
        )
        assert total == 7, "every input trade must land in exactly one bucket"
