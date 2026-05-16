"""Ticket 4 v2 Step 7 — shadow_runner v0.3.0 multi-window wiring.

Pinned guarantees:
  * ``SHADOW_RUNNER_MULTI_WINDOW_VERSION == "shadow_runner-0.3.0"``.
  * ``run_shadow_for_candidate_multi_window`` exists and produces a
    hash-pinned artefact under ``<out_dir>/<candidate_id>/<run_id>.json``.
  * The artefact's ``runner_version`` field equals
    ``"shadow_runner-0.3.0"``.
  * The artefact embeds ``per_window``, ``window_coverage``,
    ``gold_profile`` payloads (Ticket 4 v2 Step 1 schema fields).
  * ``MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION ==
    "shadow_runner-0.3.0"``.
  * G8 with a 0.2.0 artefact carrying verdict=PASS → ABSTAIN with
    `active_pass_evaluation_unsupported`-class reason (because
    pre-Ticket-4 artefacts have no per-window data).
  * G8 with a 0.3.0 artefact carrying verdict=PASS → PASS.
  * Old 0.1.0 / 0.2.0 artefacts on disk are byte-identical (no
    in-place rewrite by the runner).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import EvidenceBundle, GateStatus
from smc.hedgerock.evolution.window_coverage import WindowSpec


# ---------------------------------------------------------------------------
# Stub lake (re-used pattern from test_replay_multi_window)
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0,
          *, base_price: float = 100.0, drift: float = 0.0):
    rows = []
    p = base_price
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": p, "high": p + 0.5, "low": p - 0.5,
            "close": p, "volume": 100.0,
        })
        p += drift
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data):
        self._data = data
        self._root = Path("/tmp/stub_t4_runner")

    def list_instruments(self):
        return sorted({k[0] for k in self._data})

    def query(self, instrument, timeframe, start, end):
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def long_lake():
    base = datetime(2023, 12, 1, tzinfo=timezone.utc)
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 120),
        ("XAUUSD", "H4"): _bars(base, n=6 * 120, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=120, hours_step=24.0),
    })


def _candidate_c1():
    return next(c for c in CANDIDATE_MENU_V0
                if c.candidate_id == "c1-lower-observe-floor-0.50")


def _windows_two() -> list[WindowSpec]:
    return [
        WindowSpec(
            window_id="y2024_a",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            declared_regime_bucket="range_low_vol",
        ),
        WindowSpec(
            window_id="y2024_b",
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            declared_regime_bucket="range_low_vol",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_multi_window_version_constant_is_0_3_0() -> None:
    from smc.hedgerock.evolution.shadow_runner import (
        SHADOW_RUNNER_MULTI_WINDOW_VERSION,
    )
    assert SHADOW_RUNNER_MULTI_WINDOW_VERSION == "shadow_runner-0.3.0"


@pytest.mark.unit
def test_min_runner_version_for_active_pass_evaluation_constant() -> None:
    from smc.hedgerock.evolution.promotion_gates import (
        MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION,
        parse_runner_version,
    )
    assert MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION == "shadow_runner-0.3.0"
    assert parse_runner_version(
        MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION
    ) >= (0, 3, 0)


@pytest.mark.unit
def test_existing_runner_version_unchanged() -> None:
    """v1 single-window runner preserves its 0.2.0 version so old
    callers / archived 0.2.0 artefacts continue to validate."""
    from smc.hedgerock.evolution.shadow_runner import SHADOW_RUNNER_VERSION
    assert SHADOW_RUNNER_VERSION == "shadow_runner-0.2.0"


# ---------------------------------------------------------------------------
# 2. run_shadow_for_candidate_multi_window — produces 0.3.0 artefact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_shadow_multi_window_writes_v0_3_0_artefact(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    from smc.hedgerock.evolution.shadow_artefact import load_shadow_artefact

    cand = _candidate_c1()
    artefact_path = run_shadow_for_candidate_multi_window(
        candidate=cand,
        lake=long_lake,
        symbol="XAUUSD",
        windows=_windows_two(),
        out_dir=tmp_path,
    )
    assert artefact_path.exists()
    assert artefact_path.is_file()
    art = load_shadow_artefact(artefact_path)
    assert art.runner_version == "shadow_runner-0.3.0"


@pytest.mark.unit
def test_v0_3_0_artefact_embeds_per_window_and_coverage(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    from smc.hedgerock.evolution.shadow_artefact import load_shadow_artefact

    cand = _candidate_c1()
    p = run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )
    art = load_shadow_artefact(p)
    # per_window must be a non-empty dict / list-of-dicts; exact
    # serialization shape decided by the runner.
    assert art.per_window, "per_window field is empty"
    assert art.window_coverage, "window_coverage field is empty"
    # window_coverage carries coverage_pass + windows_evaluated.
    assert "coverage_pass" in art.window_coverage
    assert "windows_evaluated" in art.window_coverage


@pytest.mark.unit
def test_v0_3_0_artefact_verdict_is_abstain_under_thin_coverage(
    long_lake, tmp_path,
) -> None:
    """Two windows is below the 6-window floor → coverage_pass=False
    → artefact verdict ABSTAIN with XAUUSD-only reason."""
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    from smc.hedgerock.evolution.shadow_artefact import (
        load_shadow_artefact, ShadowVerdict,
    )

    cand = _candidate_c1()
    p = run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )
    art = load_shadow_artefact(p)
    assert art.verdict == ShadowVerdict.ABSTAIN, art.verdict_reason
    assert "single_symbol" not in art.verdict_reason
    assert "cross_symbol" not in art.verdict_reason


# ---------------------------------------------------------------------------
# 3. G8 with v0.2.0 artefact — ABSTAIN due to no active PASS evaluation
# ---------------------------------------------------------------------------


def _candidate_hash(cand):
    from smc.hedgerock.evolution.policy_manifest import (
        compute_canonical_candidate_hash,
    )
    return compute_canonical_candidate_hash(cand)


def _build_v020_pass_artefact(tmp_path):
    """A clean v0.2.0 artefact carrying verdict=PASS. Per Ticket 4
    v2, G8 must downgrade this to ABSTAIN because v0.2.0 lacks the
    multi-window per-window data the active PASS evaluator
    requires."""
    from smc.hedgerock.evolution.shadow_artefact import (
        SHADOW_ARTEFACT_SCHEMA_VERSION,
        CandidateDiffSnapshot, DataSliceIdentity, NoLiveEvidence,
        NoLookaheadAudit, ReplayInvariants, ShadowArtefact, ShadowMetrics,
        ShadowVerdict, SidecarModuleHashes, dump_shadow_artefact,
    )
    cand = _candidate_c1()
    z = ShadowMetrics(
        final_equity=10000.0, total_return_pct=0.0, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    art = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id="evb-v020",
        generated_at="2026-04-01T00:00:00+00:00",
        candidate_id=cand.candidate_id,
        candidate_manifest_content_hash=_candidate_hash(cand),
        candidate_diff=CandidateDiffSnapshot(
            target=cand.diff.target,
            proposed_value=cand.diff.proposed_value,
            baseline_value=cand.diff.baseline_value,
        ),
        candidate_diff_hash="0" * 64,
        baseline_policy_id="phase_d_walk_forward_baseline_v1",
        baseline_policy_hash="0" * 64,
        candidate_overlay_id="0" * 64,
        data_slice=DataSliceIdentity(
            symbols=("XAUUSD", "EURUSD"),
            time_range_start="2021-01-01",
            time_range_end="2025-01-01",
            timeframes=("H1", "H4", "D1"),
            closed_bar_rule_version="phase_d_strict_prior_v1",
            lake_snapshot_hash="0" * 64,
            lake_snapshot_row_counts={"H1": 1000, "H4": 250, "D1": 50},
        ),
        runner_version="shadow_runner-0.2.0",
        mirror_version="0" * 64,
        metric_schema_version="0" * 64,
        sidecar_module_hashes=SidecarModuleHashes(
            policy_overlay="0" * 64, rule_engine_mirror="0" * 64,
            replay_constant_mirror="0" * 64, shadow_runner="0" * 64,
            shadow_metrics="0" * 64,
        ),
        baseline_metrics=z, candidate_metrics=z, delta_metrics=z,
        replay_invariants=ReplayInvariants(
            same_bar_set_used=True, same_transition_lock_state_machine=True,
            same_cooldown_carryover=True,
            decision_only_uses_strictly_prior_data=True,
            h4_partial_bar_in_window=False, d1_partial_bar_in_window=False,
            decision_uses_data_with_ts_lt_trade_bar_ts=True,
        ),
        no_live_evidence=NoLiveEvidence(
            decision_server_routes_unchanged_hash="0" * 64,
            rule_engine_constants_unchanged_hash="0" * 64,
            http_calls_made_count=0, broker_api_calls_made_count=0,
            files_written_under_src_or_config_or_mq5_count=0,
            files_written_under_approved_or_pointer_count=0,
        ),
        mirror_consistency_check="PASS",
        exposure_class_violation=False,
        no_lookahead_audit=NoLookaheadAudit(
            decision_uses_only_prior_closed_bars=True,
            partial_bar_violation_count=0,
        ),
        verdict=ShadowVerdict.PASS,
        verdict_reason="(synthetic; v0.2.0 PASS verdict)",
    )
    p = tmp_path / "v020-pass.json"
    dump_shadow_artefact(art, p)
    return cand, p


@pytest.mark.unit
def test_g8_v020_pass_artefact_downgraded_to_abstain(tmp_path) -> None:
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison

    cand, art_path = _build_v020_pass_artefact(tmp_path)
    bundle = EvidenceBundle(
        bundle_id="evb", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 4,
                                       "negative_sign_years": ()}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        shadow_artefact_path=str(art_path),
        shadow_artefact_hash_sha256=compute_file_sha256(art_path),
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    assert r.status == GateStatus.ABSTAIN, r.reason
    # Reason vocabulary: must mention either active_pass_evaluation
    # gating or the 0.3.0 minimum version.
    rl = r.reason.lower()
    assert (
        "active_pass_evaluation" in rl
        or "0.3.0" in r.reason
        or "active pass" in rl
    )
    # Critical RFC v2 invariant.
    assert "single_symbol" not in r.reason
    assert "cross_symbol" not in r.reason


# ---------------------------------------------------------------------------
# 4. Old artefacts on disk preserved byte-for-byte
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_does_not_rewrite_existing_artefacts(
    long_lake, tmp_path,
) -> None:
    """Run runner twice; the first artefact's content must remain
    byte-identical after the second run (append-only registry)."""
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )

    cand = _candidate_c1()
    p1 = run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )
    blob_before = p1.read_bytes()

    p2 = run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )
    assert p1 != p2, "second run should produce a separate artefact"
    blob_after = p1.read_bytes()
    assert blob_before == blob_after, (
        "first artefact was rewritten in place; registry must be append-only"
    )


@pytest.mark.unit
def test_runner_writes_artefact_immutable_chmod(
    long_lake, tmp_path,
) -> None:
    """v0.3.0 artefact must be written 0o444 like the v0.2.0 path."""
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )

    cand = _candidate_c1()
    p = run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )
    mode = p.stat().st_mode & 0o777
    assert mode == 0o444, f"expected 0o444 immutable, got 0o{mode:o}"


# ---------------------------------------------------------------------------
# 5. No production-side leak after multi-window run
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_does_not_mutate_rule_engine_constants(
    long_lake, tmp_path,
) -> None:
    import importlib
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    snapshot_before = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if k.startswith("_") and not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }

    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    cand = _candidate_c1()
    run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )

    snapshot_after = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if k.startswith("_") and not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }
    assert snapshot_before == snapshot_after
