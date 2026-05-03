"""Ticket 4 v2 T4-F1 — registry-audit state plumbed end-to-end.

Pinned guarantees:

  * :class:`RegistryAuditState` carries the required fields:
    ``audit_log_path``, ``audit_log_present``,
    ``stale_v030_deleted_during_this_session``, ``lost_sha_count``,
    ``lost_sha256``, ``registry_append_only_violation``.
  * The loader parses the real audit log at
    ``policy_registry/shadow_artefacts/_audit.md`` and reports
    violation=True with lost_sha_count=4 (the four SHAs of the
    2026-05-02 incident).
  * :class:`EvidenceBundle` accepts a ``registry_audit`` field with
    None default for back-compat.
  * ``g8_shadow_comparison`` ABSTAINs when
    ``registry_append_only_violation=True`` regardless of the
    candidate artefact's verdict (even PASS).
  * ``g8_shadow_comparison`` is NOT spuriously blocked when
    ``registry_audit=None`` or violation=False — other gates run
    normally.
  * ``evaluate_pass_xauusd_multi_window`` ABSTAINs when called
    with ``registry_append_only_violation=True``.
  * The report renderer emits a per-candidate violation block
    when audit state is supplied with violation=True; no false
    positive when violation=False.
  * Smoke fixtures NEVER write to the real registry — every
    constructive runner test uses ``tmp_path``.
  * The append-only delete-prevention tests added in Step 9
    closeout remain effective (re-asserted here as a regression
    sentinel).

These tests must remain XAUUSD-only — no ``single_symbol`` /
``cross_symbol`` text in any rendered string or returned reason.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import (
    EvidenceBundle, GateStatus,
)
from smc.hedgerock.evolution.registry_audit import (
    DEFAULT_REGISTRY_AUDIT_LOG,
    REGISTRY_VIOLATION_GATE_REASON_PREFIX,
    RegistryAuditState,
    load_registry_audit_state,
)


_LOST_SHAS_2026_05_02 = (
    "f923fc24c3f1ecd7c2bae21a30b745791baeff6b55c2a4b530df4203c60bd186",
    "c50a2f9b28cdd841a9d62cc1816bba24ad8d65f5dfc2b050fdf5dff5df544d65",
    "913269e479ae57c96d579ba730b0601d8ac1f13fc5dfe1e28ead635716e5b933",
    "6916b7902fa93c5ed8bc755d6b7c973a92e14aa820d57da2bf10a06ceff5de86",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_c1():
    return next(c for c in CANDIDATE_MENU_V0
                if c.candidate_id == "c1-lower-observe-floor-0.50")


def _bars(start: datetime, n: int, hours_step: float = 1.0,
          *, base_price: float = 100.0):
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": base_price, "high": base_price + 0.5,
            "low": base_price - 0.5, "close": base_price, "volume": 100.0,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data):
        self._data = data
        self._root = Path("/tmp/stub_t4_f1")

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


def _violating_audit_state(
    *, audit_log_path: str = "/tmp/synthetic_audit.md",
) -> RegistryAuditState:
    return RegistryAuditState(
        audit_log_path=audit_log_path,
        audit_log_present=True,
        stale_v030_deleted_during_this_session=True,
        lost_sha_count=len(_LOST_SHAS_2026_05_02),
        lost_sha256=_LOST_SHAS_2026_05_02,
        registry_append_only_violation=True,
    )


def _clean_audit_state() -> RegistryAuditState:
    return RegistryAuditState(
        audit_log_path="/tmp/clean_audit.md",
        audit_log_present=True,
        stale_v030_deleted_during_this_session=False,
        lost_sha_count=0,
        lost_sha256=(),
        registry_append_only_violation=False,
    )


# ---------------------------------------------------------------------------
# 1. RegistryAuditState dataclass shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_audit_state_has_required_fields() -> None:
    s = _violating_audit_state()
    assert s.audit_log_path
    assert s.audit_log_present is True
    assert s.stale_v030_deleted_during_this_session is True
    assert s.lost_sha_count == 4
    assert s.lost_sha256 == _LOST_SHAS_2026_05_02
    assert s.registry_append_only_violation is True


@pytest.mark.unit
def test_registry_audit_state_is_frozen() -> None:
    from dataclasses import FrozenInstanceError
    s = _clean_audit_state()
    with pytest.raises(FrozenInstanceError):
        s.lost_sha_count = 99   # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Loader behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loader_returns_clean_state_when_audit_log_missing(
    tmp_path: Path,
) -> None:
    p = tmp_path / "absent_audit.md"
    s = load_registry_audit_state(p)
    assert s.audit_log_present is False
    assert s.registry_append_only_violation is False
    assert s.lost_sha_count == 0
    assert s.lost_sha256 == ()


@pytest.mark.unit
def test_loader_detects_violation_in_synthetic_audit_log(
    tmp_path: Path,
) -> None:
    p = tmp_path / "audit.md"
    p.write_text(
        "# Registry Audit Log\n\n"
        "## 2099-01-01 — Stale v0.3.0 artefacts deleted (test fixture)\n\n"
        "- "
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "- "
        "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210\n",
        encoding="utf-8",
    )
    s = load_registry_audit_state(p)
    assert s.audit_log_present is True
    assert s.stale_v030_deleted_during_this_session is True
    assert s.lost_sha_count == 2
    assert s.registry_append_only_violation is True


@pytest.mark.unit
def test_loader_returns_clean_when_log_has_no_deletion_section(
    tmp_path: Path,
) -> None:
    p = tmp_path / "audit.md"
    p.write_text(
        "# Registry Audit Log\n\n"
        "This file is append-only. No incidents on file.\n",
        encoding="utf-8",
    )
    s = load_registry_audit_state(p)
    assert s.audit_log_present is True
    assert s.registry_append_only_violation is False
    assert s.lost_sha_count == 0


@pytest.mark.unit
def test_loader_against_real_audit_log_reports_violation() -> None:
    """The real audit log at policy_registry/shadow_artefacts/_audit.md
    records the 2026-05-02 incident with all four lost SHAs."""
    if not DEFAULT_REGISTRY_AUDIT_LOG.exists():
        pytest.skip("real audit log absent (fresh checkout?)")
    s = load_registry_audit_state(DEFAULT_REGISTRY_AUDIT_LOG)
    assert s.audit_log_present is True
    assert s.registry_append_only_violation is True
    assert s.lost_sha_count == 4
    assert set(s.lost_sha256) == set(_LOST_SHAS_2026_05_02)


# ---------------------------------------------------------------------------
# 3. EvidenceBundle carries registry_audit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evidence_bundle_accepts_registry_audit_field() -> None:
    state = _violating_audit_state()
    b = EvidenceBundle(
        bundle_id="evb-t4f1", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 4,
                                       "negative_sign_years": ()}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        registry_audit=state,
    )
    assert b.registry_audit is state
    assert b.registry_audit.registry_append_only_violation is True


@pytest.mark.unit
def test_evidence_bundle_registry_audit_defaults_to_none() -> None:
    """Back-compat: existing call sites that don't supply
    registry_audit must continue to work."""
    b = EvidenceBundle(
        bundle_id="evb-default", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
    )
    assert b.registry_audit is None


# ---------------------------------------------------------------------------
# 4. G8 ABSTAINs on violation regardless of artefact verdict
# ---------------------------------------------------------------------------


def _build_pass_artefact_in_tmp(tmp_path: Path, *, runner_version: str):
    """Plant a clean artefact whose internal verdict is PASS — the
    G8 layer will still ABSTAIN when the bundle's audit state
    reports a violation."""
    from smc.hedgerock.evolution.shadow_artefact import (
        SHADOW_ARTEFACT_SCHEMA_VERSION,
        CandidateDiffSnapshot, DataSliceIdentity, NoLiveEvidence,
        NoLookaheadAudit, ReplayInvariants, ShadowArtefact, ShadowMetrics,
        ShadowVerdict, SidecarModuleHashes, dump_shadow_artefact,
    )
    from smc.hedgerock.evolution.policy_manifest import (
        compute_canonical_candidate_hash,
    )

    cand = _candidate_c1()
    z = ShadowMetrics(
        final_equity=10000.0, total_return_pct=0.0, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    art = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id=f"evb-t4f1-{runner_version}",
        generated_at="2026-04-01T00:00:00+00:00",
        candidate_id=cand.candidate_id,
        candidate_manifest_content_hash=compute_canonical_candidate_hash(cand),
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
            symbols=("XAUUSD",), time_range_start="2024-01-01",
            time_range_end="2024-01-31", timeframes=("H1", "H4", "D1"),
            closed_bar_rule_version="phase_d_strict_prior_v1",
            lake_snapshot_hash="0" * 64,
            lake_snapshot_row_counts={"H1": 100, "H4": 25, "D1": 5},
        ),
        runner_version=runner_version,
        mirror_version="0" * 64, metric_schema_version="0" * 64,
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
        verdict_reason="(synthetic; t4-f1)",
    )
    p = tmp_path / f"art-{runner_version}.json"
    dump_shadow_artefact(art, p)
    return cand, p


@pytest.mark.unit
def test_g8_abstains_when_registry_violation_true(tmp_path: Path) -> None:
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison

    cand, art_path = _build_pass_artefact_in_tmp(
        tmp_path, runner_version="shadow_runner-0.3.0",
    )
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
        registry_audit=_violating_audit_state(),
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    assert r.status == GateStatus.ABSTAIN, r.reason
    assert REGISTRY_VIOLATION_GATE_REASON_PREFIX in r.reason
    # Audit log path surfaced in details for the report.
    assert r.details.get("registry_audit_log_path")
    assert r.details.get("lost_sha_count") == 4
    assert "single_symbol" not in r.reason
    assert "cross_symbol" not in r.reason


@pytest.mark.unit
def test_g8_does_not_abstain_when_registry_audit_none(tmp_path: Path) -> None:
    """Back-compat: no registry_audit on the bundle → G8 evaluates
    using the existing rules (PASS-verdict v0.3.0 artefact still
    PASSes other gates)."""
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison

    cand, art_path = _build_pass_artefact_in_tmp(
        tmp_path, runner_version="shadow_runner-0.3.0",
    )
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
        registry_audit=None,   # no audit → no violation gate
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    # Without violation, G8 should evaluate normally — the v0.3.0
    # artefact has internal verdict=PASS.
    assert REGISTRY_VIOLATION_GATE_REASON_PREFIX not in r.reason
    assert r.status == GateStatus.PASS, (r.status, r.reason)


@pytest.mark.unit
def test_g8_does_not_abstain_when_violation_false(tmp_path: Path) -> None:
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison

    cand, art_path = _build_pass_artefact_in_tmp(
        tmp_path, runner_version="shadow_runner-0.3.0",
    )
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
        registry_audit=_clean_audit_state(),
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    assert REGISTRY_VIOLATION_GATE_REASON_PREFIX not in r.reason
    assert r.status == GateStatus.PASS, (r.status, r.reason)


# ---------------------------------------------------------------------------
# 5. Active multi-window PASS evaluator picks up the new param
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_active_pass_evaluator_abstains_on_registry_violation() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_worst_window_summary,
    )
    from smc.hedgerock.evolution.window_coverage import CoverageReport

    coverage = CoverageReport(
        coverage_pass=True,
        windows_evaluated=("w0",),
        regime_buckets_covered=("range_low_vol",),
        halt_event_windows=1,
        no_trade_windows=(),
        shortfall_reasons=(),
        declared_vs_observed_mismatches=(),
    )
    summary = compute_worst_window_summary([])
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=coverage,
        worst_summary=summary,
        per_window_metrics=[],
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
        registry_append_only_violation=True,
        registry_audit_log_path="/tmp/synthetic_audit.md",
    )
    assert out.eligible_for_pass is False
    assert REGISTRY_VIOLATION_GATE_REASON_PREFIX in out.abstain_reason
    assert out.details.get("registry_audit_log_path") == \
           "/tmp/synthetic_audit.md"


# ---------------------------------------------------------------------------
# 6. Report renderer per-candidate violation block
# ---------------------------------------------------------------------------


def _produce_artefact_in_tmp(long_lake, tmp_path: Path) -> Path:
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    from smc.hedgerock.evolution.window_coverage import WindowSpec
    cand = _candidate_c1()
    return run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=[
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
        ],
        out_dir=tmp_path,
    )


@pytest.mark.unit
def test_report_renders_per_candidate_violation_block(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact_in_tmp(long_lake, tmp_path)
    out = render_multi_window_report(
        [p], registry_audit=_violating_audit_state(),
    )
    # Block fires inside the candidate region (above the footer).
    body, _, footer = out.partition("## Hard-boundary status")
    assert "registry_append_only_violation (G8 hard block)" in body
    # And lost SHAs are visible in the candidate block.
    for sha in _LOST_SHAS_2026_05_02:
        assert sha in body, f"lost sha {sha} missing from per-candidate block"


@pytest.mark.unit
def test_report_no_per_candidate_block_when_audit_clean(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact_in_tmp(long_lake, tmp_path)
    out = render_multi_window_report(
        [p], registry_audit=_clean_audit_state(),
    )
    # No false alarm.
    assert "registry_append_only_violation (G8 hard block)" not in out


@pytest.mark.unit
def test_report_no_per_candidate_block_when_audit_none(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact_in_tmp(long_lake, tmp_path)
    out = render_multi_window_report([p])  # no registry_audit
    assert "registry_append_only_violation (G8 hard block)" not in out


# ---------------------------------------------------------------------------
# 7. Smoke / runner tests stay isolated from the real registry
# ---------------------------------------------------------------------------


_REAL_REGISTRY = Path(
    "HedgeRock/policy_registry/shadow_artefacts"
)


@pytest.mark.unit
def test_runner_test_helper_does_not_write_to_real_registry(
    long_lake, tmp_path,
) -> None:
    """Sanity: the test fixture's tmp_path is not the real registry,
    and running the multi-window runner against tmp_path leaves the
    real registry's file count unchanged."""
    assert tmp_path != _REAL_REGISTRY
    assert not str(tmp_path).startswith(str(_REAL_REGISTRY))

    pre_count = sum(
        1 for _ in _REAL_REGISTRY.rglob("*.json")
    ) if _REAL_REGISTRY.exists() else 0

    _produce_artefact_in_tmp(long_lake, tmp_path)

    post_count = sum(
        1 for _ in _REAL_REGISTRY.rglob("*.json")
    ) if _REAL_REGISTRY.exists() else 0

    assert pre_count == post_count, (
        f"runner test wrote into real registry: "
        f"pre={pre_count} post={post_count}"
    )


# ---------------------------------------------------------------------------
# 8. Append-only delete-prevention sentinel
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step9_append_only_delete_tests_are_still_present() -> None:
    """Regression sentinel: Step 9 closeout introduced
    test_artefact_registry_append_only.py to scan production code
    for delete primitives. Make sure that file still exists and
    tests are runnable from this T4-F1 module too."""
    p = Path(__file__).parent / "test_artefact_registry_append_only.py"
    assert p.exists(), "Step 9 closeout test file missing"
    body = p.read_text(encoding="utf-8")
    assert "_FORBIDDEN_NAMES" in body
    assert "test_no_delete_primitive_in_production_module" in body
