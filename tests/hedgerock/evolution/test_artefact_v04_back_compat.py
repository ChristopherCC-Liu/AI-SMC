"""Ticket 4 Step 1 — schema v1 back-compat.

Adds three OPTIONAL fields to ShadowArtefact:
  - per_window: dict[window_id, dict[role, dict[metric, value]]]
  - window_coverage: dict[str, Any]
  - gold_profile: dict[str, Any]

Pinned guarantees:
  - Old 0.1.0 / 0.2.0 artefacts load successfully under v0.3.0 loader
    with empty {} defaults for the new fields.
  - New 0.3.0 artefacts can populate the fields and round-trip
    byte-identical.
  - Tampering any new field still triggers ShadowArtefactIntegrityError.
"""

from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.shadow_artefact import (

    SHADOW_ARTEFACT_SCHEMA_VERSION,
    CandidateDiffSnapshot,
    DataSliceIdentity,
    NoLiveEvidence,
    NoLookaheadAudit,
    ReplayInvariants,
    ShadowArtefact,
    ShadowArtefactIntegrityError,
    ShadowMetrics,
    ShadowVerdict,
    SidecarModuleHashes,
    artefact_from_dict,
    artefact_to_dict,
    dump_shadow_artefact,
    load_shadow_artefact,
)


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

def _z_metrics() -> ShadowMetrics:
    return ShadowMetrics(
        final_equity=10_000.0, total_return_pct=0.0, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )


def _bare_artefact_v03(
    *,
    per_window: dict | None = None,
    window_coverage: dict | None = None,
    gold_profile: dict | None = None,
    runner_version: str = "shadow_runner-0.3.0",
) -> ShadowArtefact:
    return ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id="t4-test",
        generated_at="2026-05-02T00:00:00+00:00",
        candidate_id="c1-lower-observe-floor-0.50",
        candidate_manifest_content_hash="a" * 64,
        candidate_diff=CandidateDiffSnapshot(
            target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            proposed_value=0.50, baseline_value=0.55,
        ),
        candidate_diff_hash="b" * 64,
        baseline_policy_id="p", baseline_policy_hash="c" * 64,
        candidate_overlay_id="d" * 64,
        data_slice=DataSliceIdentity(
            symbols=("XAUUSD",), time_range_start="2021-01-01",
            time_range_end="2025-01-01",
            timeframes=("H1", "H4", "D1"),
            closed_bar_rule_version="phase_d_strict_prior_v1",
            lake_snapshot_hash="e" * 64,
            lake_snapshot_row_counts={"H1": 28798, "H4": 7962, "D1": 1554},
        ),
        runner_version=runner_version,
        mirror_version="f" * 64, metric_schema_version="g" * 64,
        sidecar_module_hashes=SidecarModuleHashes(
            policy_overlay="h" * 64, rule_engine_mirror="i" * 64,
            replay_constant_mirror="j" * 64, shadow_runner="k" * 64,
            shadow_metrics="l" * 64,
        ),
        baseline_metrics=_z_metrics(), candidate_metrics=_z_metrics(),
        delta_metrics=_z_metrics(),
        replay_invariants=ReplayInvariants(
            same_bar_set_used=True, same_transition_lock_state_machine=True,
            same_cooldown_carryover=True,
            decision_only_uses_strictly_prior_data=True,
            h4_partial_bar_in_window=False, d1_partial_bar_in_window=False,
            decision_uses_data_with_ts_lt_trade_bar_ts=True,
        ),
        no_live_evidence=NoLiveEvidence(
            decision_server_routes_unchanged_hash="m" * 64,
            rule_engine_constants_unchanged_hash="n" * 64,
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
        verdict=ShadowVerdict.ABSTAIN,
        verdict_reason="t4 fixture",
        per_window=per_window or {},
        window_coverage=window_coverage or {},
        gold_profile=gold_profile or {},
    )


# ---------------------------------------------------------------------------
# 1. New fields default to {} — old artefact shape still loads
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_new_optional_fields_default_to_empty_dict() -> None:
    a = _bare_artefact_v03()
    assert a.per_window == {}
    assert a.window_coverage == {}
    assert a.gold_profile == {}


@pytest.mark.unit
def test_old_artefact_shape_loads_with_empty_optional_fields(tmp_path: Path) -> None:
    """Construct a JSON envelope WITHOUT the three new fields; loader
    must still strict-load it with empty-dict defaults."""
    a = _bare_artefact_v03()
    payload = artefact_to_dict(a)
    # Simulate an old artefact: drop the three new fields entirely.
    payload.pop("per_window", None)
    payload.pop("window_coverage", None)
    payload.pop("gold_profile", None)

    import hashlib
    canonical = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False,
    )
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    wrapper = {"content_sha256": h, "artefact": payload}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    loaded = load_shadow_artefact(p)
    assert loaded.per_window == {}
    assert loaded.window_coverage == {}
    assert loaded.gold_profile == {}


# ---------------------------------------------------------------------------
# 2. New fields populate + round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_new_fields_round_trip_byte_identical(tmp_path: Path) -> None:
    pw = {
        "y2024_q1": {
            "baseline": {"final_equity": 9950.0, "n_trades": 1},
            "candidate": {"final_equity": 9950.0, "n_trades": 1},
        },
        "y2024_q2": {
            "baseline": {"final_equity": 10100.0, "n_trades": 2},
            "candidate": {"final_equity": 10100.0, "n_trades": 2},
        },
    }
    wc = {
        "windows_evaluated": ["y2024_q1", "y2024_q2"],
        "regime_buckets_covered": ["range_low_vol", "trend_up"],
        "halt_event_windows": 0,
        "coverage_pass": False,
        "shortfall_reasons": [
            "insufficient_xauusd_window_coverage: have 2, need >= 6",
        ],
    }
    gp = {"version": "v0", "operator_curated": True}

    a = _bare_artefact_v03(per_window=pw, window_coverage=wc, gold_profile=gp)
    p1 = tmp_path / "a.json"
    dump_shadow_artefact(a, p1)
    loaded = load_shadow_artefact(p1)
    p2 = tmp_path / "b.json"
    dump_shadow_artefact(loaded, p2)
    assert p1.read_bytes() == p2.read_bytes()
    assert loaded.per_window == pw
    assert loaded.window_coverage == wc
    assert loaded.gold_profile == gp


@pytest.mark.unit
def test_dump_writes_chmod_0444_for_v03_artefact(tmp_path: Path) -> None:
    a = _bare_artefact_v03(per_window={"y2024": {"baseline": {}, "candidate": {}}})
    p = tmp_path / "ok.json"
    dump_shadow_artefact(a, p)
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o444


# ---------------------------------------------------------------------------
# 3. Tampering new fields trips integrity error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_rejects_tampered_per_window_field(tmp_path: Path) -> None:
    a = _bare_artefact_v03(per_window={
        "y2024": {"baseline": {"final_equity": 10000.0},
                  "candidate": {"final_equity": 10100.0}},
    })
    p = tmp_path / "tamper.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["per_window"]["y2024"]["candidate"]["final_equity"] = 99999.0
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


@pytest.mark.unit
def test_load_rejects_tampered_window_coverage(tmp_path: Path) -> None:
    a = _bare_artefact_v03(window_coverage={"halt_event_windows": 0})
    p = tmp_path / "tamper-wc.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["window_coverage"]["halt_event_windows"] = 999
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


# ---------------------------------------------------------------------------
# 4. Real on-disk 0.1.0 / 0.2.0 artefacts still load
# ---------------------------------------------------------------------------


_REAL_REG = (_real_shadow_p())


@pytest.mark.integration
def test_real_old_010_artefact_loads_under_v03_loader() -> None:
    """The ticket 2 (0.1.0) artefact at known path must still strict-load
    under the v0.3.0 loader with empty new fields."""
    p = _REAL_REG / "c1-lower-observe-floor-0.50" / "20260501T133702-325440.json"
    if not p.exists():
        pytest.skip("real 0.1.0 artefact not present")
    a = load_shadow_artefact(p)
    assert a.runner_version == "shadow_runner-0.1.0"
    assert a.per_window == {}
    assert a.window_coverage == {}
    assert a.gold_profile == {}


@pytest.mark.integration
def test_real_020_artefact_loads_under_v03_loader() -> None:
    """The ticket 3 (0.2.0) artefact at known path must still strict-load."""
    p = _REAL_REG / "c1-lower-observe-floor-0.50" / "20260502T010803-304527.json"
    if not p.exists():
        pytest.skip("real 0.2.0 artefact not present")
    a = load_shadow_artefact(p)
    assert a.runner_version == "shadow_runner-0.2.0"
    assert a.per_window == {}
    assert a.window_coverage == {}
    assert a.gold_profile == {}
    assert a.baseline_metrics.n_bars_envelope_decided == 22651
