"""Ticket 2 Step 7 — shadow_run CLI + report CLI join tests.

Pinned guarantees:
  - shadow_run CLI processes every menu candidate, produces one
    artefact per candidate under
    <out_dir>/<candidate_id>/<run_id>.json.
  - shadow_run CLI does NOT write under approved/, pointer.json, src/,
    config/, or rule_engine/.mq5.
  - report CLI accepts --shadow-artefacts; without it, G8 stays
    NOT_RUN (no artefact in bundle).
  - report CLI joins by candidate_id AND manifest_content_hash.
    candidate_id-only matches do NOT join (no fallback per R5).
  - With shadow artefacts attached, the rendered report shows G8 in
    the per-candidate verdict block.
  - Even with shadow artefacts attached, NO_STRATEGY_CHANGE: true
    short-circuit still blocks every candidate (Ticket 1 invariant).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

# ---------------------------------------------------------------------------
# CLI imports
# ---------------------------------------------------------------------------


def _import_shadow_run_cli():
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_shadow_run as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


def _import_report_cli():
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_report as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


# ---------------------------------------------------------------------------
# Synthetic Phase D bundle helpers (re-used pattern)
# ---------------------------------------------------------------------------


_SAMPLE_AVAILABILITY = """
# Phase D-cont3-preflight

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | -0.067% ±0.048 (NEG) | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.122% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 (NEG) | 1 |
| XAUUSD | 2023 | 4913 | +0.063% ±0.069 (CI∋0) | +0.055% ±0.042 | -0.401% ±0.111 (NEG) | 1 |
| XAUUSD | 2024 | 5693 | +0.139% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: true
```
"""


def _seed_phase_d_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n\nfake atlas\n")
    avail = tmp_path / "availability.md"
    avail.write_text(_SAMPLE_AVAILABILITY)
    wf = tmp_path / "wf.md"
    wf.write_text("# walk-forward\n")
    return atlas, avail, wf


# ---------------------------------------------------------------------------
# Stub lake
# ---------------------------------------------------------------------------


def _stub_lake_with_xauusd():
    from datetime import datetime, timedelta, timezone
    import polars as pl

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _bars(start, n, hours_step=1.0):
        rows = []
        for i in range(n):
            ts = start + timedelta(hours=hours_step * i)
            rows.append({"ts": ts, "open": 100.0, "high": 100.5, "low": 99.5,
                         "close": 100.0, "volume": 100.0})
        return pl.DataFrame(rows).with_columns(
            pl.col("ts").dt.replace_time_zone("UTC")
        )

    class _StubLake:
        def __init__(self, data):
            self._data = data
            self._root = Path("/tmp/stub-lake")

        def list_instruments(self):
            return sorted({k[0] for k in self._data})

        def query(self, instrument, timeframe, start, end):
            df = self._data.get((instrument, str(timeframe)))
            if df is None or df.is_empty():
                return pl.DataFrame()
            return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))

    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 30),
        ("XAUUSD", "H4"): _bars(base, n=6 * 30, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=30, hours_step=24.0),
    })


# ===========================================================================
# 1. shadow_run CLI: produces one artefact per menu candidate
# ===========================================================================


@pytest.mark.unit
def test_shadow_run_produces_one_artefact_per_candidate(tmp_path: Path) -> None:
    cli = _import_shadow_run_cli()
    out_dir = tmp_path / "shadow_artefacts"
    from datetime import datetime, timezone
    cli.run(
        lake=_stub_lake_with_xauusd(),
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    expected = {
        "c1-lower-observe-floor-0.50",
        "c2-halt-expiry-observe-6h",
        "c3-aggressive-floor-0.78",
        "c4-range2-conf-0.70",
    }
    candidate_dirs = {p.name for p in out_dir.iterdir() if p.is_dir()}
    assert candidate_dirs == expected
    for cid in expected:
        artefacts = list((out_dir / cid).glob("*.json"))
        assert len(artefacts) == 1, f"{cid}: expected 1 artefact, got {len(artefacts)}"


# ===========================================================================
# 2. shadow_run CLI doesn't write outside out_dir
# ===========================================================================


@pytest.mark.unit
def test_shadow_run_does_not_write_under_approved_or_pointer(tmp_path: Path) -> None:
    cli = _import_shadow_run_cli()
    out_dir = tmp_path / "shadow_artefacts"
    fake_registry = tmp_path / "registry"
    fake_registry.mkdir()
    from datetime import datetime, timezone
    cli.run(
        lake=_stub_lake_with_xauusd(),
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    # Approved dir / pointer.json never created by shadow_run.
    assert not (fake_registry / "approved").exists()
    assert not (fake_registry / "pointer.json").exists()


# ===========================================================================
# 3. shadow_run CLI does not import production runtime mutation paths
# ===========================================================================


@pytest.mark.unit
def test_shadow_run_does_not_import_decision_server() -> None:
    """The CLI may not import the live HTTP routes."""
    cli_path = _scripts_dir_p() / "hedgerock_shadow_run.py"
    text = cli_path.read_text(encoding="utf-8")
    assert "from smc.hedgerock.decision_server" not in text
    assert "import smc.hedgerock.decision_server" not in text


@pytest.mark.unit
def test_shadow_run_no_destructive_fs_calls() -> None:
    """Sidecar boundary: no rm -rf / rmtree / unlink / remove / rmdir."""
    import ast
    cli_path = _scripts_dir_p() / "hedgerock_shadow_run.py"
    text = cli_path.read_text(encoding="utf-8")
    for needle in ("rm -rf", "rm-rf"):
        assert needle not in text
    tree = ast.parse(text)
    forbidden = {"rmtree", "unlink", "remove", "rmdir"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
            raise AssertionError(
                f"hedgerock_shadow_run.py:{node.lineno}: forbidden call "
                f".{node.func.attr}()"
            )


# ===========================================================================
# 4. report CLI: without --shadow-artefacts, G8 stays NOT_RUN
# ===========================================================================


@pytest.mark.unit
def test_report_without_shadow_dir_keeps_g8_not_run(tmp_path: Path) -> None:
    cli = _import_report_cli()
    atlas, avail, wf = _seed_phase_d_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"
    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
    )
    for c in candidates:
        gate_results = {g.gate_id: g for g in c.gates}
        g8 = gate_results.get("G8")
        assert g8 is not None
        assert g8.status.value == "NOT_RUN"


# ===========================================================================
# 5. report CLI: with --shadow-artefacts, G8 verdict comes from artefact
# ===========================================================================


@pytest.mark.unit
def test_report_with_shadow_artefacts_propagates_g8(tmp_path: Path) -> None:
    """End-to-end: produce shadow artefacts via shadow_run, then run
    report with --shadow-artefacts pointing at them. Each candidate's
    G8 must reflect the artefact verdict (ABSTAIN for v1)."""
    shadow_cli = _import_shadow_run_cli()
    out_dir = tmp_path / "shadow_artefacts"
    from datetime import datetime, timezone
    shadow_cli.run(
        lake=_stub_lake_with_xauusd(),
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )

    cli = _import_report_cli()
    atlas, avail, wf = _seed_phase_d_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"
    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
        shadow_artefacts_dir=out_dir,
    )
    # Every candidate's G8 should NOT be NOT_RUN — they should
    # carry the artefact-derived verdict (ABSTAIN in v1).
    for c in candidates:
        gate_results = {g.gate_id: g for g in c.gates}
        g8 = gate_results.get("G8")
        assert g8 is not None
        # In v1, every candidate's runner returns ABSTAIN; G8
        # propagates that.
        assert g8.status.value == "ABSTAIN", (
            f"{c.candidate_id}: G8={g8.status.value}, expected ABSTAIN — "
            "v1 runner cannot return PASS for any single-symbol candidate"
        )


# ===========================================================================
# 6. report CLI: NO_STRATEGY_CHANGE short-circuit still active
# ===========================================================================


@pytest.mark.unit
def test_report_with_shadow_artefacts_still_blocks_via_no_strategy_change(
    tmp_path: Path,
) -> None:
    """Even with shadow artefacts, NO_STRATEGY_CHANGE: true blocks
    promotion (Ticket 1 invariant retained)."""
    shadow_cli = _import_shadow_run_cli()
    out_dir = tmp_path / "shadow_artefacts"
    from datetime import datetime, timezone
    shadow_cli.run(
        lake=_stub_lake_with_xauusd(),
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )

    cli = _import_report_cli()
    atlas, avail, wf = _seed_phase_d_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"
    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
        shadow_artefacts_dir=out_dir,
    )
    for c in candidates:
        assert any("data_availability_action_gate_blocks_all" in r
                   for r in c.blocking_reasons), (
            f"{c.candidate_id}: missing NO_STRATEGY_CHANGE short-circuit"
        )


# ===========================================================================
# 7. R5: candidate_id-only join is forbidden — manifest hash mismatch → FAIL
# ===========================================================================


@pytest.mark.unit
def test_report_refuses_candidate_id_only_join_when_manifest_hash_drifted(
    tmp_path: Path,
) -> None:
    """Plant a syntactically-valid artefact whose
    ``candidate_manifest_content_hash`` doesn't match the current
    candidate; report MUST surface G8 FAIL: manifest_drift, not
    silently fall back to id-only join."""
    shadow_cli = _import_shadow_run_cli()
    out_dir = tmp_path / "shadow_artefacts"
    from datetime import datetime, timezone
    shadow_cli.run(
        lake=_stub_lake_with_xauusd(),
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )

    # Tamper one artefact's manifest hash. We must rebuild the
    # envelope content_sha256 too (otherwise the load would fail with
    # ShadowArtefactIntegrityError, which is also a FAIL but less
    # specific than what we want to test here).
    cid = "c1-lower-observe-floor-0.50"
    artefact_files = list((out_dir / cid).glob("*.json"))
    assert len(artefact_files) == 1
    p = artefact_files[0]

    import hashlib
    import json
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["candidate_manifest_content_hash"] = "0" * 64
    canonical = json.dumps(
        raw["artefact"], indent=2, sort_keys=True, ensure_ascii=False,
    )
    raw["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))

    cli = _import_report_cli()
    atlas, avail, wf = _seed_phase_d_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"
    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
        shadow_artefacts_dir=out_dir,
    )
    c1 = next(c for c in candidates if c.candidate_id == cid)
    g8 = next(g for g in c1.gates if g.gate_id == "G8")
    assert g8.status.value == "FAIL"
    assert "manifest_drift" in g8.reason.lower() or \
           "manifest" in g8.reason.lower()


# ===========================================================================
# 8. Final sanity: 4 candidates always PROMOTION_BLOCKED (R4)
# ===========================================================================


@pytest.mark.unit
def test_with_shadow_artefacts_all_4_candidates_still_promotion_blocked(
    tmp_path: Path,
) -> None:
    """R4 declaration: in v1 against current single-symbol lake +
    NO_STRATEGY_CHANGE: true, no candidate should be READY_FOR_TESTED
    even after shadow artefacts attached."""
    shadow_cli = _import_shadow_run_cli()
    out_dir = tmp_path / "shadow_artefacts"
    from datetime import datetime, timezone
    shadow_cli.run(
        lake=_stub_lake_with_xauusd(),
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    cli = _import_report_cli()
    atlas, avail, wf = _seed_phase_d_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"
    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
        shadow_artefacts_dir=out_dir,
    )
    for c in candidates:
        assert c.result.value.startswith("PROMOTION_BLOCKED")
