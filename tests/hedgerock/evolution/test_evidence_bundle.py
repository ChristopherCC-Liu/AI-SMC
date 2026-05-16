"""Phase D-cont3 / Ticket 1 — evidence_bundle tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from smc.hedgerock.evolution.evidence_bundle import (

    EvidenceBundleArtefacts,
    compute_bundle_content_hash,
    compute_file_sha256,
    load_evidence_bundle,
    parse_data_availability_report,
    verify_bundle_hash,
)
from smc.hedgerock.evolution.policy_manifest import EvidenceBundle


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

SAMPLE_AVAILABILITY = """
# Phase D-cont3-preflight — Data availability + year-replication

## Lake scan

- Lake root: `/foo/bar`
- Instruments found: 1

| Instrument | Timeframe | Start | End | Bars | Span (days) | Intra-week gaps | Largest gap (h) | Completeness | ≥3y? |
|---|---|---|---|---|---|---|---|---|---|
| XAUUSD | H1 | 2020-01-01 | 2024-12-31 | 28798 | 1825 | 1579 | 29.0 | 0.92 | ✅ |

## Year-replication summary

For each (symbol, year) combination, the existing atlas was re-run.

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | -0.067% ±0.048 (NEG) | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.122% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 (NEG) | 1 |
| XAUUSD | 2023 | 4913 | +0.063% ±0.069 (CI∋0) | +0.055% ±0.042 | -0.401% ±0.111 (NEG) | 2 |
| XAUUSD | 2024 | 5693 | +0.139% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: true
Reason:
  - single-symbol lake
```
"""

SAMPLE_AVAILABILITY_GATE_FALSE = SAMPLE_AVAILABILITY.replace(
    "NO_STRATEGY_CHANGE: true", "NO_STRATEGY_CHANGE: false"
)


def _write_artefacts(tmp_path: Path, availability: str = SAMPLE_AVAILABILITY) -> tuple[Path, Path, Path]:
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n\nfake contents for hash test\n")
    avail = tmp_path / "availability.md"
    avail.write_text(availability)
    wf = tmp_path / "walkforward.md"
    wf.write_text("# walk-forward\n")
    return atlas, avail, wf


# ---------------------------------------------------------------------------
# 1. SHA-256 helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello world\n")
    expected = hashlib.sha256(b"hello world\n").hexdigest()
    assert compute_file_sha256(p) == expected


@pytest.mark.unit
def test_compute_bundle_content_hash_deterministic(tmp_path: Path) -> None:
    a, b, c = _write_artefacts(tmp_path)
    h1 = compute_bundle_content_hash([a, b, c])
    h2 = compute_bundle_content_hash([c, b, a])  # different order; sorted internally
    assert h1 == h2


@pytest.mark.unit
def test_compute_bundle_content_hash_changes_on_file_edit(tmp_path: Path) -> None:
    a, b, c = _write_artefacts(tmp_path)
    h1 = compute_bundle_content_hash([a, b, c])
    a.write_bytes(b"tampered")
    h2 = compute_bundle_content_hash([a, b, c])
    assert h1 != h2


# ---------------------------------------------------------------------------
# 2. data-availability parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_data_availability_extracts_year_replication() -> None:
    parsed = parse_data_availability_report(SAMPLE_AVAILABILITY)
    yr = parsed["year_replication"]
    assert "XAUUSD" in yr
    assert yr["XAUUSD"]["years_total"] == 4
    # 2/4 pass: 2022 (+0.122 ±0.074) and 2024 (+0.139 ±0.067) are mean > CI no marker.
    # 2021 is (NEG); 2023 is (CI∋0). 2022 and 2024 pass.
    assert yr["XAUUSD"]["years_passing"] == 2
    assert tuple(yr["XAUUSD"]["negative_sign_years"]) == (2021,)


@pytest.mark.unit
def test_parse_data_availability_sums_halt_events() -> None:
    parsed = parse_data_availability_report(SAMPLE_AVAILABILITY)
    # 1 + 1 + 2 + 1 = 5
    assert parsed["halt_event_count"] == 5


@pytest.mark.unit
def test_parse_data_availability_counts_unique_symbols() -> None:
    parsed = parse_data_availability_report(SAMPLE_AVAILABILITY)
    assert parsed["cross_symbol_count"] == 1


@pytest.mark.unit
def test_parse_data_availability_detects_no_strategy_change_true() -> None:
    parsed = parse_data_availability_report(SAMPLE_AVAILABILITY)
    assert parsed["no_strategy_change"] is True


@pytest.mark.unit
def test_parse_data_availability_detects_no_strategy_change_false() -> None:
    parsed = parse_data_availability_report(SAMPLE_AVAILABILITY_GATE_FALSE)
    assert parsed["no_strategy_change"] is False


# ---------------------------------------------------------------------------
# 3. load_evidence_bundle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_evidence_bundle_populates_all_fields(tmp_path: Path) -> None:
    a, b, c = _write_artefacts(tmp_path)
    bundle = load_evidence_bundle(EvidenceBundleArtefacts(
        bundle_id="evb-test",
        atlas_report_path=a,
        data_availability_report_path=b,
        walk_forward_run_paths=(c,),
    ))
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.bundle_id == "evb-test"
    assert bundle.atlas_report_hash_sha256 == compute_file_sha256(a)
    assert bundle.data_availability_report_hash_sha256 == compute_file_sha256(b)
    assert bundle.cross_symbol_count == 1
    assert bundle.halt_event_count == 5
    assert bundle.no_strategy_change is True
    assert bundle.year_replication["XAUUSD"]["years_total"] == 4


# ---------------------------------------------------------------------------
# 4. verify_bundle_hash — tamper rejection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_bundle_hash_pass_unchanged(tmp_path: Path) -> None:
    a, b, c = _write_artefacts(tmp_path)
    bundle = load_evidence_bundle(EvidenceBundleArtefacts(
        bundle_id="evb-test",
        atlas_report_path=a,
        data_availability_report_path=b,
        walk_forward_run_paths=(c,),
    ))
    assert verify_bundle_hash(bundle, base_dir=tmp_path) is True


@pytest.mark.unit
def test_verify_bundle_hash_rejects_tampered_atlas(tmp_path: Path) -> None:
    a, b, c = _write_artefacts(tmp_path)
    bundle = load_evidence_bundle(EvidenceBundleArtefacts(
        bundle_id="evb-test",
        atlas_report_path=a,
        data_availability_report_path=b,
        walk_forward_run_paths=(c,),
    ))
    a.write_text("# atlas\n\nTAMPERED\n")
    assert verify_bundle_hash(bundle, base_dir=tmp_path) is False


@pytest.mark.unit
def test_verify_bundle_hash_rejects_tampered_data_availability(tmp_path: Path) -> None:
    a, b, c = _write_artefacts(tmp_path)
    bundle = load_evidence_bundle(EvidenceBundleArtefacts(
        bundle_id="evb-test",
        atlas_report_path=a,
        data_availability_report_path=b,
        walk_forward_run_paths=(c,),
    ))
    b.write_text(SAMPLE_AVAILABILITY + "\n# tampered\n")
    assert verify_bundle_hash(bundle, base_dir=tmp_path) is False


# ---------------------------------------------------------------------------
# 5. Real Phase D bundle integration (skips when artefacts not present)
# ---------------------------------------------------------------------------


_REAL_DOCS = (_hedgerock_home_p() / 'docs')


@pytest.mark.integration
def test_load_real_phase_d_bundle() -> None:
    atlas = _REAL_DOCS / "phase-d-regime-opportunity-atlas.md"
    avail = _REAL_DOCS / "phase-d-data-availability.md"
    wf = _REAL_DOCS / "phase-d-walk-forward-report.md"
    if not (atlas.exists() and avail.exists() and wf.exists()):
        pytest.skip("real Phase D bundle not present")

    bundle = load_evidence_bundle(EvidenceBundleArtefacts(
        bundle_id="evb-2024-XAUUSD",
        atlas_report_path=atlas,
        data_availability_report_path=avail,
        walk_forward_run_paths=(wf,),
    ))
    assert "XAUUSD" in bundle.year_replication
    # 4 years (2021–2024) — 2020 is partial and excluded by the
    # data-availability runner.
    assert bundle.year_replication["XAUUSD"]["years_total"] == 4
    assert bundle.cross_symbol_count == 1
    # NO_STRATEGY_CHANGE: true must currently be set on the real lake
    # (until a second symbol is ingested).
    assert bundle.no_strategy_change is True
