"""Ticket 2 Step 5 — shadow_runner tests.

The runner is the integration layer. It must:

  - Produce a complete ShadowArtefact (passing strict load) on every
    code path: PASS / FAIL / ABSTAIN. NOT_RUN is the runner *not
    being invoked*; the runner itself never returns NOT_RUN.
  - Detect mirror drift before producing PASS/FAIL → emit ABSTAIN
    artefact with mirror_consistency_check=FAIL OR raise.
  - Detect unsupported_target → emit ABSTAIN: unsupported_target
    artefact, no replay performed.
  - Snapshot production rule_engine constants before AND after
    replay; emit hashes into NoLiveEvidence; **assert** no change.
  - Use mirror snapshot_params() + apply_overlay() for candidate
    parameter substitution; never setattr on production module.
  - Refuse to write artefact at a path that already exists
    (sidecar registry append-only invariant).
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.shadow_artefact import (
    ShadowArtefact,
    ShadowVerdict,
    load_shadow_artefact,
)
from smc.hedgerock.evolution.shadow_runner import (
    SHADOW_RUNNER_VERSION,
    run_shadow_for_candidate,
)


# ---------------------------------------------------------------------------
# Stub lake (mirrors test_data_slice)
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0) -> pl.DataFrame:
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": 100.0, "high": 100.5, "low": 99.5,
            "close": 100.0, "volume": 100.0,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data: dict[tuple[str, str], pl.DataFrame]):
        self._data = data
        self._root = Path("/tmp/stub-lake")

    def list_instruments(self) -> list[str]:
        return sorted({k[0] for k in self._data})

    def query(self, instrument: str, timeframe, start, end) -> pl.DataFrame:
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def stub_lake():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 30),
        ("XAUUSD", "H4"): _bars(base, n=6 * 30, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=30, hours_step=24.0),
    })


def _candidate_by_id(cid: str):
    return next(c for c in CANDIDATE_MENU_V0 if c.candidate_id == cid)


# ---------------------------------------------------------------------------
# 1. Runner version is pinned
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_version_is_pinned() -> None:
    assert isinstance(SHADOW_RUNNER_VERSION, str)
    assert SHADOW_RUNNER_VERSION.startswith("shadow_runner-")


# ---------------------------------------------------------------------------
# 2. ABSTAIN: unsupported target
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unsupported_target_yields_abstain_artefact(
    stub_lake, tmp_path: Path,
) -> None:
    """c2 (target in phase_d_walk_forward, not whitelisted in v1) and
    c4 (target hypothetical, not in production) → ABSTAIN."""
    out_dir = tmp_path / "shadow_artefacts"
    for cid in ("c2-halt-expiry-observe-6h", "c4-range2-conf-0.70"):
        cand = _candidate_by_id(cid)
        artefact_path = run_shadow_for_candidate(
            candidate=cand,
            lake=stub_lake,
            symbol="XAUUSD",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            out_dir=out_dir,
        )
        artefact = load_shadow_artefact(artefact_path)
        assert artefact.verdict == ShadowVerdict.ABSTAIN
        assert "unsupported_target" in artefact.verdict_reason


# ---------------------------------------------------------------------------
# 3. Single-symbol → ABSTAIN even when target is whitelisted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_whitelisted_target_single_symbol_yields_abstain(
    stub_lake, tmp_path: Path,
) -> None:
    """c1 target IS whitelisted (Class A) so the runner CAN produce
    a meaningful replay artefact, but the data slice is single
    symbol → verdict must be ABSTAIN: single_symbol."""
    out_dir = tmp_path / "shadow_artefacts"
    cand = _candidate_by_id("c1-lower-observe-floor-0.50")
    artefact_path = run_shadow_for_candidate(
        candidate=cand,
        lake=stub_lake,
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    artefact = load_shadow_artefact(artefact_path)
    assert artefact.verdict == ShadowVerdict.ABSTAIN
    assert "single_symbol" in artefact.verdict_reason or \
           "single-symbol" in artefact.verdict_reason


# ---------------------------------------------------------------------------
# 4. c3 (raises_gross_exposure) → ABSTAIN or FAIL, never PASS
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_c3_aggressive_floor_never_passes(stub_lake, tmp_path: Path) -> None:
    """c3 raises_gross_exposure=True; no PASS allowed in v1 even if
    metrics theoretically would qualify."""
    out_dir = tmp_path / "shadow_artefacts"
    cand = _candidate_by_id("c3-aggressive-floor-0.78")
    artefact_path = run_shadow_for_candidate(
        candidate=cand,
        lake=stub_lake,
        symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    artefact = load_shadow_artefact(artefact_path)
    assert artefact.verdict in (ShadowVerdict.ABSTAIN, ShadowVerdict.FAIL)


# ---------------------------------------------------------------------------
# 5. Production rule_engine constants unchanged after run
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_does_not_mutate_rule_engine_constants(
    stub_lake, tmp_path: Path,
) -> None:
    """The most critical invariant: shadow run must not leak overlay
    into production rule_engine."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    before = (
        rule_engine._CONFIDENCE_OBSERVE_FLOOR,
        rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR,
    )

    out_dir = tmp_path / "shadow_artefacts"
    cand = _candidate_by_id("c1-lower-observe-floor-0.50")
    run_shadow_for_candidate(
        candidate=cand, lake=stub_lake, symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )

    after = (
        rule_engine._CONFIDENCE_OBSERVE_FLOOR,
        rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR,
    )
    assert before == after, (
        "runner leaked overlay into production rule_engine — fail-closed "
        "invariant violated"
    )


# ---------------------------------------------------------------------------
# 6. Artefact strict-loads + has all R2 fields populated
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_artefact_strict_load_succeeds(stub_lake, tmp_path: Path) -> None:
    """Whatever verdict path the runner takes, the artefact MUST be
    strict-loadable (envelope + content_sha256 verify)."""
    out_dir = tmp_path / "shadow_artefacts"
    cand = _candidate_by_id("c1-lower-observe-floor-0.50")
    p = run_shadow_for_candidate(
        candidate=cand, lake=stub_lake, symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    art = load_shadow_artefact(p)
    assert isinstance(art, ShadowArtefact)
    # Identity fields populated
    assert art.candidate_id == cand.candidate_id
    assert art.candidate_diff.target == cand.diff.target
    assert art.candidate_diff.proposed_value == cand.diff.proposed_value
    assert art.candidate_diff.baseline_value == cand.diff.baseline_value
    # Versioning fields populated
    assert art.runner_version == SHADOW_RUNNER_VERSION
    assert len(art.mirror_version) == 64
    assert len(art.metric_schema_version) == 64
    # Data slice identity populated
    assert art.data_slice.symbols == ("XAUUSD",)
    # No-live evidence: zero counters
    assert art.no_live_evidence.http_calls_made_count == 0
    assert art.no_live_evidence.broker_api_calls_made_count == 0


# ---------------------------------------------------------------------------
# 7. Artefact dump path is append-only
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_refuses_overwrite(stub_lake, tmp_path: Path) -> None:
    """Second run with same out_dir + candidate → second artefact at
    a different filename (run_id is unique). Existing files are
    untouched."""
    out_dir = tmp_path / "shadow_artefacts"
    cand = _candidate_by_id("c1-lower-observe-floor-0.50")
    p1 = run_shadow_for_candidate(
        candidate=cand, lake=stub_lake, symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    mtime_before = p1.stat().st_mtime_ns
    bytes_before = p1.read_bytes()
    p2 = run_shadow_for_candidate(
        candidate=cand, lake=stub_lake, symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    assert p1 != p2
    # Original file untouched.
    assert p1.exists()
    assert p1.stat().st_mtime_ns == mtime_before
    assert p1.read_bytes() == bytes_before


# ---------------------------------------------------------------------------
# 8. Mirror drift detected at runtime → ABSTAIN, never PASS/FAIL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runtime_mirror_drift_yields_abstain(
    stub_lake, tmp_path: Path, monkeypatch,
) -> None:
    """Inject a value drift in production rule_engine; runner must
    detect, refuse to produce PASS/FAIL, and emit an ABSTAIN
    artefact with mirror_consistency_check=FAIL."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    monkeypatch.setattr(
        rule_engine, "_CONFIDENCE_OBSERVE_FLOOR", 0.99, raising=True,
    )
    out_dir = tmp_path / "shadow_artefacts"
    cand = _candidate_by_id("c1-lower-observe-floor-0.50")
    p = run_shadow_for_candidate(
        candidate=cand, lake=stub_lake, symbol="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        out_dir=out_dir,
    )
    art = load_shadow_artefact(p)
    assert art.verdict == ShadowVerdict.ABSTAIN
    assert art.mirror_consistency_check == "FAIL"
    assert "mirror_drift" in art.verdict_reason or \
           "mirror" in art.verdict_reason.lower()
