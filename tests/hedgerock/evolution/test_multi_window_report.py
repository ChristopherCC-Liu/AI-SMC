"""Ticket 4 v2 Step 8 — multi-window report rendering.

Pinned guarantees:
  * ``render_multi_window_report`` accepts a list of v0.3.0 artefact
    paths and emits a markdown-style string carrying:
      - per-artefact absolute path, sha256, runner_version, verdict,
        verdict_reason, n_bars sum, worst-window DD, coverage_pass.
      - per-window table (window_id, observed_buckets, n_bars,
        delta_pnl_pp, candidate_max_dd_pct, max_h1_gap_bars).
      - coverage shortfall reasons block (when present).
      - forbidden file mtime/absence proof footer (live runtime
        files: approved/ pointer.json src/ config/ .mq5).
  * Renders v0.2.0 artefacts as ``no_per_window_data`` placeholders.
  * Reason vocabulary stays XAUUSD-only — no `single_symbol` /
    `cross_symbol` text in any rendered string.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.window_coverage import WindowSpec


# ---------------------------------------------------------------------------
# Stub lake (re-used)
# ---------------------------------------------------------------------------


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
        self._root = Path("/tmp/stub_t4_report")

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


def _produce_artefact(long_lake, tmp_path) -> Path:
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    cand = _candidate_c1()
    return run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=_windows_two(), out_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. Module + entry point exist
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_multi_window_report_function_exists() -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    assert callable(render_multi_window_report)


# ---------------------------------------------------------------------------
# 2. Per-artefact required fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_includes_absolute_path_and_sha256(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    # Absolute path appears verbatim.
    assert str(p.resolve()) in out
    # sha256 of the artefact file appears (64 hex chars somewhere).
    import re
    assert re.search(r"\b[0-9a-f]{64}\b", out), "no sha256 in report"


@pytest.mark.unit
def test_report_includes_runner_version_and_verdict(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    assert "shadow_runner-0.3.0" in out
    # Verdict (one of PASS/FAIL/ABSTAIN/NOT_RUN) must surface.
    assert any(v in out for v in ("PASS", "FAIL", "ABSTAIN", "NOT_RUN"))


@pytest.mark.unit
def test_report_includes_coverage_pass_and_n_bars(
    long_lake, tmp_path,
) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    # Coverage status surfaced.
    assert "coverage_pass" in out
    # Total n_bars across windows surfaced (each stub window has
    # ≥ ~24*30 = 720 hours, two windows ≈ 1440 H1 bars or so).
    assert "n_bars" in out


@pytest.mark.unit
def test_report_includes_worst_window_dd(long_lake, tmp_path) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    assert "worst_window" in out
    assert "dd" in out.lower()


# ---------------------------------------------------------------------------
# 3. Per-window table is present
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_includes_per_window_table(long_lake, tmp_path) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    # Both window IDs from _windows_two() surface.
    assert "y2024_a" in out
    assert "y2024_b" in out
    # Per-window axis columns present.
    assert "delta_pnl_pp" in out
    assert "candidate_max_dd_pct" in out


# ---------------------------------------------------------------------------
# 4. Coverage shortfall reasons surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_surfaces_coverage_shortfall_reasons(
    long_lake, tmp_path,
) -> None:
    """Two windows < 6-window floor → coverage shortfall reasons
    propagate into the rendered report."""
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    assert "insufficient_xauusd_window_coverage" in out


# ---------------------------------------------------------------------------
# 5. XAUUSD-only vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_no_legacy_single_symbol_text(long_lake, tmp_path) -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    assert "single_symbol" not in out
    assert "cross_symbol" not in out
    assert "single symbol" not in out.lower()


# ---------------------------------------------------------------------------
# 6. Forbidden file mtime/absence proof footer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_includes_forbidden_file_proof_footer(
    long_lake, tmp_path,
) -> None:
    """Final report MUST end with a proof block enumerating the
    live-runtime files that the sidecar is forbidden to touch (and
    showing they are absent or have unchanged mtime)."""
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    p = _produce_artefact(long_lake, tmp_path)
    out = render_multi_window_report([p])
    # The proof block surfaces the forbidden paths by name.
    for token in ("approved", "pointer.json"):
        assert token in out, f"forbidden-path token {token!r} missing"
    assert "forbidden" in out.lower() or "untouched" in out.lower() or \
           "absent" in out.lower()


# ---------------------------------------------------------------------------
# 7. v0.2.0 artefacts gracefully labeled (no per-window data)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_handles_v020_artefact_gracefully(tmp_path) -> None:
    """A v0.2.0 artefact (no per_window/window_coverage) is still
    rendered — just labeled with `no_per_window_data` rather than
    crashing the report."""
    from smc.hedgerock.evolution.shadow_artefact import (
        SHADOW_ARTEFACT_SCHEMA_VERSION,
        CandidateDiffSnapshot, DataSliceIdentity, NoLiveEvidence,
        NoLookaheadAudit, ReplayInvariants, ShadowArtefact, ShadowMetrics,
        ShadowVerdict, SidecarModuleHashes, dump_shadow_artefact,
    )
    from smc.hedgerock.evolution.policy_manifest import (
        compute_canonical_candidate_hash,
    )
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )

    cand = _candidate_c1()
    z = ShadowMetrics(
        final_equity=10000.0, total_return_pct=0.0, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    art = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id="evb-v020-no-pw",
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
        runner_version="shadow_runner-0.2.0",
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
        verdict=ShadowVerdict.ABSTAIN,
        verdict_reason="(synthetic v0.2.0)",
    )
    p = tmp_path / "v020-no-pw.json"
    dump_shadow_artefact(art, p)

    out = render_multi_window_report([p])
    assert "shadow_runner-0.2.0" in out
    # Render path: per-window block placeholder rather than crash.
    assert "no_per_window_data" in out or "no per-window data" in out.lower()


# ---------------------------------------------------------------------------
# 8. Empty artefact list — non-empty header but explicit "no artefacts"
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_handles_empty_input() -> None:
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    out = render_multi_window_report([])
    assert "no artefacts" in out.lower() or "empty" in out.lower()
    # Forbidden-file proof footer still emitted.
    assert "approved" in out
