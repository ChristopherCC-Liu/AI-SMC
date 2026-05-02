"""Stage 6-followup-2 task 1 — replay-based candidate validator tests
(report-only, read-only over shadow artefacts).

Pinned guarantees:

  * The validator reads existing shadow artefact JSON files under
    ``<shadow_artefacts_root>/<candidate_id>/``. It NEVER creates,
    deletes, or rewrites any artefact.
  * The validator imports no live runtime module.
  * The validator does NOT simulate strategy execution. It is
    explicitly an aggregator over already-recorded per-window
    deltas. Reports carry the
    ``heuristic_projection_only / not_a_simulation`` banner.
  * Output is a frozen :class:`ReplayValidationReport` dataclass.
  * Validator refuses paths that don't exist (raises
    ``FileNotFoundError``); refuses an empty artefact directory
    (returns a report with ``n_windows_replayed = 0`` and a
    ``no_artefacts_found`` blocking condition — never raises).
  * Validator skips entries lacking ``per_window["windows"]``;
    surfaces the skip count in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smc.hedgerock.evolution.replay_validator import (
    ReplayValidationReport,
    summarise_replay,
    render_replay_report,
)


_REPO = Path(__file__).resolve().parents[3]
_REAL_REGISTRY_ROOT = Path("/Users/christopher/HedgeRock/policy_registry")


def _seed_artefact(
    *,
    cand_dir: Path,
    artefact_id: str,
    windows: list[dict],
    verdict: str = "ABSTAIN",
) -> Path:
    cand_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artefact": {
            "artefact_id": artefact_id,
            "candidate_id": cand_dir.name,
            "verdict": verdict,
            "verdict_reason": "synthetic test fixture",
            "runner_version": "shadow_runner-0.3.0",
            "per_window": {"windows": windows},
        }
    }
    p = cand_dir / f"{artefact_id}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _window(*, window_id: str, delta_pnl_pp: float = 0.0,
            delta_dd_pp: float = 0.0, n_trades: int = 4) -> dict:
    return {
        "window_id": window_id,
        "delta_pnl_pp": delta_pnl_pp,
        "delta_dd_pp": delta_dd_pp,
        "candidate_n_trades": n_trades,
        "baseline_total_return_pct": 1.0,
        "candidate_total_return_pct": 1.0 + delta_pnl_pp,
        "baseline_max_dd_pct": 0.05,
        "candidate_max_dd_pct": 0.05 + delta_dd_pp,
        "observed_buckets": ["trend_up"],
    }


# ---------------------------------------------------------------------------
# 1. Aggregates deltas across multiple artefacts.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_summarise_aggregates_window_deltas(tmp_path: Path) -> None:
    cand_dir = tmp_path / "shadow_artefacts" / "c1-lower-observe-floor-0.50"
    _seed_artefact(
        cand_dir=cand_dir, artefact_id="run-1",
        windows=[
            _window(window_id="y2021_h1", delta_pnl_pp=0.5, delta_dd_pp=0.1),
            _window(window_id="y2021_h2", delta_pnl_pp=-0.2, delta_dd_pp=0.0),
        ],
    )
    _seed_artefact(
        cand_dir=cand_dir, artefact_id="run-2",
        windows=[
            _window(window_id="y2022_h1", delta_pnl_pp=0.3, delta_dd_pp=-0.05),
        ],
    )

    report = summarise_replay(
        candidate_id="c1-lower-observe-floor-0.50",
        shadow_artefacts_root=tmp_path / "shadow_artefacts",
    )

    assert isinstance(report, ReplayValidationReport)
    assert report.candidate_id == "c1-lower-observe-floor-0.50"
    assert report.n_artefacts_read == 2
    assert report.n_windows_replayed == 3
    # Mean delta pnl: (0.5 + -0.2 + 0.3) / 3 = 0.2
    assert report.delta_pnl_pp_mean == pytest.approx(0.2, abs=1e-6)
    # Worst delta_dd_pp = max(0.1, 0.0, -0.05) = 0.1 (most adverse positive)
    assert report.delta_dd_pp_worst == pytest.approx(0.1, abs=1e-6)
    assert report.heuristic_projection_only is True
    assert report.report_only is True
    assert report.windows_passing == 2  # delta_pnl_pp > 0
    assert report.windows_regressing == 1


# ---------------------------------------------------------------------------
# 2. Empty artefact directory → zero-count report, no exception.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_artefact_dir_yields_zero_windows_report(tmp_path: Path) -> None:
    (tmp_path / "shadow_artefacts" / "c1-lower-observe-floor-0.50").mkdir(
        parents=True
    )
    report = summarise_replay(
        candidate_id="c1-lower-observe-floor-0.50",
        shadow_artefacts_root=tmp_path / "shadow_artefacts",
    )
    assert report.n_artefacts_read == 0
    assert report.n_windows_replayed == 0
    assert "no_artefacts_found" in report.blocking_conditions


# ---------------------------------------------------------------------------
# 3. Missing root → FileNotFoundError.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_shadow_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarise_replay(
            candidate_id="c1-lower-observe-floor-0.50",
            shadow_artefacts_root=tmp_path / "does-not-exist",
        )


# ---------------------------------------------------------------------------
# 4. Validator does not mutate artefacts.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_does_not_mutate_artefacts(tmp_path: Path) -> None:
    cand_dir = tmp_path / "shadow_artefacts" / "c1-lower-observe-floor-0.50"
    p = _seed_artefact(
        cand_dir=cand_dir, artefact_id="r",
        windows=[_window(window_id="y2021_h1", delta_pnl_pp=0.1)],
    )
    pre_bytes = p.read_bytes()
    pre_mtime = p.stat().st_mtime_ns

    summarise_replay(
        candidate_id="c1-lower-observe-floor-0.50",
        shadow_artefacts_root=tmp_path / "shadow_artefacts",
    )

    assert p.read_bytes() == pre_bytes
    assert p.stat().st_mtime_ns == pre_mtime


# ---------------------------------------------------------------------------
# 5. Frozen report; report_only=True invariant.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_validation_report_is_frozen() -> None:
    r = ReplayValidationReport(
        candidate_id="c1",
        n_artefacts_read=0, n_windows_replayed=0,
        delta_pnl_pp_mean=0.0, delta_pnl_pp_p25=0.0,
        delta_pnl_pp_p75=0.0,
        delta_dd_pp_worst=0.0,
        windows_passing=0, windows_regressing=0,
        skipped_artefact_ids=(),
        observed_buckets=(),
        blocking_conditions=("no_artefacts_found",),
    )
    assert r.report_only is True
    assert r.heuristic_projection_only is True
    with pytest.raises((AttributeError, TypeError)):
        r.delta_pnl_pp_mean = 9.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. Skips malformed artefacts; surfaces skip ids.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_malformed_artefact_is_skipped(tmp_path: Path) -> None:
    cand_dir = tmp_path / "shadow_artefacts" / "c1-lower-observe-floor-0.50"
    cand_dir.mkdir(parents=True)
    # Valid artefact
    _seed_artefact(
        cand_dir=cand_dir, artefact_id="valid",
        windows=[_window(window_id="y2021_h1", delta_pnl_pp=0.4)],
    )
    # Broken artefact — not even valid JSON
    (cand_dir / "broken.json").write_text("not json", encoding="utf-8")
    # Artefact missing per_window
    (cand_dir / "no-per-window.json").write_text(
        json.dumps({"artefact": {"candidate_id": cand_dir.name}}),
        encoding="utf-8",
    )

    report = summarise_replay(
        candidate_id="c1-lower-observe-floor-0.50",
        shadow_artefacts_root=tmp_path / "shadow_artefacts",
    )
    # Only 1 valid + windows; the others go into skipped list.
    assert report.n_artefacts_read == 1
    assert report.n_windows_replayed == 1
    assert "broken" in " ".join(report.skipped_artefact_ids)
    assert "no-per-window" in " ".join(report.skipped_artefact_ids)


# ---------------------------------------------------------------------------
# 7. Markdown render carries banners.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_replay_report_carries_banners(tmp_path: Path) -> None:
    cand_dir = tmp_path / "shadow_artefacts" / "c1-lower-observe-floor-0.50"
    _seed_artefact(
        cand_dir=cand_dir, artefact_id="r",
        windows=[_window(window_id="y2021_h1", delta_pnl_pp=0.1)],
    )
    report = summarise_replay(
        candidate_id="c1-lower-observe-floor-0.50",
        shadow_artefacts_root=tmp_path / "shadow_artefacts",
    )
    body = render_replay_report(report)
    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body
    assert "heuristic_projection_only" in body
    assert "not a simulation" in body.lower()
    assert "c1-lower-observe-floor-0.50" in body
    assert "delta_pnl_pp_mean" in body
    assert "windows_passing" in body


# ---------------------------------------------------------------------------
# 8. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_validator_module_does_not_import_live_runtime() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "replay_validator.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 9. Real-registry integration smoke (skips when registry absent).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_registry_smoke_replay_does_not_mutate(tmp_path: Path) -> None:
    real_root = _REAL_REGISTRY_ROOT / "shadow_artefacts"
    if not real_root.exists():
        pytest.skip("real shadow_artefacts root not present")

    pre = sum(1 for _ in real_root.rglob("*.json"))
    pre_mtimes = {
        p: p.stat().st_mtime_ns for p in real_root.rglob("*.json")
    }
    report = summarise_replay(
        candidate_id="c1-lower-observe-floor-0.50",
        shadow_artefacts_root=real_root,
    )
    assert report.n_artefacts_read >= 1
    post = sum(1 for _ in real_root.rglob("*.json"))
    assert pre == post, "validator changed the shadow-artefact json count"
    for p, m in pre_mtimes.items():
        assert p.stat().st_mtime_ns == m, f"mtime drift on {p}"
