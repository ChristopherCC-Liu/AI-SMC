"""Stage 6-followup task 5 — end-to-end demo tests.

Pinned guarantees:

  * The demo script orchestrates the full report-only loop:
      observe → detect → recommend → queue → inspect → (optional
      paper-test seed + dry-run promotion packet)
  * All outputs land under a tmp ``--workspace`` directory.
  * The real production registry is never touched (json count
    invariant before/after).
  * The demo writes:
      - <workspace>/report/phase-d-evolution-report.md
      - <workspace>/report/hedgerock-evolution-recommendation.md
      - <workspace>/queue/shadow_test_queue.jsonl
      - <workspace>/queue/queue_inspection.md
      - <workspace>/ledger/paper_test_ledger.jsonl  (when --seed-paper-trades)
      - <workspace>/promotion/packet.md             (when --produce-packet)
  * Demo fails loudly if anything tries to write into the real
    registry root.
  * The demo's stdout summary names every stage it ran.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
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


_REPO = Path(__file__).resolve().parents[3]
_REAL_REGISTRY_ROOT = (_real_registry_p())


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_demo as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


def _real_registry_json_count() -> int:
    if not _REAL_REGISTRY_ROOT.exists():
        return 0
    return sum(1 for _ in _REAL_REGISTRY_ROOT.rglob("*.json"))


# ---------------------------------------------------------------------------
# 1. Demo runs end-to-end and produces every expected artefact.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_full_pipeline_produces_every_expected_artefact(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    workspace = tmp_path / "demo"
    pre = _real_registry_json_count()

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main([
            "--workspace", str(workspace),
            "--seed-paper-trades",
            "--produce-packet",
        ])
    assert rc == 0, f"demo exited with {rc}; stdout={stdout.getvalue()!r}"

    expected = [
        workspace / "report" / "phase-d-evolution-report.md",
        workspace / "report" / "hedgerock-evolution-recommendation.md",
        workspace / "queue" / "shadow_test_queue.jsonl",
        workspace / "queue" / "queue_inspection.md",
        workspace / "ledger" / "paper_test_ledger.jsonl",
        workspace / "promotion" / "packet.md",
    ]
    for p in expected:
        assert p.exists(), f"demo failed to produce {p}"

    out = stdout.getvalue().lower()
    for stage in ("observe", "detect", "recommend", "queue", "inspect"):
        assert stage in out, f"demo stdout missing stage marker: {stage!r}"

    # Real production registry untouched.
    assert _real_registry_json_count() == pre


# ---------------------------------------------------------------------------
# 2. Without --produce-packet, no promotion packet is written.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_without_produce_packet_skips_promotion_step(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
    ])
    assert rc == 0
    assert not (workspace / "promotion" / "packet.md").exists()
    # Other artefacts still produced.
    assert (workspace / "queue" / "shadow_test_queue.jsonl").exists()
    assert (workspace / "queue" / "queue_inspection.md").exists()


# ---------------------------------------------------------------------------
# 3. Without --seed-paper-trades, no paper-test ledger is written.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_without_seed_paper_trades_skips_ledger(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
    ])
    assert rc == 0
    assert not (workspace / "ledger" / "paper_test_ledger.jsonl").exists()


# ---------------------------------------------------------------------------
# 4. Workspace pointing at the real registry root is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_rejects_workspace_under_real_registry_root() -> None:
    cli = _import_cli()
    bad = _REAL_REGISTRY_ROOT / "demo-workspace"
    stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
        rc = cli.main(["--workspace", str(bad)])
    assert rc != 0
    err = stderr.getvalue().lower()
    assert "forbidden" in err or "policy_registry" in err.lower()


# ---------------------------------------------------------------------------
# 5. Demo's recommendation report carries the same NOT-LIVE banners.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_recommendation_carries_not_live_banners(tmp_path: Path) -> None:
    cli = _import_cli()
    workspace = tmp_path / "demo"
    rc = cli.main(["--workspace", str(workspace)])
    assert rc == 0
    body = (workspace / "report"
            / "hedgerock-evolution-recommendation.md").read_text(
                encoding="utf-8"
    )
    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body
    assert "**NOT DEPLOYED**" in body


# ---------------------------------------------------------------------------
# 6. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_script_has_no_live_runtime_imports() -> None:
    src = (_REPO / "scripts" / "hedgerock_evolution_demo.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src
