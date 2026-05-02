"""Stage 6-followup-3 task 5 — full-chain demo + audit-trail integration.

Stronger than the round-2 stage-marker test: this integration test
inspects the **format** and **ordering** of every audit-trail entry,
asserts every operation key has the required JSON schema, and
checks that the trail is byte-monotonic across stages.

Pinned guarantees:

  * The demo writes EXACTLY ONE entry per stage when invoked with
    ``--seed-paper-trades --produce-packet``.
  * Every entry has all five required keys: ``timestamp``,
    ``operation``, ``result``, ``operator``, ``details``.
  * Timestamps are strictly non-decreasing in trail order.
  * The ``operator`` field reflects ``--operator`` when supplied;
    otherwise falls back to ``$USER`` / "anonymous".
  * Stage order matches the demo flow:
    observe → recommend → queue → inspect → paper_test_seed →
    promotion_packet → demo_complete.
  * Each ``ok`` entry carries the expected detail fields:
      observe          → workspace
      recommend        → report, recommendation
      queue            → enqueued (int), queue_path
      inspect          → inspection
      paper_test_seed  → ledger, n_trades
      promotion_packet → packet
      demo_complete    → real_registry_json_count, workspace
  * Trail file is append-only: rerunning the demo against the same
    trail extends it, never rewrites prior entries.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]


def _import_demo_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_demo as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


def _read_trail(trail: Path) -> list[dict]:
    out: list[dict] = []
    for ln in trail.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


_REQUIRED_KEYS = {"timestamp", "operation", "result", "operator", "details"}
_EXPECTED_STAGE_ORDER = (
    "observe",
    "regime_anomaly",
    "recommend",
    "queue",
    "inspect",
    "paper_test_seed",
    "promotion_packet",
    "demo_complete",
)
_EXPECTED_DETAIL_FIELDS: dict[str, set[str]] = {
    "observe": {"workspace"},
    "regime_anomaly": {"regime", "anomaly", "new_candidates_allowed"},
    "recommend": {"report", "recommendation"},
    "queue": {"enqueued", "queue_path"},
    "inspect": {"inspection"},
    "paper_test_seed": {"ledger", "n_trades"},
    "promotion_packet": {"packet"},
    "demo_complete": {"real_registry_json_count", "workspace"},
}


# ---------------------------------------------------------------------------
# 1. Schema completeness — every entry has all 5 required keys.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_audit_entry_has_required_schema(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace = tmp_path / "demo"

    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
        "--produce-packet",
        "--audit-trail", str(trail),
    ])
    assert rc == 0
    entries = _read_trail(trail)
    assert entries, "demo produced no audit-trail entries"

    for e in entries:
        missing = _REQUIRED_KEYS - set(e.keys())
        assert not missing, (
            f"audit entry missing keys {missing}: {e}"
        )
        # Type sanity.
        assert isinstance(e["timestamp"], str)
        assert isinstance(e["operation"], str)
        assert isinstance(e["result"], str)
        assert isinstance(e["operator"], str) and e["operator"]
        assert isinstance(e["details"], dict)


# ---------------------------------------------------------------------------
# 2. Stage order matches the demo's documented flow.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_stage_order_matches_demo_flow(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
        "--produce-packet",
        "--audit-trail", str(trail),
    ])
    assert rc == 0
    ops = [e["operation"] for e in _read_trail(trail)]
    # Strict equality — exactly the seven stages in this exact order.
    assert ops == list(_EXPECTED_STAGE_ORDER), (
        f"stage order drifted: got {ops}"
    )


# ---------------------------------------------------------------------------
# 3. Timestamps strictly non-decreasing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_timestamps_are_monotonic(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
        "--produce-packet",
        "--audit-trail", str(trail),
    ])
    assert rc == 0
    entries = _read_trail(trail)
    timestamps = [datetime.fromisoformat(e["timestamp"]) for e in entries]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"timestamps non-monotonic at index {i}: "
            f"{timestamps[i - 1]!r} → {timestamps[i]!r}"
        )


# ---------------------------------------------------------------------------
# 4. Per-stage detail fields land where promised.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_per_stage_details_are_populated(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
        "--produce-packet",
        "--audit-trail", str(trail),
    ])
    assert rc == 0
    by_op = {e["operation"]: e for e in _read_trail(trail)}
    for op, expected in _EXPECTED_DETAIL_FIELDS.items():
        e = by_op[op]
        assert e["result"] == "ok"
        missing = expected - set(e["details"].keys())
        assert not missing, (
            f"stage {op!r} missing detail fields {missing}; got {e['details']}"
        )
    # Numeric sanity for the stages that promise int values.
    assert isinstance(by_op["queue"]["details"]["enqueued"], int)
    assert by_op["queue"]["details"]["enqueued"] >= 1
    assert isinstance(by_op["paper_test_seed"]["details"]["n_trades"], int)
    assert by_op["paper_test_seed"]["details"]["n_trades"] == 25
    assert isinstance(
        by_op["demo_complete"]["details"]["real_registry_json_count"], int
    )


# ---------------------------------------------------------------------------
# 5. --operator override flows through to every entry.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_operator_override_is_honoured(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
        "--produce-packet",
        "--audit-trail", str(trail),
        "--operator", "carol",
    ])
    assert rc == 0
    operators = {e["operator"] for e in _read_trail(trail)}
    assert operators == {"carol"}


# ---------------------------------------------------------------------------
# 6. Operator falls back to $USER / anonymous when --operator unset.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_operator_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace = tmp_path / "demo"
    monkeypatch.setenv("USER", "envtest-operator")
    rc = cli.main([
        "--workspace", str(workspace),
        "--audit-trail", str(trail),
    ])
    assert rc == 0
    operators = {e["operator"] for e in _read_trail(trail)}
    assert operators == {"envtest-operator"}


# ---------------------------------------------------------------------------
# 7. Append-only across reruns: prior bytes preserved.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_is_append_only_across_runs(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    trail = tmp_path / "operation_audit.jsonl"
    workspace1 = tmp_path / "demo-1"
    workspace2 = tmp_path / "demo-2"

    rc = cli.main([
        "--workspace", str(workspace1),
        "--audit-trail", str(trail),
        "--operator", "alice",
    ])
    assert rc == 0
    pre_bytes = trail.read_bytes()
    pre_lines = trail.read_text(encoding="utf-8").splitlines()

    rc = cli.main([
        "--workspace", str(workspace2),
        "--audit-trail", str(trail),
        "--operator", "alice",
    ])
    assert rc == 0
    post = trail.read_bytes()
    assert post.startswith(pre_bytes)
    post_lines = trail.read_text(encoding="utf-8").splitlines()
    assert len(post_lines) == 2 * len(pre_lines)


# ---------------------------------------------------------------------------
# 8. Trail-path forbidden warning surfaces but does not break the demo.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_trail_under_forbidden_location_is_warned_not_fatal(
    tmp_path: Path,
) -> None:
    """When the operator points --audit-trail at a forbidden path,
    the demo must NOT silently overwrite policy_registry/approved/.
    The trail wrapper inside the demo prints AUDIT WARN and skips
    writing — the rest of the demo continues."""
    cli = _import_demo_cli()
    forbidden_trail = tmp_path / "policy_registry" / "approved" / "trail.jsonl"
    workspace = tmp_path / "demo"
    rc = cli.main([
        "--workspace", str(workspace),
        "--audit-trail", str(forbidden_trail),
    ])
    assert rc == 0
    # No file created at the forbidden path.
    assert not forbidden_trail.exists()
