"""Stage 6-followup-2 task 4 — operation audit trail tests.

Pinned guarantees:

  * ``append_operation`` writes ONE JSONL line per call. The line
    contains: timestamp (ISO 8601, UTC), operation, result, operator,
    details (free-form dict — must round-trip through json.dumps).
  * The trail file is append-only. Repeated calls grow the file
    monotonically; previous lines stay byte-identical.
  * Trail path under ``policy_registry/approved/``,
    ``policy_registry/pointer.json``, or any path that contains the
    canonical ``policy_registry/shadow_artefacts/`` substring is
    rejected with ``ValueError`` — operation-trail and the
    shadow-artefact registry audit log are deliberately separate
    files.
  * Operator defaults to ``os.environ.get("USER")`` when not given.
  * Module imports no live runtime modules.
  * Demo CLI (``hedgerock_evolution_demo``) gains an optional
    ``--audit-trail`` flag that, when supplied, appends one line per
    successful stage to the trail.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.operation_audit import (
    OperationAuditEntry,
    append_operation,
    read_trail,
)


_REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. Single-line append.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_append_writes_one_jsonl_line(tmp_path: Path) -> None:
    trail = tmp_path / "operation_audit.jsonl"
    entry = append_operation(
        trail_path=trail,
        operation="recommend",
        result="ok",
        operator="alice",
        details={"candidates": 4, "recommended": 3},
    )
    assert isinstance(entry, OperationAuditEntry)
    assert entry.operation == "recommend"
    assert entry.result == "ok"
    assert entry.operator == "alice"

    lines = trail.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["operation"] == "recommend"
    assert d["operator"] == "alice"
    assert d["details"] == {"candidates": 4, "recommended": 3}
    # Timestamp must parse and be in UTC.
    parsed = datetime.fromisoformat(d["timestamp"])
    assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# 2. Append-only — prior lines unchanged after second append.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_repeated_appends_preserve_prior_lines(tmp_path: Path) -> None:
    trail = tmp_path / "trail.jsonl"
    append_operation(
        trail_path=trail, operation="op-a", result="ok", operator="x",
        details={},
    )
    pre = trail.read_text(encoding="utf-8")
    append_operation(
        trail_path=trail, operation="op-b", result="ok", operator="x",
        details={},
    )
    post = trail.read_text(encoding="utf-8")
    assert post.startswith(pre)
    assert post.count("\n") == 2


# ---------------------------------------------------------------------------
# 3. Trail path under approved/, pointer.json, shadow_artefacts is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relpath", [
    "policy_registry/approved/trail.jsonl",
    "policy_registry/pointer.json",
    "policy_registry/shadow_artefacts/_audit.md",
])
def test_trail_path_under_forbidden_locations_is_rejected(
    tmp_path: Path, relpath: str,
) -> None:
    bad = tmp_path / relpath
    with pytest.raises(ValueError):
        append_operation(
            trail_path=bad, operation="x", result="ok", operator="y",
            details={},
        )


# ---------------------------------------------------------------------------
# 4. Operator defaults to USER env var when unset.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_operator_defaults_to_user_env(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "envtest-user")
    trail = tmp_path / "trail.jsonl"
    entry = append_operation(
        trail_path=trail, operation="x", result="ok", operator=None,
        details={},
    )
    assert entry.operator == "envtest-user"


@pytest.mark.unit
def test_operator_falls_back_when_user_env_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("USER", raising=False)
    trail = tmp_path / "trail.jsonl"
    entry = append_operation(
        trail_path=trail, operation="x", result="ok", operator=None,
        details={},
    )
    assert entry.operator  # non-empty
    assert entry.operator in {"anonymous", "unknown"}


# ---------------------------------------------------------------------------
# 5. read_trail roundtrips lines.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_trail_roundtrips_lines(tmp_path: Path) -> None:
    trail = tmp_path / "trail.jsonl"
    append_operation(trail_path=trail, operation="a", result="ok",
                     operator="x", details={"n": 1})
    append_operation(trail_path=trail, operation="b", result="fail",
                     operator="x", details={"n": 2})
    entries = read_trail(trail)
    assert len(entries) == 2
    ops = [e["operation"] for e in entries]
    assert ops == ["a", "b"]


@pytest.mark.unit
def test_read_trail_skips_malformed_lines(tmp_path: Path) -> None:
    trail = tmp_path / "trail.jsonl"
    trail.write_text("not json\n", encoding="utf-8")
    append_operation(trail_path=trail, operation="x", result="ok",
                     operator="x", details={})
    entries = read_trail(trail)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# 6. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_operation_audit_module_has_no_live_runtime_imports() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "operation_audit.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 7. Demo CLI integration — --audit-trail logs every successful stage.
# ---------------------------------------------------------------------------


def _import_demo_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_demo as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


@pytest.mark.unit
def test_demo_writes_audit_trail_on_each_stage(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    workspace = tmp_path / "demo"
    trail = tmp_path / "operation_audit.jsonl"
    rc = cli.main([
        "--workspace", str(workspace),
        "--seed-paper-trades",
        "--produce-packet",
        "--audit-trail", str(trail),
    ])
    assert rc == 0
    assert trail.exists()
    entries = read_trail(trail)
    ops = {e["operation"] for e in entries}
    # Each demo stage should log something.
    expected_substrings = (
        "observe", "recommend", "queue", "inspect",
        "paper_test_seed", "promotion_packet", "demo_complete",
    )
    for sub in expected_substrings:
        assert any(sub in op for op in ops), (
            f"audit trail missing operation matching {sub!r}; "
            f"got ops={ops}"
        )


# ---------------------------------------------------------------------------
# 8. Without --audit-trail, no trail is written.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_demo_without_audit_trail_writes_no_trail(tmp_path: Path) -> None:
    cli = _import_demo_cli()
    workspace = tmp_path / "demo"
    rc = cli.main(["--workspace", str(workspace)])
    assert rc == 0
    # No file at the conventional default; the operator opted out.
    assert not (tmp_path / "operation_audit.jsonl").exists()
