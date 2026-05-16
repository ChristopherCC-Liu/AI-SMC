"""Stage 6-followup-4 task 2 — config validator CLI tests.

Pinned guarantees:

  * The validator passes against the shipped
    ``config/safety_bounds_template.yaml`` (every menu target
    covered, ranges agree with SAFETY_CLAMPS).
  * Validator detects: missing target, lo > hi, non-numeric values,
    non-list value, range disagreement with SAFETY_CLAMPS.
  * Validator output is deterministic — same input → same exit
    code + same stdout.
  * The CLI exits 0 on PASS, 1 on FAIL. Neither path mutates the
    input file (read-only).
  * The CLI never imports live runtime modules.
  * The CLI never writes outside an optional ``--report-path``
    that the operator supplies; refuses to write under
    ``policy_registry/approved/`` or ``policy_registry/pointer.json``.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]
_TEMPLATE = _REPO / "config" / "safety_bounds_template.yaml"


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_validate_config as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


# ---------------------------------------------------------------------------
# 1. Shipped template passes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_passes_validator() -> None:
    cli = _import_cli()
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main(["--config", str(_TEMPLATE)])
    assert rc == 0
    out = stdout.getvalue()
    assert "PASS" in out
    assert "errors: 0" in out.lower() or "0 error" in out.lower()


@pytest.mark.unit
def test_template_validator_does_not_mutate_template() -> None:
    cli = _import_cli()
    pre_bytes = _TEMPLATE.read_bytes()
    pre_mtime = _TEMPLATE.stat().st_mtime_ns
    cli.main(["--config", str(_TEMPLATE)])
    assert _TEMPLATE.read_bytes() == pre_bytes
    assert _TEMPLATE.stat().st_mtime_ns == pre_mtime


# ---------------------------------------------------------------------------
# 2. Missing target → fail.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_target_fails(tmp_path: Path) -> None:
    cli = _import_cli()
    bad = tmp_path / "bounds.yaml"
    bad.write_text(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR: [0.40, 0.65]\n",
        encoding="utf-8",
    )
    rc = cli.main(["--config", str(bad)])
    assert rc != 0


# ---------------------------------------------------------------------------
# 3. lo > hi → fail with explicit message.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lo_greater_than_hi_fails(tmp_path: Path) -> None:
    cli = _import_cli()
    bad = tmp_path / "bounds.yaml"
    bad.write_text(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR: [0.65, 0.40]\n"
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR: [0.70, 0.85]\n"
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE: [0.55, 0.80]\n"
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE: [4.0, 48.0]\n",
        encoding="utf-8",
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main(["--config", str(bad)])
    assert rc != 0
    out = stdout.getvalue()
    assert "lo" in out.lower() and "hi" in out.lower()


# ---------------------------------------------------------------------------
# 4. Non-numeric range → fail.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_numeric_range_fails(tmp_path: Path) -> None:
    cli = _import_cli()
    bad = tmp_path / "bounds.yaml"
    bad.write_text(
        'smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR: ["a", "b"]\n'
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR: [0.70, 0.85]\n"
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE: [0.55, 0.80]\n"
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE: [4.0, 48.0]\n",
        encoding="utf-8",
    )
    rc = cli.main(["--config", str(bad)])
    assert rc != 0


# ---------------------------------------------------------------------------
# 5. Non-list value → fail.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_list_value_fails(tmp_path: Path) -> None:
    cli = _import_cli()
    bad = tmp_path / "bounds.yaml"
    bad.write_text(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR: 0.50\n"
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR: [0.70, 0.85]\n"
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE: [0.55, 0.80]\n"
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE: [4.0, 48.0]\n",
        encoding="utf-8",
    )
    rc = cli.main(["--config", str(bad)])
    assert rc != 0


# ---------------------------------------------------------------------------
# 6. Range disagreement with SAFETY_CLAMPS → warning by default,
#    fail under --strict.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clamp_disagreement_warns_by_default_fails_under_strict(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    bad = tmp_path / "bounds.yaml"
    # Drift the observe floor lo from 0.40 to 0.30 — within sane
    # numeric range, but out of agreement with SAFETY_CLAMPS.
    bad.write_text(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR: [0.30, 0.65]\n"
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR: [0.70, 0.85]\n"
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE: [0.55, 0.80]\n"
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE: [4.0, 48.0]\n",
        encoding="utf-8",
    )
    # Default mode: WARN, exit 0.
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main(["--config", str(bad)])
    assert rc == 0
    assert "WARN" in stdout.getvalue() or "warning" in stdout.getvalue().lower()
    # Strict mode: FAIL, exit 1.
    rc_strict = cli.main(["--config", str(bad), "--strict"])
    assert rc_strict != 0


# ---------------------------------------------------------------------------
# 7. Optional --report-path writes a markdown report.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_writes_report_when_report_path_supplied(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    report = tmp_path / "config-validation.md"
    rc = cli.main(["--config", str(_TEMPLATE), "--report-path", str(report)])
    assert rc == 0
    body = report.read_text(encoding="utf-8")
    assert "PASS" in body
    assert "menu" in body.lower() or "target" in body.lower()


# ---------------------------------------------------------------------------
# 8. Report-path under approved/ or pointer.json is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_rejects_forbidden_report_path(tmp_path: Path) -> None:
    cli = _import_cli()
    bad_report = tmp_path / "policy_registry" / "approved" / "report.md"
    rc = cli.main([
        "--config", str(_TEMPLATE),
        "--report-path", str(bad_report),
    ])
    assert rc != 0


# ---------------------------------------------------------------------------
# 9. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_script_has_no_live_runtime_imports() -> None:
    src = (
        _REPO / "scripts" / "hedgerock_evolution_validate_config.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 10. Deterministic output.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_output_is_deterministic(tmp_path: Path) -> None:
    cli = _import_cli()
    out_a, out_b = io.StringIO(), io.StringIO()
    with redirect_stdout(out_a):
        cli.main(["--config", str(_TEMPLATE)])
    with redirect_stdout(out_b):
        cli.main(["--config", str(_TEMPLATE)])
    # Strip a possible trailing timestamp line — but be defensive.
    a = out_a.getvalue().splitlines()
    b = out_b.getvalue().splitlines()
    # Filter out any line that looks like a timestamp.
    drop = lambda lines: [ln for ln in lines if "Generated" not in ln]
    assert drop(a) == drop(b)
