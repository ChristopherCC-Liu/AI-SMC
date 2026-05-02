"""Stage 6-followup task 1 — safety_bounds template integration tests.

Pinned guarantees:

  * The sidecar template lives at
    ``config/safety_bounds_template.yaml`` (note the ``_template``
    suffix). The canonical live file ``config/safety_bounds.yaml``
    remains a red-line — this template is what the operator copies.
  * The template parses to a :class:`SafetyBoundsConfig` covering
    every dotted target named in ``CANDIDATE_MENU_V0``.
  * For every menu candidate, the menu's ``proposed_value`` falls
    inside the template's band. G6 must therefore PASS — *not*
    return ``safety_bound_undefined``.
  * Template bands match the RFC §5 / Stage-3 ``SAFETY_CLAMPS`` table
    so the candidate generator's clamp and the gate's safety check
    agree.
  * Loading the template through the existing report-CLI loader
    yields the same band map.
  * The template file does NOT live at ``config/safety_bounds.yaml``
    (the red-line path). The test asserts the template path differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]
_TEMPLATE = _REPO / "config" / "safety_bounds_template.yaml"
_LIVE_PATH = _REPO / "config" / "safety_bounds.yaml"


def _import_report_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_report as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


# ---------------------------------------------------------------------------
# 1. Template exists at the sidecar path; live path stays untouched.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_lives_at_sidecar_path_not_live_path() -> None:
    assert _TEMPLATE.exists(), (
        f"template missing at {_TEMPLATE}; operators have nothing to copy"
    )
    assert _TEMPLATE.name == "safety_bounds_template.yaml"
    # Red-line: the live path remains absent. Operators must copy
    # the template manually after a security review.
    assert not _LIVE_PATH.exists(), (
        "config/safety_bounds.yaml exists — the red-line live path "
        "should remain absent until the operator manually copies the "
        "template"
    )


# ---------------------------------------------------------------------------
# 2. Template parses to a SafetyBoundsConfig covering every menu target.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_covers_every_menu_target() -> None:
    cli = _import_report_cli()
    cfg = cli.load_safety_bounds(_TEMPLATE)
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0

    targets = {c.diff.target for c in CANDIDATE_MENU_V0}
    missing = targets - set(cfg.bounds)
    assert not missing, (
        f"template missing safety bounds for: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 3. Every menu's proposed_value lies inside the template band → G6 PASS.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_menu_proposed_value_is_in_band() -> None:
    cli = _import_report_cli()
    cfg = cli.load_safety_bounds(_TEMPLATE)
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0

    for c in CANDIDATE_MENU_V0:
        lo, hi = cfg.bounds[c.diff.target]
        pv = float(c.diff.proposed_value)  # type: ignore[arg-type]
        assert lo <= pv <= hi, (
            f"{c.candidate_id}: proposed={pv} outside band [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# 4. Running G6 against the template returns PASS for every candidate
#    (instead of safety_bound_undefined).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g6_passes_for_every_candidate_with_template() -> None:
    cli = _import_report_cli()
    cfg = cli.load_safety_bounds(_TEMPLATE)
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.promotion_gates import g6_safety_bounds

    for c in CANDIDATE_MENU_V0:
        result = g6_safety_bounds(candidate=c, bounds=cfg)
        assert result.gate_id == "G6"
        assert result.status.value == "PASS", (
            f"{c.candidate_id}: G6 returned {result.status.value} "
            f"reason={result.reason!r}"
        )
        # Make sure we are NOT seeing the fail-safe undefined branch.
        assert "safety_bound_undefined" not in result.reason


# ---------------------------------------------------------------------------
# 5. Template bands agree with the Stage-3 SAFETY_CLAMPS table.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_matches_stage3_safety_clamps() -> None:
    cli = _import_report_cli()
    cfg = cli.load_safety_bounds(_TEMPLATE)
    from smc.hedgerock.evolution.candidate_generator import SAFETY_CLAMPS

    target_to_class = {
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR":
            "confidence_threshold_observe",
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR":
            "confidence_threshold_aggressive",
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE":
            "confidence_threshold_range_2",
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE":
            "halt_expiry_observe_hours",
    }
    for target, cls in target_to_class.items():
        clamp = SAFETY_CLAMPS[cls]
        lo, hi = cfg.bounds[target]
        assert (lo, hi) == (clamp.lo, clamp.hi), (
            f"template band for {target} = [{lo}, {hi}] disagrees "
            f"with SAFETY_CLAMPS[{cls!r}] = [{clamp.lo}, {clamp.hi}]"
        )


# ---------------------------------------------------------------------------
# 6. Template carries explicit XAUUSD-only header + read-only banner.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_carries_xauusd_only_and_read_only_banner() -> None:
    body = _TEMPLATE.read_text(encoding="utf-8")
    assert "XAUUSD" in body, "template missing XAUUSD-only marker"
    # Operator banner — make absolutely sure the file is not the
    # live red-line path.
    assert "TEMPLATE" in body.upper() or "template" in body
    assert "safety_bounds.yaml" in body, (
        "template should mention the live filename it pairs with"
    )
