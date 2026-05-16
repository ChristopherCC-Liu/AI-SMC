"""Stage 6-followup-4 task 2 — config validator (read-only).

Validates ``config/safety_bounds_template.yaml`` (or any operator-
supplied bounds file) before deployment:

  * Every dotted target named in ``CANDIDATE_MENU_V0`` must have a
    band entry.
  * Every band must be a ``[lo, hi]`` list of two numerics with
    ``lo < hi``.
  * Every band SHOULD agree with the Stage-3 ``SAFETY_CLAMPS``
    table; disagreement is a WARN by default, FAIL under
    ``--strict``.

Read-only: never mutates the input file. ``--report-path`` is
optional — when present, writes a markdown report; the path MUST
NOT live under ``policy_registry/approved/`` or
``policy_registry/pointer.json``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from smc.hedgerock.evolution.candidate_generator import SAFETY_CLAMPS
from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


_TARGET_TO_CLASS = {
    "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR":
        "confidence_threshold_observe",
    "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR":
        "confidence_threshold_aggressive",
    "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE":
        "confidence_threshold_range_2",
    "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE":
        "halt_expiry_observe_hours",
}


def _assert_report_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"report-path lands under a forbidden location: "
                f"{text!r} (matched {fragment!r})"
            )


def _validate(
    *,
    config_path: Path, strict: bool,
) -> tuple[int, list[str], list[str]]:
    """Return (errors_count, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not config_path.exists():
        return 1, [f"config file missing: {config_path}"], []

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return 1, [f"yaml parse failed: {e}"], []

    if not isinstance(raw, dict):
        return 1, ["config root must be a mapping (dict)"], []

    # Coverage: every menu target present.
    menu_targets = {c.diff.target for c in CANDIDATE_MENU_V0}
    missing = sorted(menu_targets - set(raw))
    for t in missing:
        errors.append(f"missing target: {t}")

    # Per-target shape + ordering.
    for target, value in raw.items():
        if target not in menu_targets:
            warnings.append(
                f"unknown target (not in CANDIDATE_MENU_V0): {target}"
            )
            continue
        if not isinstance(value, (list, tuple)):
            errors.append(
                f"{target}: value must be a 2-list [lo, hi], got "
                f"{type(value).__name__}"
            )
            continue
        if len(value) != 2:
            errors.append(
                f"{target}: value must have exactly 2 elements, got "
                f"{len(value)}"
            )
            continue
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError):
            errors.append(
                f"{target}: lo/hi must be numeric, got {value!r}"
            )
            continue
        if not (lo < hi):
            errors.append(
                f"{target}: require lo < hi; got lo={lo}, hi={hi}"
            )
            continue
        # SAFETY_CLAMPS agreement.
        cls = _TARGET_TO_CLASS.get(target)
        if cls is not None and cls in SAFETY_CLAMPS:
            clamp = SAFETY_CLAMPS[cls]
            if (lo, hi) != (clamp.lo, clamp.hi):
                msg = (
                    f"{target}: range [{lo}, {hi}] disagrees with "
                    f"SAFETY_CLAMPS[{cls!r}] = [{clamp.lo}, {clamp.hi}]"
                )
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    return len(errors), errors, warnings


def _render_report(
    *, config_path: Path, errors: list[str], warnings: list[str],
    strict: bool,
) -> str:
    out: list[str] = []
    out.append("# HedgeRock Safety-Bounds Config Validation (READ-ONLY)")
    out.append("")
    out.append(f"- config_path: `{config_path}`")
    out.append(f"- strict mode: **{strict}**")
    if not errors:
        out.append("- result: **PASS**")
    else:
        out.append("- result: **FAIL**")
    out.append(f"- errors: {len(errors)}")
    out.append(f"- warnings: {len(warnings)}")
    out.append("")
    if errors:
        out.append("## Errors")
        out.append("")
        for e in errors:
            out.append(f"- {e}")
        out.append("")
    if warnings:
        out.append("## Warnings")
        out.append("")
        for w in warnings:
            out.append(f"- {w}")
        out.append("")
    out.append("## Targets covered (CANDIDATE_MENU_V0)")
    out.append("")
    for c in CANDIDATE_MENU_V0:
        out.append(f"- `{c.diff.target}` (candidate: `{c.candidate_id}`)")
    out.append("")
    out.append("## Boundary boilerplate")
    out.append("")
    out.append("- this validator is read-only; it never modifies the "
               "input file or any production registry path")
    out.append("")
    out.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to safety_bounds YAML file.")
    parser.add_argument("--strict", action="store_true",
                        help="Treat SAFETY_CLAMPS disagreements as errors.")
    parser.add_argument("--report-path", type=Path, default=None,
                        help="Optional path for a markdown report.")
    args = parser.parse_args(argv)

    if args.report_path is not None:
        try:
            _assert_report_path_safe(Path(args.report_path))
        except ValueError as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 1

    err_count, errors, warnings = _validate(
        config_path=Path(args.config), strict=args.strict,
    )
    print(f"config: {args.config}")
    print(f"strict: {args.strict}")
    print(f"errors: {err_count}")
    print(f"warnings: {len(warnings)}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARN: {w}")
    print("PASS" if err_count == 0 else "FAIL")

    if args.report_path is not None:
        body = _render_report(
            config_path=Path(args.config),
            errors=errors, warnings=warnings, strict=args.strict,
        )
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_path).write_text(body, encoding="utf-8")

    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
