"""Calibration runner for R10 post-bake A/B promotion verdict.

Reads a journal of trade entries (jsonl), splits them into A/B arms via
:func:`smc.eval.calibration_arms.classify_trades`, computes per-arm:

- Profit factor (PF) and bootstrap 95% CI lower bound
- Win rate + chi-square p-value vs baseline (informational only)
- Trade count
- Consec-loss histogram (occurrences of each loss_count_in_window value
  at trade-open, used by P3.1 to derive its sizing decay schedule)

Then invokes :func:`smc.eval.promotion_gate.evaluate_promotion` to render
a binary PROMOTE / HOLD verdict per the R10 P4.2 statistical rule.

Output: a single JSON file at
``.scratch/round10/calibration_<bake_window>.json`` with the shape::

    {
      "control": {
        "pf": 1.05, "ci_lower": 0.92, "ci_upper": 1.18, "n_trades": 47,
        "consec_loss_distribution": {"0": 32, "1": 8, ...}
      },
      "treatment": {...},
      "verdict": {
        "promote": false, "rationale": "...", "pf_diff": 0.13,
        "pf_diff_ci": [-0.05, 0.31], "wr_chi2_p": 0.42
      },
      "diagnostics": {
        "_control_with_unexpected_flag_on": 0,
        "_treatment_with_missing_flag": 1,
        "_unmatched": 0
      }
    }

Backward-compat: gracefully handles pre-cal-0 (no ``flags``) and
pre-cal-0b (no ``loss_count_in_window_at_open``) journal entries via
defense-impl-lead's robust extractors. Mixed pre/post journals work.

Usage::

    python scripts/run_promotion_gate.py \\
        --journal data/journal/live_trades.jsonl \\
        --bake-window 2026-04-26 \\
        --control-magic 19760418 --treatment-magic 19760428
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smc.eval.calibration_arms import (
    ArmDefinition,
    ClassificationResult,
    TradeView,
    classify_trades,
)
from smc.eval.promotion_gate import PromotionVerdict, evaluate_promotion

DEFAULT_OUTPUT_DIR = Path(".scratch") / "round10"
DEFAULT_CONTROL_MAGIC = 19760418
DEFAULT_TREATMENT_MAGIC = 19760428
ARM_CONTROL = "control"
ARM_TREATMENT = "treatment"


def load_journal(path: Path) -> list[dict[str, Any]]:
    """Parse a jsonl journal file into a list of dicts.

    Skips blank lines (common at end of file). Lines that don't parse as
    JSON are surfaced as a hard error rather than silently skipped — a
    malformed journal entry indicates an upstream writer bug, and silent
    skip would bias the calibration verdict.
    """
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed journal entry at {path}:{lineno}: {exc}"
                ) from exc
    return entries


def build_default_arms(
    *,
    control_magic: int,
    treatment_magic: int,
) -> list[ArmDefinition]:
    """Construct the canonical Phase-1 control/treatment arm definitions.

    Both arms use ``"strict"`` matching: every R10 flag must equal the
    expected value. Phase 2 ablation studies will define additional arms
    with ``"subset"`` matching for cells that vary only one or two flags.
    """
    control_flags = {
        "range_trend_filter_enabled": False,
        "range_ai_regime_gate_enabled": False,
        "spread_gate_enabled": False,
        "persistent_dd_breaker_enabled": False,
        "consec_loss_window_size": 3,
    }
    treatment_flags = {
        "range_trend_filter_enabled": True,
        "range_ai_regime_gate_enabled": True,
        "spread_gate_enabled": True,
        "persistent_dd_breaker_enabled": True,
        "consec_loss_window_size": 6,
    }
    return [
        ArmDefinition(
            arm_label=ARM_CONTROL,
            required_flags=control_flags,
            required_magic=control_magic,
            matching_mode="strict",
        ),
        ArmDefinition(
            arm_label=ARM_TREATMENT,
            required_flags=treatment_flags,
            required_magic=treatment_magic,
            matching_mode="strict",
        ),
    ]


def per_arm_metrics(arm_trades: Sequence[TradeView]) -> dict[str, Any]:
    """Compute per-arm summary stats in a single iteration over the trades.

    Single-pass DRY: PF + CI come from ``evaluate_promotion`` (one bootstrap
    run per arm pair); the histogram is accumulated here in O(n) over the
    arm's trade list. Both consume the same TradeView Sequence, so we do
    NOT re-iterate the journal.

    Histogram bucket key is the ``loss_count_in_window_at_open`` cast to
    a string (JSON-friendly). Trades from pre-cal-0b legacy records (None
    value) are bucketed under the ``"unknown"`` key so the runner remains
    truthful about partial data — silently dropping them would skew the
    histogram by an unknown amount.
    """
    histogram: Counter[str] = Counter()
    for trade in arm_trades:
        bucket = (
            str(trade.loss_count_in_window_at_open)
            if trade.loss_count_in_window_at_open is not None
            else "unknown"
        )
        histogram[bucket] += 1
    return {"consec_loss_distribution": dict(histogram)}


def compose_arm_record(
    arm_trades: Sequence[TradeView],
    *,
    pf: float,
    ci_lower: float,
    ci_upper: float,
) -> dict[str, Any]:
    """Merge per-arm PF/CI/n with the consec-loss histogram into one dict."""
    record = {
        "pf": _serialize_float(pf),
        "ci_lower": _serialize_float(ci_lower),
        "ci_upper": _serialize_float(ci_upper),
        "n_trades": len(arm_trades),
    }
    record.update(per_arm_metrics(arm_trades))
    return record


def _serialize_float(value: float) -> float | str:
    """Convert ``inf`` / ``nan`` to JSON-safe sentinel strings.

    ``json.dumps(float("inf"))`` produces invalid JSON (``Infinity``),
    so we preempt by emitting the string ``"inf"`` / ``"nan"``.
    Downstream consumers (the calibration JSON reader) parse these
    strings back to floats if needed; the round-trip preserves enough
    information for the runner's purposes (decision = promote/hold).
    """
    if value != value:  # NaN sentinel
        return "nan"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"
    return value


def compose_diagnostics_record(result: ClassificationResult) -> dict[str, int]:
    """Reduce the ClassificationResult.diagnostics to a count-only summary.

    The runner output stores counts not full trade lists; the trade-level
    detail stays inside the in-memory ClassificationResult and is available
    to test fixtures via the classifier's return value. This keeps the
    JSON output small + grep-friendly for ops triage.
    """
    return {bucket: len(trades) for bucket, trades in result.diagnostics.items()}


def compose_verdict_record(verdict: PromotionVerdict) -> dict[str, Any]:
    """Reduce the PromotionVerdict dataclass to a JSON-friendly dict."""
    return {
        "promote": verdict.promote,
        "rationale": verdict.rationale,
        "pf_diff": _serialize_float(verdict.pf_diff),
        "pf_diff_ci": [
            _serialize_float(verdict.pf_diff_ci[0]),
            _serialize_float(verdict.pf_diff_ci[1]),
        ],
        "wr_chi2_p": _serialize_float(verdict.wr_chi2_p),
    }


def run_calibration(
    journal_entries: Sequence[dict[str, Any]],
    *,
    control_magic: int = DEFAULT_CONTROL_MAGIC,
    treatment_magic: int = DEFAULT_TREATMENT_MAGIC,
    n_iter: int = 10_000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Run the full calibration pipeline on a journal entry list.

    Returns the assembled output dict ready for ``json.dump``. Pure
    function — does not write to disk; the CLI wrapper handles that.
    """
    arms = build_default_arms(
        control_magic=control_magic,
        treatment_magic=treatment_magic,
    )
    classified = classify_trades(
        journal_entries,
        arm_definitions=arms,
        control_magic=control_magic,
        treatment_magic=treatment_magic,
    )
    control_trades = classified.arms.get(ARM_CONTROL, ())
    treatment_trades = classified.arms.get(ARM_TREATMENT, ())

    verdict = evaluate_promotion(
        baseline_trades=control_trades,
        treatment_trades=treatment_trades,
        n_iter=n_iter,
        rng_seed=rng_seed,
    )
    ci_lower, ci_upper = verdict.pf_diff_ci

    return {
        ARM_CONTROL: compose_arm_record(
            control_trades,
            pf=verdict.pf_baseline,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ),
        ARM_TREATMENT: compose_arm_record(
            treatment_trades,
            pf=verdict.pf_treatment,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ),
        "verdict": compose_verdict_record(verdict),
        "diagnostics": compose_diagnostics_record(classified),
    }


def write_output(
    payload: dict[str, Any],
    *,
    bake_window: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write the calibration JSON to disk + return the path written.

    Creates ``output_dir`` if missing. Filename is
    ``calibration_<bake_window>.json``; ``bake_window`` is interpreted
    as a freeform identifier (typically a UTC date) so the writer does
    not constrain its format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"calibration_{bake_window}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R10 calibration runner: split journal into A/B arms, "
        "compute PF/CI/histogram, render PROMOTE/HOLD verdict.",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        required=True,
        help="Path to the jsonl journal of trade entries.",
    )
    parser.add_argument(
        "--bake-window",
        required=True,
        help="Identifier for the bake window (used in output filename).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--control-magic",
        type=int,
        default=DEFAULT_CONTROL_MAGIC,
        help=f"MT5 magic for control leg (default: {DEFAULT_CONTROL_MAGIC}).",
    )
    parser.add_argument(
        "--treatment-magic",
        type=int,
        default=DEFAULT_TREATMENT_MAGIC,
        help=f"MT5 magic for treatment leg (default: {DEFAULT_TREATMENT_MAGIC}).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10_000,
        help="Bootstrap iterations for PF CI (default: 10000).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed for reproducibility (default: 0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on error."""
    args = _build_parser().parse_args(argv)
    if not args.journal.exists():
        print(f"ERROR: journal not found: {args.journal}", file=sys.stderr)
        return 2
    entries = load_journal(args.journal)
    payload = run_calibration(
        entries,
        control_magic=args.control_magic,
        treatment_magic=args.treatment_magic,
        n_iter=args.n_iter,
        rng_seed=args.rng_seed,
    )
    out_path = write_output(
        payload,
        bake_window=args.bake_window,
        output_dir=args.output_dir,
    )
    print(f"[CALIBRATION-VERDICT] wrote {out_path}")
    print(
        f"  promote={payload['verdict']['promote']} "
        f"pf_diff={payload['verdict']['pf_diff']} "
        f"ci={payload['verdict']['pf_diff_ci']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
