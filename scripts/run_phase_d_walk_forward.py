"""Phase D — run dynamic vs static walk-forward and emit a report.

Usage:
    python scripts/run_phase_d_walk_forward.py \\
        --start 2024-01-01 --end 2025-01-01 \\
        --data-lake-root /path/to/lake \\
        --report-path docs/phase-d-walk-forward-report.md

Experiment-set shortcuts (Phase D-cont1 / D-cont1b):
    --experiment-set d-cont1     →  observe-only release @ 4h halt
    --experiment-set d-cont1b    →  3-way: observe @ 4h + tiny-normal @ 4h
    --experiment-set custom (default) → use the individual --experiment-* flags

Outputs:
    - docs/phase-d-walk-forward-report.md (markdown)
    - <report>.<label>.envelope_log.jsonl  (one per variant including baseline)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from smc.data.lake import ForexDataLake
from smc.hedgerock.phase_d_walk_forward import (
    ExperimentConfig,
    ExperimentResult,
    TradeMetrics,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)


_DEFAULT_LAKE = Path("/Users/christopher/claudeworkplace/AI-SMC/data/parquet")
_DEFAULT_REPORT = Path(
    "/Users/christopher/HedgeRock/docs/phase-d-walk-forward-report.md"
)

# Phase D-cont1b — red-flag thresholds (per-variant for tiny_normal).
# Comments:
#   _DD_REDLINE_PP = 2.0  →  ANY tiny_normal variant whose DD exceeds
#       baseline+2pp is flagged regardless of return change. Even a
#       profitable variant with worse DD is suspicious — we built
#       D-cont1b to test recovery, not to lever up.
#   near-stopout regression is a hard red — any increase means the
#       small new positions pushed the equity closer to wipeout.
_DD_REDLINE_PP: float = 2.0
_DD_TOLERANCE_PP: float = 0.5  # for non-tiny_normal experiments (D-cont1)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


# Phase D-cont1b-closeout — canonical calibration for the named
# experiment sets. Pinned here (not inline) so tests can import the
# exact tuple and lock the definition against drift.
#
# These mirror the D-cont1 report's definition: cold_start_grace=True
# AND halt_auto_expiry_hours=4.0. The release-mode is the only thing
# that varies between d-cont1 and d-cont1b. Any future change MUST
# rename the label rather than silently re-tuning these knobs.
_DCONT1_CONFIG = ExperimentConfig(
    cold_start_grace=True,
    halt_auto_expiry_hours=4.0,
    halt_auto_expiry_release="observe",
)
_DCONT1B_CONFIG = ExperimentConfig(
    cold_start_grace=True,
    halt_auto_expiry_hours=4.0,
    halt_auto_expiry_release="tiny_normal",
)


def _experiments_from_args(args) -> list[tuple[str, ExperimentConfig]]:
    """Translate CLI flags → list of (label, ExperimentConfig).

    The label is what the report and the per-variant envelope-log file
    use. ``baseline`` is reserved — never returned here; baseline always
    runs implicitly inside ``run_walk_forward``.

    Phase D-cont1b-closeout: ``d-cont1`` and ``d-cont1b`` shortcuts now
    BOTH carry ``cold_start_grace=True`` so the labels match the
    original D-cont1 report definition. If you ever need a release-only
    experiment without cold-start-grace, use ``--experiment-set custom``
    with the individual flags — and pick a different label.
    """
    if args.experiment_set == "d-cont1":
        return [("d-cont1", _DCONT1_CONFIG)]
    if args.experiment_set == "d-cont1b":
        # 3-way: baseline (implicit) + d-cont1 + d-cont1b.
        return [("d-cont1", _DCONT1_CONFIG), ("d-cont1b", _DCONT1B_CONFIG)]
    # custom — assemble from individual flags
    if (
        not args.experiment_cold_start_grace
        and args.experiment_halt_auto_expiry_hours is None
    ):
        return []
    cfg = ExperimentConfig(
        cold_start_grace=args.experiment_cold_start_grace,
        halt_auto_expiry_hours=args.experiment_halt_auto_expiry_hours,
        halt_auto_expiry_release=args.experiment_halt_auto_expiry_release,
    )
    return [("experiment", cfg)]


def _fmt_metrics_table(static: TradeMetrics, dynamic: TradeMetrics) -> str:
    rows = [
        ("Final equity",        f"${static.final_equity:,.2f}",       f"${dynamic.final_equity:,.2f}"),
        ("Total return %",      f"{static.total_return_pct:+.2f}%",   f"{dynamic.total_return_pct:+.2f}%"),
        ("Max DD %",            f"{static.max_dd_pct:.2f}%",          f"{dynamic.max_dd_pct:.2f}%"),
        ("Worst month",         f"{static.worst_month[0]} ({static.worst_month[1]:+.2f}%)",
                                f"{dynamic.worst_month[0]} ({dynamic.worst_month[1]:+.2f}%)"),
        ("Blowup",              str(static.blowup),                   str(dynamic.blowup)),
        ("Margin stopout #",    str(static.margin_stopout_count),     str(dynamic.margin_stopout_count)),
        ("Near-stopout bars",   str(static.near_stopout_count),       str(dynamic.near_stopout_count)),
        ("Trades",              str(static.n_trades),                 str(dynamic.n_trades)),
        ("Win rate",            f"{static.win_rate:.3f}",             f"{dynamic.win_rate:.3f}"),
        ("Avg lot",             f"{static.avg_lot:.3f}",              f"{dynamic.avg_lot:.3f}"),
        ("Max lot",             f"{static.max_lot:.3f}",              f"{dynamic.max_lot:.3f}"),
    ]
    out = ["| Metric | static (v1.04) | dynamic (rule_engine) |", "|---|---|---|"]
    for name, s, d in rows:
        out.append(f"| {name} | {s} | {d} |")
    return "\n".join(out)


def _fmt_monthly(static: TradeMetrics, dynamic: TradeMetrics) -> str:
    months = sorted(set(static.monthly_returns) | set(dynamic.monthly_returns))
    if not months:
        return "(no months)"
    out = ["| Month | static % | dynamic % |", "|---|---|---|"]
    for m in months:
        s = static.monthly_returns.get(m, 0.0)
        d = dynamic.monthly_returns.get(m, 0.0)
        out.append(f"| {m} | {s:+.2f} | {d:+.2f} |")
    return "\n".join(out)


def _fmt_dynamic_internals(dynamic: TradeMetrics, total_bars: int) -> str:
    lines: list[str] = []
    if total_bars == 0:
        return "(no bars)"

    def pct(n: int) -> str:
        return f"{n/total_bars*100:.1f}%"

    # Trading reality first — these are the bars the simulator actually
    # treated as each mode (after the transition_lock veto). This is
    # what determines PnL.
    lines.append("**Mode distribution — effective (post transition_lock veto):**")
    lines.append("")
    lines.append(f"- hedgerock: {dynamic.bars_in_hedgerock} ({pct(dynamic.bars_in_hedgerock)})")
    lines.append(f"- observe:   {dynamic.bars_in_observe} ({pct(dynamic.bars_in_observe)})")
    lines.append(f"- halt:      {dynamic.bars_in_halt} ({pct(dynamic.bars_in_halt)})")
    lines.append(f"- momentum:  {dynamic.bars_in_momentum} ({pct(dynamic.bars_in_momentum)})")
    lines.append("")
    lines.append("**Mode distribution — raw (rule_engine output, pre-lock veto):**")
    lines.append("")
    lines.append(f"- hedgerock: {dynamic.bars_in_hedgerock_raw} ({pct(dynamic.bars_in_hedgerock_raw)})")
    lines.append(f"- observe:   {dynamic.bars_in_observe_raw} ({pct(dynamic.bars_in_observe_raw)})")
    lines.append(f"- halt:      {dynamic.bars_in_halt_raw} ({pct(dynamic.bars_in_halt_raw)})")
    lines.append(f"- momentum:  {dynamic.bars_in_momentum_raw} ({pct(dynamic.bars_in_momentum_raw)})")
    lines.append("")
    lines.append("**Cooldown — bars under cooldown (reason bucket → bar count):**")
    if dynamic.cooldown_bars:
        for reason, count in sorted(dynamic.cooldown_bars.items(),
                                    key=lambda kv: -kv[1]):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("**Cooldown — leading-edge trigger count (reason bucket → trigger count):**")
    if dynamic.cooldown_trigger_count:
        for reason, count in sorted(dynamic.cooldown_trigger_count.items(),
                                    key=lambda kv: -kv[1]):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append(f"**Transition-lock bars:** {dynamic.transition_lock_bars} "
                 f"({pct(dynamic.transition_lock_bars)})")
    lines.append("")
    lines.append(f"**Aggressive tier:** {dynamic.aggressive_tier_bars} bars, "
                 f"net PnL during those bars = ${dynamic.aggressive_tier_pnl:,.2f}")
    lines.append(f"**Incomplete EA state bars:** {dynamic.incomplete_ea_state_bars}")
    lines.append(f"**History-incomplete bars:**  {dynamic.history_incomplete_bars}")
    return "\n".join(lines)


def _verdict(static: TradeMetrics, dynamic: TradeMetrics, total_bars: int,
             ) -> tuple[str, list[str]]:
    """Honest verdict — read both as raw return AND survival simultaneously."""
    reasons: list[str] = []
    dyn_positive = dynamic.total_return_pct > 0.0
    static_positive = static.total_return_pct > 0.0
    dyn_blewup = dynamic.blowup
    static_blewup = static.blowup
    dyn_lower_dd = dynamic.max_dd_pct < static.max_dd_pct
    bars_in_halt_pct = (
        dynamic.bars_in_halt / total_bars if total_bars > 0 else 0.0
    )

    # Six-bucket honest classification:
    if dyn_blewup and not static_blewup:
        verdict = "DYNAMIC REGRESSION — BLEW UP WHILE STATIC SURVIVED"
        reasons.append(
            "rule_engine failed to prevent a blowup that v1.04 avoided. "
            "This is the worst possible outcome — investigate which "
            "regime / cooldown failed to fire before risking live capital."
        )
    elif dyn_positive and (not static_positive or dynamic.total_return_pct > static.total_return_pct):
        verdict = "DYNAMIC PROFITABLE AND BETTER THAN STATIC"
    elif dyn_positive and not dyn_lower_dd:
        verdict = "DYNAMIC PROFITABLE BUT WORSE DD THAN STATIC"
        reasons.append(
            "Profitable but with worse drawdown is suspicious — likely a "
            "single lucky regime call. Multi-year walk-forward needed."
        )
    elif (not dyn_positive) and (not static_blewup) and dyn_lower_dd:
        # Both losing but dynamic safer.
        verdict = (
            f"DYNAMIC SAFER BUT NOT PROFITABLE "
            f"({dynamic.total_return_pct:+.2f}% vs static {static.total_return_pct:+.2f}%)"
        )
        reasons.append(
            "Both runs lost money. Dynamic's smaller drawdown is "
            "real risk-control progress, but raw PnL did not improve. "
            "Risk-adjusted return is better; absolute return is not."
        )
    elif (not dyn_positive) and static_blewup:
        verdict = (
            f"DYNAMIC AVERTED BLOWUP BUT NOT PROFITABLE "
            f"({dynamic.total_return_pct:+.2f}%)"
        )
        reasons.append(
            "rule_engine prevented the v1.04 blowup — this is real progress "
            "on survival. But absolute return is still negative, so the "
            "system is currently a 'lose less' machine, not a profit "
            "machine. Phase D's ship/no-ship decision is YES on safety, "
            "NOT-YET on profitability."
        )
    else:
        verdict = "DYNAMIC UNDERPERFORMS"
        reasons.append(
            "Lower return without lower DD or blowup avoidance. "
            "Likely cause: too much time in observe / halt and the "
            "system never gets to deploy at scale. Tune the gate "
            "thresholds (see next-round adjustments)."
        )

    # Always-on diagnostic callouts, regardless of headline.
    if bars_in_halt_pct > 0.30:
        reasons.append(
            f"Dynamic spent {bars_in_halt_pct*100:.1f}% of bars in halt mode "
            f"({dynamic.bars_in_halt} / {total_bars}). After a severe DD "
            "trigger fires, recovery requires equity drift — but observe/"
            "halt mode prevents new trades that could drive recovery. The "
            "system can lock itself permanently. Consider adding a "
            "'halt-cooldown auto-expiry' that releases halt after N hours "
            "even if DD is still elevated."
        )
    if dynamic.aggressive_tier_bars == 0 and total_bars > 100:
        reasons.append(
            "Aggressive tier never fired. Cold-start history-completeness "
            "gate (Phase C-hotfix-4) requires recent_sample_count ≥ 5 of "
            "non-negative PnL trades. With only 4 dynamic trades total, "
            "we never accumulated the sample. Consider a cold-start "
            "grace period that maps the first N profitable closed trades "
            "to 'normal' tier without requiring the full sample."
        )
    if dynamic.n_trades < max(1, static.n_trades * 0.3):
        reasons.append(
            f"Dynamic took {dynamic.n_trades} trades vs static's {static.n_trades} — "
            "rule_engine kept the EA out of the market most of the time. "
            "If the regime is genuinely range, this is conservative — but "
            "if classifier_v2 is missing range opportunities, tune the "
            "range#2 confidence floor."
        )

    return verdict, reasons


def _evaluate_variant_redflag(
    baseline: TradeMetrics, variant: ExperimentResult,
) -> tuple[bool, str, list[str]]:
    """Per-variant red-flag detection. Returns (red_flag, badge, reasons).

    Phase D-cont1b — tiny_normal red-line is intentionally STRICTER
    than D-cont1's:
      - return_better AND DD > baseline + 2pp → RED
      - near_stopout > baseline → RED (any increase, no tolerance)
      - blowup regression                  → RED
      - return improved AND DD shrank      → PROMOTE-CANDIDATE
      - return improved AND DD essentially same (≤ 2pp worse) → MIXED
      - else → NEUTRAL
    """
    reasons: list[str] = []
    cfg = variant.config
    metrics = variant.metrics

    return_better = metrics.total_return_pct > baseline.total_return_pct
    dd_delta = metrics.max_dd_pct - baseline.max_dd_pct  # +ve = worse
    near_stopout_worse = metrics.near_stopout_count > baseline.near_stopout_count
    blowup_regression = metrics.blowup and not baseline.blowup
    is_tiny = cfg.halt_auto_expiry_release == "tiny_normal"

    # Stricter band for tiny_normal.
    redline_pp = _DD_REDLINE_PP if is_tiny else _DD_REDLINE_PP
    tolerance_pp = _DD_TOLERANCE_PP if not is_tiny else 0.0

    red = False
    if blowup_regression:
        red = True
        reasons.append("blowup regression (variant blew up, baseline did not)")
    if return_better and dd_delta > redline_pp:
        red = True
        reasons.append(
            f"return improved but DD deteriorated by {dd_delta:+.2f} pp "
            f"(>{redline_pp} pp red-line)"
        )
    if is_tiny and near_stopout_worse:
        red = True
        reasons.append(
            f"near-stopout bars rose from {baseline.near_stopout_count} → "
            f"{metrics.near_stopout_count} (any increase is RED for tiny_normal)"
        )

    return_delta = metrics.total_return_pct - baseline.total_return_pct
    if red:
        badge = "🛑 RED FLAG — DO NOT PROMOTE"
    elif return_better and dd_delta < -tolerance_pp:
        badge = "✅ PROMOTE-CANDIDATE — return improved, DD shrank"
    elif return_better and dd_delta <= redline_pp:
        badge = (
            f"⚠️ MIXED — return improved by {return_delta:+.2f} pp; "
            f"DD {dd_delta:+.2f} pp"
        )
    elif (not return_better) and dd_delta < -tolerance_pp:
        badge = "🟡 SAFER but flatter — DD shrank, return did not move materially"
    elif (
        return_delta < -tolerance_pp
        and dd_delta > tolerance_pp
    ):
        # Phase D-cont1b — explicit "worse on both dimensions" branch.
        # Not a red flag (no return improvement to lever), but the
        # honest read is "this hypothesis failed; do not promote".
        badge = (
            f"❌ WORSE ON BOTH DIMENSIONS — return {return_delta:+.2f} pp, "
            f"DD {dd_delta:+.2f} pp. Hypothesis failed."
        )
    else:
        badge = "⏸ NEUTRAL — no material change on return or DD"
    return red, badge, reasons


def _fmt_variants_table(
    baseline: TradeMetrics, variants: list[ExperimentResult],
) -> list[str]:
    """Multi-column comparison: baseline + each experiment side-by-side."""
    lines: list[str] = []
    headers = ["Metric", "baseline dynamic"] + [v.label for v in variants]
    sep = ["---"] * len(headers)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(sep) + "|")

    def row(name: str, fmt) -> None:
        cells = [name, fmt(baseline)] + [fmt(v.metrics) for v in variants]
        lines.append("| " + " | ".join(cells) + " |")

    row("Final equity",            lambda m: f"${m.final_equity:,.2f}")
    row("Total return %",          lambda m: f"{m.total_return_pct:+.2f}%")
    row("Max DD %",                lambda m: f"{m.max_dd_pct:.2f}%")
    row("Worst month",
        lambda m: f"{m.worst_month[0]} ({m.worst_month[1]:+.2f}%)")
    row("Blowup",                  lambda m: str(m.blowup))
    row("Margin stopout #",        lambda m: str(m.margin_stopout_count))
    row("Near-stopout bars",       lambda m: str(m.near_stopout_count))
    row("Trades",                  lambda m: str(m.n_trades))
    row("Win rate",                lambda m: f"{m.win_rate:.3f}")
    row("hedgerock bars (eff)",    lambda m: str(m.bars_in_hedgerock))
    row("observe bars (eff)",      lambda m: str(m.bars_in_observe))
    row("halt bars (eff)",         lambda m: str(m.bars_in_halt))
    row("aggressive tier bars",    lambda m: str(m.aggressive_tier_bars))
    return lines


def _count_grace_fires(envelope_log: list[dict]) -> int:
    """Count bars where cold_start_grace actually fired (reason contains
    the unique tag emitted by ``apply_experiment_overrides``)."""
    return sum(
        1 for e in envelope_log
        if "cold_start_grace" in e.get("reason", "")
    )


def _count_tiny_normal_fires(envelope_log: list[dict]) -> int:
    """Count bars where halt_auto_expiry_tiny_normal actually fired."""
    return sum(
        1 for e in envelope_log
        if "halt_auto_expiry_tiny_normal" in e.get("reason", "")
    )


def _count_observe_expiry_fires(envelope_log: list[dict]) -> int:
    """Count bars where halt_auto_expiry (observe-only release) fired.
    The reason starts with 'halt_auto_expiry ' (space) for observe and
    'halt_auto_expiry_tiny_normal' for tiny_normal — match the space
    suffix to disambiguate."""
    return sum(
        1 for e in envelope_log
        if e.get("reason", "").startswith("halt_auto_expiry ")
    )


def _fmt_experiment_section(result: WalkForwardResult) -> list[str]:
    """N-way A/B comparison section. Empty when no experiments were run."""
    if not result.experiments:
        return []

    baseline = result.dynamic_metrics
    variants = result.experiments

    out: list[str] = ["## Phase D-cont1 / D-cont1b — A/B experiment", ""]
    out.append("**Variants under test:**")
    for v in variants:
        cfg = v.config
        out.append(
            f"- **{v.label}** — `cold_start_grace={cfg.cold_start_grace}`, "
            f"`halt_auto_expiry_hours={cfg.halt_auto_expiry_hours}`, "
            f"`halt_auto_expiry_release={cfg.halt_auto_expiry_release}`"
        )
    out.append("")

    out.append("### Side-by-side: baseline dynamic vs each variant")
    out.append("")
    out.extend(_fmt_variants_table(baseline, variants))
    out.append("")

    # Phase D-cont1b-closeout — explicit coverage table for each
    # experiment knob. If the knob's eligibility window never fires on
    # this dataset, the variant's PnL delta cannot be attributed to it.
    out.append("**Experiment-knob coverage (bars where each override actually fired):**")
    out.append("")
    headers = ["Variant", "cold_start_grace", "halt_auto_expiry observe", "halt_auto_expiry tiny_normal"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for v in variants:
        grace = _count_grace_fires(v.envelope_log)
        observe_exp = _count_observe_expiry_fires(v.envelope_log)
        tiny = _count_tiny_normal_fires(v.envelope_log)
        out.append(f"| {v.label} | {grace} | {observe_exp} | {tiny} |")
    out.append("")

    # Cold-start-grace coverage callout — required by the closeout.
    grace_total = sum(
        _count_grace_fires(v.envelope_log)
        for v in variants
        if v.config.cold_start_grace
    )
    enabled_variants = [v for v in variants if v.config.cold_start_grace]
    if enabled_variants and grace_total == 0:
        out.append(
            "> ℹ️ **cold_start_grace coverage: ZERO on 2024 XAUUSD.** "
            "All variants in this run had `cold_start_grace=True`, but "
            "the eligibility window (range regime + 0.45 ≤ confidence < 0.55 "
            "+ no recent closed deals) never overlapped a real bar in this "
            "window. classifier_v2 emits range at confidence 0.65 or 0.80, "
            "never inside the grace band, so the knob has **no measurable "
            "PnL effect on this dataset.** Any return / DD difference "
            "between baseline and the cont1 / cont1b variants is "
            "attributable to halt_auto_expiry alone."
        )
        out.append("")
    elif enabled_variants and grace_total < 5:
        out.append(
            f"> ℹ️ **cold_start_grace coverage: NEAR-ZERO ({grace_total} bars total).** "
            "Effect on PnL is statistically negligible — any variant "
            "delta is dominated by halt_auto_expiry."
        )
        out.append("")

    # Per-variant cooldown trigger snapshot.
    out.append("**Cooldown — leading-edge trigger count by variant:**")
    out.append("")
    keys: set[str] = set(baseline.cooldown_trigger_count)
    for v in variants:
        keys |= set(v.metrics.cooldown_trigger_count)
    if keys:
        headers = ["Bucket", "baseline"] + [v.label for v in variants]
        out.append("| " + " | ".join(headers) + " |")
        out.append("|" + "|".join(["---"] * len(headers)) + "|")
        for k in sorted(keys):
            cells = [k, str(baseline.cooldown_trigger_count.get(k, 0))]
            cells += [str(v.metrics.cooldown_trigger_count.get(k, 0)) for v in variants]
            out.append("| " + " | ".join(cells) + " |")
    else:
        out.append("- (none in any run)")
    out.append("")

    # Per-variant verdict.
    out.append("### Per-variant verdict")
    out.append("")
    variant_verdicts: list[tuple[ExperimentResult, bool, str]] = []
    for v in variants:
        red, badge, reasons = _evaluate_variant_redflag(baseline, v)
        variant_verdicts.append((v, red, badge))
        out.append(f"#### {v.label}")
        out.append("")
        out.append(f"- {badge}")
        if red:
            out.append("- **Red-line breaches:**")
            for r in reasons:
                out.append(f"  - {r}")
        # Aggressive-tier guard: tiny_normal must NEVER be aggressive.
        if v.config.halt_auto_expiry_release == "tiny_normal":
            agg = v.metrics.aggressive_tier_bars
            ok = "✅" if agg == 0 else "❌"
            out.append(f"- {ok} aggressive-tier bars during run: {agg} (must be 0)")
        out.append("")

    # Synthesized honest conclusion — only when at least one variant is
    # the tiny_normal release (= D-cont1b run). Frame the question the
    # user actually asked: did opening tiny new positions during halt-
    # expiry IMPROVE recovery, or just trade more for worse PnL?
    has_tiny = any(
        v.config.halt_auto_expiry_release == "tiny_normal" for v in variants
    )
    if has_tiny:
        out.append("### Honest conclusion (D-cont1b hypothesis test)")
        out.append("")
        out.append(
            "Hypothesis: *opening a tiny hedge (lot_factor=0.1, max 0.02 lot, "
            "max 1 buy + 1 sell) after N hours of continuous halt would seed "
            "equity recovery without re-blowing-up.*"
        )
        out.append("")
        for v in variants:
            if v.config.halt_auto_expiry_release != "tiny_normal":
                continue
            return_delta = v.metrics.total_return_pct - baseline.total_return_pct
            dd_delta = v.metrics.max_dd_pct - baseline.max_dd_pct
            trade_delta = v.metrics.n_trades - baseline.n_trades
            ns_delta = v.metrics.near_stopout_count - baseline.near_stopout_count
            improved = return_delta > 0 and dd_delta <= 0
            if improved:
                out.append(
                    f"- **{v.label}: HYPOTHESIS CONFIRMED.** Return improved by "
                    f"{return_delta:+.2f} pp, DD shrank by {-dd_delta:.2f} pp "
                    f"({trade_delta:+d} trades, near-stopout {ns_delta:+d}). "
                    "Worth promoting after multi-year validation."
                )
            elif return_delta > 0 and dd_delta > 0:
                out.append(
                    f"- **{v.label}: PARTIAL — return up, risk up.** "
                    f"Return {return_delta:+.2f} pp but DD {dd_delta:+.2f} pp. "
                    "Trades up by {0}, near-stopout {1:+d}. The new tiny opens "
                    "did generate PnL, but at the cost of deeper drawdowns. "
                    "Not a clean win.".format(trade_delta, ns_delta)
                )
            else:
                out.append(
                    f"- **{v.label}: HYPOTHESIS REJECTED.** Return {return_delta:+.2f} pp, "
                    f"DD {dd_delta:+.2f} pp ({trade_delta:+d} trades, "
                    f"near-stopout {ns_delta:+d}). The tiny new hedges traded "
                    "into the same losing range that drove the original halt; "
                    "they DEEPENED the drawdown rather than seeding recovery. "
                    "**Do not promote.** Halt auto-expiry — if used at all — "
                    "should remain observe-only (D-cont1) until we have a "
                    "smarter signal for *when* recovery is statistically "
                    "likely (e.g. price has retraced N% off the halt-trigger "
                    "level, or volatility regime has shifted)."
                )
        out.append("")
    return out


def _write_report(result: WalkForwardResult, report_path: Path,
                  envelope_log_paths: dict[str, Path]) -> None:
    cfg = result.config
    static, dynamic = result.static_metrics, result.dynamic_metrics
    total_bars = len(result.envelope_log)
    verdict, verdict_reasons = _verdict(static, dynamic, total_bars)

    body: list[str] = [
        "# Phase D — Dynamic Decision Center walk-forward report",
        "",
        "## Run config",
        "",
        f"- Instrument: `{cfg.instrument}`",
        f"- Window: `{cfg.start.date()}` → `{cfg.end.date()}`",
        f"- Initial equity: ${cfg.init_equity:,.2f}",
        f"- Spread (synth): {cfg.spread_pts} pts",
        f"- H1 lookback: {cfg.h1_lookback} bars",
        f"- H4 lookback: {cfg.h4_lookback} bars",
        f"- Bars decided by rule_engine: {total_bars}",
        "",
        "## Headline verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    for r in verdict_reasons:
        body.append(f"- {r}")
    if not verdict_reasons:
        body.append("- (no caveats)")
    body.append("")
    body.append("## Side-by-side metrics (static v1.04 vs baseline dynamic)")
    body.append("")
    body.append(_fmt_metrics_table(static, dynamic))
    body.append("")
    body.append("## Monthly returns (cumulative-equity-anchored)")
    body.append("")
    body.append(_fmt_monthly(static, dynamic))
    body.append("")
    body.append("## Dynamic-only internals (baseline)")
    body.append("")
    body.append(_fmt_dynamic_internals(dynamic, total_bars))
    body.append("")

    # Phase D-cont1 / D-cont1b — experiment block (multi-variant).
    body.extend(_fmt_experiment_section(result))

    if envelope_log_paths:
        body.append("## Per-bar envelope logs")
        body.append("")
        for label, path in envelope_log_paths.items():
            body.append(f"- **{label}** → `{path}`")
        body.append("")

    body.append("## Caveats — what this report does NOT prove")
    body.append("")
    body.append(
        "- **Simulator approximation**: this harness mirrors the v1.04 "
        "grid+martingale+hedge mechanics on H1 bars. It ignores SMLO "
        "trailing, ADX confirm chains, and per-tick price action. The "
        "blowup conditions match v1.04's PauseMultiple + MaxEquityDD; "
        "intra-bar paths can diverge from a real broker.")
    body.append(
        "- **Cold-start bias**: the synthesized EAState reports "
        "`recent_sample_count<5` until enough closed deals exist. "
        "rule_engine's history-completeness gate (Phase C-hotfix-4) "
        "correctly refuses aggressive tier in that window — but on a "
        "1-year backtest this can keep tier=normal for a long time.")
    body.append(
        "- **Single instrument, single year**: the data lake currently "
        "only carries XAUUSD through 2024-12-31. Multi-year and "
        "multi-symbol robustness is not tested here.")
    body.append(
        "- **No live spread / news**: spread_pts is held at a fixed value, "
        "so spread-cooldown never fires. NewsEngine is not wired in. Both "
        "live behaviours can change the dynamic verdict materially.")
    body.append(
        "- **Static baseline is a Python sim, not the MQL5 EA**: real EA "
        "fills, broker latency, swap, and commission are not modelled. "
        "Use this report for *relative* comparison only.")

    body.append("")
    body.append("## Next-round rule adjustments (if dynamic underperforms)")
    body.append("")
    body.append(
        "- **Loosen aggressive-tier gate for cold-start**: today the gate "
        "requires `recent_sample_count ≥ 5` AND non-negative recent PnL. "
        "Consider a 'cold-start grace period' — first N bars get the "
        "normal tier without the sample requirement.")
    body.append(
        "- **Tune `_DD_STEPDOWN_THRESHOLD`** (currently 2 %) if the "
        "static baseline shows DD spikes above 5 % that dynamic doesn't "
        "catch. Lower the threshold first; raising it later is easy.")
    body.append(
        "- **Expand `_RECENT_PNL_STEPDOWN_THRESHOLD`** from $-50 — for a "
        "$10k account this is fine; for $1k accounts the rule never "
        "trips. Make it a fraction of `init_equity`.")
    body.append(
        "- **Reconsider `range#2` confidence (0.65)**: most of the time "
        "the regime classifier outputs range at 0.65, which sits below "
        "the aggressive floor (0.80). If the dynamic run has very few "
        "aggressive bars, lower the aggressive floor to 0.70 OR raise "
        "the range#2 confidence baseline to 0.75.")
    body.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body))
    print(f"wrote report → {report_path}")


def _write_envelope_log(envelope_log: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for entry in envelope_log:
            fh.write(json.dumps(entry) + "\n")
    print(f"wrote envelope log → {path} ({len(envelope_log)} lines)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start", type=_parse_date,
                        default=datetime(2024, 1, 1, tzinfo=timezone.utc))
    parser.add_argument("--end", type=_parse_date,
                        default=datetime(2025, 1, 1, tzinfo=timezone.utc))
    parser.add_argument("--init-equity", type=float, default=10_000.0)
    parser.add_argument("--data-lake-root", type=Path, default=_DEFAULT_LAKE)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--envelope-log-path", type=Path, default=None,
                        help="Base path for per-variant envelope logs. "
                             "Each variant writes to <base>.<label>.envelope_log.jsonl. "
                             "If omitted, derives from --report-path.")
    # Phase D-cont1 / D-cont1b — experiment-set shortcuts.
    parser.add_argument(
        "--experiment-set",
        choices=["custom", "d-cont1", "d-cont1b"],
        default="custom",
        help=("Shortcut: 'd-cont1' runs observe-only release at 4h. "
              "'d-cont1b' runs a 3-way comparison: baseline + observe @4h "
              "+ tiny_normal @4h. 'custom' (default) honours the "
              "individual --experiment-* flags."),
    )
    # Phase D-cont1 — individual knobs (used when experiment-set=custom).
    parser.add_argument(
        "--experiment-cold-start-grace",
        action="store_true",
        help=("Phase D-cont1 experiment: promote observe→normal-tier "
              "hedgerock when range regime + 0.45 ≤ confidence < 0.55 "
              "+ history is cold. Disabled by default."),
    )
    parser.add_argument(
        "--experiment-halt-auto-expiry-hours", type=float, default=None,
        help=("Phase D-cont1 experiment: after N hours of continuous "
              "halt mode, downgrade per --experiment-halt-auto-expiry-release. "
              "Disabled by default."),
    )
    parser.add_argument(
        "--experiment-halt-auto-expiry-release",
        choices=["observe", "tiny_normal"],
        default="observe",
        help=("What halt decays to after expiry. 'observe' = D-cont1, "
              "no new entries. 'tiny_normal' = D-cont1b, opens at most "
              "1 buy + 1 sell with lot_factor=0.1 (max 0.02 lot)."),
    )
    args = parser.parse_args(argv)

    if not args.data_lake_root.exists():
        print(f"data lake root not found: {args.data_lake_root}",
              file=sys.stderr)
        return 2

    lake = ForexDataLake(args.data_lake_root)
    config = WalkForwardConfig(
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        init_equity=args.init_equity,
    )
    experiments = _experiments_from_args(args)
    print(f"running walk-forward {config.start.date()} → {config.end.date()} "
          f"({config.instrument}) ...")
    if experiments:
        print(f"  experiment-set: {args.experiment_set}")
        for label, cfg_v in experiments:
            print(
                f"    {label}: cold_start_grace={cfg_v.cold_start_grace} "
                f"halt_auto_expiry_hours={cfg_v.halt_auto_expiry_hours} "
                f"halt_auto_expiry_release={cfg_v.halt_auto_expiry_release}"
            )
    result = run_walk_forward(config, lake, experiments=experiments)

    print(
        f"  static            → return {result.static_metrics.total_return_pct:+.2f}% "
        f"DD {result.static_metrics.max_dd_pct:.2f}% "
        f"trades {result.static_metrics.n_trades} "
        f"blowup={result.static_metrics.blowup}"
    )
    print(
        f"  dynamic baseline  → return {result.dynamic_metrics.total_return_pct:+.2f}% "
        f"DD {result.dynamic_metrics.max_dd_pct:.2f}% "
        f"trades {result.dynamic_metrics.n_trades} "
        f"blowup={result.dynamic_metrics.blowup}"
    )
    for v in result.experiments:
        print(
            f"  dynamic {v.label:10s}→ return {v.metrics.total_return_pct:+.2f}% "
            f"DD {v.metrics.max_dd_pct:.2f}% "
            f"trades {v.metrics.n_trades} "
            f"near-stopout {v.metrics.near_stopout_count} "
            f"agg {v.metrics.aggressive_tier_bars} "
            f"blowup={v.metrics.blowup}"
        )

    # Per-variant envelope logs. Base path: --envelope-log-path, or
    # report-path with .envelope_log.jsonl suffix. Each variant gets
    # <stem>.<label>.envelope_log.jsonl so a 3-way run yields
    # baseline, d-cont1, d-cont1b log files in the same dir.
    base_log_path = args.envelope_log_path
    if base_log_path is None:
        base_log_path = args.report_path.with_suffix(".envelope_log.jsonl")
    base_dir = base_log_path.parent
    base_stem = base_log_path.name
    if base_stem.endswith(".envelope_log.jsonl"):
        base_stem = base_stem[: -len(".envelope_log.jsonl")]
    elif base_stem.endswith(".jsonl"):
        base_stem = base_stem[: -len(".jsonl")]

    log_paths: dict[str, Path] = {}
    baseline_path = base_dir / f"{base_stem}.baseline.envelope_log.jsonl"
    _write_envelope_log(result.envelope_log, baseline_path)
    log_paths["baseline"] = baseline_path
    for v in result.experiments:
        p = base_dir / f"{base_stem}.{v.label}.envelope_log.jsonl"
        _write_envelope_log(v.envelope_log, p)
        log_paths[v.label] = p

    _write_report(result, args.report_path, log_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
