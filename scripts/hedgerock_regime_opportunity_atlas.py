"""Phase D-cont2 — Regime / Opportunity Atlas (DIAGNOSTIC ONLY).

Runs the atlas over a configurable date window and emits:
    - docs/phase-d-regime-opportunity-atlas.md
    - <records-path>.records.jsonl       (per-bar decision + outcome)
    - <records-path>.halt_aftermath.jsonl (per-halt-event)

The atlas does NOT touch rule_engine or trade. It exposes:
    - Regime × confidence-bucket distribution
    - Cold-start-grace eligibility waterfall
    - Range opportunity atlas
    - Trend / breakout missed-opportunity atlas
    - Halt aftermath atlas

Usage::

    python scripts/hedgerock_regime_opportunity_atlas.py \\
        --start 2024-01-01 --end 2025-01-01 \\
        --report-path docs/phase-d-regime-opportunity-atlas.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from smc.data.lake import ForexDataLake
from smc.hedgerock.regime_opportunity_atlas import (
    AtlasConfig,
    AtlasReport,
    BUCKET_LABELS,
    DecisionRecord,
    HaltAftermathRecord,
    OutcomeLabel,
    RegimeBucketStat,
    run_atlas,
)


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


_DEFAULT_LAKE = _ai_smc_home() / "data" / "parquet"
_DEFAULT_REPORT = (
    _hedgerock_home() / "docs" / "phase-d-regime-opportunity-atlas.md"
)
_REGIMES_OF_INTEREST = (
    "range", "trend_up", "trend_down", "breakout", "crisis",
)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Per-bar JSONL serializer
# ---------------------------------------------------------------------------


def _record_to_dict(d: DecisionRecord, o: OutcomeLabel | None) -> dict:
    out: dict = {
        "ts": d.ts.isoformat(),
        "regime": d.regime,
        "confidence": d.confidence,
        "confidence_bucket": d.confidence_bucket,
        "classifier_reason": d.classifier_reason,
        "rule_votes": [list(v) for v in d.rule_votes],
        "volatility_rank": d.volatility_rank,
        "h4_trend_bars": d.h4_trend_bars,
        "hh_count": d.hh_count,
        "ll_count": d.ll_count,
        # NEUTRAL-COLD raw-intent — what rule_engine wants at clean
        # baseline. NOT what the live EA executes.
        "rule_mode": d.rule_mode,
        "rule_risk_tier": d.rule_risk_tier,
        "rule_reason": d.rule_reason,
        "rule_lot_factor": d.rule_lot_factor,
        "rule_cooldown_active": d.rule_cooldown_active,
        "grace_failed_at": d.grace_failed_at,
        "grace_eligible": d.grace_eligible,
        # Live-equivalent execution from Phase D dynamic-baseline
        # replay. ``live_raw_*`` is rule_engine output BEFORE the
        # transition_lock veto, GIVEN simulator-state DD/cooldown.
        # ``live_effective_*`` is what actually drove the simulator.
        "live_raw_mode": d.live_raw_mode,
        "live_raw_reason": d.live_raw_reason,
        "live_effective_mode": d.live_effective_mode,
        "live_risk_tier": d.live_risk_tier,
        "live_lot_factor": d.live_lot_factor,
        "live_cooldown_active": d.live_cooldown_active,
        "live_cooldown_until": (
            d.live_cooldown_until.isoformat() if d.live_cooldown_until else None
        ),
        "live_transition_lock_active": d.live_transition_lock_active,
        "live_transition_lock_until_ts": (
            d.live_transition_lock_until_ts.isoformat()
            if d.live_transition_lock_until_ts else None
        ),
        "outcome_available": o is not None,
    }
    if o is not None:
        # Future bars are LABELS ONLY — explicit prefix per Phase D-cont2
        # spec so any consumer can distinguish decision vs outcome data.
        out["label_decision_close"] = o.decision_close
        out["label_returns_pct"] = {str(k): v for k, v in o.returns.items()}
        out["label_mae_24h_pct"] = o.mae_24h
        out["label_mfe_24h_pct"] = o.mfe_24h
        out["label_range_width_24h_pct"] = o.range_width_24h
    return out


def _halt_record_to_dict(h: HaltAftermathRecord) -> dict:
    return {
        "halt_ts": h.halt_ts.isoformat(),
        "halt_close": h.halt_close,
        "halt_volatility_rank": h.halt_volatility_rank,
        "halt_regime": h.halt_regime,
        "label_horizons": {str(k): v for k, v in h.horizons.items()},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _fmt_regime_x_bucket(report: AtlasReport) -> list[str]:
    out = ["## Regime × confidence-bucket distribution", ""]
    out.append("Counts of decision bars by (regime, confidence bucket).")
    out.append("Confidence boundaries align with rule_engine constants:")
    out.append("`<0.45` (below grace floor) | `0.45-0.55` (grace band) | "
               "`0.55-0.65` (above OBSERVE floor) | `0.65-0.80` (range#2) | "
               "`>=0.80` (aggressive floor).")
    out.append("")

    headers = ["Regime"] + list(BUCKET_LABELS) + ["Total"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")

    # Discover all regimes present.
    regimes = sorted({k[0] for k in report.regime_x_bucket_count.keys()})
    bucket_totals: dict[str, int] = {b: 0 for b in BUCKET_LABELS}
    for regime in regimes:
        cells = [regime]
        regime_total = 0
        for b in BUCKET_LABELS:
            count = report.regime_x_bucket_count.get((regime, b), 0)
            cells.append(str(count))
            regime_total += count
            bucket_totals[b] += count
        cells.append(str(regime_total))
        out.append("| " + " | ".join(cells) + " |")
    grand_total = sum(bucket_totals.values())
    out.append("| **Total** | " + " | ".join(
        f"**{bucket_totals[b]}**" for b in BUCKET_LABELS
    ) + f" | **{grand_total}** |")
    out.append("")

    # Diagnostic call-out: did range ever land in the grace band?
    range_grace = report.regime_x_bucket_count.get(("range", "0.45-0.55"), 0)
    range_below = report.regime_x_bucket_count.get(("range", "<0.45"), 0)
    if range_grace == 0:
        out.append(
            "> **Diagnostic — classifier_v2 discrete-confidence finding.** "
            f"`range` regime never occurred in the 0.45–0.55 grace band "
            f"({range_grace} bars), and "
            f"{range_below} bars below 0.45. classifier_v2 emits range "
            "at fixed confidences (0.65 / 0.80 / 0.85 / 0.90), so "
            "cold_start_grace's eligibility window can NEVER overlap a "
            "real bar by design — the gating is unreachable on this "
            "classifier output regardless of dataset. Lowering the "
            "OBSERVE floor or adding a graded confidence path is the "
            "only way to test that gate empirically."
        )
        out.append("")
    return out


def _fmt_raw_vs_live(report: AtlasReport) -> list[str]:
    """Phase D-cont2-hotfix-1: side-by-side count of NEUTRAL-COLD raw
    intent vs live-equivalent execution.

    The same bar can land in different modes depending on which view
    you read. The raw column is what rule_engine wants under a clean
    EAState (no DD, no cooldown carryover, no transition_lock). The
    live column is what the Phase D dynamic baseline actually executed
    given accumulated simulator state. The delta is where Phase D's
    veto / carryover machinery actually mattered.
    """
    out = ["## Raw classifier intent vs live-effective execution", ""]
    out.append("Per-bar mode counts under each view. **Reports in the "
               "rest of this document MUST cite the right column** — "
               "raw intent answers \"what does the rule engine want?\", "
               "live-effective answers \"what did the simulator run?\". "
               "These are NOT the same.")
    out.append("")
    raw_counts: dict[str, int] = {}
    live_raw_counts: dict[str, int] = {}
    live_eff_counts: dict[str, int] = {}
    transition_lock_bars = 0
    cooldown_bars = 0
    raw_to_eff_demoted = 0
    live_join_n = 0
    for d, _ in report.records:
        raw_counts[d.rule_mode] = raw_counts.get(d.rule_mode, 0) + 1
        if d.live_raw_mode is not None:
            live_join_n += 1
            live_raw_counts[d.live_raw_mode] = live_raw_counts.get(d.live_raw_mode, 0) + 1
            eff = d.live_effective_mode or d.live_raw_mode
            live_eff_counts[eff] = live_eff_counts.get(eff, 0) + 1
            if d.live_transition_lock_active:
                transition_lock_bars += 1
            if d.live_cooldown_active:
                cooldown_bars += 1
            if d.live_raw_mode == "hedgerock" and eff != "hedgerock":
                raw_to_eff_demoted += 1
    modes = sorted(set(raw_counts) | set(live_raw_counts) | set(live_eff_counts))
    out.append("| Mode | NEUTRAL-COLD rule intent | Phase D live raw | Phase D live effective |")
    out.append("|---|---|---|---|")
    for m in modes:
        out.append(
            f"| {m} | {raw_counts.get(m, 0)} | "
            f"{live_raw_counts.get(m, 0)} | {live_eff_counts.get(m, 0)} |"
        )
    out.append("")
    out.append(
        f"- Bars with a live envelope (post-warmup): **{live_join_n}** of "
        f"{len(report.records)}."
    )
    out.append(
        f"- Live transition-lock active: **{transition_lock_bars}** bars."
    )
    out.append(
        f"- Live cooldown active: **{cooldown_bars}** bars."
    )
    out.append(
        f"- Bars where live raw was `hedgerock` but live effective "
        f"demoted (transition_lock veto / etc): **{raw_to_eff_demoted}**."
    )
    out.append("")
    return out


def _fmt_waterfall(report: AtlasReport) -> list[str]:
    out = ["## Cold-start-grace eligibility waterfall (NEUTRAL-COLD raw intent)", ""]
    out.append("Survivor count after each gate is checked, in the same "
               "order as `_eligible_for_cold_start_grace`. Each bar that "
               "fails a gate is dropped; later gates only see survivors. "
               "**Inputs are the NEUTRAL-COLD raw-intent fields** (no DD, "
               "no recent samples, no transition_lock veto) — this "
               "diagnoses classifier-vs-gate interaction in isolation, "
               "NOT what a live EA would do under accumulated simulator "
               "state. Gates 5 & 6 always pass under NEUTRAL-COLD by "
               "construction; the interesting drops are gates 1–4.")
    out.append("")
    out.append("| Stage | Survivors | Drop |")
    out.append("|---|---|---|")
    prev = None
    for stage in report.waterfall.stages:
        drop_str = "—" if prev is None else str(prev - stage.survivors)
        out.append(f"| {stage.name} | {stage.survivors} | {drop_str} |")
        prev = stage.survivors
    out.append("")
    return out


def _fmt_range_opportunity(
    report: AtlasReport,
) -> list[str]:
    out = ["## Range opportunity atlas (forward-24h outcomes)", ""]
    out.append("For every `range`-regime decision bar, the table below "
               "shows the *post-hoc* H1 close-to-close return at "
               "decision_ts → decision_ts + 24 H1 bars, plus the "
               "average MAE / MFE / total range_width within the "
               "same window. **Returns are diagnostic labels — not "
               "input to any decision.**")
    out.append("")
    out.append("| Confidence bucket | n | mean ret 24h % | median % | "
               "stdev % | mean MAE % | mean MFE % | mean range_width % |")
    out.append("|---|---|---|---|---|---|---|---|")
    min_n = report.config.min_sample_for_signal
    for s in report.range_opportunity:
        flag = ""
        if s.count < min_n:
            flag = " (n<%d, INCONCLUSIVE)" % min_n
        out.append(
            f"| {s.bucket}{flag} | {s.count} | "
            f"{s.mean_return_24h:+.3f} | {s.median_return_24h:+.3f} | "
            f"{s.stdev_return_24h:.3f} | "
            f"{s.mean_mae_24h:+.3f} | {s.mean_mfe_24h:+.3f} | "
            f"{s.mean_range_width_24h:.3f} |"
        )
    out.append("")
    # Verdict
    verdicts: list[str] = []
    for s in report.range_opportunity:
        if s.count < min_n:
            verdicts.append(
                f"- **{s.bucket}**: n={s.count} below sample floor "
                f"({min_n}). **INCONCLUSIVE.**"
            )
            continue
        # "Edge" criterion: |mean| > stdev/√n × 1.96 (rough 95% CI excludes 0)
        # AND mean_mfe > |mean_mae|. Both must hold.
        from math import sqrt
        ci = 1.96 * (s.stdev_return_24h / sqrt(max(s.count, 1)))
        if s.mean_return_24h > ci and s.mean_mfe_24h > abs(s.mean_mae_24h):
            verdicts.append(
                f"- **{s.bucket}**: positive 24h drift with MFE > |MAE|. "
                f"Mean +{s.mean_return_24h:.3f}% (95% CI ±{ci:.3f}). "
                "Tradeable edge candidate."
            )
        elif s.mean_return_24h < -ci:
            verdicts.append(
                f"- **{s.bucket}**: NEGATIVE drift {s.mean_return_24h:+.3f}% "
                f"(95% CI ±{ci:.3f}). Range trades into a fading market — "
                "expanding hedgerock here would lose money."
            )
        else:
            verdicts.append(
                f"- **{s.bucket}**: mean {s.mean_return_24h:+.3f}% within 95% "
                f"CI of zero (±{ci:.3f}). No directional edge — but "
                f"mean range_width {s.mean_range_width_24h:.3f}% gives "
                "the TP target room. Hedgerock is mean-reversion-friendly "
                "*only if* MFE/MAE asymmetry is favourable, which here "
                f"is MFE {s.mean_mfe_24h:+.3f}% vs MAE {s.mean_mae_24h:+.3f}%."
            )
    out.append("**Verdict per bucket:**")
    out.append("")
    out.extend(verdicts)
    out.append("")
    return out


def _fmt_trend_opportunity(report: AtlasReport) -> list[str]:
    out = ["## Trend / breakout missed-opportunity atlas", ""]
    out.append("For each non-range regime, we re-sign the future-24h "
               "return so that *positive* values mean the regime's "
               "expected direction was followed. `trend_down` returns "
               "are negated. **breakout is split into two rows**: "
               "`breakout_magnitude` is |return| (no direction — "
               "diagnostic only, NOT an E1 candidate regardless of CI), "
               "and `breakout_signed_by_h4` uses `h4_trend_bars` sign "
               "as a direction proxy (positive H4 → expect up, negative "
               "→ expect down; bars with zero H4 trend are dropped). "
               "Only signed-direction rows are eligible for E1 candidacy.")
    out.append("")
    out.append("| Regime | n | mean directional ret 24h % | median % | "
               "stdev % | mean MAE % | mean MFE % |")
    out.append("|---|---|---|---|---|---|---|")
    min_n = report.config.min_sample_for_signal
    for s in report.trend_opportunity:
        flag = ""
        if s.count < min_n:
            flag = " (n<%d)" % min_n
        out.append(
            f"| {s.regime}{flag} | {s.count} | "
            f"{s.mean_return_24h:+.3f} | {s.median_return_24h:+.3f} | "
            f"{s.stdev_return_24h:.3f} | "
            f"{s.mean_mae_24h:+.3f} | {s.mean_mfe_24h:+.3f} |"
        )
    out.append("")
    out.append("**E1 momentum candidacy:**")
    out.append("")
    from math import sqrt
    for s in report.trend_opportunity:
        # Phase D-cont2-hotfix-2: magnitude-only rows are NEVER E1
        # candidates — there's no execution path without direction.
        if s.regime == "breakout_magnitude":
            out.append(
                f"- **{s.regime}** (n={s.count}): mean |return| "
                f"{s.mean_return_24h:+.3f}%. **Magnitude only — no "
                f"direction model, so cannot be an E1 momentum "
                f"candidate even if CI clears zero.** Reported as a "
                f"volatility/range-of-motion diagnostic, not a tradeable "
                f"signal."
            )
            continue
        if s.count < min_n:
            out.append(
                f"- **{s.regime}**: n={s.count} below sample floor "
                f"({min_n}) — **INCONCLUSIVE, no recommendation.**"
            )
            continue
        ci = 1.96 * (s.stdev_return_24h / sqrt(max(s.count, 1)))
        if s.mean_return_24h > ci and s.mean_mfe_24h > abs(s.mean_mae_24h):
            out.append(
                f"- **{s.regime}**: directional edge +{s.mean_return_24h:.3f}% "
                f"(95% CI ±{ci:.3f}, n={s.count}). "
                "**E1 momentum candidate — worth dedicated module.**"
            )
        else:
            out.append(
                f"- **{s.regime}**: mean directional {s.mean_return_24h:+.3f}% "
                f"vs 95% CI ±{ci:.3f}. **No evidence of a tradeable edge** "
                "in this window."
            )
    out.append("")
    return out


def _fmt_halt_aftermath(report: AtlasReport) -> list[str]:
    out = ["## Halt aftermath atlas", ""]
    out.append("Each row is a leading-edge halt-trigger event from the "
               "Phase D dynamic simulator (baseline run, no experiments). "
               "Forward returns are signed against `halt_close` — "
               "positive = price moved AWAY FROM the loss direction, "
               "negative = price kept moving against. MAE / MFE are "
               "in the same window. Vol-rank-at-t shows whether "
               "volatility actually subsided.")
    out.append("")
    if not report.halt_aftermath:
        out.append("- (no halt events in this window)")
        out.append("")
        return out

    horizons = sorted({
        h
        for r in report.halt_aftermath
        for h in r.horizons.keys()
    })
    headers = ["halt_ts", "halt_close", "halt_regime",
               "halt_vol_rank"] + [
        f"ret_{h}h%" for h in horizons
    ] + [f"vol_rank@{h}h" for h in horizons]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in report.halt_aftermath:
        cells = [
            r.halt_ts.strftime("%Y-%m-%d %H:%M"),
            f"{r.halt_close:.2f}",
            r.halt_regime,
            f"{r.halt_volatility_rank:.2f}",
        ]
        for h in horizons:
            cells.append(
                f"{r.horizons.get(h, {}).get('return_pct', 0.0):+.2f}"
                if h in r.horizons else "—"
            )
        for h in horizons:
            v = r.horizons.get(h, {}).get("vol_rank_at_t")
            cells.append(f"{v:.2f}" if isinstance(v, (int, float)) else "—")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # Per-horizon summary stats.
    out.append("**Forward summary across all halt events:**")
    out.append("")
    out.append("| Horizon | n | mean ret % | median % | mean MFE % | mean MAE % |")
    out.append("|---|---|---|---|---|---|")
    from statistics import mean as _mean, median as _median
    for h in horizons:
        rets = [
            r.horizons[h]["return_pct"]
            for r in report.halt_aftermath
            if h in r.horizons
        ]
        mfes = [
            r.horizons[h]["mfe_pct"]
            for r in report.halt_aftermath
            if h in r.horizons
        ]
        maes = [
            r.horizons[h]["mae_pct"]
            for r in report.halt_aftermath
            if h in r.horizons
        ]
        if not rets:
            continue
        out.append(
            f"| {h}h | {len(rets)} | {_mean(rets):+.3f} | "
            f"{_median(rets):+.3f} | {_mean(mfes):+.3f} | {_mean(maes):+.3f} |"
        )
    out.append("")
    return out


def _fmt_dcont2_verdict(report: AtlasReport) -> list[str]:
    out = ["## D-cont2 verdict", ""]
    out.append("Diagnostic synthesis based on the atlas above. "
               "**Recommendations are evidence-based — sample-size "
               "or CI failures lead to INCONCLUSIVE rather than to a "
               "rule change.**")
    out.append("")

    min_n = report.config.min_sample_for_signal
    from math import sqrt

    # 1. classifier threshold tuning candidate?
    range_grace = report.regime_x_bucket_count.get(("range", "0.45-0.55"), 0)
    range_above = sum(
        report.regime_x_bucket_count.get(("range", b), 0)
        for b in ("0.65-0.80", ">=0.80")
    )
    out.append("### 1. Classifier threshold tuning candidate?")
    out.append("")
    if range_grace == 0:
        out.append(
            f"- classifier_v2 emits `range` only at discrete confidences "
            f"(0.65 / 0.80 / 0.85 / 0.90); the 0.45–0.55 grace band has "
            f"zero coverage. Lowering `_COLD_START_GRACE_CONF_FLOOR` from "
            f"0.45 cannot help — there is no classifier output in that "
            f"range to gate. **Real options:** (a) make classifier_v2 "
            f"emit a graded range confidence between 0.45 and 0.65 for "
            f"weak-but-present range signal, OR (b) lower the OBSERVE "
            f"floor to allow the existing 0.65-tier range bars to trade "
            f"under cold-start conditions."
        )
    # Look for negative-edge buckets that we currently let trade.
    bad_buckets = [
        s for s in report.range_opportunity
        if s.count >= min_n and s.mean_return_24h < 0
    ]
    if bad_buckets:
        for s in bad_buckets:
            ci = 1.96 * (s.stdev_return_24h / sqrt(max(s.count, 1)))
            if s.mean_return_24h < -ci:
                out.append(
                    f"- **`range` @ confidence {s.bucket}** has NEGATIVE "
                    f"24h mean return ({s.mean_return_24h:+.3f}% ± {ci:.3f}) "
                    f"with n={s.count}. RAISING the floor (so this "
                    f"bucket goes observe instead of hedgerock) is a "
                    f"data-supported tuning candidate — but only after "
                    f"a multi-year confirmation."
                )
    out.append("")

    # 2. momentum module candidate?
    out.append("### 2. Momentum module (E1) candidate?")
    out.append("")
    any_trend_edge = False
    for s in report.trend_opportunity:
        if s.regime == "breakout_magnitude":
            out.append(
                f"- {s.regime} (n={s.count}): magnitude-only — no "
                "direction model. **Cannot be an E1 candidate** "
                "regardless of |return| CI. Documented as volatility "
                "diagnostic only."
            )
            continue
        if s.count < min_n:
            out.append(
                f"- {s.regime}: n={s.count} below sample floor → INCONCLUSIVE."
            )
            continue
        ci = 1.96 * (s.stdev_return_24h / sqrt(max(s.count, 1)))
        if s.mean_return_24h > ci and s.mean_mfe_24h > abs(s.mean_mae_24h):
            any_trend_edge = True
            out.append(
                f"- {s.regime}: directional mean +{s.mean_return_24h:.3f}% "
                f"(±{ci:.3f}, n={s.count}) **with** MFE > |MAE|. "
                "Build E1 — order matters: prove on multi-year first."
            )
        else:
            out.append(
                f"- {s.regime}: directional mean {s.mean_return_24h:+.3f}% "
                f"(±{ci:.3f}, n={s.count}). No evidence — no E1 here."
            )
    if not any_trend_edge:
        out.append(
            "- **Net: NO E1 candidate on 2024 XAUUSD.** Building a "
            "momentum module without a positive-expectancy regime is "
            "premature."
        )
    out.append("")

    # 3. Smarter halt-release trigger candidate?
    out.append("### 3. Smarter halt-release trigger candidate?")
    out.append("")
    if not report.halt_aftermath:
        out.append("- No halt events in window → **N/A.**")
    else:
        # Look at 24h forward — if mean is positive across events the
        # "release at 24h" trigger has SOME evidence; if negative or
        # near zero, the user's D-cont1b finding (release fails) is
        # corroborated.
        from statistics import mean as _mean
        rets24 = [
            r.horizons[24]["return_pct"]
            for r in report.halt_aftermath
            if 24 in r.horizons
        ]
        rets72 = [
            r.horizons[72]["return_pct"]
            for r in report.halt_aftermath
            if 72 in r.horizons
        ]
        n_events = len(report.halt_aftermath)
        if n_events < 5:
            out.append(
                f"- Only {n_events} halt event(s) — far below the sample "
                f"floor ({min_n}). **Cannot conclude** anything about "
                "release-trigger design from one window."
            )
        else:
            m24 = _mean(rets24) if rets24 else 0.0
            m72 = _mean(rets72) if rets72 else 0.0
            out.append(
                f"- Across {n_events} halt event(s): mean forward 24h "
                f"return = {m24:+.3f}%, 72h = {m72:+.3f}%. "
                "Inspect the per-event table — if 24h is consistently "
                "negative (price kept moving against), release timing "
                "should be tied to a *price retracement* signal "
                "rather than wall-clock hours."
            )
    out.append("")

    # Closing summary — synthesise what is and isn't supported by the
    # 2024-XAUUSD-window evidence.
    out.append("### Net D-cont2 recommendation")
    out.append("")
    # Re-derive flags so the prose stays in sync with the verdicts above.
    range_grace = report.regime_x_bucket_count.get(("range", "0.45-0.55"), 0)
    e1_candidates: list[str] = []
    range_negative_buckets: list[str] = []
    range_positive_buckets: list[str] = []
    for s in report.trend_opportunity:
        # Phase D-cont2-hotfix-2: magnitude-only never qualifies.
        if s.regime == "breakout_magnitude":
            continue
        if s.count < min_n:
            continue
        ci = 1.96 * (s.stdev_return_24h / sqrt(max(s.count, 1)))
        if s.mean_return_24h > ci and s.mean_mfe_24h > abs(s.mean_mae_24h):
            e1_candidates.append(s.regime)
    for s in report.range_opportunity:
        if s.count < min_n:
            continue
        ci = 1.96 * (s.stdev_return_24h / sqrt(max(s.count, 1)))
        if s.mean_return_24h > ci:
            range_positive_buckets.append(s.bucket)
        elif s.mean_return_24h < -ci:
            range_negative_buckets.append(s.bucket)

    out.append("**What the 2024 XAUUSD window supports:**")
    out.append("")
    if e1_candidates:
        out.append(
            f"- E1 momentum candidacy WINDOW-EVIDENCED for: "
            f"`{', '.join(e1_candidates)}`. 95% CI clears zero, MFE > |MAE|. "
            "Promote only after multi-year + multi-symbol replication."
        )
    else:
        out.append(
            "- No regime cleared the E1 bar (CI excludes zero AND "
            "MFE > |MAE|) on this window."
        )
    if range_positive_buckets:
        out.append(
            f"- range@{','.join(range_positive_buckets)} has a positive "
            f"24h drift in this window. Magnitude is small — relevant "
            f"only if hedge sizing exploits it (which v1.04 already does "
            f"via TP / grid)."
        )
    if range_negative_buckets:
        out.append(
            f"- range@{','.join(range_negative_buckets)} has a NEGATIVE "
            f"24h drift — trading hedgerock here is empirically "
            f"unfavourable. Raising the bucket's gate to observe is a "
            f"data-supported tuning candidate."
        )
    out.append("")
    out.append("**What the window does NOT support:**")
    out.append("")
    if range_grace == 0:
        out.append(
            "- **classifier_v2 threshold tuning of cold_start_grace**: "
            "the 0.45-0.55 band has 0 coverage by classifier construction "
            "— the gate is structurally unreachable. Tuning this knob "
            "without first changing the classifier is a no-op."
        )
    if len(report.halt_aftermath) < min_n:
        out.append(
            f"- **smarter halt-release trigger**: only "
            f"{len(report.halt_aftermath)} halt event(s) in this window. "
            "Designing a release rule from a single event would be "
            "curve-fit by definition. Need a larger DD-event sample."
        )
    out.append("")
    out.append("**Recommended next step (data, not code):**")
    out.append("")
    out.append(
        "1. Extend the lake to multi-year (≥3y) and at least one "
        "additional symbol (XAGUSD or EURUSD) and re-run this atlas. "
        "Replication, not single-window emphasis, decides whether E1 "
        "is real."
    )
    out.append(
        "2. Separately, evaluate whether classifier_v2 should expose a "
        "graded range confidence in [0.45, 0.65) — that's a classifier "
        "change, not a rule_engine change, and unlocks the "
        "cold_start_grace gate empirically (Phase D-cont3 candidate)."
    )
    out.append(
        "3. Do NOT add a momentum module, retune the OBSERVE floor, or "
        "build a smarter halt-release trigger on the strength of this "
        "single window alone."
    )
    out.append("")
    return out


def _write_report(report: AtlasReport, path: Path,
                  jsonl_paths: dict[str, Path]) -> None:
    cfg = report.config
    body: list[str] = [
        "# Phase D-cont2 — Regime / Opportunity Atlas (DIAGNOSTIC ONLY)",
        "",
        "> **Read-only diagnostic.** This report does NOT change "
        "production rule_engine, does NOT add a new trading knob, "
        "and does NOT trade. It re-runs classifier_v2 + rule_engine "
        "over the Phase D window with a NEUTRAL-COLD EAState, then "
        "computes future-24h outcome labels for each decision bar. "
        "Future bars are labelled outcomes only — the no-lookahead "
        "invariant is preserved by the same closed-bar windowing as "
        "Phase D walk-forward.",
        "",
        "## Run config",
        "",
        f"- Instrument: `{cfg.instrument}`",
        f"- Window: `{cfg.start.date()}` → `{cfg.end.date()}`",
        f"- H1 lookback: {cfg.h1_lookback}, H4 lookback: {cfg.h4_lookback}",
        f"- Spread (synth): {cfg.spread_pts} pts",
        f"- Future horizons (H1 bars): {cfg.future_horizons}",
        f"- Halt aftermath horizons (H1 bars): {cfg.halt_aftermath_horizons}",
        f"- Sample-size floor for tradeable signal: {cfg.min_sample_for_signal}",
        f"- Decision records emitted: {len(report.records)}",
        f"  (records with full outcome label: "
        f"{sum(1 for _, o in report.records if o is not None)})",
        f"- Halt events found: {len(report.halt_aftermath)}",
        "",
    ]

    body.extend(_fmt_regime_x_bucket(report))
    body.extend(_fmt_raw_vs_live(report))
    body.extend(_fmt_waterfall(report))
    body.extend(_fmt_range_opportunity(report))
    body.extend(_fmt_trend_opportunity(report))
    body.extend(_fmt_halt_aftermath(report))
    body.extend(_fmt_dcont2_verdict(report))

    body.append("## Per-bar decision + outcome JSONL")
    body.append("")
    for label, p in jsonl_paths.items():
        body.append(f"- **{label}** → `{p}`")
    body.append("")

    body.append("## Caveats — what this atlas does NOT prove")
    body.append("")
    body.append(
        "- **Single-window 2024 XAUUSD**: any tuning recommendation "
        "must be reproduced on multi-year + multi-symbol data before "
        "promotion."
    )
    body.append(
        "- **NEUTRAL-COLD vs live-equivalent split**: the NEUTRAL-COLD "
        "EAState is used ONLY for the raw classifier intent (`rule_*` "
        "fields) and the cold-start-grace eligibility waterfall — both "
        "answer \"what does rule_engine want at clean baseline\". The "
        "live-equivalent execution trace (DD-induced halt cascades, "
        "cooldown carryover, transition_lock veto) IS surfaced — in "
        "the **Raw classifier intent vs live-effective execution** "
        "section AND in the **Halt aftermath atlas**, both populated "
        "from the Phase D dynamic-baseline replay envelope log. Read "
        "those sections when you need real EA-state-aware numbers."
    )
    body.append(
        "- **No directional information for breakout regimes from "
        "classify_regime_v2 alone**: the classifier doesn't expose "
        "which side of the breakout the bar is on. The atlas reports "
        "two breakout views: `breakout_magnitude` (|return|, diagnostic "
        "only — never E1) and `breakout_signed_by_h4` (signed by "
        "`h4_trend_bars` as a direction proxy, E1-eligible only when "
        "its signed CI clears zero)."
    )
    body.append(
        "- **Outcome labels are H1 close-to-close**: intra-bar paths "
        "may differ. MAE / MFE are computed over the 24-bar future "
        "window's high / low, which is robust enough for relative "
        "comparison but not for tick-level execution promises."
    )
    body.append(
        "- **No suggestion of behaviour change**: this is Phase D-cont2 "
        "diagnostic only. Any rule_engine retune is a SEPARATE phase "
        "with its own A/B harness."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body))
    print(f"wrote report → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start", type=_parse_date,
                        default=datetime(2024, 1, 1, tzinfo=timezone.utc))
    parser.add_argument("--end", type=_parse_date,
                        default=datetime(2025, 1, 1, tzinfo=timezone.utc))
    parser.add_argument("--data-lake-root", type=Path, default=_DEFAULT_LAKE)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument(
        "--records-path", type=Path, default=None,
        help="Base path for per-bar decision+outcome JSONL. Defaults "
             "to <report-path>.records.jsonl.",
    )
    parser.add_argument(
        "--min-sample-for-signal", type=int, default=30,
        help="Minimum n for a bucket / regime to be reported as "
             "tradeable rather than INCONCLUSIVE.",
    )
    args = parser.parse_args(argv)

    if not args.data_lake_root.exists():
        print(f"data lake root not found: {args.data_lake_root}",
              file=sys.stderr)
        return 2

    lake = ForexDataLake(args.data_lake_root)
    cfg = AtlasConfig(
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        min_sample_for_signal=args.min_sample_for_signal,
    )
    print(
        f"running atlas {cfg.start.date()} → {cfg.end.date()} "
        f"({cfg.instrument}) ..."
    )
    report = run_atlas(cfg, lake)
    print(
        f"  decision records: {len(report.records)} "
        f"(with outcome labels: "
        f"{sum(1 for _, o in report.records if o is not None)})"
    )
    print(f"  halt events:      {len(report.halt_aftermath)}")
    print(f"  range buckets:    {sum(s.count for s in report.range_opportunity)}")
    print(f"  trend buckets:    {sum(s.count for s in report.trend_opportunity)}")

    records_path = args.records_path
    if records_path is None:
        records_path = args.report_path.with_suffix(".records.jsonl")
    halt_path = records_path.with_suffix("").with_suffix(".halt_aftermath.jsonl")

    _write_jsonl(records_path, [
        _record_to_dict(d, o) for d, o in report.records
    ])
    _write_jsonl(halt_path, [_halt_record_to_dict(h) for h in report.halt_aftermath])

    _write_report(report, args.report_path, {
        "decisions": records_path,
        "halt_aftermath": halt_path,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
