"""Phase D-cont2 — Regime / Opportunity Atlas (DIAGNOSTIC ONLY).

Goal: answer the open Phase D-cont1b question — "is dynamic losing
because the classifier is missing opportunities, because the rule
gates are too strict, or because HedgeRock has no edge in these
regimes?" — without adding any new trading knob or touching the
production rule_engine.

Architecture
------------

The atlas is **read-only** with respect to the live system. It runs
on the same Phase D loader (``phase_d_walk_forward._load_data``) so the
no-lookahead invariant is identical: every per-bar artefact uses
**only closed prior bars**. Future H1 bars are used ONLY as
post-hoc *outcome labels* — explicitly marked, never fed into a
classifier or rule_engine call.

Per-bar pipeline
~~~~~~~~~~~~~~~~

For each H1 decision bar:
    1. Build features from strict-prior H1/H4 closed-bar slices
       (delegated to compute_market_features).
    2. classify_regime_v2 → regime + confidence + rule_votes.
    3. derive_envelope_params with a NEUTRAL-COLD EAState
       (equity=10k, dd_pct=0, no recent samples). This decouples the
       diagnostic from any simulator state — we observe what the rule
       engine WANTS to do at clean baseline. Cold history is the
       relevant baseline because it triggers cold_start_grace gating.
    4. Compute the cold-start-grace eligibility waterfall —
       which gate, if any, fails first.
    5. After the main pass: compute outcome labels from H1 bars
       [i+1 .. i+max_horizon]. These are diagnostic labels only.

Aggregations
~~~~~~~~~~~~

- regime × confidence-bucket distribution
- cold_start_grace eligibility waterfall (gate-by-gate survivor count)
- range opportunity atlas (by confidence bucket → mean / median /
  MAE / MFE / range_width future-24h returns)
- trend / breakout missed-opportunity atlas (directional return,
  MAE / MFE)
- halt aftermath atlas (price retracement / vol / regime evolution
  after each first-time halt event the simulator hit)

This module is pure. CLI driver lives in
``scripts/hedgerock_regime_opportunity_atlas.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from statistics import mean, median, pstdev
from typing import Iterable, Literal

import polars as pl

from smc.hedgerock.decision_server import EAState
from smc.hedgerock.forex_data_lake_provider import compute_market_features
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.phase_d_walk_forward import (
    _COLD_START_GRACE_CONF_FLOOR,
    WalkForwardConfig,
    _load_data,
    apply_experiment_overrides,
    ExperimentConfig,
    run_walk_forward,
)
from smc.hedgerock.regime_classifier_v2 import classify_regime_v2
from smc.hedgerock.rule_engine import (
    _RECENT_PNL_MIN_SAMPLE,
    DynamicParams,
    derive_envelope_params,
)


__all__ = [
    "AtlasConfig",
    "AtlasReport",
    "DecisionRecord",
    "GraceWaterfall",
    "HaltAftermathRecord",
    "OutcomeLabel",
    "RegimeBucketStat",
    "BUCKET_LABELS",
    "BUCKET_BAND_LO",
    "BUCKET_BAND_HI",
    "BUCKET_GRACE",
    "BUCKET_RANGE_2",
    "BUCKET_AGGRESSIVE",
    "build_neutral_cold_ea_state",
    "confidence_bucket",
    "compute_outcome_labels",
    "first_grace_failure_gate",
    "merge_live_envelope_entry",
    "aggregate_grace_waterfall",
    "aggregate_regime_x_confidence",
    "aggregate_range_opportunity",
    "aggregate_trend_opportunity",
    "find_halt_aftermath",
    "run_atlas",
]


# ---------------------------------------------------------------------------
# Confidence buckets — tuned to the production thresholds
# ---------------------------------------------------------------------------

BUCKET_BAND_LO: str = "<0.45"
BUCKET_GRACE: str = "0.45-0.55"
BUCKET_BAND_HI: str = "0.55-0.65"
BUCKET_RANGE_2: str = "0.65-0.80"
BUCKET_AGGRESSIVE: str = ">=0.80"

BUCKET_LABELS: tuple[str, ...] = (
    BUCKET_BAND_LO,
    BUCKET_GRACE,
    BUCKET_BAND_HI,
    BUCKET_RANGE_2,
    BUCKET_AGGRESSIVE,
)


def confidence_bucket(confidence: float) -> str:
    """Map raw confidence ∈ [0, 1] into one of the named buckets.

    The boundaries align with rule_engine constants:
      - 0.45 = ``_COLD_START_GRACE_CONF_FLOOR``
      - 0.55 = ``_CONFIDENCE_OBSERVE_FLOOR``
      - 0.65 = range#2 confidence baseline
      - 0.80 = ``_CONFIDENCE_AGGRESSIVE_FLOOR``
    """
    if confidence < 0.45:
        return BUCKET_BAND_LO
    if confidence < 0.55:
        return BUCKET_GRACE
    if confidence < 0.65:
        return BUCKET_BAND_HI
    if confidence < 0.80:
        return BUCKET_RANGE_2
    return BUCKET_AGGRESSIVE


# ---------------------------------------------------------------------------
# Grace eligibility waterfall — names are the SAME ORDER as
# _eligible_for_cold_start_grace, so a single failure point is easy
# to attribute. "" means all gates passed.
# ---------------------------------------------------------------------------


GraceGate = Literal[
    "",  # all gates passed
    "rule_mode_not_observe",
    "cooldown_active",
    "regime_not_range",
    "confidence_below_grace_floor",
    "confidence_at_or_above_observe_floor",
    "risk_snapshot_incomplete",
    "history_already_warm",
]

GRACE_GATES: tuple[GraceGate, ...] = (
    "rule_mode_not_observe",
    "cooldown_active",
    "regime_not_range",
    "confidence_below_grace_floor",
    "confidence_at_or_above_observe_floor",
    "risk_snapshot_incomplete",
    "history_already_warm",
)


def first_grace_failure_gate(
    *, mode: str, cooldown_active: bool, regime: str, confidence: float,
    dd_pct_present: bool, spread_pts_present: bool,
    recent_sample_count: int | None,
) -> GraceGate:
    """Return the FIRST gate that fails (or "" if all pass).

    Mirrors :func:`smc.hedgerock.phase_d_walk_forward._eligible_for_cold_start_grace`
    semantics, but operates on flat scalars so it can be unit-tested in
    isolation.
    """
    if mode != "observe":
        return "rule_mode_not_observe"
    if cooldown_active:
        return "cooldown_active"
    if regime != "range":
        return "regime_not_range"
    if confidence < _COLD_START_GRACE_CONF_FLOOR:
        return "confidence_below_grace_floor"
    if confidence >= 0.55:
        return "confidence_at_or_above_observe_floor"
    if not (dd_pct_present and spread_pts_present):
        return "risk_snapshot_incomplete"
    if (
        recent_sample_count is not None
        and recent_sample_count >= _RECENT_PNL_MIN_SAMPLE
    ):
        return "history_already_warm"
    return ""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasConfig:
    instrument: str = "XAUUSD"
    start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
    h4_lookback: int = 60
    h1_lookback: int = 240
    spread_pts: int = 20
    # Future-outcome horizons in H1 bars (post-hoc labels only).
    future_horizons: tuple[int, ...] = (1, 4, 12, 24)
    # Halt aftermath horizons in H1 bars after a halt-trigger event.
    halt_aftermath_horizons: tuple[int, ...] = (4, 12, 24, 72)
    # Sample-size floor below which an aggregate is reported as
    # INCONCLUSIVE rather than as a tradeable signal.
    min_sample_for_signal: int = 30


@dataclass(frozen=True)
class DecisionRecord:
    """Per-H1 decision-side record. Uses strict prior closed bars only.

    Two parallel rule-engine traces are recorded per bar:

    1. **NEUTRAL-COLD raw intent** (``rule_*`` fields): rule_engine output
       given a clean baseline EAState (full equity, no DD, no positions,
       no recent samples), no transition_lock veto, no cooldown
       carryover. Answers "what does the system want to do here when
       nothing has gone wrong yet?". Useful for diagnosing classifier
       gating; NOT what the live EA executes.

    2. **Live-equivalent execution** (``live_*`` fields): the SAME bar's
       outcome from the Phase D dynamic baseline simulator. Captures
       cumulative simulator state — DD-induced halt cascades, cooldown
       carryover, transition_lock veto. Answers "what did/would the
       live EA actually run here?". ``None``-valued whenever the
       dynamic loop hadn't yet emitted an envelope for this bar (e.g.
       lookback warmup at start of window).

    Phase D-cont2-hotfix-1: pre-hotfix the record only had (1). Reports
    that quoted ``rule_mode`` were silently mixing raw intent with
    live execution. The two are now strictly separated and the report
    labels each by its source.
    """

    ts: datetime
    # classifier output
    regime: str
    confidence: float
    confidence_bucket: str
    classifier_reason: str
    rule_votes: tuple[tuple[str, float, str], ...]
    # legacy features (also lookahead-clean — sourced from prior bars)
    volatility_rank: float
    h4_trend_bars: int
    hh_count: int
    ll_count: int

    # ---- (1) NEUTRAL-COLD raw intent ----
    rule_mode: str
    rule_risk_tier: str
    rule_reason: str
    rule_lot_factor: float
    rule_cooldown_active: bool

    # cold-start-grace eligibility (computed against rule_* / NEUTRAL-COLD)
    grace_failed_at: GraceGate
    grace_eligible: bool

    # ---- (2) Live-equivalent execution from Phase D dynamic baseline ----
    # ``live_raw_*`` is what rule_engine returned BEFORE the
    # transition_lock veto, GIVEN the simulator's accumulated state.
    # ``live_effective_mode`` is what actually drove the simulator.
    live_raw_mode: str | None = None
    live_raw_reason: str | None = None
    live_effective_mode: str | None = None
    live_risk_tier: str | None = None
    live_lot_factor: float | None = None
    live_cooldown_active: bool = False
    live_cooldown_until: datetime | None = None
    live_transition_lock_active: bool = False
    live_transition_lock_until_ts: datetime | None = None


@dataclass(frozen=True)
class OutcomeLabel:
    """Post-hoc outcome label. NEVER fed into a decision call.

    Returns are raw price log returns over the horizon. MAE / MFE are
    measured intra-window over [ts, ts + 24*h1] using H1 high / low.
    range_width_24h is (max_high - min_low) / decision_close.
    """

    decision_close: float
    returns: dict[int, float]  # horizon (H1 bars) → log return
    mae_24h: float  # max adverse excursion (signed: <= 0 from entry close)
    mfe_24h: float  # max favorable excursion (signed: >= 0 from entry close)
    range_width_24h: float


@dataclass(frozen=True)
class GraceWaterfallStage:
    name: str
    survivors: int


@dataclass(frozen=True)
class GraceWaterfall:
    """Eligibility waterfall — survivor count after each gate.

    ``stages[0].survivors`` is the total number of bars considered.
    Each subsequent stage filters by the corresponding gate, so the
    counts are monotone non-increasing.
    """

    stages: tuple[GraceWaterfallStage, ...]


@dataclass(frozen=True)
class RegimeBucketStat:
    """Outcome-distribution summary for a (regime, confidence_bucket) cell."""

    regime: str
    bucket: str
    count: int
    mean_return_24h: float
    median_return_24h: float
    stdev_return_24h: float
    mean_mae_24h: float
    mean_mfe_24h: float
    mean_range_width_24h: float


@dataclass(frozen=True)
class HaltAftermathRecord:
    """Per-halt-event tracker — what happened in the N hours after a
    leading-edge halt-trigger fired."""

    halt_ts: datetime
    halt_close: float
    halt_volatility_rank: float
    halt_regime: str
    # For each horizon: (return, max_favorable, max_adverse, regime_at_t,
    # vol_rank_at_t)
    horizons: dict[int, dict]


@dataclass(frozen=True)
class AtlasReport:
    config: AtlasConfig
    records: list[tuple[DecisionRecord, OutcomeLabel | None]]
    waterfall: GraceWaterfall
    regime_x_bucket_count: dict[tuple[str, str], int]
    range_opportunity: list[RegimeBucketStat]
    trend_opportunity: list[RegimeBucketStat]
    halt_aftermath: list[HaltAftermathRecord]


# ---------------------------------------------------------------------------
# Neutral-cold EAState — diagnostic baseline that does NOT depend on
# any simulator runtime. Constructed once per atlas run.
# ---------------------------------------------------------------------------


def build_neutral_cold_ea_state(
    *, init_equity: float = 10_000.0, spread_pts: int = 20,
) -> EAState:
    """An EAState with full equity, no DD, no positions, no closed-deal
    history. This is the cleanest baseline for asking "what would the
    rule engine want here, untouched by past simulator events?"."""
    return EAState(
        equity=init_equity,
        balance=init_equity,
        dd_pct=0.0,
        free_margin=init_equity,
        margin_level=999.0,
        open_lots=0.0,
        open_positions=0,
        floating_pnl=0.0,
        spread_pts=spread_pts,
        consec_losses=None,
        recent_closed_pnl=None,
        recent_sample_count=None,
    )


# ---------------------------------------------------------------------------
# Per-bar pipeline
# ---------------------------------------------------------------------------


def _decide_bar(
    *,
    ts: datetime, h1_df: pl.DataFrame, h4_df: pl.DataFrame, atr: float | None,
    spread_pts: int, neutral_ea: EAState,
) -> DecisionRecord | None:
    """Single decision-side step. Returns None when features can't be
    computed (lookback short, ATR missing, etc.)."""
    if h4_df is None or h1_df is None or atr is None or atr <= 0:
        return None
    try:
        features = compute_market_features(h4_df=h4_df, h1_df=h1_df)
    except Exception:
        return None

    assessment = classify_regime_v2(
        volatility_rank=features.volatility_rank,
        h4_trend_bars=features.h4_trend_bars,
        hh_count=features.hh_count,
        ll_count=features.ll_count,
        news_intensity=None,
        spread_pts=spread_pts,
    )
    market_state = aggregate_market_state(
        symbol="XAUUSD", now=ts,
        features=features,
        regime_assessment=assessment,
        ea_state=neutral_ea,
        ea_state_recorded_at=ts - timedelta(seconds=5),
    )
    params = derive_envelope_params(market_state, prev_envelope=None)

    cooldown_active = params.cooldown_until is not None
    gate = first_grace_failure_gate(
        mode=params.mode,
        cooldown_active=cooldown_active,
        regime=assessment.regime,
        confidence=assessment.confidence,
        dd_pct_present=neutral_ea.dd_pct is not None,
        spread_pts_present=neutral_ea.spread_pts is not None,
        recent_sample_count=neutral_ea.recent_sample_count,
    )

    return DecisionRecord(
        ts=ts,
        regime=assessment.regime,
        confidence=assessment.confidence,
        confidence_bucket=confidence_bucket(assessment.confidence),
        classifier_reason=assessment.reason,
        rule_votes=tuple(
            (str(v[0]), float(v[1]), str(v[2])) for v in assessment.rule_votes
        ),
        volatility_rank=features.volatility_rank,
        h4_trend_bars=features.h4_trend_bars,
        hh_count=features.hh_count,
        ll_count=features.ll_count,
        rule_mode=params.mode,
        rule_risk_tier=params.risk_tier,
        rule_reason=params.reason,
        rule_lot_factor=params.lot_factor,
        rule_cooldown_active=cooldown_active,
        grace_failed_at=gate,
        grace_eligible=(gate == ""),
    )


def _parse_iso(value) -> datetime | None:
    """Tolerant ISO-8601 parser. Envelope-log timestamps are stringified
    via ``ts.isoformat()`` and may be ``None`` for unset fields."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def merge_live_envelope_entry(
    record: DecisionRecord, entry: dict | None,
) -> DecisionRecord:
    """Augment a NEUTRAL-COLD :class:`DecisionRecord` with live-equivalent
    fields read from a Phase D dynamic-simulator envelope-log entry.

    Phase D-cont2-hotfix-1: callers MUST pass through this helper instead
    of constructing the live_* fields ad-hoc, so the field naming and
    parsing rules stay in one place.

    Returns a *new* record (frozen dataclass). When ``entry`` is None
    (no envelope log line for this ts — e.g. lookback warmup), the
    record is returned with all live_* fields at their defaults
    (None / False).
    """
    if entry is None:
        return record
    return replace(
        record,
        live_raw_mode=entry.get("mode"),
        live_raw_reason=entry.get("reason"),
        live_effective_mode=entry.get("effective_mode"),
        live_risk_tier=entry.get("risk_tier"),
        live_lot_factor=entry.get("lot_factor"),
        live_cooldown_active=entry.get("cooldown_until") is not None,
        live_cooldown_until=_parse_iso(entry.get("cooldown_until")),
        live_transition_lock_active=bool(entry.get("transition_lock_active")),
        live_transition_lock_until_ts=_parse_iso(
            entry.get("transition_lock_until_ts")
        ),
    )


def compute_outcome_labels(
    *, decision_idx: int, h1_bars: pl.DataFrame,
    horizons: tuple[int, ...] = (1, 4, 12, 24),
) -> OutcomeLabel | None:
    """Compute future-bar outcome labels.

    NO LOOKAHEAD INVARIANT: the decision was made AT bar
    ``decision_idx`` using strict-prior data. This function uses bars
    ``[decision_idx + 1, decision_idx + max(horizons)]`` exclusively.
    The bar ``decision_idx`` itself is the entry-close anchor and is
    NOT used as a future bar.

    Returns ``None`` when there aren't enough future bars to fill the
    largest horizon — better to drop than to silently emit short
    labels.
    """
    n = h1_bars.height
    max_h = max(horizons)
    if decision_idx + max_h >= n:
        return None
    decision_close = float(h1_bars["close"][decision_idx])

    returns: dict[int, float] = {}
    closes = h1_bars["close"]
    highs = h1_bars["high"]
    lows = h1_bars["low"]
    for h in horizons:
        future_close = float(closes[decision_idx + h])
        # Simple percent return — we don't need log-precision here, and
        # raw % matches the way the report tables read.
        returns[h] = (future_close - decision_close) / decision_close * 100.0

    # MAE / MFE / range_width over the next 24 H1 bars (or whatever
    # is in horizons that's largest, but we explicitly use 24h).
    window_end = decision_idx + 24
    if window_end >= n:
        return None
    window_high = max(float(highs[i]) for i in range(decision_idx + 1, window_end + 1))
    window_low = min(float(lows[i]) for i in range(decision_idx + 1, window_end + 1))
    mfe = (window_high - decision_close) / decision_close * 100.0
    mae = (window_low - decision_close) / decision_close * 100.0
    range_width = (window_high - window_low) / decision_close * 100.0

    return OutcomeLabel(
        decision_close=decision_close,
        returns=returns,
        mae_24h=mae,
        mfe_24h=mfe,
        range_width_24h=range_width,
    )


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def aggregate_grace_waterfall(records: Iterable[DecisionRecord]) -> GraceWaterfall:
    """Build a survivor-count waterfall by re-evaluating each gate
    in declared order. Survivor count after gate K = bars that did
    NOT fail at any gate ≤ K."""
    rec_list = list(records)
    total = len(rec_list)
    stages: list[GraceWaterfallStage] = [
        GraceWaterfallStage(name="total", survivors=total),
    ]
    # For each gate G, survivor count = bars where grace_failed_at is
    # either "" (all passed) OR a gate with index > position of G in
    # GRACE_GATES (i.e. failed strictly after G).
    for k, gate in enumerate(GRACE_GATES):
        survivors = 0
        for r in rec_list:
            if r.grace_failed_at == "":
                survivors += 1
            else:
                # If failed at gate F, survives gate G iff F > G.
                f_idx = GRACE_GATES.index(r.grace_failed_at)
                if f_idx > k:
                    survivors += 1
        stages.append(GraceWaterfallStage(
            name=f"after_{gate}_check",
            survivors=survivors,
        ))
    return GraceWaterfall(stages=tuple(stages))


def aggregate_regime_x_confidence(
    records: Iterable[DecisionRecord],
) -> dict[tuple[str, str], int]:
    """(regime, confidence_bucket) → count."""
    out: dict[tuple[str, str], int] = {}
    for r in records:
        key = (r.regime, r.confidence_bucket)
        out[key] = out.get(key, 0) + 1
    return out


def _bucket_stats(
    *, regime: str, bucket: str,
    samples: list[OutcomeLabel],
) -> RegimeBucketStat:
    if not samples:
        return RegimeBucketStat(
            regime=regime, bucket=bucket, count=0,
            mean_return_24h=0.0, median_return_24h=0.0, stdev_return_24h=0.0,
            mean_mae_24h=0.0, mean_mfe_24h=0.0, mean_range_width_24h=0.0,
        )
    rets = [s.returns.get(24, 0.0) for s in samples]
    return RegimeBucketStat(
        regime=regime,
        bucket=bucket,
        count=len(samples),
        mean_return_24h=mean(rets),
        median_return_24h=median(rets),
        stdev_return_24h=pstdev(rets) if len(rets) > 1 else 0.0,
        mean_mae_24h=mean(s.mae_24h for s in samples),
        mean_mfe_24h=mean(s.mfe_24h for s in samples),
        mean_range_width_24h=mean(s.range_width_24h for s in samples),
    )


def aggregate_range_opportunity(
    records: list[tuple[DecisionRecord, OutcomeLabel | None]],
) -> list[RegimeBucketStat]:
    """Range-only outcome distribution by confidence bucket."""
    by_bucket: dict[str, list[OutcomeLabel]] = {b: [] for b in BUCKET_LABELS}
    for d, o in records:
        if d.regime != "range" or o is None:
            continue
        by_bucket[d.confidence_bucket].append(o)
    return [
        _bucket_stats(regime="range", bucket=b, samples=by_bucket[b])
        for b in BUCKET_LABELS
    ]


def aggregate_trend_opportunity(
    records: list[tuple[DecisionRecord, OutcomeLabel | None]],
) -> list[RegimeBucketStat]:
    """trend_up / trend_down / breakout outcome distribution.

    For trend_up: directional return = +1 * next_24h_return
    For trend_down: directional return = -1 * next_24h_return (so a
        trend_down regime that drops 1.5% scores +1.5% as "trend
        followed expectation").

    Phase D-cont2-hotfix-2: breakout is split into TWO rows.
        ``breakout_magnitude`` (|return| view) is documented as
        magnitude-only. The classifier doesn't tell us which side, so
        a positive |return| mean has no execution path — taking a
        position in either direction faces 50/50 outcome at entry.
        This row is reported but the verdict logic must NEVER treat
        it as an E1 momentum candidate.

        ``breakout_signed_by_h4`` uses ``h4_trend_bars`` sign as a
        direction proxy: positive H4 trend → expect breakout up;
        negative → expect breakout down. Bars where the proxy is
        zero (no H4 trend either way) are dropped — the proxy is
        ambiguous there. This row IS allowed to be E1-eligible if
        its signed CI clears zero.
    """
    out: list[RegimeBucketStat] = []
    # trend_up — direct copy
    out.append(_bucket_stats(
        regime="trend_up", bucket="all",
        samples=[o for d, o in records if d.regime == "trend_up" and o is not None],
    ))
    # trend_down — re-sign
    td_samples: list[OutcomeLabel] = []
    for d, o in records:
        if d.regime != "trend_down" or o is None:
            continue
        td_samples.append(OutcomeLabel(
            decision_close=o.decision_close,
            returns={h: -v for h, v in o.returns.items()},
            mae_24h=-o.mfe_24h,
            mfe_24h=-o.mae_24h,
            range_width_24h=o.range_width_24h,
        ))
    out.append(_bucket_stats(regime="trend_down", bucket="all", samples=td_samples))

    # breakout (magnitude) — |return|; NOT a direction-aware signal.
    bk_mag_samples: list[OutcomeLabel] = []
    for d, o in records:
        if d.regime != "breakout" or o is None:
            continue
        bk_mag_samples.append(OutcomeLabel(
            decision_close=o.decision_close,
            returns={h: abs(v) for h, v in o.returns.items()},
            mae_24h=0.0,
            mfe_24h=max(abs(o.mae_24h), abs(o.mfe_24h)),
            range_width_24h=o.range_width_24h,
        ))
    out.append(_bucket_stats(
        regime="breakout_magnitude", bucket="|return|, no direction",
        samples=bk_mag_samples,
    ))

    # breakout (signed by H4 trend proxy) — only bars where h4_trend_bars
    # has a non-zero sign. Zero-trend bars are dropped because the proxy
    # provides no direction.
    bk_signed_samples: list[OutcomeLabel] = []
    for d, o in records:
        if d.regime != "breakout" or o is None:
            continue
        sign = 0
        if d.h4_trend_bars > 0:
            sign = +1
        elif d.h4_trend_bars < 0:
            sign = -1
        if sign == 0:
            continue
        if sign > 0:
            bk_signed_samples.append(OutcomeLabel(
                decision_close=o.decision_close,
                returns={h: v for h, v in o.returns.items()},
                mae_24h=o.mae_24h, mfe_24h=o.mfe_24h,
                range_width_24h=o.range_width_24h,
            ))
        else:
            bk_signed_samples.append(OutcomeLabel(
                decision_close=o.decision_close,
                returns={h: -v for h, v in o.returns.items()},
                mae_24h=-o.mfe_24h, mfe_24h=-o.mae_24h,
                range_width_24h=o.range_width_24h,
            ))
    out.append(_bucket_stats(
        regime="breakout_signed_by_h4", bucket="h4_trend proxy",
        samples=bk_signed_samples,
    ))
    return out


def find_halt_aftermath(
    *,
    h1_bars: pl.DataFrame,
    envelope_log: list[dict],
    horizons: tuple[int, ...] = (4, 12, 24, 72),
    decision_records_by_ts: dict[datetime, DecisionRecord] | None = None,
) -> list[HaltAftermathRecord]:
    """Find each leading-edge halt-trigger event in the envelope log
    and build a forward look at price / volatility / regime evolution.

    A halt event is a transition from non-halt → halt EFFECTIVE mode.
    """
    if not envelope_log:
        return []

    # Index H1 bars by ts for fast lookup.
    ts_list = h1_bars["ts"].to_list()
    ts_to_idx: dict[datetime, int] = {ts: i for i, ts in enumerate(ts_list)}
    closes = h1_bars["close"]
    highs = h1_bars["high"]
    lows = h1_bars["low"]
    n = h1_bars.height

    out: list[HaltAftermathRecord] = []
    prev_mode: str | None = None
    for entry in envelope_log:
        mode = entry.get("effective_mode") or entry.get("mode")
        if mode == "halt" and prev_mode != "halt":
            # Leading-edge halt event.
            ts_str = entry["ts"]
            ts_dt = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
            i = ts_to_idx.get(ts_dt)
            if i is None or i + max(horizons) >= n:
                prev_mode = mode
                continue
            halt_close = float(closes[i])
            # Pull regime / vol-rank from the matching decision record
            # if provided (otherwise fall back to envelope log fields).
            dec = (decision_records_by_ts or {}).get(ts_dt)
            halt_regime = dec.regime if dec else entry.get("regime_v2", "")
            halt_volrank = dec.volatility_rank if dec else 0.0

            horizon_dict: dict[int, dict] = {}
            for h in horizons:
                if i + h >= n:
                    continue
                future_close = float(closes[i + h])
                ret = (future_close - halt_close) / halt_close * 100.0
                window_high = max(float(highs[j]) for j in range(i + 1, i + h + 1))
                window_low = min(float(lows[j]) for j in range(i + 1, i + h + 1))
                mfe = (window_high - halt_close) / halt_close * 100.0
                mae = (window_low - halt_close) / halt_close * 100.0
                # Regime / vol_rank at horizon — read from atlas decision
                # records keyed by ts of the H1 bar at i+h.
                future_ts = ts_list[i + h]
                future_dec = (decision_records_by_ts or {}).get(future_ts)
                horizon_dict[h] = {
                    "return_pct": ret,
                    "mae_pct": mae,
                    "mfe_pct": mfe,
                    "regime_at_t": future_dec.regime if future_dec else None,
                    "vol_rank_at_t": future_dec.volatility_rank if future_dec else None,
                }
            out.append(HaltAftermathRecord(
                halt_ts=ts_dt,
                halt_close=halt_close,
                halt_volatility_rank=halt_volrank,
                halt_regime=halt_regime,
                horizons=horizon_dict,
            ))
        prev_mode = mode
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_atlas(config: AtlasConfig, lake) -> AtlasReport:
    """Run the full atlas: NEUTRAL-COLD classifier+rule pass joined per-bar
    with a Phase D dynamic-baseline replay envelope log.

    Phase D-cont2-hotfix-1: each bar's :class:`DecisionRecord` carries
    BOTH the NEUTRAL-COLD raw classifier intent (``rule_*`` fields) and
    the live-equivalent execution trace (``live_*`` fields, populated
    from the dynamic simulator's envelope log). Reports must label
    each so a reader can never misread raw intent as live execution.

    The dynamic simulator runs once; its envelope log is the single
    source for both the live-equivalent join and the halt aftermath
    atlas.
    """
    wf_config = WalkForwardConfig(
        instrument=config.instrument,
        start=config.start,
        end=config.end,
        h4_lookback=config.h4_lookback,
        h1_lookback=config.h1_lookback,
        spread_pts=config.spread_pts,
    )

    # Dynamic baseline replay first — its envelope log is the source of
    # truth for the live-equivalent fields. No experiments → byte-
    # identical to Phase D baseline dynamic.
    wf_result = run_walk_forward(wf_config, lake)
    live_by_ts: dict[datetime, dict] = {}
    for entry in wf_result.envelope_log:
        ts_parsed = _parse_iso(entry.get("ts"))
        if ts_parsed is not None:
            live_by_ts[ts_parsed] = entry

    h1, atr_per_bar, h4_frames, h1_frames = _load_data(lake, wf_config)
    n = h1.height
    ts_list = h1["ts"].to_list()
    neutral_ea = build_neutral_cold_ea_state(spread_pts=config.spread_pts)

    records: list[tuple[DecisionRecord, OutcomeLabel | None]] = []
    decision_by_ts: dict[datetime, DecisionRecord] = {}
    for i in range(n):
        ts = ts_list[i]
        rec = _decide_bar(
            ts=ts,
            h1_df=h1_frames[i],
            h4_df=h4_frames[i],
            atr=atr_per_bar[i],
            spread_pts=config.spread_pts,
            neutral_ea=neutral_ea,
        )
        if rec is None:
            continue
        # Merge the live-equivalent envelope entry for this ts (or no-op
        # when the dynamic loop hadn't emitted yet — lookback warmup).
        rec = merge_live_envelope_entry(rec, live_by_ts.get(ts))
        outcome = compute_outcome_labels(
            decision_idx=i, h1_bars=h1, horizons=config.future_horizons,
        )
        records.append((rec, outcome))
        decision_by_ts[ts] = rec

    decisions_only = [d for d, _ in records]
    waterfall = aggregate_grace_waterfall(decisions_only)
    rxc = aggregate_regime_x_confidence(decisions_only)
    range_atlas = aggregate_range_opportunity(records)
    trend_atlas = aggregate_trend_opportunity(records)

    halt_after = find_halt_aftermath(
        h1_bars=h1,
        envelope_log=wf_result.envelope_log,
        horizons=config.halt_aftermath_horizons,
        decision_records_by_ts=decision_by_ts,
    )

    return AtlasReport(
        config=config,
        records=records,
        waterfall=waterfall,
        regime_x_bucket_count=rxc,
        range_opportunity=range_atlas,
        trend_opportunity=trend_atlas,
        halt_aftermath=halt_after,
    )
