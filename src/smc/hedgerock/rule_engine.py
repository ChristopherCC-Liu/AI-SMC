"""Phase C — translate :class:`MarketState` into dynamic envelope params.

The rule engine is the single place that decides:
    - ``mode`` (hedgerock / observe / halt / momentum)
    - ``hedgerock_enabled``
    - ``risk_tier`` (observe / normal / aggressive / attack)
    - ``lot_factor``
    - ``grid_multiplier``
    - ``max_next_lot``
    - ``takeprofit_points`` / ``stoploss_points``
    - ``recovery_multiplier``
    - ``max_orders_buy`` / ``max_orders_sell``
    - ``cooldown_until``
    - ``reason`` (logged for replay)

It does NOT directly emit a :class:`SignalEnvelope` — that's
build_envelope's job. The engine returns a frozen
:class:`DynamicParams` which the /signal endpoint passes through to
``build_envelope(...)`` as kwargs.

Contract (from dynamic-decision-center-rfc §3 + Q5):
    - **Output is purely a function of** ``state`` + ``prev_envelope``.
    - **Numeric output is always within the schema bounds**
      (lot_factor ∈ [0, 5], grid_multiplier ∈ (0, 10], etc.). EA-side
      ``MaxAllowed*`` clamps further. We never exceed schema; EA never
      exceeds operator inputs. Two layers of defense.
    - **Stale EA state degrades aggressively** — ``ea_state_stale=True``
      forces ``mode=observe`` regardless of regime. Phase C's job is
      to prefer being wrong about an opportunity than wrong about
      risk.

Step-up / step-down (the *dynamic* part):
    - Range + high confidence + DD low + low spread → step UP
      (lot_factor 1.5, max_next_lot 0.10, ...)
    - DD elevated OR consec_losses ≥ 3 → step DOWN
      (lot_factor 0.5, max_next_lot 0.03, ...)
    - DD severe (>= 5%) OR consec_losses ≥ 5 → cooldown 30 min
    - Spread anomalous (>= 80 pts on XAUUSD) → cooldown 5 min

These thresholds are deliberately conservative and explicit — Phase D
walk-forward will tune them; the architecture (one place to change)
matters more than the values today.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Final

from smc.hedgerock.market_state import MarketState
from smc.hedgerock.schemas import (
    Mode,
    RegimeV2,
    RiskTier,
    SignalEnvelope,
)

__all__ = [
    "DynamicParams",
    "derive_envelope_params",
]


# ---------------------------------------------------------------------------
# Thresholds — change in one place, all of Phase C reflows
# ---------------------------------------------------------------------------

# DD step-down trigger: drawdown % at which we cut sizing in half.
_DD_STEPDOWN_THRESHOLD: Final[float] = 0.02   # 2 %
# DD severe: closeall + cooldown.
_DD_SEVERE_THRESHOLD: Final[float] = 0.05     # 5 %

# Consecutive losses: step-down at 3, cooldown at 5.
_CONSEC_LOSSES_STEPDOWN: Final[int] = 3
_CONSEC_LOSSES_COOLDOWN: Final[int] = 5

# Spread anomalous (XAUUSD): pts.
_SPREAD_PTS_COOLDOWN: Final[int] = 80

# Cooldown durations.
_COOLDOWN_DD_SECONDS: Final[int] = 30 * 60       # 30 min
_COOLDOWN_LOSS_SECONDS: Final[int] = 30 * 60     # 30 min
_COOLDOWN_SPREAD_SECONDS: Final[int] = 5 * 60    # 5 min
_COOLDOWN_NEWS_SECONDS: Final[int] = 15 * 60     # 15 min

# Confidence floors — below which we don't trust the regime call.
_CONFIDENCE_OBSERVE_FLOOR: Final[float] = 0.55
_CONFIDENCE_AGGRESSIVE_FLOOR: Final[float] = 0.80

# Recent realized-PnL gates (Phase C-hotfix #4). Tunable in Phase D
# walk-forward.
_RECENT_PNL_MIN_SAMPLE: Final[int] = 5
_RECENT_PNL_STEPDOWN_THRESHOLD: Final[float] = -50.0  # USD net realized loss


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicParams:
    """Frozen kwargs payload for :func:`build_envelope`.

    The /signal endpoint expands this with ``**params.to_kwargs()``;
    fields are aligned 1:1 with build_envelope's v2.0.0 dynamic kwargs.
    """

    mode: Mode
    hedgerock_enabled: bool
    risk_tier: RiskTier
    cooldown_until: datetime | None
    lot_factor: float
    grid_multiplier: float
    max_next_lot: float
    takeprofit_points: int
    stoploss_points: int
    recovery_multiplier: float
    max_orders_buy: int
    max_orders_sell: int
    reason: str

    def to_kwargs(self) -> dict:
        """Return the kwargs dict suitable for ``build_envelope(**kwargs)``."""
        return {
            "mode": self.mode,
            "hedgerock_enabled": self.hedgerock_enabled,
            "risk_tier": self.risk_tier,
            "cooldown_until": self.cooldown_until,
            "lot_factor": self.lot_factor,
            "grid_multiplier": self.grid_multiplier,
            "max_next_lot": self.max_next_lot,
            "takeprofit_points": self.takeprofit_points,
            "stoploss_points": self.stoploss_points,
            "recovery_multiplier": self.recovery_multiplier,
            "max_orders_buy": self.max_orders_buy,
            "max_orders_sell": self.max_orders_sell,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_observe(reason: str, *, now: datetime,
                  cooldown_until: datetime | None = None) -> DynamicParams:
    """Construct a 'no new entries, manage existing' params block."""
    return DynamicParams(
        mode="observe",
        hedgerock_enabled=False,
        risk_tier="observe",
        cooldown_until=cooldown_until,
        lot_factor=0.0,
        grid_multiplier=1.0,
        max_next_lot=0.05,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=2,
        max_orders_sell=2,
        reason=reason,
    )


def _halt(reason: str, *, now: datetime,
          cooldown_until: datetime | None = None) -> DynamicParams:
    """Halt mode — closeall + block."""
    return DynamicParams(
        mode="halt",
        hedgerock_enabled=False,
        risk_tier="observe",
        cooldown_until=cooldown_until,
        lot_factor=0.0,
        grid_multiplier=1.0,
        max_next_lot=0.05,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=0,
        max_orders_sell=0,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Per-regime base profile
#
# These numbers are intentionally conservative. Phase D walk-forward
# will tune them; the architecture (one knob per dimension, all routed
# through DynamicParams + EA clamps) is what matters today.
# ---------------------------------------------------------------------------


def _hedgerock_base_for_regime(regime: RegimeV2) -> dict:
    """Per-regime base sizing — pre-step-up / pre-step-down."""
    if regime == "range":
        return dict(
            lot_factor=1.0,
            grid_multiplier=1.0,
            max_next_lot=0.05,
            takeprofit_points=600,
            stoploss_points=3900,
            recovery_multiplier=1.2,
            max_orders_buy=2,
            max_orders_sell=2,
        )
    # trend_up / trend_down / breakout / unknown — HedgeRock not enabled
    # (handled by mode selector). This branch only fires inside
    # mode==hedgerock, which we restrict to ``range`` regime.
    raise ValueError(f"_hedgerock_base_for_regime called with non-range regime {regime}")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def _apply_cooldown_carryover(
    params: DynamicParams,
    prev_envelope: SignalEnvelope | None,
    now: datetime,
) -> DynamicParams:
    """Phase C-hotfix #1: persist cooldown across /signal calls.

    If ``prev_envelope.cooldown_until`` is still in the future:
        - take ``max(prev_cd, newly_requested_cd)`` as effective
        - if the new params want ``mode=hedgerock``, demote to observe
          and carry the lock forward (do not let a single 'all clear'
          poll erase a cooldown a stricter branch set seconds ago)

    halt branches still set their own cooldown; we just merge with prev
    so the LONGER one wins.
    """
    if prev_envelope is None or prev_envelope.cooldown_until is None:
        return params
    prev_cd = prev_envelope.cooldown_until
    if prev_cd <= now:
        return params  # prev cooldown expired — no carryover

    new_cd = params.cooldown_until
    merged = max(new_cd, prev_cd) if new_cd is not None else prev_cd

    if params.mode == "hedgerock":
        # Active cooldown blocks any HedgeRock entry — fall back to observe.
        return DynamicParams(
            mode="observe",
            hedgerock_enabled=False,
            risk_tier="observe",
            cooldown_until=merged,
            lot_factor=0.0,
            grid_multiplier=1.0,
            max_next_lot=0.05,
            takeprofit_points=600,
            stoploss_points=3900,
            recovery_multiplier=1.2,
            max_orders_buy=2,
            max_orders_sell=2,
            reason=(
                f"cooldown still active until {prev_cd.isoformat()} "
                f"(deferred original={params.reason})"
            ),
        )
    # mode is observe / halt — keep params, just merge cooldown.
    return replace(params, cooldown_until=merged)


def derive_envelope_params(
    state: MarketState,
    prev_envelope: SignalEnvelope | None,
) -> DynamicParams:
    """Translate :class:`MarketState` into :class:`DynamicParams`.

    Pure function — no I/O, deterministic given inputs. Caller is
    responsible for plumbing the result into build_envelope.

    Decision tree (highest-priority branch wins):

        1. EA state stale → observe (do not trust EAState numbers)
        2. Crisis regime → halt + 30 min cooldown
        3. News regime → observe + 15 min cooldown
        4. DD severe → halt + 30 min cooldown
        5. Spread anomalous → observe + 5 min cooldown
        6. Consecutive losses ≥ COOLDOWN threshold → observe + cooldown
        7. Trend / breakout / unknown regime → observe (no HedgeRock there)
        7.5 No EA state → observe (production safety — refuse without
            runtime state) — Phase C-hotfix #2
        8. Range regime + low confidence → observe
        9. Range regime + meets confidence floor → hedgerock,
           with step-up / step-down by DD + consec_losses + recent_pnl
        10. Cooldown carryover from prev_envelope (Phase C-hotfix #1)
    """
    now = state.generated_at
    regime = state.regime_assessment.regime
    confidence = state.regime_assessment.confidence
    reason_prefix = state.regime_assessment.reason

    # ------------------------------------------------------------------
    # Helper that wraps every return so cooldown carryover never gets
    # forgotten. This is the only place that builds the final params.
    # ------------------------------------------------------------------
    def _final(p: DynamicParams) -> DynamicParams:
        return _apply_cooldown_carryover(p, prev_envelope, now)

    # 1. Stale state — never compute risk-control off old numbers.
    if state.ea_state_stale:
        return _final(_safe_observe(
            reason=f"ea_state_stale → observe; "
                   f"age={state.ea_state_age_seconds}s; regime={regime}",
            now=now,
        ))

    # 2. Crisis — close all, long cooldown.
    if regime == "crisis":
        return _final(_halt(
            reason=f"regime=crisis → halt + cooldown; {reason_prefix}",
            now=now,
            cooldown_until=now + timedelta(seconds=_COOLDOWN_DD_SECONDS),
        ))

    # 3. News — observe + cooldown.
    if regime == "news":
        return _final(_safe_observe(
            reason=f"regime=news → observe + cooldown; {reason_prefix}",
            now=now,
            cooldown_until=now + timedelta(seconds=_COOLDOWN_NEWS_SECONDS),
        ))

    # Pull EA state metrics (None-safe).
    ea = state.ea_state
    dd_pct = (ea.dd_pct if ea is not None else None)
    consec = (ea.consec_losses if ea is not None else None)
    spread_pts = (ea.spread_pts if ea is not None else None)
    recent_pnl = (ea.recent_closed_pnl if ea is not None else None)
    recent_n = (ea.recent_sample_count if ea is not None else None)

    # 4. DD severe — halt regardless of regime.
    if dd_pct is not None and dd_pct >= _DD_SEVERE_THRESHOLD:
        return _final(_halt(
            reason=f"DD={dd_pct:.4f} ≥ {_DD_SEVERE_THRESHOLD} → halt + cooldown",
            now=now,
            cooldown_until=now + timedelta(seconds=_COOLDOWN_DD_SECONDS),
        ))

    # 5. Spread anomalous — observe + short cooldown.
    if spread_pts is not None and spread_pts >= _SPREAD_PTS_COOLDOWN:
        return _final(_safe_observe(
            reason=f"spread_pts={spread_pts} ≥ {_SPREAD_PTS_COOLDOWN} → observe + cooldown",
            now=now,
            cooldown_until=now + timedelta(seconds=_COOLDOWN_SPREAD_SECONDS),
        ))

    # 6. Consecutive losses cooldown.
    if consec is not None and consec >= _CONSEC_LOSSES_COOLDOWN:
        return _final(_safe_observe(
            reason=f"consec_losses={consec} ≥ {_CONSEC_LOSSES_COOLDOWN} → observe + cooldown",
            now=now,
            cooldown_until=now + timedelta(seconds=_COOLDOWN_LOSS_SECONDS),
        ))

    # 7. Trend / breakout / unknown — HedgeRock not designed for these.
    if regime in ("trend_up", "trend_down", "breakout", "unknown"):
        return _final(_safe_observe(
            reason=f"regime={regime} → observe (HedgeRock disabled outside range)",
            now=now,
        ))

    # 7.5 Phase C-hotfix #2: refuse to enter HedgeRock without a USABLE
    # EA risk snapshot. We require:
    #   - state.ea_state present at all (no_ea_state guard)
    #   - dd_pct present  (without it we can't enforce the DD halt / step-down)
    #   - spread_pts present (without it we can't enforce the spread cooldown)
    # History fields (consec_losses / recent_closed_pnl / recent_sample_count)
    # MAY be unavailable (Patch 1 above blocks aggressive step-up in that
    # case); these two are the minimum risk-snapshot fields.
    if state.ea_state is None:
        return _final(_safe_observe(
            reason="no_ea_state → observe (refuse to enter HedgeRock without runtime state)",
            now=now,
        ))
    missing: list[str] = []
    if state.ea_state.dd_pct is None:
        missing.append("dd_pct")
    if state.ea_state.spread_pts is None:
        missing.append("spread_pts")
    if missing:
        return _final(_safe_observe(
            reason=(
                f"incomplete_ea_state (missing {', '.join(missing)}) → observe; "
                f"refuse HedgeRock without minimum risk snapshot"
            ),
            now=now,
        ))

    # 8. Range with sub-floor confidence → observe.
    if confidence < _CONFIDENCE_OBSERVE_FLOOR:
        return _final(_safe_observe(
            reason=f"confidence={confidence:.2f} < {_CONFIDENCE_OBSERVE_FLOOR} → observe",
            now=now,
        ))

    # 9. Range + good confidence → HedgeRock with step-up / step-down.
    base = _hedgerock_base_for_regime("range")

    # Step DOWN — DD, consec losses, OR realized recent loss past threshold.
    # ``recent_pnl`` of None means HistorySelect failed (sentinel) or the
    # field wasn't sent — must NOT be treated as 0.
    step_down = False
    step_down_reason = ""
    if dd_pct is not None and dd_pct >= _DD_STEPDOWN_THRESHOLD:
        step_down = True
        step_down_reason = f"DD={dd_pct:.4f} ≥ {_DD_STEPDOWN_THRESHOLD}"
    elif consec is not None and consec >= _CONSEC_LOSSES_STEPDOWN:
        step_down = True
        step_down_reason = f"consec_losses={consec} ≥ {_CONSEC_LOSSES_STEPDOWN}"
    elif (
        recent_pnl is not None
        and recent_n is not None
        and recent_n >= _RECENT_PNL_MIN_SAMPLE
        and recent_pnl <= _RECENT_PNL_STEPDOWN_THRESHOLD
    ):
        step_down = True
        step_down_reason = (
            f"recent_closed_pnl={recent_pnl:.2f} ≤ "
            f"{_RECENT_PNL_STEPDOWN_THRESHOLD} (n={recent_n})"
        )

    # Step-UP gates (Phase C-hotfix-3 #2 + Phase C-hotfix-4):
    #
    # Aggressive step-up REQUIRES *complete* positive recent evidence:
    #   - consec_losses     is not None
    #   - recent_sample_count is not None
    #   - recent_sample_count >= _RECENT_PNL_MIN_SAMPLE
    #   - recent_closed_pnl is not None
    #   - recent_closed_pnl >= 0
    #
    # Any failure → cap at normal tier with an annotated reason.
    # Sub-buckets so logs distinguish the cases:
    #   - "recent_pnl_blocks"     n ≥ MIN AND pnl < 0  (known loss)
    #   - "history_unavailable"   recent_n is None     (sentinel / never sent)
    #   - "insufficient_sample"   recent_n < MIN       (too few closed deals)
    #   - "history_incomplete"    n ≥ MIN but pnl OR consec is missing
    #                             — Phase C-hotfix-4: partial reports
    #                             must NOT be treated as positive evidence.
    history_unavailable = recent_n is None
    insufficient_sample = (
        recent_n is not None and recent_n < _RECENT_PNL_MIN_SAMPLE
    )
    recent_pnl_blocks_stepup = (
        recent_pnl is not None
        and recent_n is not None
        and recent_n >= _RECENT_PNL_MIN_SAMPLE
        and recent_pnl < 0.0
    )
    # Phase C-hotfix-4: missing-field detection separate from None-sentinel
    # so the reason string explicitly names the absent piece of evidence.
    missing_recent_pnl = (
        recent_n is not None
        and recent_n >= _RECENT_PNL_MIN_SAMPLE
        and recent_pnl is None
    )
    missing_consec_losses = consec is None

    risk_tier: RiskTier
    if step_down:
        risk_tier = "observe"
        lot_factor = 0.5
        max_next_lot = 0.03
        max_orders = 1
        derived_reason = (
            f"range + step-down ({step_down_reason}); "
            f"{reason_prefix}; conf={confidence:.2f}"
        )
    elif confidence >= _CONFIDENCE_AGGRESSIVE_FLOOR and not (
        recent_pnl_blocks_stepup
        or history_unavailable
        or insufficient_sample
        or missing_recent_pnl
        or missing_consec_losses
    ):
        # Step UP — high confidence in range, no DD pressure, AND
        # complete positive recent evidence (consec + n ≥ MIN + pnl ≥ 0).
        risk_tier = "aggressive"
        lot_factor = 1.5
        max_next_lot = 0.10
        max_orders = 3
        derived_reason = (
            f"range + step-up (conf={confidence:.2f} ≥ "
            f"{_CONFIDENCE_AGGRESSIVE_FLOOR}); {reason_prefix}"
        )
    elif confidence >= _CONFIDENCE_AGGRESSIVE_FLOOR:
        # High confidence but step-up blocked → cap at normal tier.
        # Annotate which blocker fired so logs / decision_log are useful.
        # Order: known-loss > coverage > sample > field-missing.
        if recent_pnl_blocks_stepup:
            blocker = f"recent_pnl={recent_pnl:.2f}<0 on n={recent_n}"
        elif history_unavailable:
            blocker = "history_unavailable (recent_sample_count=None)"
        elif insufficient_sample:
            blocker = (
                f"insufficient_sample (recent_n={recent_n} < "
                f"{_RECENT_PNL_MIN_SAMPLE})"
            )
        elif missing_recent_pnl:
            blocker = "history_incomplete (missing recent_closed_pnl)"
        else:  # missing_consec_losses
            blocker = "history_incomplete (missing consec_losses)"
        risk_tier = "normal"
        lot_factor = 1.0
        max_next_lot = base["max_next_lot"]
        max_orders = 2
        derived_reason = (
            f"range + step-up blocked ({blocker}); using normal tier; "
            f"{reason_prefix}"
        )
    else:
        risk_tier = "normal"
        lot_factor = 1.0
        max_next_lot = base["max_next_lot"]
        max_orders = 2
        derived_reason = f"range + normal; conf={confidence:.2f}; {reason_prefix}"

    return _final(DynamicParams(
        mode="hedgerock",
        hedgerock_enabled=True,
        risk_tier=risk_tier,
        cooldown_until=None,
        lot_factor=lot_factor,
        grid_multiplier=base["grid_multiplier"],
        max_next_lot=max_next_lot,
        takeprofit_points=base["takeprofit_points"],
        stoploss_points=base["stoploss_points"],
        recovery_multiplier=base["recovery_multiplier"],
        max_orders_buy=max_orders,
        max_orders_sell=max_orders,
        reason=derived_reason,
    ))
