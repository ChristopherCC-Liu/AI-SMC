"""Ticket 3 Step 2 — Class C envelope mirror.

**Sidecar reimplementation** of the production
``smc.hedgerock.rule_engine.derive_envelope_params`` semantics.

Hard rules (per Ticket 3 plan §R3 + §R4):

  - This module **NEVER imports** ``smc.hedgerock.rule_engine``,
    ``smc.hedgerock.decision_server``, or
    ``smc.hedgerock.phase_d_walk_forward`` decision-side code at
    runtime. It reimplements the *semantics* in sidecar code.
  - At runtime, the only knobs the mirror is allowed to read from
    production are the four scalars in ``CLASS_A_TARGET_WHITELIST``
    — and even those go through Class A mirror (read-only via
    importlib.getattr).
  - Every candidate target the mirror supports MUST be in either
    ``CLASS_A_TARGET_WHITELIST`` or ``CLASS_B_TARGET_WHITELIST``.
    Anything else → ABSTAIN: unsupported_target at the runner level.

Golden-fixture drift detection:
  - Tests under ``tests/hedgerock/evolution/test_replay_envelope_mirror.py``
    import production ``derive_envelope_params`` and confirm the
    mirror's output matches byte-for-byte on ≥ 20 representative
    fixture cases.
  - Sidecar runtime never imports production decision-side code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from smc.hedgerock.evolution import rule_engine_mirror, replay_constant_mirror


__all__ = [
    "MIRROR_C_VERSION",
    "MirroredDynamicParams",
    "is_target_supported",
    "mirror_derive_envelope_params",
    "compute_mirror_c_version",
]


# ---------------------------------------------------------------------------
# Pinned thresholds (read-only mirror of production rule_engine constants)
#
# These values are pinned at the sidecar layer. Any drift in production
# is detected by:
#   1. rule_engine_mirror.check_mirror_drift() — Class A scalar drift
#      (covers _CONFIDENCE_OBSERVE_FLOOR, _CONFIDENCE_AGGRESSIVE_FLOOR)
#   2. test_replay_envelope_mirror_matches_production — golden fixture
#      byte-equal verification on ≥ 20 cases against production.
# ---------------------------------------------------------------------------

# DD thresholds.
_DD_STEPDOWN_THRESHOLD: float = 0.02
_DD_SEVERE_THRESHOLD: float = 0.05

# Consecutive-loss thresholds.
_CONSEC_LOSSES_STEPDOWN: int = 3
_CONSEC_LOSSES_COOLDOWN: int = 5

# Spread cooldown threshold.
_SPREAD_PTS_COOLDOWN: int = 80

# Cooldown durations.
_COOLDOWN_DD_SECONDS: int = 30 * 60
_COOLDOWN_LOSS_SECONDS: int = 30 * 60
_COOLDOWN_SPREAD_SECONDS: int = 5 * 60
_COOLDOWN_NEWS_SECONDS: int = 15 * 60

# Recent-PnL thresholds.
_RECENT_PNL_MIN_SAMPLE: int = 5
_RECENT_PNL_STEPDOWN_THRESHOLD: float = -50.0


# ---------------------------------------------------------------------------
# Sidecar mirror of DynamicParams (separate dataclass — does NOT import
# production type so the mirror is wire-isolated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MirroredDynamicParams:
    """Mirror of production ``rule_engine.DynamicParams``. Field set
    is identical to allow the golden-fixture test to compare against
    production output 1:1 by iterating over field names."""

    mode: str
    hedgerock_enabled: bool
    risk_tier: str
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

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "hedgerock_enabled": self.hedgerock_enabled,
            "risk_tier": self.risk_tier,
            "cooldown_until": (
                self.cooldown_until.isoformat() if self.cooldown_until else None
            ),
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
# Whitelist queries
# ---------------------------------------------------------------------------


def is_target_supported(target: str) -> bool:
    """A candidate target is supported by the mirror iff it appears
    in either Class A (rule_engine constants) or Class B
    (replay constants) whitelist."""
    return (
        rule_engine_mirror.is_target_in_whitelist(target)
        or replay_constant_mirror.is_target_in_whitelist(target)
    )


# ---------------------------------------------------------------------------
# Helpers — _safe_observe, _halt mirrors
# ---------------------------------------------------------------------------


def _mirror_safe_observe(reason: str, *, cooldown_until: datetime | None = None) -> MirroredDynamicParams:
    return MirroredDynamicParams(
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


def _mirror_halt(reason: str, *, cooldown_until: datetime | None = None) -> MirroredDynamicParams:
    return MirroredDynamicParams(
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


def _mirror_apply_cooldown_carryover(
    params: MirroredDynamicParams,
    prev_cooldown_until: datetime | None,
    now: datetime,
) -> MirroredDynamicParams:
    if prev_cooldown_until is None or prev_cooldown_until <= now:
        return params
    new_cd = params.cooldown_until
    merged = max(new_cd, prev_cooldown_until) if new_cd is not None else prev_cooldown_until
    if params.mode == "hedgerock":
        from dataclasses import replace as _replace
        return _replace(
            _mirror_safe_observe(
                reason=(
                    f"cooldown still active until {prev_cooldown_until.isoformat()} "
                    f"(deferred original={params.reason})"
                ),
                cooldown_until=merged,
            ),
        )
    from dataclasses import replace as _replace
    return _replace(params, cooldown_until=merged)


# ---------------------------------------------------------------------------
# mirror_derive_envelope_params — the heart of Class C mirror
# ---------------------------------------------------------------------------


def mirror_derive_envelope_params(
    *,
    now: datetime,
    regime: str,                              # "range" | "trend_up" | "trend_down" | "breakout" | "crisis" | "news" | "unknown"
    confidence: float,
    regime_reason: str,
    ea_state_stale: bool,
    ea_state_age_seconds: float | None,
    dd_pct: float | None,
    consec_losses: int | None,
    spread_pts: int | None,
    recent_closed_pnl: float | None,
    recent_sample_count: int | None,
    ea_state_present: bool,
    prev_envelope_cooldown_until: datetime | None,
    overlay_params: dict[str, Any] | None = None,
) -> MirroredDynamicParams:
    """Mirror of ``derive_envelope_params``.

    ``overlay_params`` is the candidate overlay applied to the Class A
    parameter snapshot. Recognised keys:
      - "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
      - "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"

    Decision tree mirrors production exactly. Numbered comments map
    1:1 to the docstring of production's derive_envelope_params.
    """
    overlay = overlay_params or {}

    # Resolve effective confidence floors. Default: Class A current
    # production values (read fresh via the mirror, so any production
    # change is reflected — Class A drift detection stays the gate).
    a_snap = rule_engine_mirror.snapshot_params()
    effective_observe_floor = float(overlay.get(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        a_snap.get(
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            0.55,
        ),
    ))
    effective_aggressive_floor = float(overlay.get(
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR",
        a_snap.get(
            "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR",
            0.80,
        ),
    ))

    def _final(p: MirroredDynamicParams) -> MirroredDynamicParams:
        return _mirror_apply_cooldown_carryover(p, prev_envelope_cooldown_until, now)

    # 1. Stale state.
    if ea_state_stale:
        return _final(_mirror_safe_observe(
            reason=f"ea_state_stale → observe; "
                   f"age={ea_state_age_seconds}s; regime={regime}",
        ))

    # 2. Crisis.
    if regime == "crisis":
        return _final(_mirror_halt(
            reason=f"regime=crisis → halt + cooldown; {regime_reason}",
            cooldown_until=now + timedelta(seconds=_COOLDOWN_DD_SECONDS),
        ))

    # 3. News.
    if regime == "news":
        return _final(_mirror_safe_observe(
            reason=f"regime=news → observe + cooldown; {regime_reason}",
            cooldown_until=now + timedelta(seconds=_COOLDOWN_NEWS_SECONDS),
        ))

    # 4. DD severe.
    if dd_pct is not None and dd_pct >= _DD_SEVERE_THRESHOLD:
        return _final(_mirror_halt(
            reason=f"DD={dd_pct:.4f} ≥ {_DD_SEVERE_THRESHOLD} → halt + cooldown",
            cooldown_until=now + timedelta(seconds=_COOLDOWN_DD_SECONDS),
        ))

    # 5. Spread anomalous.
    if spread_pts is not None and spread_pts >= _SPREAD_PTS_COOLDOWN:
        return _final(_mirror_safe_observe(
            reason=f"spread_pts={spread_pts} ≥ {_SPREAD_PTS_COOLDOWN} → observe + cooldown",
            cooldown_until=now + timedelta(seconds=_COOLDOWN_SPREAD_SECONDS),
        ))

    # 6. Consecutive losses.
    if consec_losses is not None and consec_losses >= _CONSEC_LOSSES_COOLDOWN:
        return _final(_mirror_safe_observe(
            reason=f"consec_losses={consec_losses} ≥ {_CONSEC_LOSSES_COOLDOWN} → observe + cooldown",
            cooldown_until=now + timedelta(seconds=_COOLDOWN_LOSS_SECONDS),
        ))

    # 7. Trend / breakout / unknown.
    if regime in ("trend_up", "trend_down", "breakout", "unknown"):
        return _final(_mirror_safe_observe(
            reason=f"regime={regime} → observe (HedgeRock disabled outside range)",
        ))

    # 7.5 No EA state / incomplete EA snapshot.
    if not ea_state_present:
        return _final(_mirror_safe_observe(
            reason="no_ea_state → observe (refuse to enter HedgeRock without runtime state)",
        ))
    missing: list[str] = []
    if dd_pct is None:
        missing.append("dd_pct")
    if spread_pts is None:
        missing.append("spread_pts")
    if missing:
        return _final(_mirror_safe_observe(
            reason=(
                f"incomplete_ea_state (missing {', '.join(missing)}) → observe; "
                "refuse HedgeRock without minimum risk snapshot"
            ),
        ))

    # 8. Range with sub-floor confidence → observe.
    # Use effective_observe_floor (overlay-aware).
    if confidence < effective_observe_floor:
        return _final(_mirror_safe_observe(
            reason=f"confidence={confidence:.2f} < {effective_observe_floor} → observe",
        ))

    # 9. Range + good confidence → HedgeRock with step-up / step-down.
    # Step DOWN.
    step_down = False
    step_down_reason = ""
    if dd_pct is not None and dd_pct >= _DD_STEPDOWN_THRESHOLD:
        step_down = True
        step_down_reason = f"DD={dd_pct:.4f} ≥ {_DD_STEPDOWN_THRESHOLD}"
    elif consec_losses is not None and consec_losses >= _CONSEC_LOSSES_STEPDOWN:
        step_down = True
        step_down_reason = (
            f"consec_losses={consec_losses} ≥ {_CONSEC_LOSSES_STEPDOWN}"
        )
    elif (
        recent_closed_pnl is not None
        and recent_sample_count is not None
        and recent_sample_count >= _RECENT_PNL_MIN_SAMPLE
        and recent_closed_pnl <= _RECENT_PNL_STEPDOWN_THRESHOLD
    ):
        step_down = True
        step_down_reason = (
            f"recent_closed_pnl={recent_closed_pnl:.2f} ≤ "
            f"{_RECENT_PNL_STEPDOWN_THRESHOLD} (n={recent_sample_count})"
        )

    history_unavailable = recent_sample_count is None
    insufficient_sample = (
        recent_sample_count is not None and recent_sample_count < _RECENT_PNL_MIN_SAMPLE
    )
    recent_pnl_blocks_stepup = (
        recent_closed_pnl is not None
        and recent_sample_count is not None
        and recent_sample_count >= _RECENT_PNL_MIN_SAMPLE
        and recent_closed_pnl < 0.0
    )
    missing_recent_pnl = (
        recent_sample_count is not None
        and recent_sample_count >= _RECENT_PNL_MIN_SAMPLE
        and recent_closed_pnl is None
    )
    missing_consec_losses = consec_losses is None

    # range base profile.
    base_max_next_lot = 0.05

    if step_down:
        risk_tier = "observe"
        lot_factor = 0.5
        max_next_lot = 0.03
        max_orders = 1
        derived_reason = (
            f"range + step-down ({step_down_reason}); "
            f"{regime_reason}; conf={confidence:.2f}"
        )
    elif confidence >= effective_aggressive_floor and not (
        recent_pnl_blocks_stepup
        or history_unavailable
        or insufficient_sample
        or missing_recent_pnl
        or missing_consec_losses
    ):
        risk_tier = "aggressive"
        lot_factor = 1.5
        max_next_lot = 0.10
        max_orders = 3
        derived_reason = (
            f"range + step-up (conf={confidence:.2f} ≥ "
            f"{effective_aggressive_floor}); {regime_reason}"
        )
    elif confidence >= effective_aggressive_floor:
        if recent_pnl_blocks_stepup:
            blocker = f"recent_pnl={recent_closed_pnl:.2f}<0 on n={recent_sample_count}"
        elif history_unavailable:
            blocker = "history_unavailable (recent_sample_count=None)"
        elif insufficient_sample:
            blocker = (
                f"insufficient_sample (recent_n={recent_sample_count} < "
                f"{_RECENT_PNL_MIN_SAMPLE})"
            )
        elif missing_recent_pnl:
            blocker = "history_incomplete (missing recent_closed_pnl)"
        else:  # missing_consec_losses
            blocker = "history_incomplete (missing consec_losses)"
        risk_tier = "normal"
        lot_factor = 1.0
        max_next_lot = base_max_next_lot
        max_orders = 2
        derived_reason = (
            f"range + step-up blocked ({blocker}); using normal tier; "
            f"{regime_reason}"
        )
    else:
        risk_tier = "normal"
        lot_factor = 1.0
        max_next_lot = base_max_next_lot
        max_orders = 2
        derived_reason = (
            f"range + normal; conf={confidence:.2f}; {regime_reason}"
        )

    return _final(MirroredDynamicParams(
        mode="hedgerock",
        hedgerock_enabled=True,
        risk_tier=risk_tier,
        cooldown_until=None,
        lot_factor=lot_factor,
        grid_multiplier=1.0,
        max_next_lot=max_next_lot,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=max_orders,
        max_orders_sell=max_orders,
        reason=derived_reason,
    ))


# ---------------------------------------------------------------------------
# Mirror version — for the artefact's mirror_version field
# ---------------------------------------------------------------------------


def compute_mirror_c_version() -> str:
    """SHA-256 over the mirror's pinned thresholds + decision-tree
    structural skeleton. Drifts when the mirror's logic changes."""
    payload = {
        "class": "replay_envelope_mirror_class_C",
        "thresholds": {
            "DD_STEPDOWN_THRESHOLD": _DD_STEPDOWN_THRESHOLD,
            "DD_SEVERE_THRESHOLD": _DD_SEVERE_THRESHOLD,
            "CONSEC_LOSSES_STEPDOWN": _CONSEC_LOSSES_STEPDOWN,
            "CONSEC_LOSSES_COOLDOWN": _CONSEC_LOSSES_COOLDOWN,
            "SPREAD_PTS_COOLDOWN": _SPREAD_PTS_COOLDOWN,
            "COOLDOWN_DD_SECONDS": _COOLDOWN_DD_SECONDS,
            "COOLDOWN_LOSS_SECONDS": _COOLDOWN_LOSS_SECONDS,
            "COOLDOWN_SPREAD_SECONDS": _COOLDOWN_SPREAD_SECONDS,
            "COOLDOWN_NEWS_SECONDS": _COOLDOWN_NEWS_SECONDS,
            "RECENT_PNL_MIN_SAMPLE": _RECENT_PNL_MIN_SAMPLE,
            "RECENT_PNL_STEPDOWN_THRESHOLD": _RECENT_PNL_STEPDOWN_THRESHOLD,
        },
        "decision_tree_branches": [
            "ea_state_stale", "crisis", "news", "dd_severe", "spread_anomalous",
            "consec_losses_cooldown", "trend_breakout_unknown",
            "no_ea_state", "incomplete_ea_state",
            "sub_observe_floor",
            "range_step_down", "range_step_up_aggressive",
            "range_step_up_blocked", "range_normal",
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


MIRROR_C_VERSION: str = compute_mirror_c_version()
