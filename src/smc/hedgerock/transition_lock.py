"""Regime transition lock.

When the market regime jumps (TREND_UP → CONSOLIDATION etc.), the
HedgeRock EA should briefly stop opening new orders to let the new
regime's strategy parameters settle and avoid whipsaw entries near
inflection points.

The lock duration scales with the *distance* between regimes:

    distance 0 (same regime)           →     0 s
    distance 1 (adjacent regimes)      →   900 s (15 min)
    distance 2 (one regime apart)      →  3600 s ( 1 hr)
    distance 3 (extreme reversal)      →  7200 s ( 2 hr)

The distance graph is hand-tuned for XAUUSD; revisit if/when other
instruments are wired in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from smc.ai.models import MarketRegimeAI
from smc.hedgerock.schemas import RegimeV2

__all__ = [
    "DISTANCE_TO_SECONDS",
    "REGIME_DISTANCE",
    "REGIME_V2_DISTANCE",
    "compute_lock_seconds",
    "compute_lock_seconds_v2",
    "compute_lock_until",
    "compute_lock_until_v2",
]


# ---------------------------------------------------------------------------
# Distance graph
# ---------------------------------------------------------------------------
#
# Rationale:
#   - TREND_UP ↔ ATH_BREAKOUT: both bullish-strong, adjacent (1)
#   - TREND_UP ↔ TRANSITION: trend cooling off, adjacent (1)
#   - TREND_UP ↔ CONSOLIDATION: skips TRANSITION, distance 2
#   - TREND_UP ↔ TREND_DOWN: extreme reversal, distance 3
#   - TRANSITION is a "hub" — adjacent to every other regime (1)
#     except ATH_BREAKOUT (which is a one-off strong bullish breakout
#     and should not flip directly to TRANSITION without TREND_UP first)
#   - ATH_BREAKOUT ↔ TREND_DOWN: as extreme as it gets (3)

REGIME_DISTANCE: dict[tuple[MarketRegimeAI, MarketRegimeAI], int] = {
    # TREND_UP row
    ("TREND_UP", "TREND_UP"): 0,
    ("TREND_UP", "TREND_DOWN"): 3,
    ("TREND_UP", "CONSOLIDATION"): 2,
    ("TREND_UP", "TRANSITION"): 1,
    ("TREND_UP", "ATH_BREAKOUT"): 1,
    # TREND_DOWN row
    ("TREND_DOWN", "TREND_UP"): 3,
    ("TREND_DOWN", "TREND_DOWN"): 0,
    ("TREND_DOWN", "CONSOLIDATION"): 2,
    ("TREND_DOWN", "TRANSITION"): 1,
    ("TREND_DOWN", "ATH_BREAKOUT"): 3,
    # CONSOLIDATION row
    ("CONSOLIDATION", "TREND_UP"): 2,
    ("CONSOLIDATION", "TREND_DOWN"): 2,
    ("CONSOLIDATION", "CONSOLIDATION"): 0,
    ("CONSOLIDATION", "TRANSITION"): 1,
    ("CONSOLIDATION", "ATH_BREAKOUT"): 2,
    # TRANSITION row (hub)
    ("TRANSITION", "TREND_UP"): 1,
    ("TRANSITION", "TREND_DOWN"): 1,
    ("TRANSITION", "CONSOLIDATION"): 1,
    ("TRANSITION", "TRANSITION"): 0,
    ("TRANSITION", "ATH_BREAKOUT"): 2,
    # ATH_BREAKOUT row
    ("ATH_BREAKOUT", "TREND_UP"): 1,
    ("ATH_BREAKOUT", "TREND_DOWN"): 3,
    ("ATH_BREAKOUT", "CONSOLIDATION"): 2,
    ("ATH_BREAKOUT", "TRANSITION"): 2,
    ("ATH_BREAKOUT", "ATH_BREAKOUT"): 0,
}


DISTANCE_TO_SECONDS: dict[int, int] = {
    0: 0,
    1: 900,
    2: 3600,
    3: 7200,
}


def compute_lock_seconds(
    prev_regime: MarketRegimeAI | None,
    current_regime: MarketRegimeAI,
) -> int:
    """Return the lock duration in seconds for a regime transition.

    `prev_regime=None` means first-call (no prior state) — returns 0
    so a freshly started decision center does not gratuitously block
    the EA on startup.
    """
    if prev_regime is None:
        return 0
    distance = REGIME_DISTANCE.get((prev_regime, current_regime))
    if distance is None:
        # Defensive: any new regime added to MarketRegimeAI without updating
        # the matrix should fail loudly rather than silently default to 0.
        raise KeyError(
            f"No distance defined for regime pair ({prev_regime}, {current_regime}); "
            f"update REGIME_DISTANCE."
        )
    return DISTANCE_TO_SECONDS[distance]


def compute_lock_until(
    prev_regime: MarketRegimeAI | None,
    current_regime: MarketRegimeAI,
    now: datetime,
) -> datetime | None:
    """Return the UTC instant the lock expires, or None if no lock applies.

    `now` must be timezone-aware (UTC). Caller responsibility — we do not
    convert silently because upstream timestamps come from the journal
    in UTC and any tz mix-up causes the EA to receive an off-by-N-hour
    lock window.
    """
    if now.tzinfo is None:
        raise ValueError("'now' must be timezone-aware (UTC)")
    seconds = compute_lock_seconds(prev_regime, current_regime)
    if seconds == 0:
        return None
    return (now.astimezone(timezone.utc)) + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# v2 distance graph (Phase C-hotfix-2 #3)
# ---------------------------------------------------------------------------
#
# Same DISTANCE_TO_SECONDS scale (0/900/3600/7200) but over the 7-value
# v2 enum. Rationale per pair:
#   - same regime → 0
#   - any flip into / out of crisis → 3 (extreme)
#   - any flip into / out of news → 1 (cooldown handled separately;
#     transition lock here is just the anti-thrash guard on top)
#   - trend_up ↔ trend_down → 3 (extreme reversal)
#   - range ↔ trend_*  → 2 (regime change without intervening transition)
#   - breakout ↔ trend_up → 1 (breakout is the impulse leg of a trend)
#   - breakout ↔ trend_down → 3 (counter-breakout = extreme reversal)
#   - unknown is "we don't know" — adjacency 1 to everything (we just
#     wait briefly, no need to lock long).

REGIME_V2_DISTANCE: dict[tuple[RegimeV2, RegimeV2], int] = {
    # range row
    ("range", "range"):       0,
    ("range", "trend_up"):    2,
    ("range", "trend_down"):  2,
    ("range", "news"):        1,
    ("range", "breakout"):    1,
    ("range", "crisis"):      3,
    ("range", "unknown"):     1,
    # trend_up row
    ("trend_up", "range"):     2,
    ("trend_up", "trend_up"):  0,
    ("trend_up", "trend_down"):3,
    ("trend_up", "news"):      1,
    ("trend_up", "breakout"):  1,
    ("trend_up", "crisis"):    3,
    ("trend_up", "unknown"):   1,
    # trend_down row
    ("trend_down", "range"):     2,
    ("trend_down", "trend_up"):  3,
    ("trend_down", "trend_down"):0,
    ("trend_down", "news"):      1,
    ("trend_down", "breakout"):  3,
    ("trend_down", "crisis"):    3,
    ("trend_down", "unknown"):   1,
    # news row
    ("news", "range"):     1,
    ("news", "trend_up"):  1,
    ("news", "trend_down"):1,
    ("news", "news"):      0,
    ("news", "breakout"):  2,
    ("news", "crisis"):    2,
    ("news", "unknown"):   1,
    # breakout row
    ("breakout", "range"):     1,
    ("breakout", "trend_up"):  1,
    ("breakout", "trend_down"):3,
    ("breakout", "news"):      2,
    ("breakout", "breakout"):  0,
    ("breakout", "crisis"):    3,
    ("breakout", "unknown"):   1,
    # crisis row — flips out of crisis are always extreme
    ("crisis", "range"):     3,
    ("crisis", "trend_up"):  3,
    ("crisis", "trend_down"):3,
    ("crisis", "news"):      2,
    ("crisis", "breakout"):  3,
    ("crisis", "crisis"):    0,
    ("crisis", "unknown"):   2,
    # unknown row
    ("unknown", "range"):     1,
    ("unknown", "trend_up"):  1,
    ("unknown", "trend_down"):1,
    ("unknown", "news"):      1,
    ("unknown", "breakout"):  1,
    ("unknown", "crisis"):    2,
    ("unknown", "unknown"):   0,
}


def compute_lock_seconds_v2(
    prev_regime: RegimeV2 | None,
    current_regime: RegimeV2,
) -> int:
    """v2 counterpart of :func:`compute_lock_seconds`.

    Phase C-hotfix-2: transition_lock_until_ts is now driven by the
    v2 assessment regime so a same-legacy / different-v2 transition
    is locked, and a same-v2 / different-legacy transition is NOT.
    """
    if prev_regime is None:
        return 0
    distance = REGIME_V2_DISTANCE.get((prev_regime, current_regime))
    if distance is None:
        raise KeyError(
            f"No v2 distance defined for ({prev_regime}, {current_regime}); "
            f"update REGIME_V2_DISTANCE."
        )
    return DISTANCE_TO_SECONDS[distance]


def compute_lock_until_v2(
    prev_regime: RegimeV2 | None,
    current_regime: RegimeV2,
    now: datetime,
) -> datetime | None:
    """v2 counterpart of :func:`compute_lock_until`.

    Returns the UTC instant the v2 transition lock expires, or None
    when the regime is unchanged / first call.
    """
    if now.tzinfo is None:
        raise ValueError("'now' must be timezone-aware (UTC)")
    seconds = compute_lock_seconds_v2(prev_regime, current_regime)
    if seconds == 0:
        return None
    return (now.astimezone(timezone.utc)) + timedelta(seconds=seconds)
