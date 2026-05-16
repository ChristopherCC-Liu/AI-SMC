"""Ticket 3 Step 3 — replay executor.

**Sidecar-only.** Runs baseline + candidate replays against the same
historical data slice, emitting per-bar logs. Uses Class C envelope
mirror for decisions and the sidecar simulator for trades. Does NOT
import production decision-side code (rule_engine.derive_envelope_params,
phase_d_walk_forward._run_dynamic, etc.).

Multi-symbol skeleton:
  - ``run_pair(symbols=tuple[str, ...])`` accepts ≥ 1 symbol.
  - v1 lake has only XAUUSD; signature is multi-symbol-ready, but
    each symbol gets its own per-symbol sub-replay.
  - When the cross-symbol set has ≥ 2 entries, downstream gates
    (G3 / single_symbol abstain in G8) shift; no special path here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from smc.hedgerock.evolution import (
    rule_engine_mirror,
    replay_constant_mirror,
    replay_envelope_mirror,
)
from smc.hedgerock.evolution.data_slice import build_decision_window
from smc.hedgerock.evolution.policy_overlay import PolicyOverlay, apply_overlay
from smc.hedgerock.evolution.replay_envelope_mirror import (
    MIRROR_C_VERSION,
    MirroredDynamicParams,
    is_target_supported,
    mirror_derive_envelope_params,
)
from smc.hedgerock.evolution.replay_state import (
    PIP_PNL_PER_LOT,
    POINT,
    STATIC_ATR_MULT,
    STATIC_GRID_BASE_USD,
    SimState,
    fresh_sim_state,
    step,
    synthesize_ea_state,
)


__all__ = [
    "MultiWindowReplayResult",
    "PerSymbolReplayLog",
    "PerWindowReplayResult",
    "PerWindowStats",
    "ReplayLog",
    "ReplayPairResult",
    "run_multi_window",
    "run_pair",
]


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerSymbolReplayLog:
    """Output of one symbol's replay for one role (baseline OR candidate).

    Mode counters (Ticket 4 v2) are non-overlapping per-bar counts:
      * ``observe_mode_bars`` — bars where the envelope returned mode
        ``"observe"`` for any non-cooldown reason.
      * ``halt_mode_bars`` — bars where the envelope returned mode
        ``"halt"``.
      * ``cooldown_mode_bars`` — bars where the prior cooldown
        carryover suppressed trading even though the underlying mode
        would have been ``hedgerock``.

    Defaulted for back-compat with Ticket 3 call sites that did not
    track mode counts.
    """

    symbol: str
    final_state: SimState
    n_bars_envelope_decided: int
    halt_event_count: int
    observe_mode_bars: int = 0
    halt_mode_bars: int = 0
    cooldown_mode_bars: int = 0


@dataclass(frozen=True)
class ReplayLog:
    """Aggregate output for one role across all replayed symbols."""

    per_symbol: tuple[PerSymbolReplayLog, ...]
    final_state: SimState           # net of last symbol or aggregated; v1 = single symbol
    n_bars_envelope_decided: int
    halt_event_count: int


@dataclass(frozen=True)
class ReplayPairResult:
    """Pair of baseline + candidate logs, plus self-reported abort
    state."""

    baseline_log: ReplayLog | None
    candidate_log: ReplayLog | None
    aborted: bool
    abort_reason: str
    symbols_run: tuple[str, ...]
    mirror_consistency: bool


# Ticket 4 v2 — XAUUSD-only multi-window dataclasses.


@dataclass(frozen=True)
class PerWindowStats:
    """Per-window stats consumed by ``window_coverage`` gate.

    The seven public fields match
    :func:`window_coverage._parse_per_window_stat` byte-for-byte so
    a list of ``PerWindowStats`` can be fed directly to
    :func:`check_window_coverage` after a ``vars()``-style coercion.
    """

    window_id: str
    n_bars: int
    n_decided_bars: int
    n_trades: int
    max_h1_gap_bars: int
    halt_event_count: int
    observed_buckets: tuple[str, ...]


@dataclass(frozen=True)
class PerWindowReplayResult:
    """Baseline + candidate logs plus stats for one historical window."""

    window_id: str
    baseline_log: ReplayLog
    candidate_log: ReplayLog
    stats: PerWindowStats


@dataclass(frozen=True)
class MultiWindowReplayResult:
    """Multi-window XAUUSD replay output. Sidecar-only.

    On success ``aborted`` is False and ``per_window_results`` has one
    entry per :class:`WindowSpec` supplied. On any pre-flight or
    per-window failure the runner short-circuits with
    ``aborted=True`` and a descriptive ``abort_reason``; downstream
    callers MUST treat aborted multi-window outputs as ABSTAIN.
    """

    aborted: bool
    abort_reason: str
    symbol: str
    windows_run: tuple[str, ...]
    per_window_results: tuple[PerWindowReplayResult, ...]
    mirror_consistency: bool


# ---------------------------------------------------------------------------
# Pre-flight drift check (uses Class A + B + C mirrors)
# ---------------------------------------------------------------------------


def _check_all_mirrors_consistent() -> tuple[bool, list[str]]:
    a_ok, a_reasons = rule_engine_mirror.check_mirror_drift()
    b_ok, b_reasons = replay_constant_mirror.check_mirror_drift()
    # Class C drift is implicit in A drift (mirror C reads Class A
    # values via snapshot_params); a clean A means C is consistent
    # for the target set we currently support. Future Class C
    # drift detection (golden-fixture replay at runtime) is out of
    # scope for v1.
    return (a_ok and b_ok), (a_reasons + b_reasons)


# ---------------------------------------------------------------------------
# H4-frame ATR helper (for grid spacing) — mirrors phase_d's
# _prepare_atr_d1 + per-bar lookup, but kept minimal
# ---------------------------------------------------------------------------


def _atr_for_bar(d1_frame, decision_ts: datetime, period: int = 14) -> float | None:
    """Compute D1 ATR(period) on the prior closed-day frame. Returns
    None when the frame is too short."""
    if d1_frame is None or d1_frame.height < period + 1:
        return None
    high = d1_frame["high"].to_list()
    low = d1_frame["low"].to_list()
    close = d1_frame["close"].to_list()
    trs: list[float] = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        trs.append(max(hl, hc, lc))
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


# ---------------------------------------------------------------------------
# Per-symbol single-replay
# ---------------------------------------------------------------------------


def _run_one(
    *,
    lake: Any,
    symbol: str,
    start: datetime,
    end: datetime,
    overlay_dict: dict[str, Any] | None,
    h1_lookback: int = 240,
    h4_lookback: int = 60,
) -> tuple[PerSymbolReplayLog | None, str | None]:
    """Run one (symbol, overlay_dict) replay; returns
    (log, error_reason). overlay_dict is the snapshot dict that has
    already been overlaid at the runner level (or unmodified for
    baseline).

    Performance: pre-load full H1/H4/D1 frames once per replay (with
    appropriate warmup pre-roll), then slice in-memory by ts. This is
    the same pattern phase_d_walk_forward._load_data uses; the
    strict-prior closed-bar invariant is enforced bar-by-bar by ts
    comparisons against the pre-loaded frames, not by per-bar lake
    queries.
    """
    from smc.data.schemas import Timeframe

    # Pre-load with warmup pre-roll for H4 (-30d) and D1 (-60d) so
    # trailing windows populate at the start of [start, end).
    h1 = lake.query(symbol, Timeframe.H1, start, end)
    h4_full = lake.query(symbol, Timeframe.H4,
                         start - timedelta(days=30), end)
    d1_full = lake.query(symbol, Timeframe.D1,
                         start - timedelta(days=120), end)
    if h1.is_empty():
        return None, f"no H1 bars for {symbol} in [{start}, {end})"

    state = fresh_sim_state(init_equity=10_000.0, spread_pts=20)
    n_decided = 0
    halt_events = 0
    observe_bars = 0
    halt_mode_bars = 0
    cooldown_bars = 0
    prev_envelope_cooldown_until: datetime | None = None

    h1_ts = h1["ts"].to_list()
    h1_high = h1["high"].to_list()
    h1_low = h1["low"].to_list()
    h1_close = h1["close"].to_list()

    h4_ts = h4_full["ts"].to_list() if not h4_full.is_empty() else []
    h4_high = h4_full["high"].to_list() if not h4_full.is_empty() else []
    h4_low = h4_full["low"].to_list() if not h4_full.is_empty() else []
    h4_close = h4_full["close"].to_list() if not h4_full.is_empty() else []
    d1_ts = d1_full["ts"].to_list() if not d1_full.is_empty() else []
    d1_high = d1_full["high"].to_list() if not d1_full.is_empty() else []
    d1_low = d1_full["low"].to_list() if not d1_full.is_empty() else []
    d1_close = d1_full["close"].to_list() if not d1_full.is_empty() else []

    SIM_REGIME = "range"
    SIM_CONFIDENCE = 0.85
    SIM_REGIME_REASON = "shadow_replay_v1_default_regime"

    # Sliding pointers for H4 (advance forward) and D1 (day floor).
    h4_period = timedelta(hours=4)
    d1_atr_period = 14
    invariants_checked = False

    for i in range(len(h1_ts)):
        decision_ts = h1_ts[i]

        # Strict-prior H4: include only bars whose 4h period closed
        # ≤ decision_ts. Since h4_ts is monotone non-decreasing, we
        # do a binary-search-like scan.
        h4_count = 0
        for j in range(len(h4_ts) - 1, -1, -1):
            if h4_ts[j] + h4_period <= decision_ts:
                h4_count = j + 1
                break
        # Strict-prior D1: ts < day floor of decision_ts.
        decision_day = decision_ts.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        d1_count = 0
        for j in range(len(d1_ts) - 1, -1, -1):
            if d1_ts[j] < decision_day:
                d1_count = j + 1
                break

        # Strict-prior invariants — by construction, the slicing
        # above respects them. The runtime self-check enforces
        # "no in-progress bar in window" by the strict comparisons
        # used. Skip when frames aren't populated enough yet.
        if i < h1_lookback:
            continue
        if h4_count < h4_lookback:
            continue
        if d1_count < d1_atr_period + 1:
            continue

        # First-valid-bar invariant self-check: spot-call build_decision_window
        # so an injected poison (test fixture or production-side bug) is
        # detected before the runner advances. Subsequent bars rely on
        # the strict-prior slicing invariant maintained by the binary
        # search above; per-bar build_decision_window calls would 10x
        # this loop's cost.
        if not invariants_checked:
            window = build_decision_window(
                lake=lake, symbol=symbol, start=start, end=end,
                decision_ts=decision_ts,
                h1_lookback=h1_lookback, h4_lookback=h4_lookback,
            )
            inv = window.invariants
            if (inv.h4_partial_bar_in_window
                    or inv.d1_partial_bar_in_window
                    or not inv.same_bar_set_used
                    or not inv.decision_only_uses_strictly_prior_data
                    or not inv.decision_uses_data_with_ts_lt_trade_bar_ts):
                return None, (
                    "replay_invariant_violation at "
                    f"{decision_ts.isoformat()}: {inv}"
                )
            invariants_checked = True

        # ATR(14) on the prior closed-day window.
        trs: list[float] = []
        for j in range(d1_count - d1_atr_period - 1 + 1, d1_count):
            if j == 0:
                continue
            hl = d1_high[j] - d1_low[j]
            hc = abs(d1_high[j] - d1_close[j - 1])
            lc = abs(d1_low[j] - d1_close[j - 1])
            trs.append(max(hl, hc, lc))
        if len(trs) < d1_atr_period:
            continue
        atr = sum(trs[-d1_atr_period:]) / d1_atr_period
        if atr <= 0:
            continue

        # Synthesize EA state from current sim state.
        ea = synthesize_ea_state(state)

        # Mirror decision.
        params = mirror_derive_envelope_params(
            now=decision_ts,
            regime=SIM_REGIME,
            confidence=SIM_CONFIDENCE,
            regime_reason=SIM_REGIME_REASON,
            ea_state_stale=False,
            ea_state_age_seconds=5.0,
            dd_pct=ea["dd_pct"],
            consec_losses=ea["consec_losses"],
            spread_pts=ea["spread_pts"],
            recent_closed_pnl=ea["recent_closed_pnl"],
            recent_sample_count=ea["recent_sample_count"],
            ea_state_present=True,
            prev_envelope_cooldown_until=prev_envelope_cooldown_until,
            overlay_params=overlay_dict,
        )

        # Effective mode (transition_lock veto stays on observe).
        # v1 simulator doesn't model regime switches across bars
        # because we hold a constant SIM_REGIME; transition_lock
        # remains None throughout. Keep the field consistent with
        # the schema.
        effective_mode = params.mode

        # Mode counters (Ticket 4 v2). Cooldown is detected by an
        # active prior carryover that suppressed trading on this bar
        # (i.e. the envelope flipped to observe with cooldown_until set
        # via the carryover path). We classify in this priority:
        #   halt → halt_mode_bars
        #   observe + cooldown carryover → cooldown_mode_bars
        #   observe (other reasons) → observe_mode_bars
        if effective_mode == "halt":
            halt_mode_bars += 1
        elif effective_mode == "observe":
            cd_active = (
                prev_envelope_cooldown_until is not None
                and prev_envelope_cooldown_until > decision_ts
            )
            if cd_active:
                cooldown_bars += 1
            else:
                observe_bars += 1

        # Drive the simulator step.
        tp_usd = params.takeprofit_points * POINT
        grid_spacing = max(
            STATIC_GRID_BASE_USD * params.grid_multiplier,
            STATIC_ATR_MULT * params.grid_multiplier * atr,
        )
        gear = params.recovery_multiplier
        # start_lots: production scales by lot_factor on a 0.1 base.
        start_lots = 0.1 * params.lot_factor

        prev_halt = state.halt_streak_active
        step(
            state=state,
            ts=decision_ts,
            high=h1_high[i], low=h1_low[i], close=h1_close[i],
            grid_spacing=grid_spacing,
            tp_usd=tp_usd, gear=gear,
            max_next_lot=params.max_next_lot,
            start_lots=start_lots,
            max_orders_buy=params.max_orders_buy,
            max_orders_sell=params.max_orders_sell,
            mode=effective_mode,
        )
        if state.halt_streak_active and not prev_halt:
            halt_events += 1
        n_decided += 1

        # Cooldown carryover for next bar.
        prev_envelope_cooldown_until = params.cooldown_until

    log = PerSymbolReplayLog(
        symbol=symbol, final_state=state,
        n_bars_envelope_decided=n_decided,
        halt_event_count=halt_events,
        observe_mode_bars=observe_bars,
        halt_mode_bars=halt_mode_bars,
        cooldown_mode_bars=cooldown_bars,
    )
    return log, None


# ---------------------------------------------------------------------------
# Public entry: run_pair
# ---------------------------------------------------------------------------


def run_pair(
    *,
    lake: Any,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    candidate_overlay: PolicyOverlay,
) -> ReplayPairResult:
    """Run baseline + candidate replays against the same data slice.

    On any pre-flight drift / unsupported_target / mid-replay
    invariant violation, returns ``aborted=True`` with reason —
    callers must NOT proceed to compute PASS/FAIL artefacts in
    that case.
    """

    # Pre-flight drift check.
    consistent, drift_reasons = _check_all_mirrors_consistent()
    if not consistent:
        return ReplayPairResult(
            baseline_log=None, candidate_log=None,
            aborted=True,
            abort_reason="mirror_drift_detected_at_runtime: " + "; ".join(drift_reasons),
            symbols_run=symbols,
            mirror_consistency=False,
        )

    # Pre-flight target whitelist.
    if not is_target_supported(candidate_overlay.target):
        return ReplayPairResult(
            baseline_log=None, candidate_log=None,
            aborted=True,
            abort_reason=(
                f"unsupported_target: {candidate_overlay.target!r} "
                "not in any mirror whitelist"
            ),
            symbols_run=symbols,
            mirror_consistency=True,
        )

    # Snapshot Class A params.
    a_snapshot = rule_engine_mirror.snapshot_params()

    # Build candidate overlay dict for mirror call.
    try:
        candidate_overlay_dict = apply_overlay(a_snapshot, candidate_overlay)
    except KeyError:
        return ReplayPairResult(
            baseline_log=None, candidate_log=None,
            aborted=True,
            abort_reason=(
                f"unsupported_target: {candidate_overlay.target!r} "
                "not in Class A snapshot"
            ),
            symbols_run=symbols,
            mirror_consistency=True,
        )

    # Snapshot of production rule_engine constants for leak check
    # at the end.
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    constants_before = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if k.startswith("_") and not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }

    baseline_logs: list[PerSymbolReplayLog] = []
    candidate_logs: list[PerSymbolReplayLog] = []
    for sym in symbols:
        b_log, b_err = _run_one(
            lake=lake, symbol=sym, start=start, end=end,
            overlay_dict=None,
        )
        if b_err is not None:
            return ReplayPairResult(
                baseline_log=None, candidate_log=None,
                aborted=True,
                abort_reason=f"baseline_replay_error: {b_err}",
                symbols_run=symbols,
                mirror_consistency=True,
            )
        c_log, c_err = _run_one(
            lake=lake, symbol=sym, start=start, end=end,
            overlay_dict=candidate_overlay_dict,
        )
        if c_err is not None:
            return ReplayPairResult(
                baseline_log=None, candidate_log=None,
                aborted=True,
                abort_reason=f"candidate_replay_error: {c_err}",
                symbols_run=symbols,
                mirror_consistency=True,
            )
        if b_log is None or c_log is None:
            return ReplayPairResult(
                baseline_log=None, candidate_log=None,
                aborted=True,
                abort_reason=f"empty_replay_for_{sym}",
                symbols_run=symbols,
                mirror_consistency=True,
            )
        baseline_logs.append(b_log)
        candidate_logs.append(c_log)

    # Verify production constants unchanged (leak check).
    constants_after = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if k.startswith("_") and not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }
    if constants_before != constants_after:
        return ReplayPairResult(
            baseline_log=None, candidate_log=None,
            aborted=True,
            abort_reason="leak_detected: production rule_engine constants mutated",
            symbols_run=symbols,
            mirror_consistency=True,
        )

    # Aggregate (v1 single symbol → just take the one log).
    baseline_log = ReplayLog(
        per_symbol=tuple(baseline_logs),
        final_state=baseline_logs[0].final_state if baseline_logs else fresh_sim_state(),
        n_bars_envelope_decided=sum(l.n_bars_envelope_decided for l in baseline_logs),
        halt_event_count=sum(l.halt_event_count for l in baseline_logs),
    )
    candidate_log = ReplayLog(
        per_symbol=tuple(candidate_logs),
        final_state=candidate_logs[0].final_state if candidate_logs else fresh_sim_state(),
        n_bars_envelope_decided=sum(l.n_bars_envelope_decided for l in candidate_logs),
        halt_event_count=sum(l.halt_event_count for l in candidate_logs),
    )

    return ReplayPairResult(
        baseline_log=baseline_log,
        candidate_log=candidate_log,
        aborted=False,
        abort_reason="",
        symbols_run=symbols,
        mirror_consistency=True,
    )


# ---------------------------------------------------------------------------
# Ticket 4 v2 — XAUUSD-only multi-window runner
# ---------------------------------------------------------------------------


def _h1_bars_for_window(
    *, lake: Any, symbol: str, start: datetime, end: datetime,
) -> list[dict[str, Any]]:
    """Read the H1 frame for ``[start, end)`` and return its rows as
    plain dicts. Empty result is an empty list — callers decide
    whether to abort."""
    from smc.data.schemas import Timeframe

    df = lake.query(symbol, Timeframe.H1, start, end)
    if df is None or df.is_empty():
        return []
    return df.to_dicts()


def _max_h1_gap_bars(bars: list[dict[str, Any]]) -> int:
    """Maximum consecutive bar-gap in an H1 frame.

    Each H1 step is expected to be 1 hour; a gap of N bars means a
    timestamp delta of (N+1) hours. We return ``N`` (zero for a
    contiguous frame).
    """
    if len(bars) < 2:
        return 0
    max_gap = 0
    for i in range(1, len(bars)):
        try:
            delta = bars[i]["ts"] - bars[i - 1]["ts"]
        except TypeError:
            continue
        seconds = getattr(delta, "total_seconds", lambda: 0.0)()
        gap_h = int(seconds // 3600) - 1
        if gap_h > max_gap:
            max_gap = gap_h
    return max_gap


def run_multi_window(
    *,
    lake: Any,
    symbol: str,
    windows: list,                  # list[WindowSpec]
    candidate_overlay: PolicyOverlay,
) -> MultiWindowReplayResult:
    """Run baseline + candidate replays across N XAUUSD time windows.

    Sidecar-only. The function emits per-window stats compatible with
    :func:`window_coverage.check_window_coverage`. ``symbol`` is a
    single-string parameter — multi-symbol is explicitly out of scope
    per RFC v2 §1.1. Aborts on:

    * empty ``windows`` list (``no_windows: …``)
    * pre-flight mirror drift (``mirror_drift_detected_at_runtime: …``)
    * unsupported overlay target
    * per-window replay error or empty H1 frame
    * post-run leak detection (production constants mutated)

    The reason vocabulary stays XAUUSD-specific — the legacy
    ``single_symbol`` / ``cross_symbol`` blocker is never produced.
    """
    from smc.hedgerock.evolution.regime_classifier import classify_window

    if not windows:
        return MultiWindowReplayResult(
            aborted=True,
            abort_reason=(
                "no_windows: empty windows list — XAUUSD multi-window "
                "runner requires >= 1 WindowSpec"
            ),
            symbol=symbol,
            windows_run=(),
            per_window_results=(),
            mirror_consistency=True,
        )

    # Pre-flight drift check (Class A + B mirrors).
    consistent, drift_reasons = _check_all_mirrors_consistent()
    if not consistent:
        return MultiWindowReplayResult(
            aborted=True,
            abort_reason=(
                "mirror_drift_detected_at_runtime: " + "; ".join(drift_reasons)
            ),
            symbol=symbol,
            windows_run=(),
            per_window_results=(),
            mirror_consistency=False,
        )

    # Pre-flight target whitelist.
    if not is_target_supported(candidate_overlay.target):
        return MultiWindowReplayResult(
            aborted=True,
            abort_reason=(
                f"unsupported_target: {candidate_overlay.target!r} "
                "not in any mirror whitelist"
            ),
            symbol=symbol,
            windows_run=(),
            per_window_results=(),
            mirror_consistency=True,
        )

    # Build candidate overlay dict.
    a_snapshot = rule_engine_mirror.snapshot_params()
    try:
        candidate_overlay_dict = apply_overlay(a_snapshot, candidate_overlay)
    except KeyError:
        return MultiWindowReplayResult(
            aborted=True,
            abort_reason=(
                f"unsupported_target: {candidate_overlay.target!r} "
                "not in Class A snapshot"
            ),
            symbol=symbol,
            windows_run=(),
            per_window_results=(),
            mirror_consistency=True,
        )

    # Snapshot of production rule_engine constants for leak check.
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    constants_before = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if k.startswith("_") and not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }

    windows_run = tuple(s.window_id for s in windows)
    per_window: list[PerWindowReplayResult] = []

    for spec in windows:
        # Pull the H1 bars for the window for classifier + n_bars +
        # gap computation. _run_one will re-query internally; that's
        # OK for v0.3.0 — callers paying for the duplicate query are
        # the test stub and the production lake (which caches).
        h1_bars = _h1_bars_for_window(
            lake=lake, symbol=symbol, start=spec.start, end=spec.end,
        )
        if not h1_bars:
            return MultiWindowReplayResult(
                aborted=True,
                abort_reason=(
                    f"empty_window_for_{spec.window_id}: no H1 bars in "
                    f"[{spec.start.isoformat()}, {spec.end.isoformat()})"
                ),
                symbol=symbol,
                windows_run=windows_run,
                per_window_results=(),
                mirror_consistency=True,
            )

        b_log, b_err = _run_one(
            lake=lake, symbol=symbol,
            start=spec.start, end=spec.end,
            overlay_dict=None,
        )
        if b_err is not None:
            return MultiWindowReplayResult(
                aborted=True,
                abort_reason=(
                    f"baseline_replay_error_in_{spec.window_id}: {b_err}"
                ),
                symbol=symbol,
                windows_run=windows_run,
                per_window_results=(),
                mirror_consistency=True,
            )
        c_log, c_err = _run_one(
            lake=lake, symbol=symbol,
            start=spec.start, end=spec.end,
            overlay_dict=candidate_overlay_dict,
        )
        if c_err is not None:
            return MultiWindowReplayResult(
                aborted=True,
                abort_reason=(
                    f"candidate_replay_error_in_{spec.window_id}: {c_err}"
                ),
                symbol=symbol,
                windows_run=windows_run,
                per_window_results=(),
                mirror_consistency=True,
            )
        if b_log is None or c_log is None:
            return MultiWindowReplayResult(
                aborted=True,
                abort_reason=f"empty_replay_for_{spec.window_id}",
                symbol=symbol,
                windows_run=windows_run,
                per_window_results=(),
                mirror_consistency=True,
            )

        observed_buckets = classify_window(h1_bars)
        stats = PerWindowStats(
            window_id=spec.window_id,
            n_bars=len(h1_bars),
            n_decided_bars=c_log.n_bars_envelope_decided,
            n_trades=c_log.final_state.n_trades,
            max_h1_gap_bars=_max_h1_gap_bars(h1_bars),
            halt_event_count=c_log.halt_event_count,
            observed_buckets=observed_buckets,
        )
        b_aggregate = ReplayLog(
            per_symbol=(b_log,),
            final_state=b_log.final_state,
            n_bars_envelope_decided=b_log.n_bars_envelope_decided,
            halt_event_count=b_log.halt_event_count,
        )
        c_aggregate = ReplayLog(
            per_symbol=(c_log,),
            final_state=c_log.final_state,
            n_bars_envelope_decided=c_log.n_bars_envelope_decided,
            halt_event_count=c_log.halt_event_count,
        )
        per_window.append(PerWindowReplayResult(
            window_id=spec.window_id,
            baseline_log=b_aggregate,
            candidate_log=c_aggregate,
            stats=stats,
        ))

    # Verify production constants unchanged (leak check).
    constants_after = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if k.startswith("_") and not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }
    if constants_before != constants_after:
        return MultiWindowReplayResult(
            aborted=True,
            abort_reason="leak_detected: production rule_engine constants mutated",
            symbol=symbol,
            windows_run=windows_run,
            per_window_results=(),
            mirror_consistency=True,
        )

    return MultiWindowReplayResult(
        aborted=False,
        abort_reason="",
        symbol=symbol,
        windows_run=windows_run,
        per_window_results=tuple(per_window),
        mirror_consistency=True,
    )
