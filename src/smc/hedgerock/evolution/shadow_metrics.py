"""Ticket 2 Step 5 — pure metric computation.

Takes a small replay-summary dict and returns a frozen
:class:`ShadowMetrics`. Also computes per-field deltas between
candidate and baseline metrics with sign convention:

    delta = candidate − baseline

So a candidate that outperforms baseline (higher return, lower DD)
has positive ``total_return_pct`` and **positive** ``max_dd_pct``
when DD is **worse** (DD is a non-negative percentage; bigger =
worse). Callers that want "DD better" semantics should negate.

Ticket 4 v2 extension — per-window risk surface
================================================

The XAUUSD-only multi-window runner emits one
:class:`PerWindowRiskMetrics` per ``WindowSpec``. The PASS gate
reduces the per-window list via :class:`WorstWindowSummary`, which
selects the WORST window per axis (DD, near-stopout, halt count,
exposure, coverage gap). Aggregate-average reductions are
explicitly NOT exposed: averaging across windows would let a
strong window mask a weak one and is forbidden by RFC v2 §6.

No imports from production runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields

from smc.hedgerock.evolution.shadow_artefact import ShadowMetrics


__all__ = [
    "METRIC_SCHEMA_VERSION",
    "PerWindowRiskMetrics",
    "WorstWindowSummary",
    "compute_delta_metrics",
    "compute_exposure_class_violation",
    "compute_metrics_from_replay",
    "compute_metrics_from_replay_log",
    "compute_per_window_risk_metrics",
    "compute_worst_window_summary",
]


def _compute_metric_schema_version() -> str:
    """SHA-256 over the field set of ShadowMetrics. Bumps automatically
    when fields are added/removed/renamed."""
    payload = {
        "metric_schema": "shadow_metrics_v1",
        "fields": [(f.name, f.type if isinstance(f.type, str) else str(f.type))
                   for f in fields(ShadowMetrics)],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


METRIC_SCHEMA_VERSION: str = _compute_metric_schema_version()


# ---------------------------------------------------------------------------
# Replay summary → ShadowMetrics
# ---------------------------------------------------------------------------


def compute_metrics_from_replay(replay_summary: dict) -> ShadowMetrics:
    """Build a :class:`ShadowMetrics` from a runner-emitted replay
    summary dict.

    Required keys: ``final_equity``, ``init_equity``, ``n_trades``,
    ``near_stopout_count``, ``halt_event_count``, ``max_dd_pct``,
    ``max_open_lots``, ``max_grid_density``, ``n_bars_envelope_decided``.
    """
    final_equity = float(replay_summary["final_equity"])
    init_equity = float(replay_summary["init_equity"])
    if init_equity > 0:
        total_return_pct = (final_equity - init_equity) / init_equity * 100.0
    else:
        total_return_pct = 0.0

    return ShadowMetrics(
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        max_dd_pct=float(replay_summary["max_dd_pct"]),
        near_stopout_count=int(replay_summary["near_stopout_count"]),
        n_trades=int(replay_summary["n_trades"]),
        max_open_lots=float(replay_summary["max_open_lots"]),
        max_grid_density=int(replay_summary["max_grid_density"]),
        halt_event_count=int(replay_summary["halt_event_count"]),
        n_bars_envelope_decided=int(replay_summary["n_bars_envelope_decided"]),
    )


def compute_metrics_from_replay_log(replay_log) -> ShadowMetrics:
    """Build ShadowMetrics from a real
    :class:`replay_executor.ReplayLog` (Ticket 3 path).

    The log carries the simulator's final state + the per-bar
    envelope-decision count + halt-event count. Every metric is
    drawn from real replay outcomes, not zero-trade placeholders.
    """
    fs = replay_log.final_state
    init = float(fs.init_equity) if fs.init_equity > 0 else 10_000.0
    if init > 0:
        total_return_pct = (fs.equity - init) / init * 100.0
    else:
        total_return_pct = 0.0
    return ShadowMetrics(
        final_equity=float(fs.equity),
        total_return_pct=float(total_return_pct),
        max_dd_pct=float(fs.max_dd_pct),
        near_stopout_count=int(fs.near_stopout_count),
        n_trades=int(fs.n_trades),
        max_open_lots=float(fs.max_open_lots),
        max_grid_density=int(fs.max_grid_density),
        halt_event_count=int(replay_log.halt_event_count),
        n_bars_envelope_decided=int(replay_log.n_bars_envelope_decided),
    )


def compute_exposure_class_violation(
    *, candidate_log, baseline_log,
) -> bool:
    """Behaviour-side exposure class check (Ticket 3 R5 row 12).

    Returns True iff the candidate's actual replay opened MORE
    lots / grid density / etc than baseline. Manifest's self-reported
    raises_* flags are NOT consulted here — this is the BEHAVIOUR
    side; gates can compare with manifest declaration separately.
    """
    cand_state = candidate_log.final_state
    base_state = baseline_log.final_state
    if cand_state.max_open_lots > base_state.max_open_lots + 1e-9:
        return True
    if cand_state.max_grid_density > base_state.max_grid_density:
        return True
    return False


def compute_delta_metrics(
    *, candidate: ShadowMetrics, baseline: ShadowMetrics,
) -> ShadowMetrics:
    """Per-field delta = candidate − baseline. Sign convention:

      - ``total_return_pct``: positive = candidate did better
      - ``max_dd_pct``: positive = candidate's DD was WORSE
      - ``near_stopout_count``: positive = candidate had MORE
      - ``halt_event_count``: positive = candidate halted MORE
      - ``n_trades``: positive = candidate traded MORE
      - ``max_open_lots`` / ``max_grid_density``: positive = candidate
        opened MORE

    ``final_equity`` is the candidate's absolute equity (we keep
    candidate.final_equity rather than computing a delta — useful
    for downstream display).
    """
    return ShadowMetrics(
        final_equity=candidate.final_equity,
        total_return_pct=candidate.total_return_pct - baseline.total_return_pct,
        max_dd_pct=candidate.max_dd_pct - baseline.max_dd_pct,
        near_stopout_count=candidate.near_stopout_count - baseline.near_stopout_count,
        n_trades=candidate.n_trades - baseline.n_trades,
        max_open_lots=candidate.max_open_lots - baseline.max_open_lots,
        max_grid_density=candidate.max_grid_density - baseline.max_grid_density,
        halt_event_count=candidate.halt_event_count - baseline.halt_event_count,
        n_bars_envelope_decided=(
            candidate.n_bars_envelope_decided - baseline.n_bars_envelope_decided
        ),
    )


# ---------------------------------------------------------------------------
# Ticket 4 v2 — Per-window risk metrics + worst-window summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerWindowRiskMetrics:
    """One window's worst-window-friendly risk surface.

    All fields are per-window terminal values — never window-mean,
    never aggregated across multiple windows. The PASS gate consumes
    a *list* of these and reduces via :class:`WorstWindowSummary`,
    which picks the WORST window per axis. Averaging here would let
    a strong window mask a weak window and is forbidden per RFC v2
    §6.

    Sign convention for delta fields (``delta = candidate −
    baseline``):
      * ``delta_pnl_pp`` — positive = candidate better
      * ``delta_dd_pp`` — positive = candidate's DD WORSE (DD is a
        non-negative percentage, larger is worse)
      * ``delta_near_stopout`` / ``delta_halt_event_count`` —
        positive = candidate had MORE
      * ``delta_max_open_lots`` / ``delta_max_grid_density`` —
        positive = candidate opened MORE

    Mode-bar fields are non-overlapping per-bar counters from the
    sidecar replay (``observe_mode_bars`` + ``halt_mode_bars`` +
    ``cooldown_mode_bars`` ≤ ``n_decided_bars``).
    """

    window_id: str
    n_bars: int
    n_decided_bars: int
    max_h1_gap_bars: int
    observed_buckets: tuple[str, ...]

    candidate_final_equity: float
    candidate_total_return_pct: float
    baseline_total_return_pct: float
    delta_pnl_pp: float

    candidate_max_dd_pct: float
    baseline_max_dd_pct: float
    delta_dd_pp: float

    candidate_near_stopout_count: int
    baseline_near_stopout_count: int
    delta_near_stopout: int

    candidate_max_open_lots: float
    baseline_max_open_lots: float
    candidate_max_grid_density: int
    baseline_max_grid_density: int
    delta_max_open_lots: float
    delta_max_grid_density: int

    candidate_halt_event_count: int
    baseline_halt_event_count: int
    delta_halt_event_count: int

    candidate_observe_mode_bars: int
    baseline_observe_mode_bars: int
    candidate_halt_mode_bars: int
    baseline_halt_mode_bars: int
    candidate_cooldown_mode_bars: int
    baseline_cooldown_mode_bars: int

    candidate_n_trades: int
    baseline_n_trades: int


@dataclass(frozen=True)
class WorstWindowSummary:
    """Per-axis worst-window selection across one PASS evaluation's
    per-window metric list. Each axis carries (value, window_id) so
    the PASS gate / report CLI can identify the binding window.

    No average / mean fields are exposed — the PASS gate must read
    only worst-window values per RFC v2 §6.
    """

    n_windows: int

    worst_delta_pnl_pp: float
    worst_delta_pnl_window_id: str

    worst_delta_dd_pp: float
    worst_delta_dd_window_id: str

    worst_candidate_dd_pp: float
    worst_candidate_dd_window_id: str

    worst_delta_near_stopout: int
    worst_delta_near_stopout_window_id: str

    worst_delta_halt_event_count: int
    worst_delta_halt_event_window_id: str

    worst_candidate_max_open_lots: float
    worst_candidate_max_open_lots_window_id: str

    worst_candidate_max_grid_density: int
    worst_candidate_max_grid_density_window_id: str

    worst_max_h1_gap_bars: int
    worst_max_h1_gap_window_id: str


def compute_per_window_risk_metrics(
    *,
    baseline_log,
    candidate_log,
    stats,
) -> PerWindowRiskMetrics:
    """Pure: build one :class:`PerWindowRiskMetrics` from a baseline
    + candidate :class:`ReplayLog` pair and the runner-emitted
    :class:`PerWindowStats` for the same window.

    Inputs are never mutated. The function never imports production
    runtime modules.
    """
    bs = baseline_log.final_state
    cs = candidate_log.final_state

    # Total return % from init_equity in each side.
    b_init = float(bs.init_equity) if bs.init_equity > 0 else 10_000.0
    c_init = float(cs.init_equity) if cs.init_equity > 0 else 10_000.0
    baseline_ret = (
        (bs.equity - b_init) / b_init * 100.0 if b_init > 0 else 0.0
    )
    candidate_ret = (
        (cs.equity - c_init) / c_init * 100.0 if c_init > 0 else 0.0
    )

    # Mode counters live on the per-symbol log; we read the first
    # entry (XAUUSD-only — single-element tuple). If for some reason
    # ``per_symbol`` is empty (defensive) we substitute zero counters.
    def _mode_counts(log) -> tuple[int, int, int]:
        if not log.per_symbol:
            return 0, 0, 0
        psl = log.per_symbol[0]
        return (
            int(getattr(psl, "observe_mode_bars", 0)),
            int(getattr(psl, "halt_mode_bars", 0)),
            int(getattr(psl, "cooldown_mode_bars", 0)),
        )

    b_obs, b_halt_b, b_cd = _mode_counts(baseline_log)
    c_obs, c_halt_b, c_cd = _mode_counts(candidate_log)

    return PerWindowRiskMetrics(
        window_id=stats.window_id,
        n_bars=int(stats.n_bars),
        n_decided_bars=int(stats.n_decided_bars),
        max_h1_gap_bars=int(stats.max_h1_gap_bars),
        observed_buckets=tuple(stats.observed_buckets),

        candidate_final_equity=float(cs.equity),
        candidate_total_return_pct=float(candidate_ret),
        baseline_total_return_pct=float(baseline_ret),
        delta_pnl_pp=float(candidate_ret - baseline_ret),

        candidate_max_dd_pct=float(cs.max_dd_pct),
        baseline_max_dd_pct=float(bs.max_dd_pct),
        # DD delta normalized to percentage points so the PASS gate
        # threshold (max_delta_dd_pp = 0.5pp) reads naturally.
        delta_dd_pp=float((cs.max_dd_pct - bs.max_dd_pct) * 100.0),

        candidate_near_stopout_count=int(cs.near_stopout_count),
        baseline_near_stopout_count=int(bs.near_stopout_count),
        delta_near_stopout=int(cs.near_stopout_count - bs.near_stopout_count),

        candidate_max_open_lots=float(cs.max_open_lots),
        baseline_max_open_lots=float(bs.max_open_lots),
        candidate_max_grid_density=int(cs.max_grid_density),
        baseline_max_grid_density=int(bs.max_grid_density),
        delta_max_open_lots=float(cs.max_open_lots - bs.max_open_lots),
        delta_max_grid_density=int(cs.max_grid_density - bs.max_grid_density),

        candidate_halt_event_count=int(candidate_log.halt_event_count),
        baseline_halt_event_count=int(baseline_log.halt_event_count),
        delta_halt_event_count=int(
            candidate_log.halt_event_count - baseline_log.halt_event_count
        ),

        candidate_observe_mode_bars=c_obs,
        baseline_observe_mode_bars=b_obs,
        candidate_halt_mode_bars=c_halt_b,
        baseline_halt_mode_bars=b_halt_b,
        candidate_cooldown_mode_bars=c_cd,
        baseline_cooldown_mode_bars=b_cd,

        candidate_n_trades=int(cs.n_trades),
        baseline_n_trades=int(bs.n_trades),
    )


def compute_worst_window_summary(
    metrics: list[PerWindowRiskMetrics],
) -> WorstWindowSummary:
    """Reduce a list of :class:`PerWindowRiskMetrics` to a frozen
    :class:`WorstWindowSummary`.

    Per-axis selection rules:

    * ``worst_delta_pnl_pp`` — minimum (most-negative) value
    * ``worst_delta_dd_pp`` / ``worst_delta_near_stopout`` /
      ``worst_delta_halt_event_count`` — maximum (most-degraded
      candidate) value
    * ``worst_candidate_dd_pp`` /
      ``worst_candidate_max_open_lots`` /
      ``worst_candidate_max_grid_density`` — maximum absolute
      candidate-side value
    * ``worst_max_h1_gap_bars`` — maximum bar-gap across windows

    Empty input → all-zero summary with empty window-id strings.
    Never raises.
    """
    if not metrics:
        return WorstWindowSummary(
            n_windows=0,
            worst_delta_pnl_pp=0.0, worst_delta_pnl_window_id="",
            worst_delta_dd_pp=0.0, worst_delta_dd_window_id="",
            worst_candidate_dd_pp=0.0, worst_candidate_dd_window_id="",
            worst_delta_near_stopout=0,
            worst_delta_near_stopout_window_id="",
            worst_delta_halt_event_count=0,
            worst_delta_halt_event_window_id="",
            worst_candidate_max_open_lots=0.0,
            worst_candidate_max_open_lots_window_id="",
            worst_candidate_max_grid_density=0,
            worst_candidate_max_grid_density_window_id="",
            worst_max_h1_gap_bars=0,
            worst_max_h1_gap_window_id="",
        )

    # min / max picks. We keep the FIRST tied window in deterministic
    # iteration order — the runner already orders metrics by spec.
    min_pnl = metrics[0]
    max_dd_delta = metrics[0]
    max_cand_dd = metrics[0]
    max_ns_delta = metrics[0]
    max_halt_delta = metrics[0]
    max_lots = metrics[0]
    max_density = metrics[0]
    max_gap = metrics[0]

    for m in metrics[1:]:
        if m.delta_pnl_pp < min_pnl.delta_pnl_pp:
            min_pnl = m
        if m.delta_dd_pp > max_dd_delta.delta_dd_pp:
            max_dd_delta = m
        if m.candidate_max_dd_pct > max_cand_dd.candidate_max_dd_pct:
            max_cand_dd = m
        if m.delta_near_stopout > max_ns_delta.delta_near_stopout:
            max_ns_delta = m
        if m.delta_halt_event_count > max_halt_delta.delta_halt_event_count:
            max_halt_delta = m
        if m.candidate_max_open_lots > max_lots.candidate_max_open_lots:
            max_lots = m
        if m.candidate_max_grid_density > max_density.candidate_max_grid_density:
            max_density = m
        if m.max_h1_gap_bars > max_gap.max_h1_gap_bars:
            max_gap = m

    return WorstWindowSummary(
        n_windows=len(metrics),
        worst_delta_pnl_pp=min_pnl.delta_pnl_pp,
        worst_delta_pnl_window_id=min_pnl.window_id,
        worst_delta_dd_pp=max_dd_delta.delta_dd_pp,
        worst_delta_dd_window_id=max_dd_delta.window_id,
        # Candidate-side absolute DD also surfaced in pp scale so the
        # ceiling threshold (worst_window_dd_ceiling_pp = 1.0pp) reads
        # naturally.
        worst_candidate_dd_pp=max_cand_dd.candidate_max_dd_pct * 100.0,
        worst_candidate_dd_window_id=max_cand_dd.window_id,
        worst_delta_near_stopout=max_ns_delta.delta_near_stopout,
        worst_delta_near_stopout_window_id=max_ns_delta.window_id,
        worst_delta_halt_event_count=max_halt_delta.delta_halt_event_count,
        worst_delta_halt_event_window_id=max_halt_delta.window_id,
        worst_candidate_max_open_lots=max_lots.candidate_max_open_lots,
        worst_candidate_max_open_lots_window_id=max_lots.window_id,
        worst_candidate_max_grid_density=max_density.candidate_max_grid_density,
        worst_candidate_max_grid_density_window_id=max_density.window_id,
        worst_max_h1_gap_bars=max_gap.max_h1_gap_bars,
        worst_max_h1_gap_window_id=max_gap.window_id,
    )
