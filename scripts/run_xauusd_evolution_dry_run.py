"""End-to-end XAUUSD self-evolution dry-run orchestrator.

REPORT-ONLY. NEVER touches live EA, ``rule_engine.py`` (red-line), or
``policy_registry/approved/``. Every artefact is written under
``--output-dir`` (default ``tmp/xauusd_evolution_dry_run/<UTC ISO>``).

XAUUSD-ONLY by hard assertion. ``--symbol`` other than ``XAUUSD`` is
rejected before any IO.

**Honesty rules — what the report does NOT do:**

  * It does NOT pass off the long-only buy-and-hold baseline as a
    strategy backtest. The "benchmark" section reports what the bars
    LOOK LIKE; the "dynamic-replay" section is a SEPARATE field whose
    value is NOT_AVAILABLE / WAIT until a real strategy replay is
    plumbed in.
  * It does NOT fabricate trade counts, veto reasons, cooldown reasons,
    risk-tier distributions, or transition-lock states. Those fields
    are RESERVED — set to ``None`` until the dynamic replay produces
    them.
  * Evidence quality is surfaced in both the markdown report and the
    JSON snapshot as ``evidence_quality`` ∈ {"benchmark_only",
    "dynamic_replay"}, so downstream consumers can refuse to act on
    benchmark-only evidence.

Stages (every stage is wrapped in graceful degradation — missing
inputs surface as DEGRADED markers in the final report, not a
crash):

    1.  XAUUSD-only assertion
    2.  Health-check pre-flight (with safe auto-recovery)
    3.  Load real XAUUSD H1 / H4 / D1 bars from the data lake
    4a. Benchmark statistics — long-only baseline over real H1 closes
    4b. Dynamic replay — strategy backtest against rule_engine
        (currently NOT_AVAILABLE; reserved fields stay null)
    5.  Regime detection
    6.  Anomaly shield check
    7.  Timeframe consensus
    8.  Adaptive stops
    9.  Stress test against a neutral demo proposal
    10. Recommend CLI (calibrator + explainability + fingerprint enabled)
    11. Fingerprint chain verify
    12. Registry audit summary
    13. Approval checklist (dry-run, never approves; dynamic replay = WAIT)
    14. Final XAUUSD report (markdown) + JSON snapshot

Exit codes:

    0 — all stages OK or DEGRADED (report still produced)
    2 — XAUUSD-only constraint violated
    3 — health check returned CRITICAL/DOWN even after auto-recovery
    4 — workspace lands under a forbidden parent (production registry)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants — XAUUSD-only by design.
# ---------------------------------------------------------------------------

SYMBOL = "XAUUSD"
DEFAULT_LOOKBACK_DAYS = 365

# Master promotion-readiness gate threshold. The replay MUST produce
# at least this many CLOSED trades (entries, trades AND exits ALL
# above this floor) before PASS unlocks. 20 was chosen as the minimum
# statistically meaningful sample — operator can raise it via
# --min-trades but never lower it below this constant.
MIN_DYNAMIC_REPLAY_TRADES = 20


# ---------------------------------------------------------------------------
# Path helpers (shared with the rest of the sidecar).
# ---------------------------------------------------------------------------


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


_REAL_REGISTRY_ROOT = _hedgerock_home() / "policy_registry"
_FORBIDDEN_OUTPUT_PARENTS = (
    _REAL_REGISTRY_ROOT,
    _ai_smc_home() / "config",
)


def _assert_output_dir_safe(output_dir: Path) -> None:
    abs_out = output_dir.resolve()
    for parent in _FORBIDDEN_OUTPUT_PARENTS:
        if parent in abs_out.parents or abs_out == parent:
            raise ValueError(
                f"output dir lands under a forbidden location: "
                f"{output_dir!s}"
            )


def _assert_xauusd_only(symbol: str) -> None:
    if symbol != SYMBOL:
        raise ValueError(
            f"this orchestrator is XAUUSD-only; got symbol={symbol!r}"
        )


# ---------------------------------------------------------------------------
# Per-stage result envelope.
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    status: str  # "OK" | "DEGRADED" | "FAILED"
    details: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def _ok(name: str, **details: Any) -> StageResult:
    return StageResult(name=name, status="OK", details=dict(details))


def _degraded(name: str, note: str, **details: Any) -> StageResult:
    return StageResult(
        name=name, status="DEGRADED", note=note, details=dict(details),
    )


# ---------------------------------------------------------------------------
# Real XAUUSD bars loader.
# ---------------------------------------------------------------------------


def _bars_from_lake(
    *,
    timeframe: str,
    start: datetime,
    end: datetime,
    lake_root: Path,
) -> list[dict]:
    """Return OHLC dicts ordered by ts. Empty list on any read error."""
    try:
        from smc.data.lake import ForexDataLake
        from smc.data.schemas import Timeframe
    except Exception:  # pragma: no cover — defensive
        return []
    try:
        lake = ForexDataLake(lake_root)
        tf = Timeframe(timeframe)
        df = lake.query(SYMBOL, tf, start=start, end=end)
    except Exception:
        return []
    if df.is_empty():
        return []
    cols = {"open", "high", "low", "close"}
    if not cols.issubset(set(df.columns)):
        return []
    rows = df.select(["ts", "open", "high", "low", "close"]).to_dicts()
    out: list[dict] = []
    for r in rows:
        out.append({
            "ts": r["ts"].isoformat() if r["ts"] is not None else None,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })
    return out


# ---------------------------------------------------------------------------
# Evidence-quality flags + walk-forward outputs.
#
# We split the bar-statistics concern into two strictly-separated
# concepts:
#
#   * BenchmarkStats — long-only buy-and-hold reference computed from
#     close-to-close returns on real XAUUSD H1 bars. NOT A STRATEGY.
#     This exists only to surface the shape of the price history.
#
#   * DynamicReplayStats — a real strategy backtest against
#     ``rule_engine`` via ``phase_d_walk_forward``. RESERVED until the
#     replay is plumbed end-to-end; for now ``try_dynamic_replay()``
#     returns ``None`` and the report renders NOT_AVAILABLE / WAIT.
#
# Both objects are passed independently into the snapshot so consumers
# never confuse one for the other. ``evidence_quality`` is set to
# ``"benchmark_only"`` until DynamicReplayStats is populated.
# ---------------------------------------------------------------------------


EVIDENCE_BENCHMARK_ONLY = "benchmark_only"
EVIDENCE_DYNAMIC_REPLAY = "dynamic_replay"


@dataclass(frozen=True)
class BenchmarkStats:
    """Long-only buy-and-hold reference. NOT A STRATEGY BACKTEST."""

    n_bars: int
    pnl_pct: float
    max_drawdown_pct: float
    win_rate_per_bar: float
    sharpe_annualised: float
    bars_per_year_assumed: int
    window_start: str
    window_end: str
    note: str = (
        "long-only buy-and-hold reference; surfaces price-history shape, "
        "NOT strategy performance"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "pnl_pct": self.pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate_per_bar": self.win_rate_per_bar,
            "sharpe_annualised": self.sharpe_annualised,
            "bars_per_year_assumed": self.bars_per_year_assumed,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "note": self.note,
        }


@dataclass(frozen=True)
class DynamicReplayStats:
    """Real strategy backtest output. RESERVED until run_walk_forward
    delivers genuine trade-level results.

    All fields default to ``None`` so partial integrations can fill
    only what they have without faking the rest. Once every required
    field is populated, the orchestrator promotes ``evidence_quality``
    to ``dynamic_replay``.
    """

    available: bool = False
    reason: str = "dynamic replay not yet plumbed against rule_engine"

    # Aggregate equity-curve metrics (only meaningful with trade-level
    # output; never derived from long-only bar returns).
    pnl_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe_annualised: float | None = None

    # Trade counters.
    trade_count: int | None = None
    entry_count: int | None = None
    exit_count: int | None = None
    win_rate: float | None = None  # closed-trade win rate, NOT per-bar

    # Reason distributions — keys are reason ids, values are counts.
    veto_reasons: dict[str, int] | None = None
    cooldown_reasons: dict[str, int] | None = None
    observe_reasons: dict[str, int] | None = None
    halt_reasons: dict[str, int] | None = None

    # Dynamic parameter & state distributions.
    risk_tier_distribution: dict[str, int] | None = None
    lot_factor_distribution: dict[str, int] | None = None
    transition_lock_states: dict[str, int] | None = None

    # Event counters — number of times the gate FIRED (not the
    # per-bar state).
    transition_lock_events: int | None = None
    cooldown_events: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "pnl_pct": self.pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_annualised": self.sharpe_annualised,
            "trade_count": self.trade_count,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "win_rate": self.win_rate,
            "veto_reasons": self.veto_reasons,
            "cooldown_reasons": self.cooldown_reasons,
            "observe_reasons": self.observe_reasons,
            "halt_reasons": self.halt_reasons,
            "risk_tier_distribution": self.risk_tier_distribution,
            "lot_factor_distribution": self.lot_factor_distribution,
            "transition_lock_states": self.transition_lock_states,
            "transition_lock_events": self.transition_lock_events,
            "cooldown_events": self.cooldown_events,
        }


_BARS_PER_YEAR_H1 = 24 * 365


def _benchmark_stats(bars: list[dict]) -> BenchmarkStats | None:
    """Compute long-only reference metrics from real H1 bars.

    Explicitly NOT a strategy backtest — every metric is a property of
    the price history alone (no entries, no exits, no rule_engine).
    """
    if len(bars) < 24:
        return None
    closes = [b["close"] for b in bars if b.get("close") is not None]
    if len(closes) < 24:
        return None

    rets: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev <= 0:
            continue
        rets.append((closes[i] - prev) / prev)
    if not rets:
        return None

    pnl_pct = (closes[-1] / closes[0] - 1.0) * 100.0
    wins = sum(1 for r in rets if r > 0)
    win_rate_per_bar = wins / len(rets)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in rets:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        dd = (equity / peak) - 1.0
        max_dd = min(max_dd, dd)
    max_drawdown_pct = max_dd * 100.0

    if len(rets) >= 2:
        mu = statistics.mean(rets)
        sigma = statistics.pstdev(rets)
        sharpe = (mu / sigma) * math.sqrt(_BARS_PER_YEAR_H1) if sigma > 0 else 0.0
    else:
        sharpe = 0.0

    return BenchmarkStats(
        n_bars=len(bars),
        pnl_pct=round(pnl_pct, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        win_rate_per_bar=round(win_rate_per_bar, 4),
        sharpe_annualised=round(sharpe, 4),
        bars_per_year_assumed=_BARS_PER_YEAR_H1,
        window_start=bars[0].get("ts") or "",
        window_end=bars[-1].get("ts") or "",
    )


def try_dynamic_replay(
    *,
    bars: list[dict],
    lake_root: Path,
    window_start: datetime,
    window_end: datetime,
    h4_bars: list[dict] | None = None,
    d1_bars: list[dict] | None = None,
) -> DynamicReplayStats:
    """Attempt a real rule_engine-backed walk-forward replay.

    Calls into ``smc.hedgerock.evolution.dynamic_replay`` which performs
    a closed-bar / no-lookahead replay over real XAUUSD H1 bars,
    honouring the live ``mode``, ``cooldown_until`` and transition-lock
    gates. When the replay produces a populated result, every reserved
    field is filled and ``DynamicReplayStats.available`` is ``True``;
    the orchestrator's ``_evidence_quality()`` then promotes the run
    to ``dynamic_replay``.

    Graceful degradation: import failures, insufficient bars, or
    XAUUSD-only assertion violations all surface as
    ``DynamicReplayStats(available=False, reason=...)`` with every
    reserved field null — never a partial / fabricated metric.
    """
    if not bars:
        return DynamicReplayStats(
            available=False,
            reason="no XAUUSD H1 bars supplied to replay",
        )
    try:
        from smc.hedgerock.evolution.dynamic_replay import replay_xauusd_h1
    except Exception as e:
        return DynamicReplayStats(
            available=False,
            reason=f"replay module import failed: {e!r}",
        )

    try:
        result = replay_xauusd_h1(
            symbol=SYMBOL,
            h1_bars=bars,
            h4_bars=h4_bars or [],
            d1_bars=d1_bars or [],
        )
    except Exception as e:  # pragma: no cover — defensive
        return DynamicReplayStats(
            available=False,
            reason=f"replay raised: {e!r}",
        )

    if not result.available:
        return DynamicReplayStats(
            available=False, reason=result.reason,
        )

    return DynamicReplayStats(
        available=True,
        reason=result.reason,
        pnl_pct=result.pnl_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_annualised=result.sharpe_annualised,
        trade_count=result.trade_count,
        entry_count=result.entry_count,
        exit_count=result.exit_count,
        win_rate=result.win_rate,
        veto_reasons=result.veto_reasons,
        cooldown_reasons=result.cooldown_reasons,
        observe_reasons=result.observe_reasons,
        halt_reasons=result.halt_reasons,
        risk_tier_distribution=result.risk_tier_distribution,
        lot_factor_distribution=result.lot_factor_distribution,
        transition_lock_states=result.transition_lock_states,
        transition_lock_events=result.transition_lock_events,
        cooldown_events=result.cooldown_events,
    )


def _evidence_quality(replay: DynamicReplayStats) -> str:
    return EVIDENCE_DYNAMIC_REPLAY if replay.available else EVIDENCE_BENCHMARK_ONLY


# Four-state validation status with explicit threshold enforcement.
#
#   not_available      — replay never ran
#   no_entries_wait    — replay ran but entry_count == 0
#   insufficient_sample — entries>0 but at least one of (entry, trade,
#                         exit) count is below ``min_trades``. The
#                         numbers exist but the sample is too small for
#                         statistical validation.
#   ok                 — every count >= ``min_trades``.
REPLAY_STATUS_OK = "ok"
REPLAY_STATUS_NO_ENTRIES_WAIT = "no_entries_wait"
REPLAY_STATUS_INSUFFICIENT_SAMPLE = "insufficient_sample"
REPLAY_STATUS_NOT_AVAILABLE = "not_available"


def _replay_validation_status(
    replay: DynamicReplayStats,
    *,
    min_trades: int = MIN_DYNAMIC_REPLAY_TRADES,
) -> str:
    if not replay.available:
        return REPLAY_STATUS_NOT_AVAILABLE
    entries = replay.entry_count or 0
    trades = replay.trade_count or 0
    exits = replay.exit_count or 0
    if entries <= 0:
        return REPLAY_STATUS_NO_ENTRIES_WAIT
    if entries < min_trades or trades < min_trades or exits < min_trades:
        return REPLAY_STATUS_INSUFFICIENT_SAMPLE
    return REPLAY_STATUS_OK


# Master promotion-readiness gate. ONE source of truth.
#
# PASS *only* when:
#   - replay_validation_status == "ok"  (i.e. available AND entry_count > 0)
# Every other input (benchmark stats, regime detection, anomaly level,
# multi-TF consensus, stress survival, fingerprint, registry presence,
# operator confirmation) is INSUFFICIENT on its own. They are listed
# separately on the approval checklist so the operator sees the full
# picture, but only the replay-validation gate flips this master row.
#
# Returns (status, reason) where status ∈ {"PASS", "WAIT"}.
PROMOTION_PASS = "PASS"
PROMOTION_WAIT = "WAIT"


def _promotion_readiness(
    replay: DynamicReplayStats,
    *,
    min_trades: int = MIN_DYNAMIC_REPLAY_TRADES,
) -> tuple[str, str]:
    """Master gate. PASS requires (ALL three):
        entry_count >= min_trades
        trade_count >= min_trades
        exit_count  >= min_trades

    Otherwise WAIT, with the reason naming the failing counts and the
    threshold so the operator knows EXACTLY what is missing.
    """
    if not replay.available:
        return (
            PROMOTION_WAIT,
            f"dynamic replay NOT_AVAILABLE — {replay.reason}",
        )
    entries = replay.entry_count or 0
    trades = replay.trade_count or 0
    exits = replay.exit_count or 0
    if entries <= 0:
        return (
            PROMOTION_WAIT,
            "replay ran but entry_count=0 — no-entry replay is not "
            "performance validation; PnL/WinRate/Sharpe are trivial zeros",
        )
    if entries < min_trades or trades < min_trades or exits < min_trades:
        return (
            PROMOTION_WAIT,
            f"insufficient dynamic replay sample — entry_count={entries}, "
            f"trade_count={trades}, exit_count={exits}; "
            f"promotion requires all three >= {min_trades} "
            f"(MIN_DYNAMIC_REPLAY_TRADES)",
        )
    return (
        PROMOTION_PASS,
        f"replay_validation_status=ok, entry_count={entries}, "
        f"trade_count={trades}, exit_count={exits}, "
        f"min_trades={min_trades}",
    )


# ---------------------------------------------------------------------------
# Approval checklist (dry-run).
# ---------------------------------------------------------------------------


def _approval_checklist(
    *,
    health_ok: bool,
    benchmark_ok: bool,
    dynamic_replay: DynamicReplayStats,
    regime_ok: bool,
    anomaly_state: Any | None,
    consensus: Any | None,
    stress_all_passed: bool,
    fingerprint_ok: bool,
    registry_present: bool,
) -> list[tuple[str, str]]:
    """Return (item, status) tuples — status is PASS / WAIT / SKIP.

    The dynamic-replay row is ALWAYS WAIT until ``DynamicReplayStats``
    arrives populated; the benchmark row is informational only and
    NEVER unlocks promotion by itself.
    """
    rows: list[tuple[str, str]] = []
    pr_status, pr_reason = _promotion_readiness(dynamic_replay)
    # Master gate first — operator should see ONE-line answer at the top.
    rows.append((
        "PROMOTION READINESS (master gate)",
        f"{pr_status} — {pr_reason}",
    ))
    rows.append(("health-check pre-flight", "PASS" if health_ok else "WAIT"))
    rows.append((
        "benchmark (long-only) stats produced — INFORMATIONAL ONLY",
        "PASS (informational)" if benchmark_ok else "WAIT",
    ))
    replay_status = _replay_validation_status(dynamic_replay)
    if replay_status == REPLAY_STATUS_OK:
        replay_row = "PASS"
    elif replay_status == REPLAY_STATUS_NO_ENTRIES_WAIT:
        replay_row = (
            "WAIT (replay ran but produced no entries — "
            "no-entry replay is not performance validation)"
        )
    else:
        replay_row = f"WAIT ({dynamic_replay.reason})"
    rows.append(("dynamic replay against rule_engine", replay_row))
    rows.append((
        "regime detection produced snapshot",
        "PASS" if regime_ok else "WAIT",
    ))
    if anomaly_state is None:
        rows.append(("anomaly level below halt", "WAIT"))
    else:
        try:
            level = anomaly_state.level.value
        except Exception:
            level = str(anomaly_state)
        rows.append((
            f"anomaly level = {level}",
            "PASS" if level in ("NORMAL", "ELEVATED") else "WAIT",
        ))
    if consensus is None:
        rows.append(("multi-TF consensus reached", "WAIT"))
    else:
        rows.append((
            f"multi-TF consensus={consensus.consensus_score:.2f}",
            "PASS" if consensus.can_recommend else "WAIT",
        ))
    rows.append((
        "stress scenarios all survived",
        "PASS" if stress_all_passed else "WAIT",
    ))
    rows.append(("fingerprint chain verified", "PASS" if fingerprint_ok else "WAIT"))
    rows.append((
        "operator registry present",
        "PASS" if registry_present else "SKIP (DEGRADED — re-point flag)",
    ))
    rows.append(("operator promotion confirmation", "WAIT (DRY-RUN, never asks)"))
    return rows


# ---------------------------------------------------------------------------
# Markdown report renderer.
# ---------------------------------------------------------------------------


def _render_report(
    *,
    output_dir: Path,
    symbol: str,
    run_id: str,
    evidence_path: Path,
    evidence_hash: str,
    stages: list[StageResult],
    benchmark: BenchmarkStats | None,
    dynamic_replay: DynamicReplayStats,
    evidence_quality: str,
    regime: Any | None,
    anomaly: Any | None,
    consensus: Any | None,
    stop: Any | None,
    stress_total: int,
    stress_survived: int,
    n_proposals: int,
    n_recommend: int,
    fingerprint_verify: dict[str, Any] | None,
    approval_rows: list[tuple[str, str]],
    registry_present: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# XAUUSD Evolution — Dry-Run Report ({symbol})")
    lines.append("")
    lines.append(
        f"_Generated at {datetime.now(timezone.utc).isoformat()} — "
        "REPORT-ONLY. NOT LIVE / NOT APPROVED / NOT DEPLOYED._"
    )
    lines.append("")
    lines.append(f"**Run id:** `{run_id}`")
    lines.append(f"**Evidence payload:** `{evidence_path}`")
    lines.append(f"**Evidence hash (SHA-256):** `{evidence_hash}`")
    lines.append("")
    lines.append(
        "> The recommendation / calibrator / explainability / fingerprint "
        "chain ran against the evidence at the path above. The hash is "
        "the same canonical-JSON SHA-256 used by the fingerprint "
        "module — operators can audit which evidence the chain was "
        "bound to without re-running the orchestrator."
    )
    lines.append("")

    replay_status = _replay_validation_status(dynamic_replay)
    pr_status, pr_reason = _promotion_readiness(dynamic_replay)
    lines.append(
        f"**PROMOTION READINESS (master gate):** `{pr_status}` — {pr_reason}"
    )
    lines.append("")
    lines.append(f"**Evidence quality:** `{evidence_quality}`")
    lines.append(
        f"**Replay validation status:** `{replay_status}`"
    )
    lines.append(
        f"**Min trades threshold (PASS gate):** `{MIN_DYNAMIC_REPLAY_TRADES}`"
    )
    lines.append(
        f"**Sample counts:** entry=`{dynamic_replay.entry_count}`, "
        f"trade=`{dynamic_replay.trade_count}`, "
        f"exit=`{dynamic_replay.exit_count}`"
    )
    if replay_status == REPLAY_STATUS_NO_ENTRIES_WAIT:
        lines.append("")
        lines.append(
            "> ⚠️  The replay adapter ran end-to-end against real "
            "rule_engine, but EVERY bar landed in observe / halt / "
            "veto — **0 entries**. PnL, WinRate and Sharpe are "
            "trivially `0` and are **NOT performance validation**. "
            "Approval is held at WAIT until a replay produces real "
            "entries with real exits."
        )
    if evidence_quality == EVIDENCE_BENCHMARK_ONLY:
        lines.append("")
        lines.append(
            "> ⚠️  This run carries **benchmark-only** evidence. The "
            "dynamic-replay backtest against `rule_engine` is "
            "NOT_AVAILABLE in this build, so trade-level metrics "
            "(trade count, veto/cooldown reasons, risk-tier mix, "
            "transition locks) are reserved as `null`. The "
            "long-only baseline below describes the **price history**, "
            "**NOT** strategy performance — do not promote on its "
            "basis."
        )
    lines.append("")

    lines.append("## 1. Stage status")
    lines.append("")
    lines.append("| Stage | Status | Note |")
    lines.append("|---|---|---|")
    for s in stages:
        lines.append(f"| {s.name} | {s.status} | {s.note} |")
    lines.append("")

    lines.append("## 2A. Benchmark — long-only baseline (NOT a strategy)")
    lines.append("")
    if benchmark is not None:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Bars | {benchmark.n_bars} |")
        lines.append(
            f"| Window | {benchmark.window_start} → {benchmark.window_end} |"
        )
        lines.append(
            f"| PnL % (long-only buy-and-hold) | {benchmark.pnl_pct:+.4f} |"
        )
        lines.append(f"| Max drawdown % | {benchmark.max_drawdown_pct:.4f} |")
        lines.append(
            f"| Win rate (per-bar, NOT per-trade) "
            f"| {benchmark.win_rate_per_bar:.4f} |"
        )
        lines.append(
            f"| Sharpe (annualised) | {benchmark.sharpe_annualised:+.4f} |"
        )
        lines.append(
            f"| Annualisation factor (bars/yr) "
            f"| {benchmark.bars_per_year_assumed} |"
        )
        lines.append("")
        lines.append(f"_{benchmark.note}_")
    else:
        lines.append("_DEGRADED — no XAUUSD bars available in the lake._")
    lines.append("")

    lines.append("## 2B. Dynamic replay — strategy backtest (rule_engine)")
    lines.append("")
    if replay_status == REPLAY_STATUS_NO_ENTRIES_WAIT:
        lines.append(
            "**Validation status: `no_entries_wait`** — replay ran "
            "but produced 0 entries. The metrics below are TRIVIAL "
            "ZEROS, NOT performance evidence."
        )
        lines.append("")
    if dynamic_replay.available:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Trade count | {dynamic_replay.trade_count} |")
        lines.append(f"| Entry count | {dynamic_replay.entry_count} |")
        lines.append(f"| Win rate (closed trades) | {dynamic_replay.win_rate} |")
        lines.append(f"| PnL % | {dynamic_replay.pnl_pct} |")
        lines.append(f"| Max drawdown % | {dynamic_replay.max_drawdown_pct} |")
        lines.append(
            f"| Sharpe (annualised) | {dynamic_replay.sharpe_annualised} |"
        )
        for label, dist in (
            ("veto reasons", dynamic_replay.veto_reasons),
            ("cooldown reasons", dynamic_replay.cooldown_reasons),
            ("observe reasons", dynamic_replay.observe_reasons),
            ("halt reasons", dynamic_replay.halt_reasons),
            ("risk-tier distribution", dynamic_replay.risk_tier_distribution),
            ("lot-factor distribution", dynamic_replay.lot_factor_distribution),
            ("transition-lock states", dynamic_replay.transition_lock_states),
        ):
            if dist:
                top = ", ".join(f"{k}:{v}" for k, v in list(dist.items())[:6])
                lines.append(f"| {label} | {top} |")
    else:
        lines.append(
            f"**NOT_AVAILABLE** — {dynamic_replay.reason}"
        )
        lines.append("")
        lines.append("Reserved fields (all `null` in this run):")
        lines.append("")
        lines.append(
            "- `trade_count`, `entry_count`, `win_rate` "
            "(closed-trade, NOT per-bar)"
        )
        lines.append("- `pnl_pct`, `max_drawdown_pct`, `sharpe_annualised`")
        lines.append(
            "- `veto_reasons`, `cooldown_reasons`, `observe_reasons`, "
            "`halt_reasons`"
        )
        lines.append(
            "- `risk_tier_distribution`, `lot_factor_distribution`, "
            "`transition_lock_states`"
        )
    lines.append("")

    lines.append("## 3. Regime + anomaly (real H1 bars)")
    lines.append("")
    if regime is not None:
        lines.append(
            f"- regime = `{regime.regime.value}` "
            f"(confidence = {regime.confidence:.2f})"
        )
    else:
        lines.append("- regime = _DEGRADED_")
    if anomaly is not None:
        lines.append(f"- anomaly level = `{anomaly.level.value}`")
    else:
        lines.append("- anomaly level = _DEGRADED_")
    lines.append("")

    lines.append("## 4. Multi-timeframe consensus + adaptive stop")
    lines.append("")
    if consensus is not None:
        lines.append(f"- session = `{consensus.active_session.value}`")
        lines.append(
            f"- consensus score = {consensus.consensus_score:.2f}; "
            f"can_recommend = {consensus.can_recommend}"
        )
    else:
        lines.append("- consensus = _DEGRADED_")
    if stop is not None:
        lines.append(
            f"- vol regime = `{stop.vol_regime.value}`; "
            f"atr_mult = {stop.atr_multiplier}; "
            f"position_scale = {stop.position_scale}"
        )
    else:
        lines.append("- adaptive stop = _DEGRADED_")
    lines.append("")

    lines.append("## 5. Stress-test survival matrix")
    lines.append("")
    lines.append(
        f"- scenarios = {stress_total}; "
        f"survived = {stress_survived}; "
        f"breached = {stress_total - stress_survived}"
    )
    lines.append("")

    lines.append("## 6. Recommendation pipeline")
    lines.append("")
    lines.append(f"- evidence quality: `{evidence_quality}`")
    lines.append(f"- replay validation status: `{replay_status}`")
    if evidence_quality == EVIDENCE_BENCHMARK_ONLY:
        lines.append(
            "- ⚠️  recommendation evidence is **benchmark-only**; "
            "candidate generator ran against the price-history baseline, "
            "**not** trade-level replay output"
        )
    elif replay_status == REPLAY_STATUS_NO_ENTRIES_WAIT:
        lines.append(
            "- ⚠️  replay produced 0 entries — operator cannot see "
            "real PnL/WinRate/Sharpe yet; promotion readiness held at WAIT"
        )
    lines.append(f"- candidates produced: {n_proposals}")
    lines.append(f"- proposals with decision=RECOMMEND: {n_recommend}")
    lines.append("")

    lines.append("## 7. Fingerprint chain")
    lines.append("")
    if fingerprint_verify is None:
        lines.append("- _DEGRADED — fingerprint stage skipped or not produced._")
    else:
        lines.append(f"- entries: {fingerprint_verify.get('n_entries', 0)}")
        lines.append(f"- verify ok: {fingerprint_verify.get('ok', False)}")
        if fingerprint_verify.get("first_break_reason"):
            lines.append(
                f"- first break: {fingerprint_verify['first_break_reason']}"
            )
    lines.append("")

    lines.append("## 8. Registry audit")
    lines.append("")
    lines.append(
        f"- HedgeRock policy_registry root: `{_REAL_REGISTRY_ROOT}`"
    )
    lines.append(
        "- registry present on this machine: "
        f"{'yes' if registry_present else 'no (DEGRADED — re-point flag)'}"
    )
    lines.append("")

    lines.append("## 9. Approval checklist (DRY-RUN)")
    lines.append("")
    lines.append("| Item | Status |")
    lines.append("|---|---|")
    for item, status in approval_rows:
        lines.append(f"| {item} | {status} |")
    lines.append("")

    lines.append("---")
    lines.append("**status: NOT LIVE / NOT APPROVED / NOT DEPLOYED**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommend pipeline shim — calls the existing recommend CLI with a
# minimal evidence fixture so the chain (calibrator + explainability +
# fingerprint) runs end to end against the demo evidence skeleton.
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    """Stable, sortable per-run identifier — used as evidence-binding tag."""
    return "xauusd-dry-run-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def _evidence_payload(
    *,
    run_id: str,
    symbol: str,
    lookback_days: int,
    window_start: datetime,
    window_end: datetime,
    benchmark: BenchmarkStats | None,
    dynamic_replay: DynamicReplayStats,
    evidence_quality: str,
    n_h1: int,
    n_h4: int,
    n_d1: int,
    registry_present: bool,
    registry_root: Path,
    registry_json_count: int | None,
) -> dict[str, Any]:
    """Canonical dict shape that BOTH wf.md/availability.md/atlas.md AND
    the snapshot's evidence_hash are derived from. Single source of
    truth — operators auditing the fingerprint chain hash the same
    bytes that the markdown reflects."""
    return {
        "run_id": run_id,
        "symbol": symbol,
        "lookback_days": lookback_days,
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "evidence_quality": evidence_quality,
        "replay_validation_status": _replay_validation_status(dynamic_replay),
        "dynamic_replay_min_trades": MIN_DYNAMIC_REPLAY_TRADES,
        "dynamic_replay_sample_counts": {
            "entry_count": dynamic_replay.entry_count,
            "trade_count": dynamic_replay.trade_count,
            "exit_count": dynamic_replay.exit_count,
        },
        "bars_loaded": {"H1": n_h1, "H4": n_h4, "D1": n_d1},
        "benchmark_long_only": benchmark.to_dict() if benchmark else None,
        "dynamic_replay": dynamic_replay.to_dict(),
        "registry_audit": {
            "present": registry_present,
            "root": str(registry_root),
            "json_count": registry_json_count,
        },
        "promotion_status": "NOT LIVE / NOT APPROVED / NOT DEPLOYED",
    }


def _evidence_hash(payload: dict[str, Any]) -> str:
    """SHA-256 over canonical JSON of the evidence payload — same
    canonicalisation as the fingerprint module so chain entries can
    cross-reference this hash by value."""
    import hashlib
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _render_wf_md(payload: dict[str, Any]) -> str:
    """Render the walk-forward evidence markdown bound to this run.

    REPLACES the previous static fixture. Every field is sourced from
    the run's actual evidence so the recommend pipeline (and through
    it the calibrator + explainability + fingerprint) operates on
    THIS run's data, not a placeholder."""
    bench = payload.get("benchmark_long_only")
    replay = payload.get("dynamic_replay") or {}
    reg = payload.get("registry_audit") or {}

    lines: list[str] = []
    lines.append(
        f"# Walk-forward evidence — run `{payload['run_id']}`"
    )
    lines.append("")
    lines.append(f"- symbol: **{payload['symbol']}**")
    lines.append(f"- lookback_days: {payload['lookback_days']}")
    lines.append(
        f"- window: `{payload['window']['start']}` → "
        f"`{payload['window']['end']}`"
    )
    lines.append(f"- evidence_quality: **`{payload['evidence_quality']}`**")
    rs = payload.get("replay_validation_status")
    if rs:
        lines.append(f"- replay_validation_status: **`{rs}`**")
    mt = payload.get("dynamic_replay_min_trades")
    sc = payload.get("dynamic_replay_sample_counts") or {}
    if mt is not None:
        lines.append(
            f"- dynamic_replay_min_trades: **{mt}** (PASS gate threshold)"
        )
    if sc:
        lines.append(
            f"- sample_counts: entry={sc.get('entry_count')}, "
            f"trade={sc.get('trade_count')}, exit={sc.get('exit_count')}"
        )
    bars = payload.get("bars_loaded", {})
    lines.append(
        f"- bars_loaded: H1={bars.get('H1')}, H4={bars.get('H4')}, "
        f"D1={bars.get('D1')}"
    )
    lines.append("")

    lines.append("## Benchmark — long-only baseline (NOT a strategy)")
    lines.append("")
    if bench:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| n_bars | {bench['n_bars']} |")
        lines.append(f"| pnl_pct (long-only) | {bench['pnl_pct']:+.4f} |")
        lines.append(f"| max_drawdown_pct | {bench['max_drawdown_pct']:.4f} |")
        lines.append(
            f"| win_rate_per_bar (NOT per-trade) | "
            f"{bench['win_rate_per_bar']:.4f} |"
        )
        lines.append(
            f"| sharpe_annualised | {bench['sharpe_annualised']:+.4f} |"
        )
        lines.append(f"| window_start | {bench['window_start']} |")
        lines.append(f"| window_end | {bench['window_end']} |")
        lines.append("")
        lines.append(f"_{bench['note']}_")
    else:
        lines.append("_DEGRADED — no XAUUSD bars available._")
    lines.append("")

    lines.append("## Dynamic replay — strategy backtest (rule_engine)")
    lines.append("")
    lines.append(f"_Source: {replay.get('reason', 'unspecified')}_")
    lines.append("")
    if replay.get("available"):
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in (
            "trade_count", "entry_count", "exit_count", "win_rate",
            "pnl_pct", "max_drawdown_pct", "sharpe_annualised",
            "transition_lock_events", "cooldown_events",
        ):
            lines.append(f"| {key} | {replay.get(key)} |")
    else:
        lines.append(
            f"**NOT_AVAILABLE** — {replay.get('reason', 'unspecified')}"
        )
        lines.append("")
        lines.append(
            "Reserved trade-level fields (all `null` in this run): "
            "`trade_count`, `entry_count`, `win_rate`, `pnl_pct`, "
            "`max_drawdown_pct`, `sharpe_annualised`, `veto_reasons`, "
            "`cooldown_reasons`, `observe_reasons`, `halt_reasons`, "
            "`risk_tier_distribution`, `lot_factor_distribution`, "
            "`transition_lock_states`."
        )
    lines.append("")

    lines.append("## Registry audit")
    lines.append("")
    lines.append(f"- registry_present: {reg.get('present')}")
    lines.append(f"- registry_root: `{reg.get('root')}`")
    lines.append(f"- json_count: {reg.get('json_count')}")
    lines.append("")

    lines.append("## Action gate")
    lines.append("")
    lines.append("```yaml")
    lines.append("NO_STRATEGY_CHANGE: false")
    lines.append("```")
    lines.append("")
    lines.append(f"**status: {payload['promotion_status']}**")
    lines.append("")
    return "\n".join(lines)


def _render_availability_md(payload: dict[str, Any]) -> str:
    bars = payload.get("bars_loaded", {})
    lines: list[str] = []
    lines.append(
        f"# Phase D-cont3-preflight — run `{payload['run_id']}`"
    )
    lines.append("")
    lines.append(f"- symbol: **{payload['symbol']}**")
    lines.append(
        f"- window: `{payload['window']['start']}` → "
        f"`{payload['window']['end']}`"
    )
    lines.append(f"- evidence_quality: **`{payload['evidence_quality']}`**")
    lines.append("")
    lines.append("## Bars loaded this run")
    lines.append("")
    lines.append("| Symbol | Timeframe | Bar count |")
    lines.append("|---|---|---|")
    for tf in ("H1", "H4", "D1"):
        lines.append(f"| {payload['symbol']} | {tf} | {bars.get(tf, 0)} |")
    lines.append("")
    lines.append("## Action gate")
    lines.append("")
    lines.append("```yaml")
    lines.append("NO_STRATEGY_CHANGE: false")
    lines.append("```")
    lines.append("")
    lines.append(f"**status: {payload['promotion_status']}**")
    lines.append("")
    return "\n".join(lines)


def _render_atlas_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Atlas — run `{payload['run_id']}`")
    lines.append("")
    lines.append(f"- symbol: **{payload['symbol']}**")
    lines.append(f"- evidence_quality: **`{payload['evidence_quality']}`**")
    lines.append(
        f"- window: `{payload['window']['start']}` → "
        f"`{payload['window']['end']}`"
    )
    lines.append("")
    lines.append(
        "Atlas regime/opportunity replication is upstream of this "
        "orchestrator; this file marks the run binding so the "
        "downstream recommend / calibrator / explainability / "
        "fingerprint chain anchors to this evidence."
    )
    lines.append("")
    lines.append(f"**status: {payload['promotion_status']}**")
    lines.append("")
    return "\n".join(lines)


def _seed_evidence(
    workspace: Path,
    *,
    payload: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    """Write atlas / availability / wf / audit fixtures bound to this
    run's evidence payload. ``safety_bounds.yaml`` is intentionally
    absent so G6 fires safety_bound_undefined, which is the v0
    candidate-generator trigger that lets the chain produce proposals."""
    fixture = workspace / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    atlas = fixture / "atlas.md"
    atlas.write_text(_render_atlas_md(payload), encoding="utf-8")
    avail = fixture / "availability.md"
    avail.write_text(_render_availability_md(payload), encoding="utf-8")
    wf = fixture / "wf.md"
    wf.write_text(_render_wf_md(payload), encoding="utf-8")
    bounds = fixture / "safety_bounds.yaml"  # absent on purpose — fires G6
    audit_log = fixture / "_audit.md"
    audit_log.write_text(
        f"# Shadow-Artefact Registry Audit Log — run `{payload['run_id']}`\n\n"
        "(no incidents)\n",
        encoding="utf-8",
    )
    return atlas, avail, wf, bounds, audit_log


# ---------------------------------------------------------------------------
# Main orchestrator.
# ---------------------------------------------------------------------------


def run(
    *,
    output_dir: Path,
    symbol: str = SYMBOL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    lake_root: Path | None = None,
    registry_root: Path | None = None,
) -> int:
    _assert_xauusd_only(symbol)
    output_dir = Path(output_dir).resolve()
    try:
        _assert_output_dir_safe(output_dir)
    except ValueError as e:
        print(f"FAILED (forbidden output): {e}", file=sys.stderr)
        return 4
    output_dir.mkdir(parents=True, exist_ok=True)

    lake_root = (lake_root or (_ai_smc_home() / "data" / "parquet")).resolve()
    registry_root = (registry_root or _REAL_REGISTRY_ROOT).resolve()

    run_id = _make_run_id()
    stages: list[StageResult] = []

    # Stage 1 — XAUUSD-only assertion (already done above; record it).
    stages.append(_ok("xauusd_only", symbol=symbol, run_id=run_id))
    print(f"[1/13] XAUUSD-only assertion OK (symbol={symbol}, run_id={run_id})")

    # Stage 2 — health check.
    print("[2/13] HEALTH-CHECK")
    from smc.hedgerock.evolution.health_check import (
        HealthStatus as _HS, auto_recover as _autorecover, diagnose as _diag,
    )
    health = _diag(
        registry_root=output_dir / "registry",
        queue_path=output_dir / "queue" / "shadow_test_queue.jsonl",
        ledger_path=output_dir / "ledger" / "paper_test_ledger.jsonl",
    )
    health_ok = True
    if health.overall in (_HS.CRITICAL, _HS.DOWN):
        rec = _autorecover(
            health,
            registry_root=output_dir / "registry",
            queue_path=output_dir / "queue" / "shadow_test_queue.jsonl",
            ledger_path=output_dir / "ledger" / "paper_test_ledger.jsonl",
        )
        if rec.post_status in (_HS.CRITICAL, _HS.DOWN):
            stages.append(StageResult(
                name="health_check", status="FAILED",
                details={"overall": rec.post_status.value},
                note="health check unrecoverable",
            ))
            print(f"  FAILED — overall={rec.post_status.value}", file=sys.stderr)
            return 3
        stages.append(_ok(
            "health_check",
            pre=health.overall.value, post=rec.post_status.value,
            recovered=len(rec.actions),
        ))
        print(f"  recovered: {health.overall.value} → {rec.post_status.value}")
    elif health.overall == _HS.DEGRADED:
        stages.append(_degraded(
            "health_check",
            note="DEGRADED (acceptable on fresh machine)",
            overall=health.overall.value,
        ))
        health_ok = False
        print(f"  DEGRADED — overall={health.overall.value}")
    else:
        stages.append(_ok("health_check", overall=health.overall.value))
        print(f"  OK — overall={health.overall.value}")

    # Stage 3 — load real XAUUSD bars.
    print(f"[3/13] LOAD REAL XAUUSD BARS (lake={lake_root})")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    h1_bars = _bars_from_lake(
        timeframe="H1", start=start, end=end, lake_root=lake_root,
    )
    h4_bars = _bars_from_lake(
        timeframe="H4", start=start, end=end, lake_root=lake_root,
    )
    d1_bars = _bars_from_lake(
        timeframe="D1", start=start - timedelta(days=lookback_days),
        end=end, lake_root=lake_root,
    )
    if not h1_bars:
        # Try a wider historical window — fresh checkouts may not have
        # bars in the trailing year, but the lake has 2020-2024.
        end_fallback = datetime(2025, 1, 1, tzinfo=timezone.utc)
        start_fallback = end_fallback - timedelta(days=lookback_days)
        h1_bars = _bars_from_lake(
            timeframe="H1", start=start_fallback, end=end_fallback,
            lake_root=lake_root,
        )
        h4_bars = _bars_from_lake(
            timeframe="H4", start=start_fallback, end=end_fallback,
            lake_root=lake_root,
        )
        d1_bars = _bars_from_lake(
            timeframe="D1",
            start=start_fallback - timedelta(days=lookback_days),
            end=end_fallback, lake_root=lake_root,
        )
    if h1_bars:
        stages.append(_ok(
            "load_bars",
            n_h1=len(h1_bars), n_h4=len(h4_bars), n_d1=len(d1_bars),
        ))
        print(
            f"  loaded H1={len(h1_bars)} H4={len(h4_bars)} D1={len(d1_bars)}"
        )
    else:
        stages.append(_degraded(
            "load_bars", note=f"lake at {lake_root} has no XAUUSD bars",
        ))
        print(f"  DEGRADED — no bars under {lake_root}")

    # Stage 4a — benchmark (long-only baseline). NOT a strategy.
    print("[4a/13] BENCHMARK STATS (long-only baseline; NOT a strategy)")
    benchmark = _benchmark_stats(h1_bars) if h1_bars else None
    if benchmark is not None:
        stages.append(_ok("benchmark_long_only", **benchmark.to_dict()))
        print(
            f"  benchmark PnL={benchmark.pnl_pct:+.4f}% "
            f"MaxDD={benchmark.max_drawdown_pct:.4f}% "
            f"WR/bar={benchmark.win_rate_per_bar:.4f} "
            f"Sharpe={benchmark.sharpe_annualised:+.4f} "
            "(NOT strategy)"
        )
    else:
        stages.append(_degraded(
            "benchmark_long_only", note="insufficient bars for benchmark",
        ))
        print("  DEGRADED — insufficient bars")

    # Stage 4b — dynamic replay against rule_engine. Currently
    # NOT_AVAILABLE; reserved fields stay null.
    print("[4b/13] DYNAMIC REPLAY (rule_engine-backed walk-forward)")
    dynamic_replay = try_dynamic_replay(
        bars=h1_bars or [],
        h4_bars=h4_bars or [],
        d1_bars=d1_bars or [],
        lake_root=lake_root,
        window_start=start, window_end=end,
    )
    if dynamic_replay.available:
        stages.append(_ok("dynamic_replay", **dynamic_replay.to_dict()))
        print(
            f"  trades={dynamic_replay.trade_count} "
            f"entries={dynamic_replay.entry_count} "
            f"PnL={dynamic_replay.pnl_pct} "
            f"Sharpe={dynamic_replay.sharpe_annualised}"
        )
    else:
        stages.append(_degraded(
            "dynamic_replay",
            note=f"NOT_AVAILABLE — {dynamic_replay.reason}",
            **dynamic_replay.to_dict(),
        ))
        print(f"  NOT_AVAILABLE — {dynamic_replay.reason}")
    evidence_quality = _evidence_quality(dynamic_replay)
    print(f"  evidence_quality = {evidence_quality}")

    # Stage 5 — regime.
    print("[5/13] REGIME DETECTION")
    regime = None
    if h1_bars:
        try:
            from smc.hedgerock.evolution.regime_engine import RegimeDetector
            regime = RegimeDetector().detect(bars=h1_bars)
            stages.append(_ok(
                "regime",
                regime=regime.regime.value, confidence=regime.confidence,
            ))
            print(
                f"  regime={regime.regime.value} "
                f"confidence={regime.confidence:.2f}"
            )
        except Exception as e:
            stages.append(_degraded(
                "regime", note=f"detector raised: {e!r}",
            ))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded("regime", note="no bars"))

    # Stage 6 — anomaly.
    print("[6/13] ANOMALY SHIELD")
    anomaly = None
    if h1_bars:
        try:
            from smc.hedgerock.evolution.anomaly_shield import (
                AnomalyDetector, shield_action,
            )
            anomaly = AnomalyDetector().detect(bars=h1_bars)
            action = shield_action(anomaly)
            stages.append(_ok(
                "anomaly",
                level=anomaly.level.value,
                new_candidates_allowed=action.new_candidates_allowed,
            ))
            print(
                f"  level={anomaly.level.value} "
                f"new_candidates_allowed={action.new_candidates_allowed}"
            )
        except Exception as e:
            stages.append(_degraded("anomaly", note=f"detector raised: {e!r}"))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded("anomaly", note="no bars"))

    # Stage 7 — multi-TF consensus.
    print("[7/13] TIMEFRAME CONSENSUS")
    consensus = None
    try:
        from smc.hedgerock.evolution.multi_timeframe_state import (
            compute_consensus,
        )
        consensus = compute_consensus(
            d1_bars=d1_bars or None,
            h4_bars=h4_bars or None,
            h1_bars=h1_bars or None,
        )
        stages.append(_ok(
            "consensus",
            session=consensus.active_session.value,
            score=consensus.consensus_score,
            can_recommend=consensus.can_recommend,
        ))
        print(
            f"  session={consensus.active_session.value} "
            f"score={consensus.consensus_score:.2f} "
            f"can_recommend={consensus.can_recommend}"
        )
    except Exception as e:
        stages.append(_degraded("consensus", note=f"raised: {e!r}"))
        print(f"  DEGRADED — {e!r}")

    # Stage 8 — adaptive stops.
    print("[8/13] ADAPTIVE STOPS")
    stop = None
    if h1_bars:
        try:
            from smc.hedgerock.evolution.adaptive_stops import (
                compute_stop_recommendation,
            )
            stop = compute_stop_recommendation(bars=h1_bars)
            stages.append(_ok(
                "adaptive_stop",
                vol_regime=stop.vol_regime.value,
                atr_multiplier=stop.atr_multiplier,
                position_scale=stop.position_scale,
            ))
            print(
                f"  vol_regime={stop.vol_regime.value} "
                f"atr_mult={stop.atr_multiplier} "
                f"position_scale={stop.position_scale}"
            )
        except Exception as e:
            stages.append(_degraded("adaptive_stop", note=f"raised: {e!r}"))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded("adaptive_stop", note="no bars"))

    # Stage 9 — stress test against a neutral demo proposal.
    print("[9/13] STRESS TEST (neutral demo proposal)")
    stress_total = 0
    stress_survived = 0
    stress_all_passed = False
    try:
        from smc.hedgerock.evolution.candidate_generator import (
            CandidateProposal as _CP,
            DECISION_RECOMMEND as _DR,
            get_live_parameter_snapshot as _live_snapshot,
        )
        from smc.hedgerock.evolution.stress_tester import (
            StressTester, VERDICT_BREACHED,
        )
        demo = _CP(
            candidate_id="xauusd-dry-run-probe",
            parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            parameter_class="confidence_threshold_observe",
            baseline_value=0.55,
            proposed_value=0.55,
            triggered_by=("dry_run",),
            expected_improvement="dry-run: neutral probe",
            risks=(),
            next_validation=(),
            decision=_DR,
            decision_reason="",
        )
        results = StressTester().test_candidate(demo, _live_snapshot())
        stress_total = len(results)
        stress_survived = sum(
            1 for r in results if r.verdict != VERDICT_BREACHED
        )
        stress_all_passed = stress_survived == stress_total
        stages.append(_ok(
            "stress_test",
            scenarios=stress_total, survived=stress_survived,
            all_passed=stress_all_passed,
        ))
        print(f"  scenarios={stress_total} survived={stress_survived}")
    except Exception as e:
        stages.append(_degraded("stress_test", note=f"raised: {e!r}"))
        print(f"  DEGRADED — {e!r}")

    # Build the canonical evidence payload for this run BEFORE the
    # recommend pipeline runs. wf.md / availability.md / atlas.md are
    # rendered FROM this payload — same bytes as evidence_hash, so the
    # fingerprint chain entry references provably-matching evidence.
    registry_json_count: int | None = None
    if registry_root.exists():
        registry_json_count = sum(1 for _ in registry_root.rglob("*.json"))
    evidence_payload = _evidence_payload(
        run_id=run_id,
        symbol=symbol,
        lookback_days=lookback_days,
        window_start=start, window_end=end,
        benchmark=benchmark,
        dynamic_replay=dynamic_replay,
        evidence_quality=evidence_quality,
        n_h1=len(h1_bars), n_h4=len(h4_bars), n_d1=len(d1_bars),
        registry_present=registry_root.exists(),
        registry_root=registry_root,
        registry_json_count=registry_json_count,
    )
    evidence_hash = _evidence_hash(evidence_payload)
    evidence_payload_path = output_dir / "fixture" / "evidence_payload.json"
    evidence_payload_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_payload_path.write_text(
        json.dumps(evidence_payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    # Stage 10 — recommend CLI (with calibrator + explainability + fingerprint).
    print("[10/13] RECOMMEND (calibrator + explainability + fingerprint)")
    atlas, avail, wf, bounds, audit_log = _seed_evidence(
        output_dir, payload=evidence_payload,
    )
    report_path = output_dir / "report" / "phase-d-evolution-report.md"
    rec_path = output_dir / "report" / "hedgerock-evolution-recommendation.md"
    fingerprint_path = output_dir / "fingerprint" / "chain.jsonl"
    calibrator_path = output_dir / "calibrator" / "state.json"
    # Reserve the dirs eagerly so operators can find the slot even if a
    # particular invocation does not persist anything to it.
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)

    n_proposals = 0
    n_recommend = 0
    try:
        _this_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(_this_dir))
        try:
            import hedgerock_evolution_recommend as recommend_cli  # type: ignore
        finally:
            sys.path.pop(0)
        rc_stdout = io.StringIO()
        with redirect_stdout(rc_stdout):
            rc = recommend_cli.main([
                "--atlas-report", str(atlas),
                "--data-availability-report", str(avail),
                "--walk-forward-report", str(wf),
                "--safety-bounds", str(bounds),
                "--registry-root", str(output_dir / "registry"),
                "--report-path", str(report_path),
                "--recommendation-path", str(rec_path),
                "--registry-audit-log", str(audit_log),
                "--explainability",
                "--fingerprint", str(fingerprint_path),
                "--calibrator-state", str(calibrator_path),
            ])
        snap = output_dir / "report" / "candidate_proposals.json"
        if rc == 0 and snap.exists():
            data = json.loads(snap.read_text(encoding="utf-8"))
            proposals = data.get("proposals", [])
            n_proposals = len(proposals)
            n_recommend = sum(
                1 for p in proposals if p.get("decision") == "RECOMMEND"
            )
            stages.append(_ok(
                "recommend",
                n_proposals=n_proposals, n_recommend=n_recommend,
            ))
            print(f"  proposals={n_proposals} recommend={n_recommend}")
        else:
            stages.append(_degraded(
                "recommend", note=f"rc={rc} or snapshot missing",
            ))
            print(f"  DEGRADED — rc={rc}")
    except Exception as e:
        stages.append(_degraded("recommend", note=f"raised: {e!r}"))
        print(f"  DEGRADED — {e!r}")

    # Stage 11 — fingerprint verify.
    print("[11/13] FINGERPRINT VERIFY")
    fingerprint_verify: dict[str, Any] | None = None
    if fingerprint_path.exists():
        try:
            from smc.hedgerock.evolution.fingerprint import verify_chain
            v = verify_chain(fingerprint_path)
            fingerprint_verify = {
                "ok": v.ok,
                "n_entries": v.n_entries,
                "first_break_index": v.first_break_index,
                "first_break_reason": v.first_break_reason,
            }
            if v.ok:
                stages.append(_ok(
                    "fingerprint_verify",
                    ok=True, n_entries=v.n_entries,
                ))
                print(f"  OK — entries={v.n_entries}")
            else:
                stages.append(_degraded(
                    "fingerprint_verify",
                    note=f"break: {v.first_break_reason}",
                ))
                print(f"  DEGRADED — {v.first_break_reason}")
        except Exception as e:
            stages.append(_degraded(
                "fingerprint_verify", note=f"raised: {e!r}",
            ))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded(
            "fingerprint_verify", note=f"chain not produced at {fingerprint_path}",
        ))
        print("  DEGRADED — chain not produced")

    # Stage 12 — registry audit summary.
    print("[12/13] REGISTRY AUDIT")
    registry_present = registry_root.exists()
    if registry_present:
        json_count = sum(1 for _ in registry_root.rglob("*.json"))
        stages.append(_ok(
            "registry_audit",
            registry_root=str(registry_root), json_count=json_count,
        ))
        print(f"  registry present at {registry_root} ({json_count} JSON)")
    else:
        stages.append(_degraded(
            "registry_audit",
            note=f"registry absent at {registry_root}",
        ))
        print(f"  DEGRADED — registry absent at {registry_root}")

    # Stage 13 — approval checklist.
    print("[13/13] APPROVAL CHECKLIST (DRY-RUN)")
    approval_rows = _approval_checklist(
        health_ok=health_ok,
        benchmark_ok=benchmark is not None,
        dynamic_replay=dynamic_replay,
        regime_ok=regime is not None,
        anomaly_state=anomaly,
        consensus=consensus,
        stress_all_passed=stress_all_passed,
        fingerprint_ok=bool(fingerprint_verify and fingerprint_verify.get("ok")),
        registry_present=registry_present,
    )
    stages.append(_ok("approval_checklist", n_items=len(approval_rows)))

    # Final report.
    body = _render_report(
        output_dir=output_dir,
        symbol=symbol,
        run_id=run_id,
        evidence_path=evidence_payload_path,
        evidence_hash=evidence_hash,
        stages=stages,
        benchmark=benchmark,
        dynamic_replay=dynamic_replay,
        evidence_quality=evidence_quality,
        regime=regime,
        anomaly=anomaly,
        consensus=consensus,
        stop=stop,
        stress_total=stress_total,
        stress_survived=stress_survived,
        n_proposals=n_proposals,
        n_recommend=n_recommend,
        fingerprint_verify=fingerprint_verify,
        approval_rows=approval_rows,
        registry_present=registry_present,
    )
    final_md = output_dir / "xauusd_dry_run_report.md"
    final_md.write_text(body, encoding="utf-8")
    snap_path = output_dir / "xauusd_dry_run_snapshot.json"
    snap_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "evidence_quality": evidence_quality,
                "replay_validation_status": _replay_validation_status(
                    dynamic_replay,
                ),
                "dynamic_replay_min_trades": MIN_DYNAMIC_REPLAY_TRADES,
                "dynamic_replay_sample_counts": {
                    "entry_count": dynamic_replay.entry_count,
                    "trade_count": dynamic_replay.trade_count,
                    "exit_count": dynamic_replay.exit_count,
                },
                "promotion_readiness": {
                    "status": _promotion_readiness(dynamic_replay)[0],
                    "reason": _promotion_readiness(dynamic_replay)[1],
                },
                "evidence_path": str(evidence_payload_path),
                "evidence_hash": evidence_hash,
                "evidence_artefacts": {
                    "atlas": str(atlas),
                    "availability": str(avail),
                    "wf": str(wf),
                    "audit_log": str(audit_log),
                },
                "stages": [
                    {
                        "name": s.name, "status": s.status,
                        "note": s.note, "details": s.details,
                    } for s in stages
                ],
                "benchmark_long_only": (
                    benchmark.to_dict() if benchmark else None
                ),
                "dynamic_replay": dynamic_replay.to_dict(),
                "stress_total": stress_total,
                "stress_survived": stress_survived,
                "n_proposals": n_proposals,
                "n_recommend": n_recommend,
                "fingerprint_verify": fingerprint_verify,
                "registry_present": registry_present,
                "approval_rows": [
                    {"item": i, "status": st} for i, st in approval_rows
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n== XAUUSD dry-run complete ==")
    print(f"  run_id:        {run_id}")
    print(f"  evidence:      {evidence_payload_path}")
    print(f"  evidence_hash: {evidence_hash}")
    print(f"  report:        {final_md}")
    print(f"  snapshot:      {snap_path}")
    print(f"  status: NOT LIVE / NOT APPROVED / NOT DEPLOYED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_out = (
        _ai_smc_home() / "tmp" / "xauusd_evolution_dry_run"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_out,
        help="Where to write the report + snapshot. "
             "Defaults to tmp/xauusd_evolution_dry_run/<UTC ISO>/.",
    )
    parser.add_argument(
        "--symbol", default=SYMBOL,
        help="MUST be XAUUSD; the orchestrator refuses everything else.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="How many days of bars to load (rolling window).",
    )
    parser.add_argument(
        "--lake-root", type=Path, default=None,
        help="Path to the parquet data lake root (defaults to "
             "$AI_SMC_HOME/data/parquet).",
    )
    parser.add_argument(
        "--registry-root", type=Path, default=None,
        help="HedgeRock policy_registry root (defaults to "
             "$HEDGEROCK_HOME/policy_registry).",
    )
    args = parser.parse_args(argv)

    try:
        return run(
            output_dir=Path(args.output_dir),
            symbol=args.symbol,
            lookback_days=args.lookback_days,
            lake_root=args.lake_root,
            registry_root=args.registry_root,
        )
    except ValueError as e:
        # XAUUSD-only assertion failure.
        print(f"FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
