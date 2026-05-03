"""Ticket 4 Step 2 — XAUUSD-only window coverage helper.

Pure dataclasses + pure functions. **Sidecar layer.** Reads
``policy_registry/config/gold_profile.yaml`` read-only (or any
caller-supplied path); never writes to disk.

The coverage check answers a single question:
    "Does the supplied set of XAUUSD historical windows + per-window
     replay stats clear every floor in RFC §1.2?"

It returns a :class:`CoverageReport` with ``coverage_pass`` boolean
+ a list of shortfall reasons. The PASS evaluator (Step 6) treats a
``False`` ``coverage_pass`` as ABSTAIN with the reasons propagated
verbatim.

Critically: this module **never** emits the legacy v1-era
``single_symbol shadow window`` blocker. XAUUSD-only is the
expected state of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


__all__ = [
    "CoverageReport",
    "CoverageThresholds",
    "REGIME_BUCKETS",
    "WindowSpec",
    "check_window_coverage",
    "load_gold_profile",
]


# Operator-/runner-known regime buckets the coverage gate counts.
REGIME_BUCKETS: tuple[str, ...] = (
    "range_low_vol",
    "range_high_vol",
    "trend_up",
    "trend_down",
    "breakout",
    "news_crisis",
    "weekend_gap",
)


@dataclass(frozen=True)
class CoverageThresholds:
    """Hard floors per RFC §1.2 of Ticket 4 v2. Lowering is a
    human-only config commit (same treatment as G2 / G5)."""

    min_windows: int = 6
    min_bar_count_per_window: int = 1000
    min_decided_bars_per_window: int = 500
    min_trade_count_per_window: int = 4
    min_regime_buckets_covered: int = 4
    min_halt_event_windows: int = 1
    max_closed_bar_gap_h1: int = 168


@dataclass(frozen=True)
class WindowSpec:
    """Operator-curated window definition. Cross-checked against
    mirror-derived bucket distribution at run time."""

    window_id: str
    start: datetime
    end: datetime
    declared_regime_bucket: str


@dataclass(frozen=True)
class CoverageReport:
    coverage_pass: bool
    windows_evaluated: tuple[str, ...]
    regime_buckets_covered: tuple[str, ...]
    halt_event_windows: int
    no_trade_windows: tuple[str, ...]
    shortfall_reasons: tuple[str, ...]
    declared_vs_observed_mismatches: tuple[str, ...] = ()


def _parse_per_window_stat(stat: dict[str, Any]) -> tuple[
    str, int, int, int, int, int, tuple[str, ...]
]:
    """Pull the seven fields the coverage gate consumes from a
    runner-supplied per-window stats dict."""
    return (
        str(stat["window_id"]),
        int(stat["n_bars"]),
        int(stat["n_decided_bars"]),
        int(stat["n_trades"]),
        int(stat["max_h1_gap_bars"]),
        int(stat["halt_event_count"]),
        tuple(stat.get("observed_buckets", ())),
    )


def check_window_coverage(
    *,
    specs: list[WindowSpec],
    per_window_stats: list[dict[str, Any]],
    candidate_affects_halt_mode: bool,
    thresholds: CoverageThresholds | None = None,
) -> CoverageReport:
    """Evaluate the XAUUSD multi-window coverage floors.

    Returns a frozen ``CoverageReport``. ``coverage_pass`` is True
    only when every floor in RFC §1.2 clears AND the candidate's
    halt-mode requirement is satisfied if applicable.

    Reason vocabulary is XAUUSD-only — never produces
    ``single_symbol`` / ``cross_symbol`` text.
    """
    t = thresholds or CoverageThresholds()
    reasons: list[str] = []
    no_trade: list[str] = []
    declared_mismatches: list[str] = []

    if len(specs) != len(per_window_stats):
        reasons.append(
            f"window_spec_stat_length_mismatch: {len(specs)} specs vs "
            f"{len(per_window_stats)} stats"
        )
    spec_by_id = {s.window_id: s for s in specs}

    if len(specs) < t.min_windows:
        reasons.append(
            f"insufficient_xauusd_window_coverage: have {len(specs)}, "
            f"need >= {t.min_windows}"
        )

    bucket_set: set[str] = set()
    halt_window_count = 0
    for stat in per_window_stats:
        try:
            (window_id, n_bars, n_decided, n_trades, max_gap, halt_n,
             observed_buckets) = _parse_per_window_stat(stat)
        except (KeyError, TypeError, ValueError) as e:
            reasons.append(f"per_window_stat_malformed: {e}")
            continue

        if n_bars < t.min_bar_count_per_window:
            reasons.append(
                f"window_too_thin: {window_id} has {n_bars} bars < "
                f"{t.min_bar_count_per_window}"
            )
            continue
        if n_decided < t.min_decided_bars_per_window:
            reasons.append(
                f"window_decision_thin: {window_id} has {n_decided} "
                f"decided bars < {t.min_decided_bars_per_window}"
            )
        if n_trades < t.min_trade_count_per_window:
            no_trade.append(window_id)
            # RFC v2 §1.2 — hard trade-coverage floor. Without this,
            # a candidate could clear coverage with most windows
            # holding fewer trades than the operator floor; the
            # PASS gate would then be averaging across thin
            # evidence. Surface as a shortfall so coverage_pass
            # flips False and the runner must ABSTAIN truthfully.
            reasons.append(
                f"insufficient_trade_coverage_in_window_{window_id}: "
                f"n_trades={n_trades} < {t.min_trade_count_per_window}"
            )
        if max_gap > t.max_closed_bar_gap_h1:
            reasons.append(
                f"data_gap_in_window_{window_id}: max_h1_gap={max_gap} "
                f"bars > {t.max_closed_bar_gap_h1}"
            )

        for b in observed_buckets:
            if b in REGIME_BUCKETS:
                bucket_set.add(b)
        if halt_n > 0:
            halt_window_count += 1

        # Cross-check declared vs observed.
        spec = spec_by_id.get(window_id)
        if spec is not None and spec.declared_regime_bucket not in observed_buckets:
            declared_mismatches.append(
                f"declared_bucket_mismatch_observed: {window_id} "
                f"declared={spec.declared_regime_bucket!r}, "
                f"observed={list(observed_buckets)}"
            )

    if len(bucket_set) < t.min_regime_buckets_covered:
        reasons.append(
            f"insufficient_regime_bucket_coverage: covered={sorted(bucket_set)}, "
            f"need >= {t.min_regime_buckets_covered}"
        )

    if candidate_affects_halt_mode and halt_window_count < t.min_halt_event_windows:
        reasons.append(
            f"halt_corpus_insufficient: {halt_window_count} halt-bearing "
            f"windows < {t.min_halt_event_windows} for halt-mode candidate"
        )

    coverage_pass = (len(reasons) == 0)
    return CoverageReport(
        coverage_pass=coverage_pass,
        windows_evaluated=tuple(s.window_id for s in specs),
        regime_buckets_covered=tuple(sorted(bucket_set)),
        halt_event_windows=halt_window_count,
        no_trade_windows=tuple(no_trade),
        shortfall_reasons=tuple(reasons),
        declared_vs_observed_mismatches=tuple(declared_mismatches),
    )


def load_gold_profile(path: Path) -> dict[str, Any]:
    """Read-only YAML loader for the operator-curated gold profile.

    Missing file → empty dict (no error). Malformed YAML → empty dict
    + the caller can log; runner falls back to no-profile behaviour.
    Never writes to disk.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw
