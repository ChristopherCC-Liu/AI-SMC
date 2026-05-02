"""Stage 6-followup-2 task 1 — replay-based candidate validator.

Two complementary entry points:

  * :func:`summarise_replay` — legacy path. Aggregates per-window
    deltas already recorded under
    ``policy_registry/shadow_artefacts/<candidate_id>/``.
  * :func:`summarise_replay_with_backtest` — wired path. Calls the
    public :func:`phase_d_walk_forward.run_walk_forward_backtest`
    interface against a supplied historical corpus, computing
    per-window deltas vs the live baseline read from
    :func:`decision_server.get_live_parameters`.

What this module still does NOT do:
    - Modify any artefact, registry entry, or live config.
    - Touch ``policy_registry/approved/`` or ``pointer.json``.
    - Import ``rule_engine`` (still red-line). The Tier-1 unseal
      whitelist is exactly ``phase_d_walk_forward`` +
      ``decision_server`` and is exercised here read-only.

Output is a frozen :class:`ReplayValidationReport`. The text
renderer carries explicit
``heuristic_projection_only / not a simulation`` banners so an
operator never reads the projection as a live forecast.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# Tier-1 read-only imports authorised by RFC §1 (post-unseal):
# walk_forward provides the public backtest function;
# decision_server provides the live-parameter snapshot.
from smc.hedgerock import decision_server, phase_d_walk_forward


__all__ = [
    "ReplayValidationReport",
    "summarise_replay",
    "summarise_replay_with_backtest",
    "render_replay_report",
]


@dataclass(frozen=True)
class ReplayValidationReport:
    candidate_id: str
    n_artefacts_read: int
    n_windows_replayed: int
    delta_pnl_pp_mean: float
    delta_pnl_pp_p25: float
    delta_pnl_pp_p75: float
    delta_dd_pp_worst: float  # most adverse positive (DD increase) value
    windows_passing: int  # delta_pnl_pp > 0
    windows_regressing: int  # delta_pnl_pp < 0
    skipped_artefact_ids: tuple[str, ...]
    observed_buckets: tuple[str, ...]
    blocking_conditions: tuple[str, ...]
    report_only: bool = True
    heuristic_projection_only: bool = True
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _load_artefact_windows(
    *, path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return the per-window list (or empty) plus a skip-reason if
    the artefact is unusable. Skipped artefacts return ``([], reason)``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], "malformed_json"
    artefact = raw.get("artefact", raw)
    pw = artefact.get("per_window")
    if not isinstance(pw, dict):
        return [], "missing_per_window"
    windows = pw.get("windows")
    if not isinstance(windows, list):
        return [], "per_window_windows_not_list"
    return windows, None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    # Standard linear interpolation between closest ranks.
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def summarise_replay(
    *,
    candidate_id: str,
    shadow_artefacts_root: Path,
) -> ReplayValidationReport:
    """Aggregate per-window deltas across all artefacts on disk for
    ``candidate_id``. Read-only.

    Raises :class:`FileNotFoundError` if ``shadow_artefacts_root``
    does not exist; returns a zero-count report when the root exists
    but the candidate's subdir is empty.
    """
    root = Path(shadow_artefacts_root)
    if not root.exists():
        raise FileNotFoundError(f"shadow_artefacts_root missing: {root}")

    cand_dir = root / candidate_id
    if not cand_dir.exists():
        return ReplayValidationReport(
            candidate_id=candidate_id,
            n_artefacts_read=0, n_windows_replayed=0,
            delta_pnl_pp_mean=0.0, delta_pnl_pp_p25=0.0,
            delta_pnl_pp_p75=0.0, delta_dd_pp_worst=0.0,
            windows_passing=0, windows_regressing=0,
            skipped_artefact_ids=(),
            observed_buckets=(),
            blocking_conditions=("no_artefacts_found",),
        )

    artefact_paths = sorted(cand_dir.glob("*.json"))
    if not artefact_paths:
        return ReplayValidationReport(
            candidate_id=candidate_id,
            n_artefacts_read=0, n_windows_replayed=0,
            delta_pnl_pp_mean=0.0, delta_pnl_pp_p25=0.0,
            delta_pnl_pp_p75=0.0, delta_dd_pp_worst=0.0,
            windows_passing=0, windows_regressing=0,
            skipped_artefact_ids=(),
            observed_buckets=(),
            blocking_conditions=("no_artefacts_found",),
        )

    deltas_pnl: list[float] = []
    deltas_dd: list[float] = []
    bucket_set: set[str] = set()
    n_artefacts_read = 0
    skipped: list[str] = []

    for p in artefact_paths:
        windows, skip = _load_artefact_windows(path=p)
        if skip is not None:
            skipped.append(f"{p.stem}:{skip}")
            continue
        n_artefacts_read += 1
        for w in windows:
            try:
                deltas_pnl.append(float(w.get("delta_pnl_pp", 0.0)))
                deltas_dd.append(float(w.get("delta_dd_pp", 0.0)))
                for b in w.get("observed_buckets", ()):
                    bucket_set.add(str(b))
            except (TypeError, ValueError):
                continue

    n_windows = len(deltas_pnl)
    blockers: list[str] = []
    if n_artefacts_read == 0:
        blockers.append("no_artefacts_found")
    if n_windows < 8:
        blockers.append(f"insufficient_window_coverage:n={n_windows}<8")

    mean_pnl = statistics.fmean(deltas_pnl) if deltas_pnl else 0.0
    p25 = _percentile(deltas_pnl, 0.25) if deltas_pnl else 0.0
    p75 = _percentile(deltas_pnl, 0.75) if deltas_pnl else 0.0
    # "worst" DD delta = most adverse positive number (DD growing).
    worst_dd = max(deltas_dd) if deltas_dd else 0.0
    passing = sum(1 for v in deltas_pnl if v > 0)
    regressing = sum(1 for v in deltas_pnl if v < 0)

    return ReplayValidationReport(
        candidate_id=candidate_id,
        n_artefacts_read=n_artefacts_read,
        n_windows_replayed=n_windows,
        delta_pnl_pp_mean=mean_pnl,
        delta_pnl_pp_p25=p25,
        delta_pnl_pp_p75=p75,
        delta_dd_pp_worst=worst_dd,
        windows_passing=passing,
        windows_regressing=regressing,
        skipped_artefact_ids=tuple(skipped),
        observed_buckets=tuple(sorted(bucket_set)),
        blocking_conditions=tuple(blockers),
    )


# ---------------------------------------------------------------------------
# Backtest-wired path — uses phase_d_walk_forward + decision_server
# ---------------------------------------------------------------------------


def summarise_replay_with_backtest(
    *,
    candidate_id: str,
    parameter_class: str,
    proposed_value: float,
    history: Iterable[Mapping[str, Any]],
) -> ReplayValidationReport:
    """Run the public walk-forward backtest against ``history`` under
    the proposed parameter, then again under the live baseline read
    from :func:`decision_server.get_live_parameters`. Aggregate the
    per-window deltas into a :class:`ReplayValidationReport`.

    This call is read-only end-to-end: ``phase_d_walk_forward``'s
    public backtest is a pure function, and ``decision_server`` only
    exposes a getter. No artefacts, no live config, no EA file is
    touched.
    """
    history_list = list(history)
    live = decision_server.get_live_parameters()

    if parameter_class not in live:
        return ReplayValidationReport(
            candidate_id=candidate_id,
            n_artefacts_read=0, n_windows_replayed=0,
            delta_pnl_pp_mean=0.0, delta_pnl_pp_p25=0.0,
            delta_pnl_pp_p75=0.0, delta_dd_pp_worst=0.0,
            windows_passing=0, windows_regressing=0,
            skipped_artefact_ids=(),
            observed_buckets=(),
            blocking_conditions=(
                f"unsupported_parameter_class:{parameter_class}",
            ),
        )

    proposed_params = dict(live)
    proposed_params[parameter_class] = float(proposed_value)

    proposed_run = phase_d_walk_forward.run_walk_forward_backtest(
        parameters=proposed_params, history=history_list,
    )
    baseline_run = phase_d_walk_forward.run_walk_forward_backtest(
        parameters=live, history=history_list,
    )

    deltas_pnl: list[float] = []
    deltas_dd: list[float] = []
    bucket_set: set[str] = set()
    baseline_by_id = {w.window_id: w for w in baseline_run.windows}
    for w in proposed_run.windows:
        b = baseline_by_id.get(w.window_id)
        if b is None:
            continue
        deltas_pnl.append(round(w.pnl_pp - b.pnl_pp, 6))
        deltas_dd.append(round(w.dd_pp - b.dd_pp, 6))
        if w.regime_bucket:
            bucket_set.add(w.regime_bucket)

    n_windows = len(deltas_pnl)
    blockers: list[str] = []
    if n_windows < 8:
        blockers.append(f"insufficient_window_coverage:n={n_windows}<8")

    mean_pnl = statistics.fmean(deltas_pnl) if deltas_pnl else 0.0
    p25 = _percentile(deltas_pnl, 0.25) if deltas_pnl else 0.0
    p75 = _percentile(deltas_pnl, 0.75) if deltas_pnl else 0.0
    worst_dd = max(deltas_dd) if deltas_dd else 0.0
    passing = sum(1 for v in deltas_pnl if v > 0)
    regressing = sum(1 for v in deltas_pnl if v < 0)

    return ReplayValidationReport(
        candidate_id=candidate_id,
        n_artefacts_read=0,
        n_windows_replayed=n_windows,
        delta_pnl_pp_mean=mean_pnl,
        delta_pnl_pp_p25=p25,
        delta_pnl_pp_p75=p75,
        delta_dd_pp_worst=worst_dd,
        windows_passing=passing,
        windows_regressing=regressing,
        skipped_artefact_ids=(),
        observed_buckets=tuple(sorted(bucket_set)),
        blocking_conditions=tuple(blockers),
    )


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_replay_report(report: ReplayValidationReport) -> str:
    out: list[str] = []
    out.append("# HedgeRock Replay Validation (READ-ONLY HEURISTIC PROJECTION)")
    out.append("")
    out.append("> This is **not a simulation**. It aggregates per-window "
               "deltas already recorded by the shadow runner. Numbers are "
               "a `heuristic_projection_only` of past evidence, not a "
               "forward forecast.")
    out.append("")
    out.append("- status: **NOT LIVE**")
    out.append("- approval: **NOT APPROVED**")
    out.append("- deployment: **NOT DEPLOYED**")
    out.append(f"- mode: `heuristic_projection_only`")
    out.append("")
    out.append("## Subject")
    out.append("")
    out.append(f"- candidate_id: `{report.candidate_id}`")
    out.append(f"- artefacts read: **{report.n_artefacts_read}**")
    out.append(f"- windows replayed: **{report.n_windows_replayed}**")
    out.append("")
    out.append("## Aggregate deltas (PnL pp, DD pp)")
    out.append("")
    out.append(f"- delta_pnl_pp_mean: **{report.delta_pnl_pp_mean:+.3f}**")
    out.append(f"- delta_pnl_pp_p25: **{report.delta_pnl_pp_p25:+.3f}**")
    out.append(f"- delta_pnl_pp_p75: **{report.delta_pnl_pp_p75:+.3f}**")
    out.append(f"- delta_dd_pp_worst: **{report.delta_dd_pp_worst:+.3f}** "
               "(most adverse DD increase)")
    out.append(f"- windows_passing (delta_pnl_pp > 0): "
               f"**{report.windows_passing}**")
    out.append(f"- windows_regressing (delta_pnl_pp < 0): "
               f"**{report.windows_regressing}**")
    out.append("")
    out.append("## Coverage")
    out.append("")
    if report.observed_buckets:
        out.append("- observed regime buckets:")
        for b in report.observed_buckets:
            out.append(f"    - `{b}`")
    else:
        out.append("- observed regime buckets: (none recorded)")
    out.append("")
    out.append("## Skipped artefacts")
    out.append("")
    if report.skipped_artefact_ids:
        for s in report.skipped_artefact_ids:
            out.append(f"- `{s}`")
    else:
        out.append("- (none)")
    out.append("")
    out.append("## Blocking conditions")
    out.append("")
    if report.blocking_conditions:
        for b in report.blocking_conditions:
            out.append(f"- `{b}`")
    else:
        out.append("- (none — artefact corpus is large enough to project)")
    out.append("")
    out.append("## Boundary boilerplate")
    out.append("")
    out.append("- this report is a read-only aggregation of existing "
               "`policy_registry/shadow_artefacts/` JSON files")
    out.append("- this report does not execute the trading runtime")
    out.append("- this report does not promote, approve, or deploy "
               "anything")
    out.append("")
    out.append(f"Generated at: {report.generated_at}")
    out.append("")
    return "\n".join(out)
