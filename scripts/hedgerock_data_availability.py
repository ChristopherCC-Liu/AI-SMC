"""Phase D-cont3-preflight — Data availability + replication CLI.

DIAGNOSTIC ONLY. Does NOT touch rule_engine, decision_server, .mq5
or any production trading code.

Two passes:

    1. Lake scan — list each instrument's H1 / H4 / D1 coverage
       (start, end, bar count, gaps, completeness ratio).
    2. Year-replication — for each instrument that passes the 3y
       bar AND each calendar year, run the existing atlas
       (``run_atlas``) and extract per-year stats for trend_up,
       range@>=0.80, breakout_signed_by_h4, plus halt-event count.
       Cells where the underlying bucket has < min_sample_for_signal
       samples are flagged INCONCLUSIVE.

Output: ``docs/phase-d-data-availability.md``
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from typing import Iterable

from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.hedgerock.data_availability import (
    AvailabilityReport,
    REPLICATION_MIN_YEARS,
    SymbolCoverage,
    scan_lake,
)
from smc.hedgerock.regime_opportunity_atlas import (
    AtlasConfig,
    AtlasReport,
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
_DEFAULT_REPORT = _hedgerock_home() / "docs" / "phase-d-data-availability.md"


# ---------------------------------------------------------------------------
# Replication cell — one (symbol, year) row in the per-year table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicationCell:
    """One (symbol, year) replication row.

    ``trend_up_*`` reads from ``aggregate_trend_opportunity``'s row
    where ``regime == 'trend_up'``. ``range_aggressive_*`` reads from
    the ``>= 0.80`` bucket of ``aggregate_range_opportunity``.
    ``breakout_signed_*`` reads from the ``breakout_signed_by_h4`` row.
    ``halt_event_count`` is ``len(report.halt_aftermath)``.
    """

    symbol: str
    year: int
    decision_bars: int
    trend_up_n: int
    trend_up_mean: float
    trend_up_ci_95: float
    range_aggressive_n: int
    range_aggressive_mean: float
    range_aggressive_ci_95: float
    breakout_signed_n: int
    breakout_signed_mean: float
    breakout_signed_ci_95: float
    halt_event_count: int
    min_sample_for_signal: int

    def _verdict(self, n: int, mean: float, ci: float) -> str:
        if n < self.min_sample_for_signal:
            return f"INCONCLUSIVE (n={n})"
        if mean > ci:
            return f"+{mean:.3f}% ±{ci:.3f}"
        if mean < -ci:
            return f"{mean:.3f}% ±{ci:.3f} (NEG)"
        return f"{mean:+.3f}% ±{ci:.3f} (CI∋0)"

    @property
    def trend_up_verdict(self) -> str:
        return self._verdict(self.trend_up_n, self.trend_up_mean, self.trend_up_ci_95)

    @property
    def range_aggressive_verdict(self) -> str:
        return self._verdict(
            self.range_aggressive_n, self.range_aggressive_mean,
            self.range_aggressive_ci_95,
        )

    @property
    def breakout_signed_verdict(self) -> str:
        return self._verdict(
            self.breakout_signed_n, self.breakout_signed_mean,
            self.breakout_signed_ci_95,
        )


def _ci_95(stdev: float, n: int) -> float:
    if n < 2:
        return 0.0
    return 1.96 * (stdev / sqrt(n))


def _find_stat(stats: list[RegimeBucketStat], *, regime: str | None = None,
               bucket: str | None = None) -> RegimeBucketStat | None:
    for s in stats:
        if regime is not None and s.regime != regime:
            continue
        if bucket is not None and s.bucket != bucket:
            continue
        return s
    return None


def build_replication_cell(
    *, symbol: str, year: int, atlas: AtlasReport, min_sample: int,
) -> ReplicationCell:
    """Extract a per-(symbol, year) row from a fully-built AtlasReport."""
    trend_up = _find_stat(atlas.trend_opportunity, regime="trend_up")
    range_aggr = _find_stat(atlas.range_opportunity, bucket=">=0.80")
    breakout_signed = _find_stat(
        atlas.trend_opportunity, regime="breakout_signed_by_h4",
    )
    return ReplicationCell(
        symbol=symbol, year=year,
        decision_bars=len(atlas.records),
        trend_up_n=trend_up.count if trend_up else 0,
        trend_up_mean=trend_up.mean_return_24h if trend_up else 0.0,
        trend_up_ci_95=_ci_95(
            trend_up.stdev_return_24h, trend_up.count,
        ) if trend_up else 0.0,
        range_aggressive_n=range_aggr.count if range_aggr else 0,
        range_aggressive_mean=range_aggr.mean_return_24h if range_aggr else 0.0,
        range_aggressive_ci_95=_ci_95(
            range_aggr.stdev_return_24h, range_aggr.count,
        ) if range_aggr else 0.0,
        breakout_signed_n=breakout_signed.count if breakout_signed else 0,
        breakout_signed_mean=breakout_signed.mean_return_24h if breakout_signed else 0.0,
        breakout_signed_ci_95=_ci_95(
            breakout_signed.stdev_return_24h, breakout_signed.count,
        ) if breakout_signed else 0.0,
        halt_event_count=len(atlas.halt_aftermath),
        min_sample_for_signal=min_sample,
    )


def replication_e1_verdict(cells: list[ReplicationCell]) -> tuple[str, list[str]]:
    """Synthesise an E1 / promote verdict from the per-year table.

    The original Phase D-cont2 finding was that trend_up has a
    *single-window* edge. The hardness of E1 promotion is replication:
    we want the trend_up edge to repeat across multiple INDEPENDENT
    windows (and ideally symbols).

    Returns (headline, supporting bullets).
    """
    if not cells:
        return "INSUFFICIENT DATA — no replication cells.", []
    by_symbol: dict[str, list[ReplicationCell]] = {}
    for c in cells:
        by_symbol.setdefault(c.symbol, []).append(c)

    n_symbols = len(by_symbol)
    bullets: list[str] = []

    # For each symbol, count years where trend_up CI clears zero.
    trend_up_pass_per_symbol: dict[str, int] = {}
    trend_up_cells_per_symbol: dict[str, int] = {}
    for sym, sym_cells in by_symbol.items():
        passing = sum(
            1 for c in sym_cells
            if c.trend_up_n >= c.min_sample_for_signal
            and c.trend_up_mean > c.trend_up_ci_95
        )
        trend_up_pass_per_symbol[sym] = passing
        trend_up_cells_per_symbol[sym] = len(sym_cells)

    bullets.append(
        f"trend_up window-evidence count: "
        + ", ".join(
            f"{sym} {trend_up_pass_per_symbol[sym]}/{trend_up_cells_per_symbol[sym]} years"
            for sym in by_symbol
        )
    )

    if n_symbols < 2:
        bullets.append(
            f"Only {n_symbols} symbol present in lake — cross-symbol "
            f"robustness CANNOT be tested. Year-replication is the only "
            f"available robustness signal."
        )

    # Headline verdict logic:
    only_symbol = next(iter(by_symbol)) if n_symbols == 1 else None
    if only_symbol:
        passing = trend_up_pass_per_symbol[only_symbol]
        total = trend_up_cells_per_symbol[only_symbol]
        if passing >= 3 and passing >= total // 2 + 1:
            headline = (
                f"E1 trend_up shows year-replication on {only_symbol} "
                f"({passing}/{total} years pass), but **NO second-symbol "
                f"evidence**. Promotion remains GATED on adding a "
                f"second symbol."
            )
        elif passing >= 2:
            headline = (
                f"E1 trend_up year-replication on {only_symbol} is "
                f"PARTIAL ({passing}/{total} years pass) — short of "
                f"a clean replication. Do not promote yet; investigate "
                f"why some years fail."
            )
        else:
            headline = (
                f"E1 trend_up does NOT replicate across years on "
                f"{only_symbol} ({passing}/{total} years pass). The "
                f"2024 single-window edge was likely noise."
            )
    else:
        # Multiple symbols. (Aspirational — current lake has 1.)
        headline = (
            "Multi-symbol replication possible — see per-symbol bullets."
        )

    return headline, bullets


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _fmt_lake_scan(report: AvailabilityReport) -> list[str]:
    out = ["## Lake scan", ""]
    out.append(f"- Lake root: `{report.lake_root}`")
    out.append(f"- Instruments found: {len(report.instruments)}")
    out.append(f"- Replication candidates (≥{REPLICATION_MIN_YEARS}y "
               f"H1+H4+D1 with completeness ≥ 0.50): "
               f"{', '.join(report.replication_candidates()) or '(none)'}")
    out.append("")
    out.append("| Instrument | Timeframe | Start | End | Bars | Span (days) | "
               "Intra-week gaps | Largest gap (h) | Completeness | ≥3y? |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in report.coverages:
        start = c.start_ts.date().isoformat() if c.start_ts else "—"
        end = c.end_ts.date().isoformat() if c.end_ts else "—"
        out.append(
            f"| {c.instrument} | {c.timeframe} | {start} | {end} | "
            f"{c.bar_count} | {c.span_days} | "
            f"{c.intraweek_gap_count} | {c.largest_intraweek_gap_hours:.1f} | "
            f"{c.completeness_ratio:.2f} | "
            f"{'✅' if c.sufficient_for_3y_replication else '❌'} |"
        )
    out.append("")
    return out


def _fmt_replication(
    cells: list[ReplicationCell], min_sample: int,
) -> list[str]:
    out = ["## Year-replication summary", ""]
    if not cells:
        out.append("- (no replication runs — no instrument cleared the "
                   "≥3y / completeness bar)")
        out.append("")
        return out
    out.append(
        f"For each (symbol, year) combination, the existing atlas was "
        f"re-run with `--instrument <symbol> --start <year>-01-01 "
        f"--end <year+1>-01-01`. The cells below extract the same "
        f"buckets the atlas already aggregates. **INCONCLUSIVE** when "
        f"the underlying bucket sample is below the n={min_sample} "
        f"floor; **(NEG)** when CI excludes zero on the negative side; "
        f"**(CI∋0)** when CI brackets zero (no edge either way)."
    )
    out.append("")
    out.append("| Symbol | Year | Bars | trend_up | range@≥0.80 | "
               "breakout_signed_by_h4 | halt events |")
    out.append("|---|---|---|---|---|---|---|")
    for c in cells:
        out.append(
            f"| {c.symbol} | {c.year} | {c.decision_bars} | "
            f"{c.trend_up_verdict} | "
            f"{c.range_aggressive_verdict} | "
            f"{c.breakout_signed_verdict} | "
            f"{c.halt_event_count} |"
        )
    out.append("")
    return out


def _fmt_e1_verdict(cells: list[ReplicationCell]) -> list[str]:
    out = ["## E1 trend_up promotion verdict", ""]
    headline, bullets = replication_e1_verdict(cells)
    out.append(f"**{headline}**")
    out.append("")
    for b in bullets:
        out.append(f"- {b}")
    out.append("")
    return out


def _fmt_action_gate(
    *, availability: AvailabilityReport, cells: list[ReplicationCell],
) -> list[str]:
    """Phase D-cont3-preflight closeout — explicit action gate.

    The "PARTIAL" wording in the E1 verdict can be misread by a future
    session as "close to promote". The Action gate makes the
    no-strategy-change rule machine-greppable and human-unambiguous.

    Returns a markdown block with three required keys:

      - ``NO_STRATEGY_CHANGE``: hard boolean. True until the gate
        clears.
      - ``Reason``: enumerated failure modes (single-symbol /
        replication-fail / negative-sign year / halt-event-count).
      - ``Allowed next work``: enumerated whitelist. Anything outside
        this list is OUT OF SCOPE.
    """
    n_symbols = len(availability.instruments)
    n_cells = len(cells)
    by_symbol: dict[str, list[ReplicationCell]] = {}
    for c in cells:
        by_symbol.setdefault(c.symbol, []).append(c)

    # Enumerate failure modes that gate strategy work.
    reasons: list[str] = []
    if n_symbols < 2:
        reasons.append(
            f"**single-symbol lake** ({n_symbols} symbol present): "
            "cross-symbol robustness untestable"
        )
    # Year-replication for trend_up.
    for sym, sym_cells in by_symbol.items():
        passing = sum(
            1 for c in sym_cells
            if c.trend_up_n >= c.min_sample_for_signal
            and c.trend_up_mean > c.trend_up_ci_95
        )
        total = len(sym_cells)
        if total > 0 and passing < total:
            reasons.append(
                f"**trend_up year-replication FAILS on {sym}**: "
                f"{passing}/{total} years pass"
            )
        # Reverse-signed years: trend_up CI excludes zero on the
        # negative side. Even one such year is a flat veto.
        neg_years = [
            c.year for c in sym_cells
            if c.trend_up_n >= c.min_sample_for_signal
            and c.trend_up_mean < -c.trend_up_ci_95
        ]
        if neg_years:
            reasons.append(
                f"**trend_up REVERSE-signed on {sym} in "
                f"{', '.join(str(y) for y in neg_years)}** "
                "(classifier was systematically wrong, not just no edge)"
            )
    # Halt-event sample.
    total_halt = sum(c.halt_event_count for c in cells)
    floor = cells[0].min_sample_for_signal if cells else 30
    if total_halt < floor:
        reasons.append(
            f"**halt-event sample n={total_halt} < {floor}**: "
            "smarter-halt-release design would be curve-fit"
        )

    no_strategy_change = bool(reasons)

    out = ["## Action gate", ""]
    out.append("Machine-readable gate — a follow-on session should "
               "consult these keys before starting strategy work. "
               "**`NO_STRATEGY_CHANGE: true` overrides any softer "
               "wording in the verdict above.** The gate clears only "
               "when every reason below is resolved by additional "
               "DATA (a second symbol, more halt events, etc.) — not "
               "by re-running with different sample-size floors.")
    out.append("")
    out.append("```yaml")
    out.append(f"NO_STRATEGY_CHANGE: {str(no_strategy_change).lower()}")
    if reasons:
        out.append("Reason:")
        for r in reasons:
            # Strip markdown bold for YAML-friendliness; keep prose.
            out.append(f"  - {r.replace('**', '')}")
    else:
        out.append("Reason: []")
    out.append("Allowed_next_work:")
    if no_strategy_change:
        out.append("  - data_acquisition_second_symbol  # XAGUSD / EURUSD / etc")
        out.append(
            "  - classifier_diagnostic_negative_sign_year  "
            "# explain why trend_up flips in reverse-signed years"
        )
        out.append(
            "  - halt_event_corpus_expansion  "
            "# more years / more symbols until n >= 30 halt events"
        )
        out.append("Disallowed_next_work:")
        out.append("  - momentum_module_E1")
        out.append("  - observe_floor_retune")
        out.append("  - smarter_halt_release_trigger")
        out.append("  - any_rule_engine_or_decision_server_change")
    else:
        out.append("  - any (gate cleared)")
    out.append("```")
    out.append("")
    return out


def _write_report(
    availability: AvailabilityReport,
    cells: list[ReplicationCell],
    *, min_sample: int, report_path: Path,
) -> None:
    body: list[str] = [
        "# Phase D-cont3-preflight — Data availability + year-replication",
        "",
        "> **Read-only diagnostic.** This report does NOT modify "
        "production rule_engine, decision_server, or .mq5. It scans "
        "the data lake for symbol / timeframe coverage and re-runs "
        "the existing Phase D-cont2 atlas year-by-year on every "
        "symbol that meets the ≥3y bar. Output answers: \"do we have "
        "enough data to push E1 trend_up promotion forward, or do we "
        "need to source additional data first?\".",
        "",
    ]
    body.extend(_fmt_lake_scan(availability))
    body.extend(_fmt_replication(cells, min_sample))
    body.extend(_fmt_e1_verdict(cells))
    body.extend(_fmt_action_gate(availability=availability, cells=cells))

    body.append("## Caveats")
    body.append("")
    body.append(
        "- **Year-replication is one form of robustness — not all of it.** "
        "Cross-symbol replication (XAGUSD / EURUSD / etc.) is a stronger "
        "signal than year-replication on the same instrument because "
        "regime structure can be symbol-specific."
    )
    body.append(
        "- **Completeness ratio is a 24/5 forex approximation.** "
        "Holidays and broker-specific outages reduce theoretical bar "
        "counts; a completeness < 1.0 is normal. Completeness < 0.50 "
        "is the threshold below which we refuse to use a (symbol, "
        "timeframe) for replication."
    )
    body.append(
        "- **Atlas year boundaries lose ~10 days of warmup** (h1_lookback "
        "= 240 H1 bars). Per-year decision_bars counts already exclude "
        "this warmup region."
    )
    body.append(
        "- **No code changes from Phase D-cont3 yet.** This is a "
        "preflight to decide WHETHER to start D-cont3 work. The next "
        "step depends on the headline verdict above."
    )
    body.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body))
    print(f"wrote report → {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _years_in_coverage(c: SymbolCoverage) -> list[int]:
    if c.start_ts is None or c.end_ts is None:
        return []
    return list(range(c.start_ts.year, c.end_ts.year + 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-lake-root", type=Path, default=_DEFAULT_LAKE)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument(
        "--min-sample-for-signal", type=int, default=30,
        help="Per-bucket sample floor. Cells below this are INCONCLUSIVE.",
    )
    parser.add_argument(
        "--skip-replication", action="store_true",
        help="Scan only — do NOT run the per-year atlas pass.",
    )
    args = parser.parse_args(argv)

    if not args.data_lake_root.exists():
        print(f"data lake root not found: {args.data_lake_root}",
              file=sys.stderr)
        return 2

    lake = ForexDataLake(args.data_lake_root)
    print(f"scanning lake at {args.data_lake_root} ...")
    availability = scan_lake(lake)
    print(f"  instruments: {len(availability.instruments)}")
    for inst in availability.instruments:
        for c in availability.by_instrument(inst):
            sufficiency = "OK" if c.sufficient_for_3y_replication else "thin"
            print(
                f"    {inst} / {c.timeframe}: "
                f"{c.bar_count} bars, span {c.span_days}d, "
                f"completeness {c.completeness_ratio:.2f} → {sufficiency}"
            )

    cells: list[ReplicationCell] = []
    candidates = availability.replication_candidates()
    if args.skip_replication or not candidates:
        if not candidates:
            print("  no replication candidates — skipping per-year atlas pass.")
    else:
        # Use the H1 coverage to decide which years to cover per symbol.
        # Year is "covered" if both start_ts.year and end_ts.year span it.
        for symbol in candidates:
            tf_h1 = next(
                c for c in availability.by_instrument(symbol)
                if c.timeframe == Timeframe.H1
            )
            for year in _years_in_coverage(tf_h1):
                start = datetime(year, 1, 1, tzinfo=timezone.utc)
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                # Skip year if it's not fully inside the coverage —
                # partial years confuse n / CI.
                if tf_h1.start_ts and start < tf_h1.start_ts:
                    continue
                if tf_h1.end_ts and end > tf_h1.end_ts + timedelta(days=1):
                    continue
                print(f"  replication: {symbol} {year} ...")
                atlas_cfg = AtlasConfig(
                    instrument=symbol, start=start, end=end,
                    min_sample_for_signal=args.min_sample_for_signal,
                )
                atlas = run_atlas(atlas_cfg, lake)
                cell = build_replication_cell(
                    symbol=symbol, year=year, atlas=atlas,
                    min_sample=args.min_sample_for_signal,
                )
                cells.append(cell)
                print(
                    f"    bars={cell.decision_bars} "
                    f"trend_up={cell.trend_up_verdict} "
                    f"range>=0.80={cell.range_aggressive_verdict} "
                    f"halt={cell.halt_event_count}"
                )

    _write_report(
        availability, cells,
        min_sample=args.min_sample_for_signal,
        report_path=args.report_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
