"""
Worst Indicator 3D Scan: regime x trigger x direction
=====================================================
Scans ALL gate results (v5, v6, v7, v8_v1, v8_v2) to find:
1. INVERSION CANDIDATES: (regime, trigger, direction) combos with WR < 30%
2. PROVEN GOOD: combos with WR > 60% and N >= 5
3. CYCLE MAP: how dominant regimes and best inversions shift across market cycles

Output: data/worst_indicator_matrix.json
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Market cycle definitions based on XAUUSD macro context
CYCLE_MAP = {
    "2020_covid": {"start": "2020-01-01", "end": "2020-09-30", "label": "COVID crash + recovery"},
    "2020_post_covid": {"start": "2020-10-01", "end": "2021-06-30", "label": "post-COVID range"},
    "2021_taper": {"start": "2021-07-01", "end": "2021-12-31", "label": "taper tantrum / range"},
    "2022_rate_hike": {"start": "2022-01-01", "end": "2022-12-31", "label": "rate hike regime"},
    "2023_plateau": {"start": "2023-01-01", "end": "2023-12-31", "label": "rate hike plateau"},
    "2024_ath": {"start": "2024-01-01", "end": "2024-12-31", "label": "ATH run"},
}


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, handling Infinity values."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle JSON Infinity which is not valid JSON
            line = line.replace(": Infinity", ": 1e9").replace(":-Infinity", ": -1e9")
            records.append(json.loads(line))
    return records


def get_regime_for_window(window: dict) -> str:
    """Extract regime label from a window record, normalizing across versions."""
    # v7 has atr_regime_at_start
    if "atr_regime_at_start" in window:
        return window["atr_regime_at_start"]
    # v5/v6 have regime_at_start
    if "regime_at_start" in window:
        return window["regime_at_start"]
    # v8 has no regime — label as "unknown"
    return "unknown"


def get_cycle_for_trade(open_ts: str) -> str:
    """Map a trade's open timestamp to a market cycle."""
    dt = datetime.fromisoformat(open_ts.replace("+00:00", ""))
    date_str = dt.strftime("%Y-%m-%d")
    for cycle_id, cycle_info in CYCLE_MAP.items():
        if cycle_info["start"] <= date_str <= cycle_info["end"]:
            return cycle_id
    return "other"


def extract_trades(all_windows: list[dict]) -> list[dict]:
    """Flatten all windows into individual trade records with metadata."""
    trades = []
    for w in all_windows:
        version = w.get("version", "unknown")
        regime = get_regime_for_window(w)
        window_id = w.get("window", 0)
        test_start = w.get("test_start", "")
        test_end = w.get("test_end", "")

        for t in w.get("trade_details", []):
            trade = {
                "version": version,
                "window": window_id,
                "regime": regime,
                "direction": t["direction"],
                "trigger_type": t["trigger_type"],
                "pnl_usd": t["pnl_usd"],
                "is_win": t["pnl_usd"] > 0,
                "close_reason": t["close_reason"],
                "open_ts": t["open_ts"],
                "close_ts": t["close_ts"],
                "test_start": test_start,
                "test_end": test_end,
                "cycle": get_cycle_for_trade(t["open_ts"]),
            }
            trades.append(trade)
    return trades


def compute_group_stats(trades: list[dict]) -> dict:
    """Compute WR, PF, avg_pnl, consistency for a group of trades."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0, "pf": 0, "avg_pnl": 0, "total_pnl": 0}

    wins = [t for t in trades if t["is_win"]]
    losses = [t for t in trades if not t["is_win"]]
    wr = len(wins) / n
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (1e9 if gross_profit > 0 else 0)
    avg_pnl = sum(t["pnl_usd"] for t in trades) / n
    total_pnl = sum(t["pnl_usd"] for t in trades)

    # Consistency: % of unique windows where WR < 50%
    window_groups = defaultdict(list)
    for t in trades:
        key = (t["version"], t["window"])
        window_groups[key].append(t)
    bad_windows = sum(
        1 for group in window_groups.values()
        if sum(1 for t in group if t["is_win"]) / len(group) < 0.5
    )
    consistency_bad = bad_windows / len(window_groups) if window_groups else 0

    return {
        "n": n,
        "wr": round(wr, 4),
        "pf": round(min(pf, 999), 4),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "n_windows": len(window_groups),
        "bad_window_pct": round(consistency_bad, 4),
        "versions": sorted(set(t["version"] for t in trades)),
    }


def run_3d_scan(trades: list[dict]) -> dict:
    """Group by (regime, trigger, direction) and compute stats."""
    groups = defaultdict(list)
    for t in trades:
        key = (t["regime"], t["trigger_type"], t["direction"])
        groups[key].append(t)

    results = {}
    for (regime, trigger, direction), group_trades in groups.items():
        stats = compute_group_stats(group_trades)
        results[(regime, trigger, direction)] = stats
    return results


def run_2d_scan(trades: list[dict], dim1: str, dim2: str) -> dict:
    """Group by any 2 dimensions and compute stats."""
    groups = defaultdict(list)
    for t in trades:
        key = (t[dim1], t[dim2])
        groups[key].append(t)

    results = {}
    for key, group_trades in groups.items():
        results[key] = compute_group_stats(group_trades)
    return results


def run_1d_scan(trades: list[dict], dim: str) -> dict:
    """Group by 1 dimension and compute stats."""
    groups = defaultdict(list)
    for t in trades:
        groups[t[dim]].append(t)

    results = {}
    for key, group_trades in groups.items():
        results[key] = compute_group_stats(group_trades)
    return results


def classify_combos(scan_3d: dict) -> tuple[list, list, list]:
    """Classify into inversion candidates, proven good, and neutral."""
    inversion_candidates = []
    proven_good = []
    neutral = []

    for (regime, trigger, direction), stats in scan_3d.items():
        entry = {
            "regime": regime,
            "trigger": trigger,
            "direction": direction,
            **stats,
        }

        if stats["n"] >= 3 and stats["wr"] < 0.30:
            entry["inverted_wr"] = round(1.0 - stats["wr"], 4)
            entry["inverted_avg_pnl"] = round(-stats["avg_pnl"], 2)
            inversion_candidates.append(entry)
        elif stats["n"] >= 5 and stats["wr"] > 0.55:
            proven_good.append(entry)
        else:
            neutral.append(entry)

    # Sort: worst first for inversions, best first for proven
    inversion_candidates.sort(key=lambda x: x["wr"])
    proven_good.sort(key=lambda x: -x["wr"])

    return inversion_candidates, proven_good, neutral


def build_cycle_analysis(trades: list[dict]) -> dict:
    """Analyze how patterns shift across market cycles."""
    cycle_groups = defaultdict(list)
    for t in trades:
        cycle_groups[t["cycle"]].append(t)

    cycle_analysis = {}
    for cycle_id, cycle_trades in sorted(cycle_groups.items()):
        if cycle_id == "other" or len(cycle_trades) < 3:
            continue

        # Dominant regime
        regime_counts = defaultdict(int)
        for t in cycle_trades:
            regime_counts[t["regime"]] += 1
        dominant_regime = max(regime_counts, key=regime_counts.get)

        # Per-trigger-direction stats within this cycle
        td_groups = defaultdict(list)
        for t in cycle_trades:
            td_groups[(t["trigger_type"], t["direction"])].append(t)

        worst_combo = None
        worst_wr = 1.0
        best_combo = None
        best_wr = 0.0

        td_breakdown = {}
        for (trigger, direction), group in td_groups.items():
            stats = compute_group_stats(group)
            td_breakdown[f"{trigger}_{direction}"] = stats
            if stats["n"] >= 3 and stats["wr"] < worst_wr:
                worst_wr = stats["wr"]
                worst_combo = f"{trigger}_{direction}"
            if stats["n"] >= 3 and stats["wr"] > best_wr:
                best_wr = stats["wr"]
                best_combo = f"{trigger}_{direction}"

        cycle_info = CYCLE_MAP.get(cycle_id, {})
        cycle_analysis[cycle_id] = {
            "label": cycle_info.get("label", cycle_id),
            "n_trades": len(cycle_trades),
            "overall_wr": round(sum(1 for t in cycle_trades if t["is_win"]) / len(cycle_trades), 4),
            "overall_pnl": round(sum(t["pnl_usd"] for t in cycle_trades), 2),
            "dominant_regime": dominant_regime,
            "regime_distribution": dict(regime_counts),
            "worst_combo": worst_combo,
            "worst_combo_wr": round(worst_wr, 4) if worst_combo else None,
            "best_combo": best_combo,
            "best_combo_wr": round(best_wr, 4) if best_combo else None,
            "trigger_direction_breakdown": td_breakdown,
        }

    return cycle_analysis


def compute_cross_version_stability(trades: list[dict]) -> dict:
    """Check if the worst combos are stable across gate versions."""
    version_groups = defaultdict(list)
    for t in trades:
        version_groups[t["version"]].append(t)

    # For each (trigger, direction), check WR across versions
    td_version_stats = defaultdict(lambda: defaultdict(dict))
    for version, v_trades in version_groups.items():
        td_groups = defaultdict(list)
        for t in v_trades:
            td_groups[(t["trigger_type"], t["direction"])].append(t)
        for (trigger, direction), group in td_groups.items():
            stats = compute_group_stats(group)
            td_version_stats[(trigger, direction)][version] = stats

    stability = {}
    for (trigger, direction), version_data in td_version_stats.items():
        key = f"{trigger}_{direction}"
        wrs = [s["wr"] for s in version_data.values() if s["n"] >= 2]
        if len(wrs) >= 2:
            avg_wr = sum(wrs) / len(wrs)
            wr_std = (sum((w - avg_wr) ** 2 for w in wrs) / len(wrs)) ** 0.5
            stability[key] = {
                "versions": {v: {"wr": s["wr"], "n": s["n"], "pf": s["pf"]}
                             for v, s in version_data.items()},
                "avg_wr": round(avg_wr, 4),
                "wr_std": round(wr_std, 4),
                "is_stable": wr_std < 0.15,
                "total_n": sum(s["n"] for s in version_data.values()),
                "verdict": "STABLE_BAD" if avg_wr < 0.35 and wr_std < 0.15
                          else "STABLE_GOOD" if avg_wr > 0.55 and wr_std < 0.15
                          else "UNSTABLE" if wr_std >= 0.15
                          else "NEUTRAL",
            }

    return stability


def estimate_inversion_impact(trades: list[dict], inversion_candidates: list[dict]) -> dict:
    """Estimate additional trades per year and projected PF from inversions."""
    inversion_set = {
        (c["regime"], c["trigger"], c["direction"])
        for c in inversion_candidates
    }

    # Original trades that match inversion candidates
    inverted_trades = [
        t for t in trades
        if (t["regime"], t["trigger_type"], t["direction"]) in inversion_set
    ]

    if not inverted_trades:
        return {"inverted_trades_count": 0, "note": "no trades match inversion candidates"}

    # Time span
    dates = [datetime.fromisoformat(t["open_ts"].replace("+00:00", "")) for t in inverted_trades]
    min_date = min(dates)
    max_date = max(dates)
    years = max((max_date - min_date).days / 365.25, 0.5)

    # Inverted PnL
    inverted_pnl = sum(-t["pnl_usd"] for t in inverted_trades)
    inverted_wins = sum(1 for t in inverted_trades if t["pnl_usd"] < 0)  # losses become wins
    inverted_wr = inverted_wins / len(inverted_trades)

    # Gross profit/loss of inverted
    inv_gross_profit = sum(-t["pnl_usd"] for t in inverted_trades if t["pnl_usd"] < 0)
    inv_gross_loss = sum(t["pnl_usd"] for t in inverted_trades if t["pnl_usd"] > 0)
    inv_pf = inv_gross_profit / inv_gross_loss if inv_gross_loss > 0 else 999

    # Non-inversion trades (keep as-is)
    non_inv_trades = [
        t for t in trades
        if (t["regime"], t["trigger_type"], t["direction"]) not in inversion_set
    ]
    non_inv_pnl = sum(t["pnl_usd"] for t in non_inv_trades)
    non_inv_profit = sum(t["pnl_usd"] for t in non_inv_trades if t["pnl_usd"] > 0)
    non_inv_loss = abs(sum(t["pnl_usd"] for t in non_inv_trades if t["pnl_usd"] < 0))
    non_inv_pf = non_inv_profit / non_inv_loss if non_inv_loss > 0 else 999

    # Hybrid: non-inverted + inverted
    hybrid_gross_profit = non_inv_profit + inv_gross_profit
    hybrid_gross_loss = non_inv_loss + inv_gross_loss
    hybrid_pf = hybrid_gross_profit / hybrid_gross_loss if hybrid_gross_loss > 0 else 999

    return {
        "inverted_trades_count": len(inverted_trades),
        "inverted_trades_per_year": round(len(inverted_trades) / years, 1),
        "inverted_total_pnl": round(inverted_pnl, 2),
        "inverted_wr": round(inverted_wr, 4),
        "inverted_pf": round(min(inv_pf, 999), 4),
        "original_total_pnl": round(sum(t["pnl_usd"] for t in trades), 2),
        "non_inverted_pf": round(min(non_inv_pf, 999), 4),
        "hybrid_pf": round(min(hybrid_pf, 999), 4),
        "hybrid_total_pnl": round(non_inv_pnl + inverted_pnl, 2),
        "pnl_improvement": round((non_inv_pnl + inverted_pnl) - sum(t["pnl_usd"] for t in trades), 2),
        "time_span_years": round(years, 2),
    }


def compute_ai_needs_assessment(cycle_analysis: dict, stability: dict) -> dict:
    """Determine if inversions need AI cycle detection or simple rules."""
    # Check if worst patterns are cycle-dependent
    worst_by_cycle = {}
    for cycle_id, data in cycle_analysis.items():
        if data["worst_combo"]:
            worst_by_cycle[cycle_id] = data["worst_combo"]

    unique_worsts = set(worst_by_cycle.values())

    # Check stability verdict distribution
    stable_bad = [k for k, v in stability.items() if v["verdict"] == "STABLE_BAD"]
    unstable = [k for k, v in stability.items() if v["verdict"] == "UNSTABLE"]

    assessment = {
        "worst_combos_across_cycles": worst_by_cycle,
        "unique_worst_patterns": list(unique_worsts),
        "pattern_shifts": len(unique_worsts) > 1,
        "stable_bad_patterns": stable_bad,
        "unstable_patterns": unstable,
    }

    if len(unique_worsts) <= 1 and len(stable_bad) > 0:
        assessment["recommendation"] = "SIMPLE_RULE"
        assessment["rationale"] = (
            "The worst pattern is consistent across cycles. "
            "A simple static inversion rule suffices — no AI needed."
        )
    elif len(unique_worsts) > 2 or len(unstable) > len(stable_bad):
        assessment["recommendation"] = "ADAPTIVE_AI"
        assessment["rationale"] = (
            "Worst patterns shift across cycles. "
            "AI cycle detection is needed for dynamic inversion."
        )
    else:
        assessment["recommendation"] = "REGIME_CONDITIONAL"
        assessment["rationale"] = (
            "Some patterns are regime-dependent. "
            "A regime-conditional rule (regime -> inversion mapping) works, "
            "with optional AI for regime edge cases."
        )

    return assessment


def main():
    print("=" * 70)
    print("WORST INDICATOR 3D SCAN: regime x trigger x direction")
    print("=" * 70)

    # Load all gate data
    files = {
        "v5": DATA_DIR / "gate1_v5_windows.jsonl",
        "v6": DATA_DIR / "gate1_v6_windows.jsonl",
        "v7": DATA_DIR / "gate1_v7_windows.jsonl",
        "v8_v1": DATA_DIR / "gate_v8_v1.jsonl",
        "v8_v2": DATA_DIR / "gate_v8_v2.jsonl",
    }

    all_windows = []
    for label, path in files.items():
        if path.exists():
            windows = load_jsonl(path)
            all_windows.extend(windows)
            n_trades = sum(len(w.get("trade_details", [])) for w in windows)
            print(f"  Loaded {label}: {len(windows)} windows, {n_trades} trades")

    # Extract flat trade list
    all_trades = extract_trades(all_windows)
    print(f"\nTotal trades extracted: {len(all_trades)}")

    # 3D scan: (regime, trigger, direction)
    print("\n" + "-" * 70)
    print("3D SCAN: (regime, trigger, direction)")
    print("-" * 70)
    scan_3d = run_3d_scan(all_trades)
    inversion_candidates, proven_good, neutral = classify_combos(scan_3d)

    print(f"\nINVERSION CANDIDATES (WR < 30%, N >= 3):")
    for c in inversion_candidates:
        print(f"  [{c['regime']:>13}] {c['trigger']:>25} {c['direction']:>5} | "
              f"WR={c['wr']:.1%} N={c['n']:>3} PnL={c['total_pnl']:>8.1f} | "
              f"inverted_WR={c['inverted_wr']:.1%}")

    print(f"\nPROVEN GOOD (WR > 55%, N >= 5):")
    for c in proven_good:
        print(f"  [{c['regime']:>13}] {c['trigger']:>25} {c['direction']:>5} | "
              f"WR={c['wr']:.1%} N={c['n']:>3} PF={c['pf']:>6.2f} PnL={c['total_pnl']:>8.1f}")

    # 2D scans for broader view
    print("\n" + "-" * 70)
    print("2D SCAN: (trigger, direction) — regime-agnostic")
    print("-" * 70)
    scan_td = run_2d_scan(all_trades, "trigger_type", "direction")
    for (trigger, direction), stats in sorted(scan_td.items(), key=lambda x: x[1]["wr"]):
        marker = "<<<INVERT" if stats["n"] >= 5 and stats["wr"] < 0.35 else ""
        marker = "  GOOD>>>" if stats["n"] >= 5 and stats["wr"] > 0.55 else marker
        print(f"  {trigger:>25} {direction:>5} | "
              f"WR={stats['wr']:.1%} N={stats['n']:>3} PF={stats['pf']:>6.2f} "
              f"PnL={stats['total_pnl']:>8.1f} {marker}")

    # 1D scan: trigger only
    print("\n" + "-" * 70)
    print("1D SCAN: trigger only")
    print("-" * 70)
    scan_trigger = run_1d_scan(all_trades, "trigger_type")
    for trigger, stats in sorted(scan_trigger.items(), key=lambda x: x[1]["wr"]):
        print(f"  {trigger:>25} | WR={stats['wr']:.1%} N={stats['n']:>3} "
              f"PF={stats['pf']:>6.2f} PnL={stats['total_pnl']:>8.1f}")

    # Cross-version stability
    print("\n" + "-" * 70)
    print("CROSS-VERSION STABILITY")
    print("-" * 70)
    stability = compute_cross_version_stability(all_trades)
    for key, data in sorted(stability.items(), key=lambda x: x[1]["avg_wr"]):
        print(f"  {key:>35} | avg_WR={data['avg_wr']:.1%} std={data['wr_std']:.3f} "
              f"N={data['total_n']:>3} [{data['verdict']}]")
        for v, vs in sorted(data["versions"].items()):
            print(f"    {v:>8}: WR={vs['wr']:.1%} N={vs['n']:>3} PF={vs['pf']:>6.2f}")

    # Cycle analysis
    print("\n" + "-" * 70)
    print("CYCLE ANALYSIS")
    print("-" * 70)
    cycle_analysis = build_cycle_analysis(all_trades)
    for cycle_id, data in sorted(cycle_analysis.items()):
        print(f"\n  {cycle_id} ({data['label']}):")
        print(f"    Trades: {data['n_trades']}, WR: {data['overall_wr']:.1%}, "
              f"PnL: {data['overall_pnl']:.1f}")
        print(f"    Dominant regime: {data['dominant_regime']}")
        print(f"    Worst: {data['worst_combo']} (WR={data['worst_combo_wr']})")
        print(f"    Best:  {data['best_combo']} (WR={data['best_combo_wr']})")

    # Inversion impact estimation
    print("\n" + "-" * 70)
    print("INVERSION IMPACT ESTIMATION")
    print("-" * 70)
    impact = estimate_inversion_impact(all_trades, inversion_candidates)
    for k, v in impact.items():
        print(f"  {k}: {v}")

    # AI needs assessment
    print("\n" + "-" * 70)
    print("AI NEEDS ASSESSMENT")
    print("-" * 70)
    ai_assessment = compute_ai_needs_assessment(cycle_analysis, stability)
    print(f"  Recommendation: {ai_assessment['recommendation']}")
    print(f"  Rationale: {ai_assessment['rationale']}")
    print(f"  Pattern shifts across cycles: {ai_assessment['pattern_shifts']}")
    print(f"  Stable bad patterns: {ai_assessment['stable_bad_patterns']}")
    print(f"  Unstable patterns: {ai_assessment['unstable_patterns']}")

    # Build output JSON
    output = {
        "scan_timestamp": datetime.now().isoformat(),
        "total_trades_analyzed": len(all_trades),
        "total_windows_analyzed": len(all_windows),
        "gate_versions": list(files.keys()),
        "inversion_candidates": inversion_candidates,
        "proven_good": proven_good,
        "neutral": [n for n in neutral if n["n"] >= 3],  # only meaningful neutrals
        "trigger_direction_2d": {
            f"{k[0]}_{k[1]}": v for k, v in scan_td.items()
        },
        "trigger_1d": dict(scan_trigger),
        "cross_version_stability": stability,
        "cycle_map": cycle_analysis,
        "inversion_impact": impact,
        "ai_needs_assessment": ai_assessment,
        "key_findings": [],  # populated below
    }

    # Summarize key findings
    findings = []
    if inversion_candidates:
        worst = inversion_candidates[0]
        findings.append(
            f"WORST: [{worst['regime']}] {worst['trigger']} {worst['direction']} "
            f"has {worst['wr']:.0%} WR across {worst['n']} trades — "
            f"inverting gives {worst['inverted_wr']:.0%} WR"
        )

    stable_bads = [k for k, v in stability.items() if v["verdict"] == "STABLE_BAD"]
    if stable_bads:
        findings.append(
            f"STABLE BAD across all versions: {', '.join(stable_bads)} — "
            f"safe to invert with simple rule"
        )

    if ai_assessment["pattern_shifts"]:
        findings.append(
            f"Pattern shifts detected across cycles — "
            f"worst combo changes: {ai_assessment['worst_combos_across_cycles']}"
        )

    if impact.get("hybrid_pf", 0) > 1.0:
        findings.append(
            f"Hybrid v1+inversions projected PF: {impact['hybrid_pf']:.2f} "
            f"(improvement: +{impact.get('pnl_improvement', 0):.1f} USD)"
        )

    output["key_findings"] = findings

    # Write output
    output_path = DATA_DIR / "worst_indicator_matrix.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Output written to: {output_path}")
    print(f"Key findings:")
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")


if __name__ == "__main__":
    main()
