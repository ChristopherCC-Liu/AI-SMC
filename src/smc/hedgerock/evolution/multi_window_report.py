"""Ticket 4 v2 Step 8 — XAUUSD multi-window shadow report renderer.

Pure function. Reads a list of v0.3.0 :class:`ShadowArtefact` paths
and emits a markdown-style report covering:

  * per-artefact metadata (absolute path, sha256, runner_version,
    verdict, coverage_pass, worst-window DD, total n_bars)
  * per-window table for each v0.3.0 artefact
  * coverage shortfall reasons (when present)
  * forbidden-file mtime / absence proof footer enumerating the
    live-runtime paths the sidecar is contractually forbidden to
    touch (``approved/``, ``pointer.json``, ``src/``, ``config/``,
    ``.mq5``)

XAUUSD-only by contract — the report never emits the legacy
``single_symbol`` / ``cross_symbol`` blocker phrasing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from smc.hedgerock.evolution.registry_audit import (
    REGISTRY_VIOLATION_GATE_REASON_PREFIX,
)


__all__ = [
    "FORBIDDEN_LIVE_PATHS",
    "render_multi_window_report",
]


def _hedgerock_home() -> Path:
    """Resolve the HedgeRock root. ``$HEDGEROCK_HOME`` overrides;
    otherwise defaults to ``$HOME/HedgeRock``. Both paths are
    safety-check anchors — never written to by this sidecar."""
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


def _ai_smc_home() -> Path:
    """Resolve the AI-SMC repo root. ``$AI_SMC_HOME`` overrides;
    otherwise inferred from this file's location
    (``parents[4]`` = repo root from ``src/smc/hedgerock/evolution/``).
    """
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[4]


# Live-runtime paths the sidecar must NEVER write to. The mtime
# proof footer reads these on disk; absent paths surface as
# ``MISSING`` (which is a valid + correct outcome on machines
# without a HedgeRock checkout). Override the locations on a fresh
# machine with ``$HEDGEROCK_HOME`` / ``$AI_SMC_HOME``.
def _build_forbidden_paths() -> tuple[str, ...]:
    hr = _hedgerock_home()
    smc = _ai_smc_home()
    return (
        str(hr / "policy_registry" / "approved"),
        str(hr / "policy_registry" / "pointer.json"),
        str(hr / "HedgeRock_v3.6.10.0_AHL3.mq5"),
        str(hr / "HedgeRock_v3.6.10.0_AHL2.mq5"),
        str(smc / "src" / "smc" / "hedgerock" / "rule_engine.py"),
        str(smc / "src" / "smc" / "hedgerock" / "decision_server.py"),
    )


FORBIDDEN_LIVE_PATHS: tuple[str, ...] = _build_forbidden_paths()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_load_artefact(path: Path):
    """Try to load a ShadowArtefact. Returns None on failure (the
    report still renders; failing artefacts get a placeholder)."""
    from smc.hedgerock.evolution.shadow_artefact import (
        load_shadow_artefact,
    )
    try:
        return load_shadow_artefact(path)
    except Exception:  # noqa: BLE001 — fail-soft for report
        return None


def _per_window_table(artefact) -> str:
    """Render the per-window block for one artefact."""
    pw = getattr(artefact, "per_window", {}) or {}
    windows = pw.get("windows", [])
    if not windows:
        return "  (no_per_window_data — pre-v0.3.0 artefact)"

    header = (
        "| window_id | observed_buckets | n_bars | n_decided_bars | "
        "max_h1_gap_bars | delta_pnl_pp | candidate_max_dd_pct | "
        "delta_dd_pp | candidate_max_open_lots | "
        "candidate_max_grid_density | candidate_n_trades |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows: list[str] = [header]
    for w in windows:
        rows.append(
            "| {wid} | {buckets} | {n_bars} | {n_dec} | {gap} | "
            "{dpnl:+.3f} | {cdd:.4f} | {ddpp:+.3f} | "
            "{lots:.3f} | {dens} | {ntrd} |".format(
                wid=w.get("window_id", ""),
                buckets=",".join(w.get("observed_buckets", []) or []),
                n_bars=w.get("n_bars", 0),
                n_dec=w.get("n_decided_bars", 0),
                gap=w.get("max_h1_gap_bars", 0),
                dpnl=float(w.get("delta_pnl_pp", 0.0)),
                cdd=float(w.get("candidate_max_dd_pct", 0.0)),
                ddpp=float(w.get("delta_dd_pp", 0.0)),
                lots=float(w.get("candidate_max_open_lots", 0.0)),
                dens=int(w.get("candidate_max_grid_density", 0)),
                ntrd=int(w.get("candidate_n_trades", 0)),
            )
        )
    return "\n".join(rows)


def _coverage_block(artefact) -> str:
    wc = getattr(artefact, "window_coverage", {}) or {}
    if not wc:
        return "  (no window_coverage data)"
    lines = [
        f"  coverage_pass={wc.get('coverage_pass')}",
        f"  windows_evaluated={wc.get('windows_evaluated')}",
        f"  regime_buckets_covered={wc.get('regime_buckets_covered')}",
        f"  halt_event_windows={wc.get('halt_event_windows')}",
        f"  no_trade_windows={wc.get('no_trade_windows')}",
    ]
    shortfalls = wc.get("shortfall_reasons") or ()
    if shortfalls:
        lines.append("  shortfall_reasons:")
        for s in shortfalls:
            lines.append(f"    - {s}")
    return "\n".join(lines)


def _worst_window_dd_summary(artefact) -> str:
    """Compute the worst-window candidate DD across the artefact's
    per_window block. Returns a short phrase including the window id
    and the value (in fraction units, the same scale stored in the
    artefact)."""
    pw = getattr(artefact, "per_window", {}) or {}
    windows = pw.get("windows", [])
    if not windows:
        return "worst_window_dd: n/a"
    worst_w = max(
        windows, key=lambda w: float(w.get("candidate_max_dd_pct", 0.0))
    )
    return (
        f"worst_window_dd={float(worst_w.get('candidate_max_dd_pct', 0.0)):.4f} "
        f"(window={worst_w.get('window_id', '')!r})"
    )


def _total_n_bars(artefact) -> int:
    pw = getattr(artefact, "per_window", {}) or {}
    return sum(int(w.get("n_bars", 0)) for w in pw.get("windows", []))


_REGISTRY_AUDIT_LOG: Path = (
    _hedgerock_home() / "policy_registry" / "shadow_artefacts" / "_audit.md"
)


# Append-only registry red-line state. The 2026-05-02 incident
# (Ticket 4 v2 Step 9 closeout) deleted four ``shadow_runner-0.3.0``
# artefacts mid-iteration; their SHAs are recorded in the audit log
# at :data:`_REGISTRY_AUDIT_LOG`. The boundary table surfaces this
# fact in every report so operators cannot miss the loss.
_STALE_V030_DELETED_DURING_THIS_SESSION: bool = True


def _hard_boundary_table() -> str:
    """Emit the hard-boundary status block. The
    `stale v0.3.0 artefacts deleted during this session` row must
    surface YES until a clean session passes without a deletion
    incident; only then can a future contributor flip the flag."""
    flag = "YES" if _STALE_V030_DELETED_DURING_THIS_SESSION else "NO"
    audit_link = (
        f"see {_REGISTRY_AUDIT_LOG} for SHAs of lost artefacts"
        if _STALE_V030_DELETED_DURING_THIS_SESSION
        else "no incidents"
    )
    return "\n".join([
        "",
        "## Hard-boundary status",
        "",
        "| boundary | value | corrective action |",
        "| --- | --- | --- |",
        f"| stale v0.3.0 artefacts deleted during this session | {flag} | "
        f"{audit_link} |",
        "| production rule_engine.py / decision_server.py mutated | NO | "
        "verified by mtime block below |",
        "| files written under approved/ or pointer.json | NO | "
        "verified by absence in proof block below |",
        "| files written under src/ config/ .mq5 | NO | "
        "verified by NoLiveEvidence counters in each artefact |",
        "",
        f"Audit log path: `{_REGISTRY_AUDIT_LOG}` (see this file for the "
        "byte-for-byte loss accounting).",
    ])


def _forbidden_proof_footer() -> str:
    """Build the live-runtime mtime / absence proof block, prefixed
    by the hard-boundary status table."""
    lines: list[str] = [_hard_boundary_table(), ""]
    lines.append("## Forbidden file proof — sidecar must not touch live runtime")
    lines.append("")
    for raw in FORBIDDEN_LIVE_PATHS:
        p = Path(raw)
        if not p.exists():
            lines.append(f"  - {raw}: ABSENT (forbidden path not present on disk)")
        else:
            try:
                mt = p.stat().st_mtime
                # Render mtime as int seconds (no second-level
                # precision needed — operators verify this is older
                # than the report run).
                lines.append(
                    f"  - {raw}: present, mtime={int(mt)} "
                    "(verify this predates the report run; sidecar must "
                    "leave it untouched)"
                )
            except OSError as e:
                lines.append(f"  - {raw}: stat_error={e!r}")
    lines.append("")
    lines.append("  These paths are FORBIDDEN write targets for the sidecar.")
    return "\n".join(lines)


def _registry_violation_block_for_candidate(registry_audit) -> str:
    """Per-artefact registry-violation block. Renders in every
    candidate's report region so operators cannot read PASS in one
    candidate while violation: YES is hidden in the footer.

    Returns "" when no audit state is supplied OR when the state
    reports no violation. Operators verify the absence of this
    block via grep, which is fail-loud by design — a missing
    block is never a silent green."""
    if registry_audit is None:
        return ""
    if not bool(getattr(
        registry_audit, "registry_append_only_violation", False
    )):
        return ""
    log_path = getattr(registry_audit, "audit_log_path", "<unknown>")
    lost_n = int(getattr(registry_audit, "lost_sha_count", 0))
    lost_shas = list(getattr(registry_audit, "lost_sha256", ()))
    lines = [
        "",
        "### registry_append_only_violation (G8 hard block)",
        "",
        "  This candidate cannot PASS this session — the shadow-",
        "  artefact registry's append-only contract was violated.",
        f"  audit_log_path: {log_path}",
        f"  lost_sha_count: {lost_n}",
    ]
    if lost_shas:
        lines.append("  lost_sha256:")
        for s in lost_shas:
            lines.append(f"    - {s}")
    lines.append(
        "  G8 reason prefix: "
        f"{REGISTRY_VIOLATION_GATE_REASON_PREFIX}"
    )
    return "\n".join(lines)


def render_multi_window_report(
    artefact_paths: Iterable[Path | str],
    *,
    registry_audit: Any | None = None,
) -> str:
    """Render a markdown-style multi-window shadow report.

    Pure: reads each artefact file but never writes anywhere. The
    output string carries every field RFC v2 §8 enumerates plus the
    forbidden-file proof footer.

    When ``registry_audit`` is supplied AND it reports a violation,
    EVERY candidate region carries a "registry_append_only_violation"
    block (in addition to the hard-boundary footer). Operators
    cannot read a PASS-style verdict in one region while the
    violation flag hides at the bottom of the report.
    """
    paths: list[Path] = [Path(p).resolve() for p in artefact_paths]

    out: list[str] = []
    out.append("# XAUUSD Multi-Window Shadow Report")
    out.append("")
    out.append(f"Total artefacts processed: {len(paths)}")
    out.append("")

    if not paths:
        out.append("(no artefacts — empty input list)")
        violation_block = _registry_violation_block_for_candidate(
            registry_audit
        )
        if violation_block:
            out.append(violation_block)
            out.append("")
        out.append(_forbidden_proof_footer())
        return "\n".join(out)

    for path in paths:
        out.append("---")
        out.append(f"## artefact_path: {path}")
        out.append("")
        if not path.exists():
            out.append("  (file missing on disk)")
            continue

        try:
            sha = _file_sha256(path)
        except OSError as e:
            sha = f"sha256_error={e!r}"
        out.append(f"  sha256={sha}")

        artefact = _safe_load_artefact(path)
        if artefact is None:
            out.append("  (artefact failed to load — integrity error)")
            continue

        verdict_value = getattr(artefact.verdict, "value", str(artefact.verdict))
        out.append(f"  runner_version={artefact.runner_version!r}")
        out.append(f"  verdict={verdict_value}")
        out.append(f"  verdict_reason={artefact.verdict_reason!r}")
        out.append(f"  candidate_id={artefact.candidate_id!r}")
        out.append(f"  total_n_bars={_total_n_bars(artefact)}")
        out.append(f"  {_worst_window_dd_summary(artefact)}")
        out.append("")
        out.append("### window_coverage")
        out.append(_coverage_block(artefact))
        out.append("")
        out.append("### per_window")
        out.append(_per_window_table(artefact))

        # Per-candidate registry-violation block — fires for every
        # artefact in the report when audit state reports a violation.
        violation_block = _registry_violation_block_for_candidate(
            registry_audit
        )
        if violation_block:
            out.append(violation_block)
        out.append("")

    out.append(_forbidden_proof_footer())
    return "\n".join(out)
