"""Stage 6-followup-2 task 3 — ASCII visualisations for the
recommendation report.

Pure functions; no FS, no live runtime imports. Three renderers:

  * :func:`render_parameter_comparison_table` — fixed-width table of
    candidate_id × baseline → proposed × decision × visual band.
  * :func:`render_gate_matrix` — rows-per-candidate × G1..G8 matrix
    with PASS / FAIL / OTHER glyphs.
  * :func:`render_heat_ranking` — ordered list with RECOMMEND first,
    then by trigger count, then by candidate_id (deterministic).
"""

from __future__ import annotations

from typing import Iterable, Mapping

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_RECOMMEND,
    SAFETY_CLAMPS,
)
from smc.hedgerock.evolution.policy_manifest import GateStatus


__all__ = [
    "render_gate_matrix",
    "render_heat_ranking",
    "render_parameter_comparison_table",
]


_GATE_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")


# ---------------------------------------------------------------------------
# Parameter comparison table
# ---------------------------------------------------------------------------


def _band_marker(*, baseline: float, proposed: float,
                 lo: float, hi: float, width: int = 10) -> str:
    """Return a fixed-width ASCII band marker.

    The band spans ``[lo .. hi]``; the proposed value's position
    within the band is rendered as ``·``; baseline is ``b``;
    proposed is ``p``. Out-of-band values clip to the edges.
    """
    if hi <= lo or width < 4:
        return "[no band]"
    cells = ["-"] * width

    def _idx(v: float) -> int:
        if v <= lo:
            return 0
        if v >= hi:
            return width - 1
        return int(round((v - lo) / (hi - lo) * (width - 1)))

    b_idx = _idx(baseline)
    p_idx = _idx(proposed)
    cells[b_idx] = "b"
    if p_idx == b_idx:
        cells[b_idx] = "B"  # baseline & proposed coincide
    else:
        cells[p_idx] = "p"
    return "[" + "".join(cells) + "]"


def render_parameter_comparison_table(
    proposals: Iterable[CandidateProposal],
) -> str:
    """Markdown-flavoured fixed-width table of parameter comparisons."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    for p in proposals:
        cls = p.parameter_class or ""
        clamp = SAFETY_CLAMPS.get(cls)
        if clamp is None:
            band = "(no band)"
            band_str = f"{band}"
        else:
            band = f"[{clamp.lo}, {clamp.hi}]"
            band_str = (
                f"{band} "
                f"{_band_marker(baseline=float(p.baseline_value), proposed=float(p.proposed_value), lo=clamp.lo, hi=clamp.hi)}"
            )
        rows.append((
            p.candidate_id,
            p.parameter_target,
            f"{p.baseline_value} → {p.proposed_value}",
            p.decision,
            p.decision_reason or "",
            band_str,
        ))

    header = (
        "| candidate_id | target | baseline → proposed | decision | reason | band visual |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| `{r[0]}` | `{r[1]}` | `{r[2]}` | `{r[3]}` | `{r[4]}` | `{r[5]}` |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate matrix
# ---------------------------------------------------------------------------


_GLYPH_PASS = "✓"
_GLYPH_FAIL = "✗"
_GLYPH_OTHER = "·"
_GLYPH_MISSING = "·"


def _gate_glyph(status: GateStatus | None) -> str:
    if status is None:
        return _GLYPH_MISSING
    if status == GateStatus.PASS:
        return _GLYPH_PASS
    if status == GateStatus.FAIL:
        return _GLYPH_FAIL
    return _GLYPH_OTHER


def render_gate_matrix(
    gate_results_per_candidate: Mapping[
        str, Mapping[str, GateStatus | object]
    ],
) -> str:
    """Render a markdown table with one row per candidate and one
    column per gate id (G1..G8). Missing or non-status entries get a
    placeholder glyph.
    """
    header = (
        "| candidate_id | "
        + " | ".join(_GATE_IDS)
        + " |"
    )
    sep = "|---|" + "|".join(["---"] * len(_GATE_IDS)) + "|"
    lines = [header, sep]
    for cid in sorted(gate_results_per_candidate):
        gates = gate_results_per_candidate[cid]
        cells: list[str] = []
        for gid in _GATE_IDS:
            raw = gates.get(gid)  # type: ignore[arg-type]
            status: GateStatus | None
            if isinstance(raw, GateStatus):
                status = raw
            elif raw is None:
                status = None
            else:
                # Some callers pass a PromotionGateResult-like object;
                # try to read its `status` attribute.
                status = getattr(raw, "status", None)
            cells.append(_gate_glyph(status))
        lines.append(f"| `{cid}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heat ranking
# ---------------------------------------------------------------------------


def render_heat_ranking(proposals: Iterable[CandidateProposal]) -> str:
    """Sort proposals by (RECOMMEND first, len(triggered_by) desc,
    candidate_id asc) and render a numbered table."""

    def _key(p: CandidateProposal) -> tuple[int, int, str]:
        return (
            0 if p.decision == DECISION_RECOMMEND else 1,
            -len(p.triggered_by),
            p.candidate_id,
        )

    ordered = sorted(proposals, key=_key)
    lines = [
        "| rank | candidate_id | decision | triggers | reason |",
        "|---|---|---|---|---|",
    ]
    for i, p in enumerate(ordered, start=1):
        lines.append(
            f"| {i} | `{p.candidate_id}` | `{p.decision}` | "
            f"{len(p.triggered_by)} | `{p.decision_reason or ''}` |"
        )
    return "\n".join(lines)
