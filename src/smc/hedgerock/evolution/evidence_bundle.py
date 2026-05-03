"""Phase D-cont3 / Ticket 1 — evidence-bundle loader.

**Read-only with respect to the Phase D artefacts.** Loads the
existing atlas + data-availability + walk-forward Markdown reports,
extracts the structured fields the gates need, and computes
hash-pinned references so any later tampering breaks bundle
verification.

The parsers are regex-based — the source reports are auto-generated
by Phase D scripts with stable, table-shaped layouts, so a tight
regex is more honest than a full Markdown parser.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smc.hedgerock.evolution.policy_manifest import EvidenceBundle


__all__ = [
    "EvidenceBundleArtefacts",
    "compute_bundle_content_hash",
    "compute_file_sha256",
    "load_evidence_bundle",
    "parse_data_availability_report",
    "verify_bundle_hash",
]


_HASH_BLOCK = 1 << 16  # 64 KiB


def compute_file_sha256(path: Path) -> str:
    """Stream-hash a file. Sha-256 of the raw bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bundle_content_hash(paths: list[Path] | tuple[Path, ...]) -> str:
    """Deterministic SHA-256 across a set of artefact files. Files are
    hashed in name-sorted order so the bundle hash is independent of
    caller-supplied order."""
    ordered = sorted([Path(p) for p in paths], key=lambda p: p.name)
    h = hashlib.sha256()
    for p in ordered:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# data-availability parser
# ---------------------------------------------------------------------------


# Match a data row of the year-replication table:
#   | XAUUSD | 2024 | 5693 | +0.139% ±0.067 | ... | 1 |
_REPL_ROW = re.compile(
    r"^\|\s*([A-Za-z0-9_]+)\s*\|\s*(\d{4})\s*\|\s*\d+\s*\|\s*"
    r"([^|]+?)\s*\|\s*[^|]+\|\s*[^|]+\|\s*(\d+)\s*\|\s*$"
)

# Match the trend_up cell content. Examples:
#   "+0.139% ±0.067"
#   "-0.067% ±0.048 (NEG)"
#   "+0.063% ±0.069 (CI∋0)"
#   "INCONCLUSIVE (n=10)"
_TREND_CELL = re.compile(
    r"^([+-]?\d+\.\d+)%\s*±\s*(\d+\.\d+)\s*(\(NEG\)|\(CI∋0\))?\s*$"
)


def parse_data_availability_report(text: str) -> dict[str, Any]:
    """Extract the fields the gates read.

    Returns::

        {
          "year_replication": {
              "<SYMBOL>": {
                  "years_total": int,
                  "years_passing": int,
                  "negative_sign_years": tuple[int, ...],
              },
              ...
          },
          "halt_event_count": int,   # sum across all rows
          "cross_symbol_count": int, # distinct symbols seen
          "no_strategy_change": bool,
        }
    """
    by_symbol: dict[str, dict[str, Any]] = {}
    halt_total = 0
    in_year_table = False
    for line in text.splitlines():
        if line.strip().startswith("| Symbol | Year | Bars | trend_up"):
            in_year_table = True
            continue
        if in_year_table:
            if not line.strip().startswith("|"):
                in_year_table = False
                continue
            if line.strip().startswith("|---"):
                continue
            m = _REPL_ROW.match(line)
            if m is None:
                continue
            symbol, year_str, trend_cell, halt_str = m.groups()
            year = int(year_str)
            halt = int(halt_str)
            halt_total += halt

            entry = by_symbol.setdefault(symbol, {
                "years_total": 0,
                "years_passing": 0,
                "negative_sign_years": [],
            })
            entry["years_total"] += 1
            cell = trend_cell.strip()
            tm = _TREND_CELL.match(cell)
            if tm is not None:
                mean = float(tm.group(1))
                ci = float(tm.group(2))
                marker = tm.group(3)  # None | "(NEG)" | "(CI∋0)"
                if marker == "(NEG)":
                    entry["negative_sign_years"].append(year)
                elif marker is None:
                    # No marker AND mean clears CI on positive side ⇒ pass
                    if mean > ci:
                        entry["years_passing"] += 1
                # "(CI∋0)" → no edge, not counted as passing
            # INCONCLUSIVE / unrecognised cells: don't count as passing,
            # don't count as negative.

    # Freeze the negative_sign_years lists into tuples for caller
    # immutability.
    yr_clean: dict[str, dict[str, Any]] = {}
    for sym, body in by_symbol.items():
        yr_clean[sym] = {
            "years_total": body["years_total"],
            "years_passing": body["years_passing"],
            "negative_sign_years": tuple(body["negative_sign_years"]),
        }

    no_strategy_change = _detect_no_strategy_change(text)

    return {
        "year_replication": yr_clean,
        "halt_event_count": halt_total,
        "cross_symbol_count": len(yr_clean),
        "no_strategy_change": no_strategy_change,
    }


def _detect_no_strategy_change(text: str) -> bool:
    """Match ``NO_STRATEGY_CHANGE: true|false`` inside the Action gate
    YAML block. Default is True (block) when the marker is absent —
    fail-closed."""
    m = re.search(r"NO_STRATEGY_CHANGE:\s*(true|false)\b", text)
    if m is None:
        return True  # fail-closed: missing marker = treat as blocked
    return m.group(1) == "true"


# ---------------------------------------------------------------------------
# load + verify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceBundleArtefacts:
    """Caller-supplied list of artefact paths to wrap into a bundle.
    Read-only — the loader never writes back."""

    bundle_id: str
    atlas_report_path: Path
    data_availability_report_path: Path
    walk_forward_run_paths: tuple[Path, ...] = field(default_factory=tuple)


def load_evidence_bundle(artefacts: EvidenceBundleArtefacts) -> EvidenceBundle:
    """Load the artefacts, parse the data-availability fields, and
    compute hash references. Raises ``FileNotFoundError`` on any
    missing artefact."""
    atlas_path = Path(artefacts.atlas_report_path)
    avail_path = Path(artefacts.data_availability_report_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"atlas report missing: {atlas_path}")
    if not avail_path.exists():
        raise FileNotFoundError(f"data-availability report missing: {avail_path}")

    atlas_hash = compute_file_sha256(atlas_path)
    avail_hash = compute_file_sha256(avail_path)
    wf_paths = tuple(Path(p) for p in artefacts.walk_forward_run_paths)
    for p in wf_paths:
        if not p.exists():
            raise FileNotFoundError(f"walk-forward artefact missing: {p}")

    bundle_paths: list[Path] = [atlas_path, avail_path, *wf_paths]
    bundle_hash = compute_bundle_content_hash(bundle_paths)

    parsed = parse_data_availability_report(avail_path.read_text(encoding="utf-8"))

    return EvidenceBundle(
        bundle_id=artefacts.bundle_id,
        bundle_hash_sha256=bundle_hash,
        atlas_report_path=str(atlas_path),
        atlas_report_hash_sha256=atlas_hash,
        data_availability_report_path=str(avail_path),
        data_availability_report_hash_sha256=avail_hash,
        walk_forward_run_paths=tuple(str(p) for p in wf_paths),
        year_replication=parsed["year_replication"],
        cross_symbol_count=parsed["cross_symbol_count"],
        halt_event_count=parsed["halt_event_count"],
        no_strategy_change=parsed["no_strategy_change"],
    )


def verify_bundle_hash(bundle: EvidenceBundle, *, base_dir: Path | None = None) -> bool:
    """Recompute hashes and compare against the recorded values.

    ``base_dir`` is unused but accepted for forward compatibility — the
    bundle stores absolute paths today.
    """
    try:
        atlas_now = compute_file_sha256(Path(bundle.atlas_report_path))
        avail_now = compute_file_sha256(Path(bundle.data_availability_report_path))
    except FileNotFoundError:
        return False
    if atlas_now != bundle.atlas_report_hash_sha256:
        return False
    if avail_now != bundle.data_availability_report_hash_sha256:
        return False
    paths_for_bundle = [
        Path(bundle.atlas_report_path),
        Path(bundle.data_availability_report_path),
        *(Path(p) for p in bundle.walk_forward_run_paths),
    ]
    if compute_bundle_content_hash(paths_for_bundle) != bundle.bundle_hash_sha256:
        return False
    return True
