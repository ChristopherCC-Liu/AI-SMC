"""Manifest generator for data-lake reproducibility.

Every parquet source has a manifest at ``data/manifests/{slug}.json``::

    {
      "source": "mt5:XAUUSD",
      "sha256": "…",
      "source_url": "mt5://XAUUSD",
      "fetched_at": "2026-04-15T08:00:00+00:00",
      "schema_version": 1,
      "row_count": 50000,
      "date_min": "2020-01-02T00:00:00+00:00",
      "date_max": "2026-04-14T22:00:00+00:00"
    }

A backtest that cannot reproduce its manifest hash fails its quality gate.
The hash is computed over the *content* of the parquet files in a stable order
so that a rewrite with identical rows produces an identical hash.

All timestamps are stored in UTC (``FOREX_TZ = "UTC"``).  No timezone
conversion is performed; the system speaks UTC throughout.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from smc.data.schemas import FOREX_TZ, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """Immutable record of a completed parquet ingest.

    Attributes:
        source: Source identifier, e.g. ``"mt5:XAUUSD"`` or ``"csv:XAUUSD"``.
        sha256: Hex digest of the parquet file contents (see
            :func:`compute_content_sha256`).
        source_url: Human-visitable or machine-parseable URL of the upstream
            data, e.g. ``"mt5://XAUUSD"`` or a path to a CSV directory.
        fetched_at: ISO-8601 string of the fetch time in UTC.
        schema_version: ``SCHEMA_VERSION`` at the time of ingest.
        row_count: Total rows across all parquet files for this source.
        date_min: ISO-8601 string of the earliest ``ts`` value.
        date_max: ISO-8601 string of the latest ``ts`` value.
    """

    source: str
    sha256: str
    source_url: str
    fetched_at: str
    schema_version: int
    row_count: int
    date_min: str
    date_max: str

    def to_json(self) -> str:
        """Serialise to a pretty-printed JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Deserialise from a JSON string produced by :meth:`to_json`."""
        data = json.loads(text)
        return cls(**data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def source_to_dir_slug(source: str) -> str:
    """Convert a source identifier to a filesystem-safe directory slug.

    Examples::

        source_to_dir_slug("mt5:XAUUSD")   -> "mt5_xauusd"
        source_to_dir_slug("csv:XAUUSD")   -> "csv_xauusd"
        source_to_dir_slug("yfinance:^VIX") -> "yfinance__vix"

    Rules:
    - Convert to lower-case.
    - Replace ``:`` with ``_``.
    - Replace any character that is not ``[a-z0-9_-]`` with ``_``.
    - Collapse consecutive underscores to a single ``_``.
    """
    slug = source.lower()
    slug = slug.replace(":", "_")
    slug = re.sub(r"[^a-z0-9_\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def compute_content_sha256(files: list[Path]) -> str:
    """SHA-256 hash over the byte contents of *files*, in the given order.

    We hash raw bytes rather than in-memory representations so that any
    future run that produces byte-identical parquet files gets the same hash.
    Ordering matters: callers must pass files in a stable order (e.g. sorted
    alphabetically by path).

    Args:
        files: Non-empty list of paths to parquet files.

    Returns:
        Lowercase hex digest string.

    Raises:
        ValueError: if *files* is empty.
        FileNotFoundError: if any path in *files* does not exist.
    """
    if not files:
        raise ValueError("Cannot hash an empty file list.")
    hasher = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_manifest(
    *,
    source: str,
    source_url: str,
    files: list[Path],
    row_count: int,
    date_min: datetime,
    date_max: datetime,
    fetched_at: datetime | None = None,
) -> Manifest:
    """Build a :class:`Manifest` for a completed parquet write.

    All datetime arguments must be tz-aware.  The manifest stores them as
    UTC ISO-8601 strings; any offset is normalised to ``+00:00``.

    Args:
        source: Source identifier, e.g. ``"mt5:XAUUSD"``.
        source_url: Human-readable or machine-parseable URL / path of the
            upstream data source.
        files: Every parquet file that belongs to this source, in a stable
            order (e.g. sorted by path).  Other sources must not be included.
        row_count: Total rows across all *files*.
        date_min: Earliest ``ts`` in the dataset.  Must be tz-aware.
        date_max: Latest ``ts`` in the dataset.  Must be tz-aware.
        fetched_at: Optional fetch timestamp.  Defaults to ``datetime.now(UTC)``.

    Returns:
        An immutable :class:`Manifest` instance.

    Raises:
        ValueError: if any datetime argument is tz-naive.
    """
    if fetched_at is None:
        fetched_at = datetime.now(tz=timezone.utc)

    for label, dt in [("date_min", date_min), ("date_max", date_max), ("fetched_at", fetched_at)]:
        if dt.tzinfo is None:
            raise ValueError(f"'{label}' must be tz-aware, got naive datetime.")

    # Normalise to UTC offset string "+00:00"
    def _to_utc_iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat()

    return Manifest(
        source=source,
        sha256=compute_content_sha256(files),
        source_url=source_url,
        fetched_at=_to_utc_iso(fetched_at),
        schema_version=SCHEMA_VERSION,
        row_count=row_count,
        date_min=_to_utc_iso(date_min),
        date_max=_to_utc_iso(date_max),
    )


def write_manifest(manifest: Manifest, manifests_dir: Path) -> Path:
    """Write *manifest* to ``{manifests_dir}/{slug}.json``.

    The slug is derived from :func:`source_to_dir_slug` so manifest file
    names align with the parquet partition directories created by
    :func:`~smc.data.writers.write_forex_partitioned`.

    Args:
        manifest: The manifest to persist.
        manifests_dir: Directory to write into (created if absent).

    Returns:
        Absolute path to the written JSON file.
    """
    slug = source_to_dir_slug(manifest.source)
    out = manifests_dir / f"{slug}.json"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return out.resolve()


__all__ = [
    "FOREX_TZ",
    "SCHEMA_VERSION",
    "Manifest",
    "compute_content_sha256",
    "build_manifest",
    "write_manifest",
    "source_to_dir_slug",
]
