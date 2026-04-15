"""Integration tests for the CSV ingest pipeline.

Tests the full chain: CSV files on disk -> ingest_csv_files() ->
partitioned Parquet in the data lake -> manifest JSON on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.data.ingest import ingest_csv_files
from smc.data.lake import ForexDataLake
from smc.data.manifest import Manifest
from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_csv_fixture(csv_dir: Path, instrument: str, timeframe: Timeframe) -> Path:
    """Create a minimal MT5-format CSV file under the expected directory layout.

    Layout: {csv_dir}/{timeframe}/{instrument}.csv
    """
    tf_dir = csv_dir / str(timeframe)
    tf_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tf_dir / f"{instrument}.csv"

    # MT5 tab-separated format with realistic XAUUSD data
    lines = [
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>",
        "2024.01.02\t00:00\t2062.00\t2065.50\t2060.10\t2064.30\t1500.0\t0\t3.0",
        "2024.01.02\t01:00\t2064.30\t2066.00\t2063.00\t2065.80\t1200.0\t0\t2.5",
        "2024.01.02\t02:00\t2065.80\t2068.20\t2064.50\t2067.10\t1800.0\t0\t3.2",
        "2024.01.02\t03:00\t2067.10\t2069.00\t2066.00\t2068.50\t900.0\t0\t2.8",
        "2024.01.02\t04:00\t2068.50\t2070.30\t2067.20\t2069.80\t1100.0\t0\t3.0",
        "2024.01.02\t05:00\t2069.80\t2071.00\t2068.50\t2070.50\t1300.0\t0\t2.7",
        "2024.01.02\t06:00\t2070.50\t2072.80\t2069.00\t2071.90\t1600.0\t0\t3.1",
        "2024.01.02\t07:00\t2071.90\t2073.50\t2070.80\t2072.60\t1400.0\t0\t2.9",
        "2024.01.02\t08:00\t2072.60\t2074.00\t2071.50\t2073.30\t1700.0\t0\t3.3",
        "2024.01.02\t09:00\t2073.30\t2075.20\t2072.00\t2074.80\t2000.0\t0\t2.6",
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestPipeline:
    """Integration tests for ingest_csv_files."""

    def test_ingest_creates_parquet_and_manifest(self, tmp_path: Path) -> None:
        """Verify that ingest creates parquet files and a manifest."""
        csv_dir = tmp_path / "csv"
        data_dir = tmp_path / "lake"

        _create_csv_fixture(csv_dir, "XAUUSD", Timeframe.H1)

        manifest = ingest_csv_files(
            csv_dir=csv_dir,
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        # Manifest is returned
        assert isinstance(manifest, Manifest)
        assert manifest.row_count == 10
        assert manifest.source == "csv:XAUUSD"
        assert manifest.schema_version == 1

        # Manifest file written to disk
        manifest_files = list((data_dir / "manifests").glob("*.json"))
        assert len(manifest_files) == 1

        # Parquet files written to disk
        parquet_files = list(data_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_ingest_data_queryable_from_lake(self, tmp_path: Path) -> None:
        """Verify that ingested data can be queried from the data lake."""
        from datetime import datetime, timezone

        csv_dir = tmp_path / "csv"
        data_dir = tmp_path / "lake"

        _create_csv_fixture(csv_dir, "XAUUSD", Timeframe.H1)

        ingest_csv_files(
            csv_dir=csv_dir,
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        # Query the lake
        lake = ForexDataLake(data_dir)
        df = lake.query(
            "XAUUSD",
            Timeframe.H1,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

        assert not df.is_empty()
        assert len(df) == 10
        assert "ts" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns

    def test_ingest_manifest_sha256_is_deterministic(self, tmp_path: Path) -> None:
        """Verify that re-ingesting the same CSV produces the same SHA-256."""
        csv_dir = tmp_path / "csv"
        data_dir_1 = tmp_path / "lake1"
        data_dir_2 = tmp_path / "lake2"

        _create_csv_fixture(csv_dir, "XAUUSD", Timeframe.H1)

        m1 = ingest_csv_files(
            csv_dir=csv_dir,
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir_1,
        )
        m2 = ingest_csv_files(
            csv_dir=csv_dir,
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir_2,
        )

        assert m1.sha256 == m2.sha256
        assert m1.row_count == m2.row_count

    def test_ingest_manifest_roundtrips_from_disk(self, tmp_path: Path) -> None:
        """Verify that the manifest written to disk can be read back."""
        csv_dir = tmp_path / "csv"
        data_dir = tmp_path / "lake"

        _create_csv_fixture(csv_dir, "XAUUSD", Timeframe.H1)

        original = ingest_csv_files(
            csv_dir=csv_dir,
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            data_dir=data_dir,
        )

        manifest_files = list((data_dir / "manifests").glob("*.json"))
        loaded = Manifest.from_json(manifest_files[0].read_text(encoding="utf-8"))

        assert loaded.source == original.source
        assert loaded.sha256 == original.sha256
        assert loaded.row_count == original.row_count

    def test_ingest_missing_csv_raises(self, tmp_path: Path) -> None:
        """Verify that a missing CSV file raises an error."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir(parents=True)
        data_dir = tmp_path / "lake"

        with pytest.raises(Exception):  # ForexAdapterError
            ingest_csv_files(
                csv_dir=csv_dir,
                instrument="XAUUSD",
                timeframe=Timeframe.H1,
                data_dir=data_dir,
            )
