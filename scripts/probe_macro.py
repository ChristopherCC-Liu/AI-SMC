"""Probe all 3 macro sources directly on VPS. Correct API + UTF-8 env.

Usage (on VPS):
    cd C:\\AI-SMC && .venv\\Scripts\\python.exe scripts\\probe_macro.py

Forces PYTHONIOENCODING=utf-8 to bypass smartmoneyconcepts emoji GBK issue.
"""
from __future__ import annotations

import os

# Force UTF-8 to bypass smartmoneyconcepts emoji GBK encoding issue
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Reconfigure stdout/stderr to UTF-8 on Windows
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def probe_cftc_raw() -> None:
    print("=" * 60)
    print("CFTC RAW HTTP (with correct 088691 query)")
    print("=" * 60)
    try:
        import requests

        resp = requests.get(
            "https://publicreporting.cftc.gov/resource/gpe5-46if.json",
            params={
                "$where": "cftc_commodity_code='088691'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": "3",
            },
            timeout=15,
        )
        print(f"  status: {resp.status_code}")
        print(f"  body[:500]: {resp.text[:500]}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")


def probe_fred_raw() -> None:
    print()
    print("=" * 60)
    print("FRED RAW HTTP (no api_key → expect 400)")
    print("=" * 60)
    try:
        import requests

        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "DFII10", "file_type": "json", "limit": 1},
            timeout=15,
        )
        print(f"  status: {resp.status_code}")
        print(f"  body[:300]: {resp.text[:300]}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")


def probe_cot_fetcher() -> None:
    print()
    print("=" * 60)
    print("COT FETCHER (production code path)")
    print("=" * 60)
    try:
        from smc.ai.cot_fetcher import COTFetcher

        fetcher = COTFetcher(cache_path=PROJECT_ROOT / "data" / "macro" / "probe_cot.parquet")
        df = fetcher.fetch()
        if df is None:
            print("  fetch() returned None (fetch failed or cache empty)")
        else:
            print(f"  fetch() returned DataFrame, rows={len(df)}")
            print(f"  columns: {df.columns}")
            if len(df) > 0:
                print(f"  latest row: {df.tail(3).to_dicts()}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()


def probe_tips_fetcher() -> None:
    print()
    print("=" * 60)
    print("TIPS FETCHER (production code path)")
    print("=" * 60)
    try:
        from smc.ai.tips_fetcher import TIPSFetcher

        fetcher = TIPSFetcher(cache_path=PROJECT_ROOT / "data" / "macro" / "probe_tips.parquet")
        hist = fetcher.fetch_history()
        print(f"  fetch_history() returned {type(hist).__name__}, len={len(hist) if hasattr(hist, '__len__') else '?'}")
        if hasattr(hist, "__len__") and len(hist) > 0:
            print(f"  first 3: {hist[:3]}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()


def probe_dxy_fetcher() -> None:
    print()
    print("=" * 60)
    print("DXY (ExternalContextFetcher)")
    print("=" * 60)
    try:
        from smc.ai.external_context import ExternalContextFetcher

        fetcher = ExternalContextFetcher()
        dxy = fetcher.fetch_dxy()
        print(f"  fetch_dxy() returned: {dxy!r}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()


def probe_macro_layer() -> None:
    print()
    print("=" * 60)
    print("MACRO LAYER aggregation")
    print("=" * 60)
    try:
        from smc.ai.macro_layer import MacroLayer

        layer = MacroLayer(cache_dir=PROJECT_ROOT / "data" / "macro")
        bias = layer.compute_macro_bias(instrument="XAUUSD")
        print(f"  total_bias: {bias.total_bias:+.4f}")
        print(f"  direction:  {bias.direction}")
        print(f"  sources_available: {bias.sources_available}")
        print(f"  cot_bias:   {bias.cot_bias:+.4f}")
        print(f"  yield_bias: {bias.yield_bias:+.4f}")
        print(f"  dxy_bias:   {bias.dxy_bias:+.4f}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print(f"PYTHONIOENCODING={os.environ.get('PYTHONIOENCODING')}")
    print(f"sys.stdout.encoding={sys.stdout.encoding}")
    print()
    probe_cftc_raw()
    probe_fred_raw()
    probe_cot_fetcher()
    probe_tips_fetcher()
    probe_dxy_fetcher()
    probe_macro_layer()
    print()
    print("=" * 60)
    print("PROBE COMPLETE")
    print("=" * 60)
