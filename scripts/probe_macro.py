"""Probe all 3 macro sources directly on VPS to diagnose why they return 0.

Usage (on VPS):
    cd C:\\AI-SMC && .venv\\Scripts\\python.exe scripts\\probe_macro.py

Prints detailed reachability + parse status for each source.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def probe_dxy() -> None:
    print("=" * 60)
    print("DXY PROBE (yfinance)")
    print("=" * 60)
    try:
        import yfinance  # type: ignore

        print(f"  yfinance version: {yfinance.__version__}")
    except ImportError as e:
        print(f"  FAIL: yfinance not installed: {e}")
        return

    try:
        from smc.ai.external_context import ExternalContextFetcher

        fetcher = ExternalContextFetcher()
        snap = fetcher.fetch_dxy()
        print(f"  ExternalContextFetcher.fetch_dxy() returned: {snap!r}")
    except Exception as e:
        print(f"  ExternalContextFetcher.fetch_dxy() EXCEPTION: {e}")
        traceback.print_exc()


def probe_cot() -> None:
    print()
    print("=" * 60)
    print("COT PROBE (CFTC Socrata)")
    print("=" * 60)
    try:
        import requests

        resp = requests.get(
            "https://publicreporting.cftc.gov/resource/gpe5-46if.json",
            params={
                "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
                "$limit": 1,
            },
            timeout=15,
        )
        print(f"  raw HTTP status: {resp.status_code}")
        print(f"  body preview: {resp.text[:300]}")
    except Exception as e:
        print(f"  raw requests EXCEPTION: {e}")

    try:
        from smc.ai.cot_fetcher import COTFetcher

        fetcher = COTFetcher(cache_path=PROJECT_ROOT / "data" / "macro" / "probe_cot.parquet")
        hist = fetcher.fetch_history()
        print(f"  COTFetcher.fetch_history() len={len(hist)}")
        if hist:
            print(f"    latest 3: {hist[-3:]}")
    except Exception as e:
        print(f"  COTFetcher EXCEPTION: {e}")
        traceback.print_exc()


def probe_tips() -> None:
    print()
    print("=" * 60)
    print("TIPS PROBE (FRED DFII10)")
    print("=" * 60)
    try:
        import requests

        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DFII10",
                "file_type": "json",
                "limit": 1,
            },
            timeout=15,
        )
        print(f"  raw HTTP status: {resp.status_code}")
        print(f"  body preview: {resp.text[:300]}")
    except Exception as e:
        print(f"  raw requests EXCEPTION: {e}")

    try:
        from smc.ai.tips_fetcher import TIPSFetcher

        fetcher = TIPSFetcher(
            cache_path=PROJECT_ROOT / "data" / "macro" / "probe_tips.parquet",
        )
        hist = fetcher.fetch_history()
        print(f"  TIPSFetcher.fetch_history() len={len(hist)}")
        if hist:
            print(f"    latest 3: {hist[:3]}")
    except Exception as e:
        print(f"  TIPSFetcher EXCEPTION: {e}")
        traceback.print_exc()

    # Alternative: yfinance TIP ETF as TIPS proxy
    print()
    print("  --- yfinance TIP ETF fallback probe ---")
    try:
        import yfinance

        tip = yfinance.Ticker("TIP")
        hist = tip.history(period="30d")
        if len(hist) > 0:
            print(f"  TIP ETF history rows: {len(hist)}, latest close: {hist['Close'].iloc[-1]:.2f}")
            print(f"  TIP ETF 5d change: {(hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100:.3f}%")
        else:
            print("  TIP ETF empty history")
    except Exception as e:
        print(f"  TIP ETF EXCEPTION: {e}")


def probe_macro_layer() -> None:
    print()
    print("=" * 60)
    print("MACRO LAYER AGGREGATION PROBE")
    print("=" * 60)
    try:
        from smc.ai.macro_layer import MacroLayer

        layer = MacroLayer(cache_dir=PROJECT_ROOT / "data" / "macro")
        bias = layer.compute_macro_bias(instrument="XAUUSD")
        print(f"  MacroLayer.compute_macro_bias(XAUUSD): value={bias.value:.4f}")
        print(f"    direction: {bias.direction}")
        print(f"    sources_available: {bias.sources_available}")
        print(f"    dxy_component: {bias.dxy_component:.4f}")
        print(f"    cot_component: {bias.cot_component:.4f}")
        print(f"    yield_component: {bias.yield_component:.4f}")
    except Exception as e:
        print(f"  MacroLayer EXCEPTION: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    probe_dxy()
    probe_cot()
    probe_tips()
    probe_macro_layer()
    print()
    print("=" * 60)
    print("PROBE COMPLETE")
    print("=" * 60)
