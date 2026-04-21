"""R7-B2 — rebuild setup cache with synthetic_zones_enabled=True.

The existing `.scratch/round4/setup_cache/*.pkl` was generated before
`synthetic_zones_enabled` landed (Round 5 A3-R).  This leaves 2024
Mar-Oct with zero setups because historical SMC zones don't exist at
ATH levels.  This script regenerates every walk-forward window with
synthetic-zone augmentation turned on so the A/B harness can measure
the P0-1 router on the real ATH-rally bars.

Strategy
--------
- Cache the regenerated windows to `.scratch/round7/setup_cache_synth/`
  (NOT overwrite the round4 cache — we want an A/B on *cache* too).
- Point `backtest_mode_router_ab.py` at the new directory via the
  `--setup-cache-dir` flag (added in this change, backward-compat).
- 2024-only fast mode: `--years=2024` rebuilds W13/W14/W15.
- Full rebuild: ~12-15 minutes per window × 15 windows.

Usage
-----
    /opt/anaconda3/bin/python scripts/rebuild_setup_cache_synthetic.py              # full 2021-2024
    /opt/anaconda3/bin/python scripts/rebuild_setup_cache_synthetic.py --years=2024 # 2024 only (fast)
    /opt/anaconda3/bin/python scripts/rebuild_setup_cache_synthetic.py --no-skip    # force regen even if cache exists
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.getLogger("smc").setLevel(logging.WARNING)
logging.getLogger("smc.ai").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

import polars as pl

from smc.ai.regime_cache import RegimeCacheLookup, build_regime_cache
from smc.backtest.adapter_fast import FastSMCStrategyAdapter
from smc.backtest.walk_forward import _add_months
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator


# Match backtest_trailing_grid window structure so the caches are drop-in.
_TRAIN_MONTHS = 12
_TEST_MONTHS = 3
_STEP_MONTHS = 3

_OUTPUT_CACHE_DIR = PROJECT_ROOT / ".scratch" / "round7" / "setup_cache_synth"
_REGIME_CACHE_PATH = PROJECT_ROOT / "data" / "regime_cache.parquet"

# Mirror scripts/backtest_trailing_grid.py window numbering so keys align.
_WINDOW_YEAR_MAP: dict[int, list[int]] = {
    2021: [1, 2, 3, 4],
    2022: [5, 6, 7, 8],
    2023: [9, 10, 11, 12],
    2024: [13, 14, 15],
}


@dataclass
class WindowSpec:
    window_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def key(self) -> str:
        return f"W{self.window_num:02d}_{self.test_start.strftime('%Y%m%d')}"

    @property
    def test_year(self) -> int:
        return self.test_start.year


def _build_windows(lake: ForexDataLake) -> list[WindowSpec]:
    available = lake.available_range("XAUUSD", Timeframe.M15)
    if available is None:
        raise RuntimeError("No M15 data")
    data_start = available[0]

    windows: list[WindowSpec] = []
    num = 1
    train_start = data_start
    while True:
        train_end = _add_months(train_start, _TRAIN_MONTHS)
        test_end = _add_months(train_end, _TEST_MONTHS)
        if test_end > available[1]:
            break
        windows.append(WindowSpec(
            window_num=num,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
        ))
        train_start = _add_months(train_start, _STEP_MONTHS)
        num += 1
        if len(windows) > 50:
            break
    return windows


def _ensure_regime_cache(lake: ForexDataLake) -> RegimeCacheLookup:
    if not _REGIME_CACHE_PATH.exists():
        print("Regime cache missing; rebuilding…", flush=True)
        build_regime_cache(lake, _REGIME_CACHE_PATH, frequency_hours=4)
    return RegimeCacheLookup(_REGIME_CACHE_PATH)


def _regenerate_window(
    spec: WindowSpec,
    lake: ForexDataLake,
    regime_cache: RegimeCacheLookup,
    *,
    force: bool = False,
) -> dict:
    _OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _OUTPUT_CACHE_DIR / f"{spec.key}.pkl"

    if cache_path.exists() and not force:
        print(f"  [{spec.key}] cache exists, skipping (use --no-skip to force)", flush=True)
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        n_setups = sum(len(v) for v in data["setups"].values())
        return {
            "key": spec.key,
            "bars": len(data["bars"]),
            "setups": n_setups,
            "synthetic_hit": data.get("synthetic_hit", 0),
            "elapsed_s": 0.0,
        }

    test_bars = lake.query("XAUUSD", Timeframe.M15, spec.test_start, spec.test_end)
    if test_bars.is_empty():
        print(f"  [{spec.key}] no test bars", flush=True)
        return {"key": spec.key, "bars": 0, "setups": 0, "synthetic_hit": 0, "elapsed_s": 0.0}

    # R7-B2 core change: synthetic_zones_enabled=True.
    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(
        detector=detector,
        ai_regime_enabled=False,
        regime_cache=regime_cache,
        synthetic_zones_enabled=True,
        synthetic_zones_min_historical=2,
    )
    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator,
        lake=lake,
        instrument="XAUUSD",
    )

    train_bars = lake.query("XAUUSD", Timeframe.M15, spec.train_start, spec.train_end)
    strategy.train(train_bars)
    aggregator.clear_cooldowns()
    aggregator.clear_active_zones()

    t0 = time.time()
    setups = strategy.generate_setups(test_bars)
    elapsed = time.time() - t0
    n_setups = sum(len(v) for v in setups.values())

    # Count how many setups came from synthetic zones. Provenance lives on
    # the zone dataclass (R5 A3-R added `provenance`); we probe safely.
    synth_hits = 0
    for tup in setups.values():
        for s in tup:
            prov = getattr(s.zone, "provenance", None) or ""
            if isinstance(prov, str) and prov.startswith("synthetic"):
                synth_hits += 1

    print(
        f"  [{spec.key}] setups={n_setups} (synthetic={synth_hits}) "
        f"in {elapsed:.0f}s ({len(test_bars):,} bars)",
        flush=True,
    )

    with open(cache_path, "wb") as f:
        pickle.dump({
            "setups": setups,
            "bars": test_bars,
            "synthetic_hit": synth_hits,
            "synthetic_zones_enabled": True,
            "synthetic_zones_min_historical": 2,
        }, f, protocol=4)

    return {
        "key": spec.key,
        "bars": len(test_bars),
        "setups": n_setups,
        "synthetic_hit": synth_hits,
        "elapsed_s": elapsed,
    }


def main() -> None:
    years = list(_WINDOW_YEAR_MAP.keys())
    force = "--no-skip" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--years="):
            raw = arg.split("=", 1)[1]
            if "-" in raw:
                a, b = raw.split("-")
                years = list(range(int(a), int(b) + 1))
            else:
                years = [int(raw)]

    print(f"R7-B2 setup cache rebuild with synthetic_zones_enabled=True", flush=True)
    print(f"Years: {years} | Force regen: {force}", flush=True)
    print(f"Output: {_OUTPUT_CACHE_DIR}", flush=True)

    lake = ForexDataLake(PROJECT_ROOT / "data" / "parquet")
    regime_cache = _ensure_regime_cache(lake)

    windows = _build_windows(lake)
    target_window_nums = set()
    for yr in years:
        target_window_nums.update(_WINDOW_YEAR_MAP.get(yr, []))

    selected = [w for w in windows if w.window_num in target_window_nums]
    print(f"Selected windows: {len(selected)}", flush=True)

    t0 = time.time()
    results: list[dict] = []
    for spec in selected:
        results.append(_regenerate_window(spec, lake, regime_cache, force=force))

    print(f"\n=== Summary ===", flush=True)
    total_setups = 0
    total_synth = 0
    for r in results:
        yr = r["key"].split("_")[1][:4] if "_" in r["key"] else "?"
        total_setups += r["setups"]
        total_synth += r["synthetic_hit"]
        print(
            f"  {r['key']}: bars={r['bars']:,} setups={r['setups']} "
            f"synthetic={r['synthetic_hit']} ({r['elapsed_s']:.0f}s)",
            flush=True,
        )
    print(f"\nTotal setups: {total_setups} (synthetic contribution: {total_synth})", flush=True)
    print(f"Total wall time: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
