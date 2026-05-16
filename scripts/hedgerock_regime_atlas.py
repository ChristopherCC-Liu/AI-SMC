"""HedgeRock × Market Regime Empirical Atlas — Capability 1 first cut.

Goal: empirically characterize XAUUSD price behavior in each of the 5 regimes
labeled by AI-SMC's regime_classifier, then estimate HedgeRock-style averaging
fitness per regime.

HedgeRock real-XAUUSD.set key parameters:
- TakeprofitI = 250 points
- PointsBetweenLevels = 250 points (grid spacing)
- ATRMultiplier = 4.0 (grid spacing scales with ATR)
- GearRH = 2.0 (martingale gear)
- MaxLotMultiply = 0.667
- MaxNextLot = 0.5

Output: per-regime statistics on
- bar volatility (ATR-proxy via high-low range)
- forward 12/24/48-bar return distributions
- max favorable / adverse excursion (MFE / MAE) within forward window
- regime persistence (run length)
- HedgeRock-style sweet-spot probability heuristic
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


DATA_ROOT = _ai_smc_home() / "data"
REGIME_CACHE = DATA_ROOT / "regime_cache.parquet"
H1_DIR = DATA_ROOT / "parquet" / "XAUUSD" / "H1"


def load_h1_data() -> pd.DataFrame:
    files = sorted(H1_DIR.glob("*/*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_regime() -> pd.DataFrame:
    df = pd.read_parquet(REGIME_CACHE)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[["ts", "regime"]]


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def forward_excursions(
    df: pd.DataFrame, horizon_bars: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute per-bar forward (return, max favorable, max adverse) over horizon."""
    fwd_ret = df["close"].shift(-horizon_bars) - df["close"]
    fwd_high = (
        df["high"].rolling(horizon_bars, min_periods=1).max().shift(-horizon_bars + 1)
    )
    fwd_low = (
        df["low"].rolling(horizon_bars, min_periods=1).min().shift(-horizon_bars + 1)
    )
    mfe = fwd_high - df["close"]
    mae = df["close"] - fwd_low
    return fwd_ret, mfe, mae


def main() -> None:
    print("Loading XAUUSD H1 data + regime cache...")
    h1 = load_h1_data()
    regime = load_regime()

    print(f"H1 bars: {len(h1)} | regime bars: {len(regime)}")
    print(f"H1 range: {h1['ts'].min()} → {h1['ts'].max()}")
    print(f"Regime range: {regime['ts'].min()} → {regime['ts'].max()}")

    # Inner join — only keep H1 bars that have regime labels
    df = h1.merge(regime, on="ts", how="inner")
    print(f"\nJoined: {len(df)} bars with regime labels")
    print(f"Joined range: {df['ts'].min()} → {df['ts'].max()}")

    # ATR computation
    df["atr14"] = compute_atr(df, 14)

    # Forward excursions over multiple horizons
    for h in (12, 24, 48):
        fwd_ret, mfe, mae = forward_excursions(df, h)
        df[f"fwd_ret_{h}"] = fwd_ret
        df[f"mfe_{h}"] = mfe
        df[f"mae_{h}"] = mae

    # HedgeRock real-XAUUSD.set TP target = 250 points
    # On XAUUSD with _Point=0.01: 250 pts = $2.50 (close range scale)
    # On XAUUSD with _Point=0.10: 250 pts = $25 (wider grid)
    # Most live brokers use 0.01 so we test both interpretations
    TP_USD_001 = 2.5  # if _Point = 0.01
    TP_USD_010 = 25.0  # if _Point = 0.10

    print("\n" + "=" * 80)
    print("REGIME CHARACTERIZATION — XAUUSD H1")
    print("=" * 80)

    # Per-regime stats
    regime_stats = []
    for regime_name in sorted(df["regime"].dropna().unique()):
        sub = df[df["regime"] == regime_name].copy()
        n = len(sub)
        if n < 50:
            continue

        avg_atr = sub["atr14"].mean()
        atr_pctile = sub["atr14"].quantile([0.1, 0.5, 0.9]).values

        # Forward 24-bar (1 day) excursion statistics
        mfe24 = sub["mfe_24"].dropna()
        mae24 = sub["mae_24"].dropna()
        fwd24 = sub["fwd_ret_24"].dropna()

        # Probability HedgeRock would reach TP in 24 bars
        # (assuming entry at current bar, TP=$2.50 above for buy / below for sell)
        p_tp_buy = (mfe24 >= TP_USD_001).mean()  # buy MFE reached $2.50 above
        p_tp_sell = (mae24 >= TP_USD_001).mean()  # sell MFE reached $2.50 below

        # Prob extreme adverse — heuristic for "blowup risk"
        # MAE > 4 * ATR means price moved 4 ATRs against us = grid heavy load
        p_extreme_adverse = (mae24 > 4 * avg_atr).mean()

        regime_stats.append(
            {
                "regime": regime_name,
                "n_bars": n,
                "avg_atr": avg_atr,
                "atr_p10": atr_pctile[0],
                "atr_p50": atr_pctile[1],
                "atr_p90": atr_pctile[2],
                "fwd24_mean": fwd24.mean(),
                "fwd24_std": fwd24.std(),
                "mfe24_mean": mfe24.mean(),
                "mae24_mean": mae24.mean(),
                "p_tp_buy_24": p_tp_buy,
                "p_tp_sell_24": p_tp_sell,
                "p_extreme_adverse_24": p_extreme_adverse,
            }
        )

    stats_df = pd.DataFrame(regime_stats)
    print(stats_df.to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 80)
    print("HEDGEROCK SWEET-SPOT HEURISTIC")
    print("=" * 80)
    print("Sweet spot ≈ high P(reach TP) + low P(extreme adverse)")
    print("Danger ≈ low P(reach TP) and/or high P(extreme adverse)")
    print()
    for _, row in stats_df.iterrows():
        # Heuristic score: average of (p_tp_buy, p_tp_sell) - p_extreme_adverse
        avg_tp = (row["p_tp_buy_24"] + row["p_tp_sell_24"]) / 2
        score = avg_tp - row["p_extreme_adverse_24"]
        verdict = (
            "SWEET" if score > 0.5 else ("OK" if score > 0.2 else ("CAUTION" if score > 0 else "DANGER"))
        )
        print(
            f"  {row['regime']:15s}  avg_TP={avg_tp:.3f}  "
            f"adverse={row['p_extreme_adverse_24']:.3f}  score={score:+.3f}  "
            f"→ {verdict}"
        )

    # Save full atlas
    out_path = DATA_ROOT.parent / "reports" / "hedgerock_regime_atlas.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(out_path, index=False)
    print(f"\nAtlas saved: {out_path}")


if __name__ == "__main__":
    main()
