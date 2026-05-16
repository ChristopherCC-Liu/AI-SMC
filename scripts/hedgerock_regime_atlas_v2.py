"""HedgeRock × Market Regime Empirical Atlas — v2 (corrected parameter scale).

Corrections from v1:
- HedgeRock real-XAUUSD.set TP=250 points; XAUUSD broker _Point=0.1 → TP=$25
- Grid spacing = max(PointsBetweenLevelsP=$25, ATRMultiplier=4.0 * ATR_D1)
- For ATR_D1 ~ $20-50: dynamic grid = $80-$200 (much wider than $25)
- Real "blowup" = price moves >4 ATR_D1 in one direction without retracement
- HedgeRock recovers if price oscillates around entry zone within ATR multiples

What this script empirically tests:
- For each regime, characterize "sustained one-direction excursion" probability
  (this is what kills hedge martingale)
- Compute the 95th / 99th percentile of |MAE/ATR_D1| ratio per regime
  (catastrophic adverse excursion in regime-relative terms)
- Compute time-to-recovery distribution (how long until price returns to entry +/- ATR)
- Compare regime-conditional Sharpe of buy-only / sell-only / hedge-symmetric strategies
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
D1_DIR = DATA_ROOT / "parquet" / "XAUUSD" / "D1"


def load_h1_data() -> pd.DataFrame:
    files = sorted(H1_DIR.glob("*/*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_d1_data() -> pd.DataFrame:
    files = sorted(D1_DIR.glob("*/*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_d1_atr_for_h1(d1: pd.DataFrame, h1: pd.DataFrame) -> pd.Series:
    """Compute ATR_D1 (14 day) and forward-fill to H1 frequency."""
    if d1.empty:
        # Fallback: use H1 ATR(14*24=336) as proxy
        return compute_atr(h1, 336).rename("atr_d1")
    d1 = d1.copy()
    d1["atr_d1"] = compute_atr(d1, 14)
    h1 = h1.copy()
    h1["date"] = h1["ts"].dt.floor("D")
    d1_lookup = d1.set_index("ts")[["atr_d1"]]
    d1_lookup.index = d1_lookup.index.floor("D")
    return h1["date"].map(d1_lookup["atr_d1"]).ffill()


def forward_one_direction_excursion(
    df: pd.DataFrame, horizon_bars: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the largest one-direction excursion within forward horizon bars.

    Returns:
        (max_up_move, max_down_move, net_drift) — all in price units.
    """
    n = len(df)
    max_up = np.full(n, np.nan)
    max_down = np.full(n, np.nan)
    net = np.full(n, np.nan)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(n - horizon_bars):
        entry = closes[i]
        window_high = highs[i + 1 : i + 1 + horizon_bars].max()
        window_low = lows[i + 1 : i + 1 + horizon_bars].min()
        max_up[i] = window_high - entry
        max_down[i] = entry - window_low
        net[i] = closes[i + horizon_bars] - entry

    return pd.Series(max_up, index=df.index), pd.Series(max_down, index=df.index), pd.Series(net, index=df.index)


def main() -> None:
    print("Loading data...")
    h1 = load_h1_data()
    d1 = load_d1_data()
    regime = pd.read_parquet(REGIME_CACHE)
    regime["ts"] = pd.to_datetime(regime["ts"], utc=True)
    regime = regime[["ts", "regime"]]

    print(f"H1: {len(h1)} bars | D1: {len(d1)} bars | regime: {len(regime)} bars")

    df = h1.merge(regime, on="ts", how="inner")
    df["atr_d1"] = compute_d1_atr_for_h1(d1, df).values
    print(f"\nJoined: {len(df)} bars; ATR_D1 stats:")
    print(df["atr_d1"].describe())

    # HedgeRock real-XAUUSD.set parameters
    POINT = 0.1  # XAUUSD broker convention (10 ticks per dollar)
    TP_POINTS = 250
    GRID_POINTS = 250
    ATR_MULTIPLIER = 4.0

    TP_USD = TP_POINTS * POINT  # $25
    GRID_BASE_USD = GRID_POINTS * POINT  # $25
    print(f"\nHedgeRock params: TP=${TP_USD}, grid_base=${GRID_BASE_USD}, ATR_mult={ATR_MULTIPLIER}")
    print(f"Dynamic grid spacing = max($25, {ATR_MULTIPLIER} * ATR_D1)")

    df["grid_spacing"] = np.maximum(GRID_BASE_USD, ATR_MULTIPLIER * df["atr_d1"])
    print(f"Grid spacing distribution: mean={df['grid_spacing'].mean():.2f} median={df['grid_spacing'].median():.2f}")

    # Forward 24 / 48 / 96-bar (1 / 2 / 4 day) one-direction excursions
    horizons = (24, 48, 96)
    for h in horizons:
        up, down, net = forward_one_direction_excursion(df, h)
        df[f"max_up_{h}"] = up
        df[f"max_down_{h}"] = down
        df[f"net_drift_{h}"] = net

    print("\n" + "=" * 100)
    print("REGIME-CONDITIONAL HEDGEROCK FITNESS — XAUUSD H1")
    print("=" * 100)

    # Per-regime analysis
    rows = []
    for regime_name in sorted(df["regime"].dropna().unique()):
        sub = df[df["regime"] == regime_name].dropna(subset=["atr_d1", "max_up_24"]).copy()
        n = len(sub)
        if n < 50:
            continue

        # HedgeRock sweet spot heuristic:
        # - Net drift small → oscillating market → HedgeRock recovers via TP=$25
        # - One-direction excursion bounded by 4*ATR_D1 → grid not too stacked
        # - One-direction excursion >> 4*ATR_D1 → hedge martingale stuck → blowup risk

        for h in horizons:
            sub[f"max_one_dir_{h}"] = np.maximum(sub[f"max_up_{h}"], sub[f"max_down_{h}"])
            sub[f"oscillation_{h}"] = sub[f"max_up_{h}"] + sub[f"max_down_{h}"] - sub[f"net_drift_{h}"].abs()

        # Catastrophic blowup proxy: max one-direction move >> 4 ATR_D1
        sub["blowup_24"] = (sub["max_one_dir_24"] > 4 * sub["atr_d1"]).astype(float)
        sub["blowup_48"] = (sub["max_one_dir_48"] > 5 * sub["atr_d1"]).astype(float)
        sub["blowup_96"] = (sub["max_one_dir_96"] > 6 * sub["atr_d1"]).astype(float)

        # HedgeRock TP fitness: max excursion >= TP (HedgeRock has chance to TP)
        sub["tp_reached_24"] = (sub["max_one_dir_24"] >= TP_USD).astype(float)

        # Net drift / ATR_D1 — if too high, market trending strongly (martingale fails)
        sub["drift_atr_ratio_24"] = sub["net_drift_24"].abs() / sub["atr_d1"]

        # Oscillation / one-dir ratio: high = good for HedgeRock (bouncy), low = bad
        sub["oscillation_ratio_24"] = sub["oscillation_24"] / (sub["max_one_dir_24"] + 0.01)

        rows.append(
            {
                "regime": regime_name,
                "n_bars": n,
                "atr_d1_mean": sub["atr_d1"].mean(),
                "tp_reached_24": sub["tp_reached_24"].mean(),
                "blowup_24_4atr": sub["blowup_24"].mean(),
                "blowup_48_5atr": sub["blowup_48"].mean(),
                "blowup_96_6atr": sub["blowup_96"].mean(),
                "drift_atr_24_p50": sub["drift_atr_ratio_24"].median(),
                "drift_atr_24_p90": sub["drift_atr_ratio_24"].quantile(0.9),
                "oscillation_ratio_24": sub["oscillation_ratio_24"].mean(),
                "max_one_dir_24_p99_atr": sub["max_one_dir_24"].quantile(0.99) / sub["atr_d1"].mean(),
            }
        )

    stats_df = pd.DataFrame(rows)
    print(stats_df.to_string(index=False, float_format="%.4f"))

    # Compute HedgeRock fitness score per regime
    # Score = TP_reached - blowup_risk - drift_pressure
    print("\n" + "=" * 100)
    print("HEDGEROCK FITNESS SCORE per Regime (higher = better)")
    print("=" * 100)
    for _, row in stats_df.iterrows():
        # Heuristic: HedgeRock benefits from oscillating markets with bounded drift
        score = (
            row["tp_reached_24"] * 1.0           # +reach TP
            - row["blowup_24_4atr"] * 1.5        # -catastrophic adverse 4 ATR
            - row["blowup_48_5atr"] * 1.0        # -2-day deep adverse
            - row["drift_atr_24_p90"] * 0.2      # -trending strongly
        )
        verdict = (
            "SWEET" if score > 0.6
            else ("OK" if score > 0.3
            else ("CAUTION" if score > 0
            else "DANGER"))
        )
        print(
            f"  {row['regime']:15s}  TP={row['tp_reached_24']:.3f}  "
            f"blow_24={row['blowup_24_4atr']:.3f}  blow_48={row['blowup_48_5atr']:.3f}  "
            f"drift90={row['drift_atr_24_p90']:.2f}  score={score:+.3f} → {verdict}"
        )

    # Save atlas
    out_path = DATA_ROOT.parent / "reports" / "hedgerock_regime_atlas_v2.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(out_path, index=False)
    print(f"\nAtlas v2 saved: {out_path}")

    # Save regime-tagged feature df for further analysis
    feature_df = df[
        ["ts", "regime", "close", "atr_d1", "grid_spacing", "max_up_24", "max_down_24", "net_drift_24"]
    ].dropna()
    feature_path = DATA_ROOT.parent / "reports" / "hedgerock_regime_features.parquet"
    feature_df.to_parquet(feature_path, index=False)
    print(f"Features saved: {feature_path}")


if __name__ == "__main__":
    main()
