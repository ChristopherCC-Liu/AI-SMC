"""Deterministic, no-look-ahead indicator primitives — the single source of
truth shared by the Python backtester/engine and the MQL5 EA.

DESIGN CONTRACT
---------------
* Every function is a pure transform of array-like price inputs to a NumPy
  float64 series of the SAME length, with ``np.nan`` during the warm-up.
* ``indicator[t]`` depends ONLY on inputs ``[0 .. t]`` — never the future.
  This is what makes the IC validation (``smc.research.ic``) honest and the
  backtest faithful to live.
* Wilder-smoothed indicators (ATR/RSI/ADX) match MetaTrader's ``iATR/iRSI/
  iADX`` so the Python engine and the EA agree bar-for-bar.
* The "latest" value an engine needs is simply ``series[-1]``.

This module consolidates ATR / SMA / slope / Donchian / VWAP that were
previously re-implemented inline in ``strategy/regime.py``,
``strategy/htf_bias.py``, ``ai/regime_classifier.py``,
``strategy/range_trader.py`` and ``smc_core/synthetic_zones.py`` (DRY fix),
and adds the prioritised new primitives: RSI, ADX/+DI/-DI, EMA, narrow-range
-bar breakout, Williams fractals, and the HedgeRock-style MA-slope cascade.

All prices are in instrument price units (XAUUSD: 1 point = $0.01).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

FloatArray: TypeAlias = NDArray[np.float64]
ArrayLike: TypeAlias = "NDArray[np.float64] | list[float] | tuple[float, ...]"

__all__ = [
    "as_array",
    "sma",
    "ema",
    "rma",
    "true_range",
    "atr",
    "atr_pct",
    "rsi",
    "adx",
    "slope_pct_per_bar",
    "sma_slope_pct",
    "classify_slope",
    "donchian",
    "vwap_bands",
    "narrow_range_bar",
    "nrb_breakout",
    "williams_fractals",
    "ma_slope_cascade",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def as_array(values: ArrayLike) -> FloatArray:
    """Coerce any array-like (list, tuple, np.ndarray, polars Series) to a
    contiguous float64 NumPy array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D series, got shape {arr.shape}")
    return arr


def _check_period(period: int) -> None:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")


def _same_length(*arrays: FloatArray) -> None:
    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError("all input series must have the same length")


def _rolling(
    values: FloatArray, period: int, fn: Callable[..., FloatArray]
) -> FloatArray:
    """Generic point-in-time rolling reduction with NaN warm-up."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, period)
    out[period - 1 :] = fn(windows, axis=1)
    return out


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(values: ArrayLike, period: int) -> FloatArray:
    """Simple moving average (NaN before ``period`` samples)."""
    _check_period(period)
    return _rolling(as_array(values), period, np.mean)


def ema(values: ArrayLike, period: int) -> FloatArray:
    """Exponential moving average, seeded with the SMA of the first window.

    alpha = 2 / (period + 1).  Matches MT5 ``iMA(MODE_EMA)``.
    """
    _check_period(period)
    v = as_array(values)
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = v[:period].mean()
    for i in range(period, n):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def rma(values: ArrayLike, period: int) -> FloatArray:
    """Wilder's smoothing (a.k.a. RMA/SMMA), seeded with the first SMA.

    alpha = 1 / period.  This is the smoothing MetaTrader uses internally for
    ATR/RSI/ADX, so engine and EA stay in parity.
    """
    _check_period(period)
    v = as_array(values)
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    out[period - 1] = v[:period].mean()
    inv = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] + inv * (v[i] - out[i - 1])
    return out


# ---------------------------------------------------------------------------
# Volatility: True Range / ATR
# ---------------------------------------------------------------------------


def true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> FloatArray:
    """True Range series.  TR[0] = high[0] - low[0]; thereafter
    max(H-L, |H-Cprev|, |L-Cprev|)."""
    h, low_a, c = as_array(high), as_array(low), as_array(close)
    _same_length(h, low_a, c)
    n = len(h)
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = h[0] - low_a[0]
    if n > 1:
        prev_c = c[:-1]
        hl = h[1:] - low_a[1:]
        hc = np.abs(h[1:] - prev_c)
        lc = np.abs(low_a[1:] - prev_c)
        tr[1:] = np.maximum.reduce([hl, hc, lc])
    return tr


def atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
    method: Literal["wilder", "sma"] = "wilder",
) -> FloatArray:
    """Average True Range.

    ``method="wilder"`` (default) matches MT5 ``iATR``; ``method="sma"``
    reproduces the legacy inline implementation in ``strategy/regime.py``
    (SMA of the last ``period`` true ranges) for regression parity.
    """
    tr = true_range(high, low, close)
    if method == "wilder":
        return rma(tr, period)
    if method == "sma":
        return sma(tr, period)
    raise ValueError(f"unknown atr method {method!r}")


def atr_pct(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
    method: Literal["wilder", "sma"] = "wilder",
) -> FloatArray:
    """ATR expressed as a percentage of close — the regime volatility metric."""
    a = atr(high, low, close, period, method)
    c = as_array(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0.0, a / c * 100.0, np.nan)


# ---------------------------------------------------------------------------
# Momentum: RSI
# ---------------------------------------------------------------------------


def rsi(close: ArrayLike, period: int = 14) -> FloatArray:
    """Wilder's RSI (0-100).  Matches MT5 ``iRSI``.

    RSI = 100 - 100 / (1 + RMA(gains) / RMA(losses)).  When average loss is
    zero the value saturates at 100.
    """
    _check_period(period)
    c = as_array(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    delta = np.diff(c)
    gains = np.where(delta > 0.0, delta, 0.0)
    losses = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    # rma over the diff series (length n-1); align back to close index (+1).
    for i in range(period, n):
        g = avg_gain[i - 1]
        loss_val = avg_loss[i - 1]
        if np.isnan(g) or np.isnan(loss_val):
            continue
        if loss_val == 0.0:
            out[i] = 100.0
        else:
            rs = g / loss_val
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


# ---------------------------------------------------------------------------
# Trend strength: ADX / +DI / -DI (Wilder)
# ---------------------------------------------------------------------------


def adx(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wilder's ADX with directional indicators.  Matches MT5 ``iADX``.

    Returns ``(adx, plus_di, minus_di)``, each a NaN-padded float64 series.
    +DI/-DI become available after ``period`` bars; ADX after ``2*period``.
    """
    _check_period(period)
    h, low_a, c = as_array(high), as_array(low), as_array(close)
    _same_length(h, low_a, c)
    n = len(h)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    adx_out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return adx_out, plus_di, minus_di

    up_move = h[1:] - h[:-1]
    down_move = low_a[:-1] - low_a[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    tr = true_range(h, low_a, c)[1:]  # align to the diff series

    atr_s = rma(tr, period)
    plus_dm_s = rma(plus_dm, period)
    minus_dm_s = rma(minus_dm, period)

    dx = np.full(len(tr), np.nan, dtype=np.float64)
    for i in range(len(tr)):
        a = atr_s[i]
        if np.isnan(a) or a == 0.0:
            continue
        pdi = 100.0 * plus_dm_s[i] / a
        mdi = 100.0 * minus_dm_s[i] / a
        plus_di[i + 1] = pdi
        minus_di[i + 1] = mdi
        denom = pdi + mdi
        if denom > 0.0:
            dx[i] = 100.0 * abs(pdi - mdi) / denom

    adx_s = rma(dx, period)
    adx_out[1:] = adx_s
    return adx_out, plus_di, minus_di


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------


def slope_pct_per_bar(values: ArrayLike, lookback: int = 5) -> FloatArray:
    """Normalised slope (%/bar): (v[t] - v[t-lookback]) / v[t] * 100 / lookback.

    Scale-free rate matching the SMA50-slope formula used across the codebase
    (``htf_bias._sma50_slope_pct_per_bar``).
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    v = as_array(values)
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback, n):
        now = v[i]
        past = v[i - lookback]
        if np.isnan(now) or np.isnan(past) or now == 0.0:
            continue
        out[i] = (now - past) / now * 100.0 / lookback
    return out


def sma_slope_pct(close: ArrayLike, period: int = 50, lookback: int = 5) -> FloatArray:
    """SMA(period) normalised slope in %/bar — the HTF trend-direction metric.

    Reproduces ``htf_bias._sma50_slope_pct_per_bar`` /
    ``regime_classifier._sma50_direction_and_slope`` exactly.
    """
    return slope_pct_per_bar(sma(close, period), lookback)


def classify_slope(
    slope: float, up: float = 0.02, down: float = -0.02
) -> Literal["up", "down", "flat"]:
    """Map a %/bar slope to a direction label (defaults match the codebase)."""
    if np.isnan(slope):
        return "flat"
    if slope > up:
        return "up"
    if slope < down:
        return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Channels: Donchian / VWAP bands
# ---------------------------------------------------------------------------


def donchian(high: ArrayLike, low: ArrayLike, lookback: int = 48) -> tuple[FloatArray, FloatArray]:
    """Donchian channel: rolling ``(max(high), min(low))`` over ``lookback`` bars.

    Reproduces ``range_trader`` Donchian (max H / min L over N bars).
    """
    _check_period(lookback)
    h, low_a = as_array(high), as_array(low)
    _same_length(h, low_a)
    upper = _rolling(h, lookback, np.max)
    lower = _rolling(low_a, lookback, np.min)
    return upper, lower


def vwap_bands(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike | None = None,
    period: int = 20,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Rolling VWAP ± 1 volume-weighted std band over ``period`` bars.

    Typical price = (H+L+C)/3; volume-weighted variance around VWAP.
    Reproduces ``synthetic_zones._vwap_bands`` for the latest window and
    extends it to a full point-in-time series.  Returns ``(vwap, upper, lower)``.
    If ``volume`` is None, unit volume is assumed.
    """
    _check_period(period)
    h, low_a, c = as_array(high), as_array(low), as_array(close)
    _same_length(h, low_a, c)
    n = len(h)
    typical = (h + low_a + c) / 3.0
    vol = np.ones(n, dtype=np.float64) if volume is None else as_array(volume)
    _same_length(typical, vol)

    vwap = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return vwap, upper, lower

    tp_win = np.lib.stride_tricks.sliding_window_view(typical, period)
    v_win = np.lib.stride_tricks.sliding_window_view(vol, period)
    total = v_win.sum(axis=1)
    safe = total > 0.0
    vw = np.full(len(total), np.nan, dtype=np.float64)
    vw[safe] = (tp_win[safe] * v_win[safe]).sum(axis=1) / total[safe]
    var = np.full(len(total), np.nan, dtype=np.float64)
    diff2 = (tp_win - vw[:, None]) ** 2
    var[safe] = (v_win[safe] * diff2[safe]).sum(axis=1) / total[safe]
    std = np.sqrt(np.clip(var, 0.0, None))
    vwap[period - 1 :] = vw
    upper[period - 1 :] = vw + std
    lower[period - 1 :] = vw - std
    return vwap, upper, lower


# ---------------------------------------------------------------------------
# Breakout / structure: Narrow-Range-Bar, Williams fractals, MA-slope cascade
# ---------------------------------------------------------------------------


def narrow_range_bar(high: ArrayLike, low: ArrayLike, lookback: int = 5) -> NDArray[np.bool_]:
    """True where the current bar's range is the smallest over the last
    ``lookback`` bars (an NRB — volatility compression).  Point-in-time."""
    _check_period(lookback)
    h, low_a = as_array(high), as_array(low)
    _same_length(h, low_a)
    rng = h - low_a
    n = len(rng)
    out = np.zeros(n, dtype=np.bool_)
    if n < lookback:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(rng, lookback)
    out[lookback - 1 :] = rng[lookback - 1 :] <= windows.min(axis=1)
    return out


def nrb_breakout(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    lookback: int = 5,
    mult: float = 2.0,
) -> NDArray[np.int8]:
    """HedgeRock-style narrow-range-bar volatility-expansion breakout.

    For each bar, anchor on the smallest-range bar in the trailing ``lookback``
    window (the NRB).  Signal:
      +1 when close breaks above ``nrb_low + mult * nrb_range``
      -1 when close breaks below ``nrb_high - mult * nrb_range``
       0 otherwise.
    Point-in-time: the NRB anchor uses only the trailing window.
    """
    _check_period(lookback)
    h, low_a, c = as_array(high), as_array(low), as_array(close)
    _same_length(h, low_a, c)
    n = len(h)
    out = np.zeros(n, dtype=np.int8)
    if n < lookback:
        return out
    rng = h - low_a
    hw = np.lib.stride_tricks.sliding_window_view(h, lookback)
    lw = np.lib.stride_tricks.sliding_window_view(low_a, lookback)
    rw = np.lib.stride_tricks.sliding_window_view(rng, lookback)
    for t in range(lookback - 1, n):
        w = t - (lookback - 1)
        j = int(np.argmin(rw[w]))
        nrb_high = hw[w][j]
        nrb_low = lw[w][j]
        nrb_range = nrb_high - nrb_low
        if nrb_range <= 0.0:
            continue
        if c[t] > nrb_low + mult * nrb_range:
            out[t] = 1
        elif c[t] < nrb_high - mult * nrb_range:
            out[t] = -1
    return out


def williams_fractals(
    high: ArrayLike, low: ArrayLike, n: int = 2
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Bill Williams fractals, CONFIRMED with no look-ahead.

    An up (resistance) fractal is centred on a bar whose high is the strict max
    of the ``n`` bars each side; a down (support) fractal likewise for lows.
    Confirmation requires ``n`` future bars, so the boolean is placed at the
    confirmation bar ``center + n`` (NOT the centre) — i.e. ``up[t] = True``
    means "an up-fractal centred at t-n is confirmed as of bar t".  This keeps
    the series strictly point-in-time.

    Returns ``(up_fractal_confirmed, down_fractal_confirmed)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    h, low_a = as_array(high), as_array(low)
    _same_length(h, low_a)
    length = len(h)
    up = np.zeros(length, dtype=np.bool_)
    down = np.zeros(length, dtype=np.bool_)
    for center in range(n, length - n):
        seg_h = h[center - n : center + n + 1]
        seg_l = low_a[center - n : center + n + 1]
        if h[center] == seg_h.max() and (seg_h == h[center]).sum() == 1:
            up[center + n] = True
        if low_a[center] == seg_l.min() and (seg_l == low_a[center]).sum() == 1:
            down[center + n] = True
    return up, down


def ma_slope_cascade(
    close: ArrayLike,
    primary_period: int = 192,
    fast_period: int = 6,
    lookback: int = 5,
    grad_min_pct: float = 0.0,
) -> NDArray[np.int8]:
    """HedgeRock ``FindMA``-style cascade: agreement of a slow trend MA and a
    fast confirmation MA's slope.

    +1 when BOTH the primary SMA slope and the fast SMA slope exceed
    ``+grad_min_pct`` (%/bar); -1 when both fall below ``-grad_min_pct``;
    0 otherwise.  Deterministic, point-in-time.
    """
    slow = slope_pct_per_bar(sma(close, primary_period), lookback)
    fast = slope_pct_per_bar(sma(close, fast_period), lookback)
    n = len(slow)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        s, f = slow[i], fast[i]
        if np.isnan(s) or np.isnan(f):
            continue
        if s > grad_min_pct and f > grad_min_pct:
            out[i] = 1
        elif s < -grad_min_pct and f < -grad_min_pct:
            out[i] = -1
    return out
