"""Tests for the deterministic indicator core library.

Three things matter most and are all covered here:
  1. NO LOOK-AHEAD — indicator[t] computed on the full series equals the same
     indicator computed on the prefix series[:t+1] (its last value).  This is
     the property that makes IC validation and the backtest honest.
  2. REGRESSION PARITY — ATR(sma), SMA-slope, Donchian, VWAP reproduce the
     legacy inline formulas they consolidate.
  3. CORRECTNESS — Wilder RSI/ATR/ADX, NRB, fractals, MA-slope cascade behave.
"""

from __future__ import annotations

import numpy as np
import pytest

from smc.indicators import core

# --- deterministic OHLCV fixture (no RNG) -----------------------------------

_N = 120
_IDX = np.arange(_N, dtype=np.float64)
_BASE = 2000.0 + 30.0 * np.sin(_IDX / 7.0) + 0.5 * _IDX
_CLOSE = _BASE
_HIGH = _BASE + 2.0 + np.abs(np.cos(_IDX / 3.0))
_LOW = _BASE - 2.0 - np.abs(np.sin(_IDX / 5.0))
_VOL = 100.0 + (_IDX % 10) * 5.0

_SAMPLE_T = [30, 55, 80, 99, 115, _N - 1]


# ---------------------------------------------------------------------------
# 1. NO LOOK-AHEAD — the load-bearing property
# ---------------------------------------------------------------------------


def _assert_no_lookahead(fn, series: list[np.ndarray]) -> None:
    full = np.asarray(fn(*series), dtype=np.float64)
    for t in _SAMPLE_T:
        partial = np.asarray(fn(*[s[: t + 1] for s in series]), dtype=np.float64)
        a, b = full[t], partial[-1]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"lookahead leak at t={t}: {a} != {b}"


def test_no_lookahead_moving_averages() -> None:
    _assert_no_lookahead(lambda c: core.sma(c, 20), [_CLOSE])
    _assert_no_lookahead(lambda c: core.ema(c, 20), [_CLOSE])
    _assert_no_lookahead(lambda c: core.rma(c, 14), [_CLOSE])


def test_no_lookahead_atr_variants() -> None:
    _assert_no_lookahead(lambda h, lo, c: core.atr(h, lo, c, 14, "wilder"), [_HIGH, _LOW, _CLOSE])
    _assert_no_lookahead(lambda h, lo, c: core.atr(h, lo, c, 14, "sma"), [_HIGH, _LOW, _CLOSE])
    _assert_no_lookahead(lambda h, lo, c: core.atr_pct(h, lo, c, 14), [_HIGH, _LOW, _CLOSE])


def test_no_lookahead_rsi() -> None:
    _assert_no_lookahead(lambda c: core.rsi(c, 14), [_CLOSE])


def test_no_lookahead_slope() -> None:
    _assert_no_lookahead(lambda c: core.slope_pct_per_bar(c, 5), [_CLOSE])
    _assert_no_lookahead(lambda c: core.sma_slope_pct(c, 50, 5), [_CLOSE])


def test_no_lookahead_channels() -> None:
    _assert_no_lookahead(lambda h, lo: core.donchian(h, lo, 48)[0], [_HIGH, _LOW])
    _assert_no_lookahead(lambda h, lo: core.donchian(h, lo, 48)[1], [_HIGH, _LOW])
    _assert_no_lookahead(
        lambda h, lo, c, v: core.vwap_bands(h, lo, c, v, 20)[0], [_HIGH, _LOW, _CLOSE, _VOL]
    )


def test_no_lookahead_breakout_and_structure() -> None:
    _assert_no_lookahead(lambda h, lo: core.narrow_range_bar(h, lo, 5), [_HIGH, _LOW])
    _assert_no_lookahead(
        lambda h, lo, c: core.nrb_breakout(h, lo, c, 5, 2.0), [_HIGH, _LOW, _CLOSE]
    )
    _assert_no_lookahead(lambda h, lo: core.williams_fractals(h, lo, 2)[0], [_HIGH, _LOW])
    _assert_no_lookahead(lambda h, lo: core.williams_fractals(h, lo, 2)[1], [_HIGH, _LOW])
    _assert_no_lookahead(lambda c: core.ma_slope_cascade(c, 50, 6, 5), [_CLOSE])


def test_no_lookahead_adx() -> None:
    _assert_no_lookahead(lambda h, lo, c: core.adx(h, lo, c, 14)[0], [_HIGH, _LOW, _CLOSE])
    _assert_no_lookahead(lambda h, lo, c: core.adx(h, lo, c, 14)[1], [_HIGH, _LOW, _CLOSE])
    _assert_no_lookahead(lambda h, lo, c: core.adx(h, lo, c, 14)[2], [_HIGH, _LOW, _CLOSE])


# ---------------------------------------------------------------------------
# 2. REGRESSION PARITY with the legacy inline implementations
# ---------------------------------------------------------------------------


def test_atr_sma_matches_legacy_regime_formula() -> None:
    # strategy/regime.py: TR from i=1, atr = mean(last 14 TRs), pct = atr/close*100
    high, low, close = _HIGH.tolist(), _LOW.tolist(), _CLOSE.tolist()
    tr = []
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    legacy_atr = sum(tr[-14:]) / 14
    legacy_pct = legacy_atr / close[-1] * 100.0

    assert core.atr(_HIGH, _LOW, _CLOSE, 14, "sma")[-1] == pytest.approx(legacy_atr)
    assert core.atr_pct(_HIGH, _LOW, _CLOSE, 14, "sma")[-1] == pytest.approx(legacy_pct)


def test_sma_slope_matches_legacy_htf_formula() -> None:
    # htf_bias._sma50_slope_pct_per_bar
    closes = _CLOSE.tolist()
    sma_now = sum(closes[-50:]) / 50
    sma_5ago = sum(closes[-55:-5]) / 50
    legacy = (sma_now - sma_5ago) / sma_now * 100.0 / 5.0
    assert core.sma_slope_pct(_CLOSE, 50, 5)[-1] == pytest.approx(legacy)


def test_donchian_matches_max_min() -> None:
    upper, lower = core.donchian(_HIGH, _LOW, 48)
    assert upper[-1] == pytest.approx(_HIGH[-48:].max())
    assert lower[-1] == pytest.approx(_LOW[-48:].min())


def test_vwap_bands_matches_legacy_synthetic_zone_formula() -> None:
    period = 20
    h, low, c, v = _HIGH[-period:], _LOW[-period:], _CLOSE[-period:], _VOL[-period:]
    typical = (h + low + c) / 3.0
    total = v.sum()
    vwap = float((typical * v).sum() / total)
    var = float((v * (typical - vwap) ** 2).sum() / total)
    std = max(var, 0.0) ** 0.5
    out_vwap, out_up, out_low = core.vwap_bands(_HIGH, _LOW, _CLOSE, _VOL, period)
    assert out_vwap[-1] == pytest.approx(vwap)
    assert out_up[-1] == pytest.approx(vwap + std)
    assert out_low[-1] == pytest.approx(vwap - std)


# ---------------------------------------------------------------------------
# 3. CORRECTNESS
# ---------------------------------------------------------------------------


def test_sma_known_values() -> None:
    out = core.sma([1, 2, 3, 4, 5], 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(4.0)


def test_ema_rma_seed_with_sma() -> None:
    vals = [1, 2, 3, 4, 5, 6]
    assert core.ema(vals, 3)[2] == pytest.approx(2.0)  # seed = SMA(first 3)
    assert core.rma(vals, 3)[2] == pytest.approx(2.0)


def test_rsi_bounds_and_all_up() -> None:
    r = core.rsi(_CLOSE, 14)
    valid = r[~np.isnan(r)]
    assert valid.min() >= 0.0 and valid.max() <= 100.0
    # strictly rising series → RSI saturates at 100 (no losses)
    rising = np.arange(1, 40, dtype=np.float64)
    assert core.rsi(rising, 14)[-1] == pytest.approx(100.0)


def test_adx_non_negative_and_directional() -> None:
    adx, pdi, mdi = core.adx(_HIGH, _LOW, _CLOSE, 14)
    for s in (adx, pdi, mdi):
        valid = s[~np.isnan(s)]
        assert (valid >= -1e-9).all()
    # On a clean uptrend, +DI should dominate -DI at the end.
    up = np.arange(1, 60, dtype=np.float64)
    a_up, p_up, m_up = core.adx(up + 1, up - 1, up, 14)
    assert p_up[-1] > m_up[-1]


def test_nrb_breakout_signal() -> None:
    # Flat compression then a sharp upside break.
    high = np.array([10, 10.1, 10.05, 10.1, 10.0, 12.0])
    low = np.array([9.9, 9.95, 9.9, 9.95, 9.9, 11.5])
    close = np.array([9.95, 10.0, 9.95, 10.0, 9.95, 11.9])
    sig = core.nrb_breakout(high, low, close, lookback=5, mult=2.0)
    assert sig[-1] == 1  # broke well above the NRB expansion band


def test_williams_fractals_confirmed_with_lag() -> None:
    # A clear peak at index 4 with n=2 must be CONFIRMED at index 6, not 4.
    high = np.array([1, 2, 3, 4, 9, 4, 3, 2, 1], dtype=np.float64)
    low = np.array([0, 1, 2, 3, 8, 3, 2, 1, 0], dtype=np.float64)
    up, _ = core.williams_fractals(high, low, n=2)
    assert up[4] == np.bool_(False)  # NOT marked at the centre (would be look-ahead)
    assert up[6] == np.bool_(True)  # confirmed n=2 bars later
    # Last n bars can never host a confirmed fractal.
    assert not up[-1] and not up[-2]


def test_ma_slope_cascade_directions() -> None:
    rising = 2000.0 + np.arange(_N, dtype=np.float64)  # monotone up
    falling = 2000.0 - np.arange(_N, dtype=np.float64)
    assert core.ma_slope_cascade(rising, 50, 6, 5)[-1] == 1
    assert core.ma_slope_cascade(falling, 50, 6, 5)[-1] == -1


def test_classify_slope_thresholds() -> None:
    assert core.classify_slope(0.05) == "up"
    assert core.classify_slope(-0.05) == "down"
    assert core.classify_slope(0.0) == "flat"
    assert core.classify_slope(float("nan")) == "flat"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_bad_period() -> None:
    with pytest.raises(ValueError):
        core.sma(_CLOSE, 0)
    with pytest.raises(ValueError):
        core.atr(_HIGH, _LOW, _CLOSE, -1)


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        core.true_range(_HIGH, _LOW[:-1], _CLOSE)


def test_warmup_is_nan_not_zero() -> None:
    out = core.sma(_CLOSE, 50)
    assert np.isnan(out[:49]).all()
    assert not np.isnan(out[49])
