# Apr 21 2026 09:47 UTC — Range-Break BUY Incident Fixture

## What this fixture proves

That `smc.strategy.range_trader._trend_filter_should_block`, when wired
through `RangeTrader(trend_filter_enabled=True)` (R9 P0-A, enabled in
production via `SMC_RANGE_TREND_FILTER_ENABLED=true`), correctly blocks a
long support_bounce setup when the D1 trend is materially down:

- D1 SMA50 slope ≤ -0.05 %/bar
- D1 close ≤ -1.0% below SMA50

Both conditions held on Apr 21 09:47 UTC (slope -0.057, close -2.1%),
yet RangeTrader still emitted a long BUY because the trend filter was
not enabled at the time. The R9 P0-A patch added the filter; this
fixture pins that behaviour into the regression suite.

## State semantics

`state.json` records the pre-computed trend signals so the test does
not need to ship 50+ D1 bars. The two floats are what
`_sma50_slope_pct_per_bar` would have returned for the production D1
sequence at 09:47 UTC.

| Field                                  | Meaning                                  |
| -------------------------------------- | ---------------------------------------- |
| `candidate_setup.direction`            | "long" — the would-be RANGE BUY.         |
| `trend_signals.slope_pct_per_bar`      | -0.057 (bearish, exceeds 0.05 threshold) |
| `trend_signals.close_vs_sma50_pct`     | -2.1   (close below SMA50, exceeds 1.0)  |
| `thresholds.slope_threshold_pct_per_bar` | 0.05 — for documentation only          |
| `thresholds.close_threshold_pct`       | 1.0   — for documentation only           |

The fixture is **synthetic** — production OHLCV bars are private and
out of scope for this repo. The numerical values match the production
incident; the bars themselves do not.

## How to regenerate

`state.json` is hand-edited. If R-future updates the thresholds, edit
the `trend_signals` numbers to keep both magnitudes safely above the
new thresholds and update `thresholds.*` for documentation.

## Assertions

`test_apr_21_range_break.py` runs two complementary tests:

1. **Fix enabled** — calls `_trend_filter_should_block(direction,
   slope, close)`; outcome must be: `True` (block). Wrapped in a
   stable assertion that does NOT depend on threshold constants.
2. **Baseline without fix** — calls a version that ignores the trend
   signals (mirroring pre-R9-P0-A behaviour) and asserts the long BUY
   is blocked. Marked `pytest.mark.xfail(strict=True, reason="documents
   the bug")`. Strict ensures that if a future round adds another
   defensive layer that catches this case via a different path, we get
   notified for explicit review.
