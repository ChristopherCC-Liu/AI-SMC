"""Round 4.6-X connectivity test: verify mt5.order_send() chain reachable.

Usage (on VPS):
    .venv\\Scripts\\python.exe scripts\\test_mt5_order_connectivity.py

Expected outcomes:
- Market CLOSED (weekend): retcode=10018 (TRADE_RETCODE_MARKET_CLOSED) ← SUCCESS for connectivity
- Market OPEN but invalid: retcode=10030/10027 (various) ← still SUCCESS for connectivity
- Exception raised: FAILURE → investigate env / symbol / credentials

This test does NOT commit a real order when market is closed. It proves the
full request construction + transmission path works end-to-end.
"""
from __future__ import annotations

import os

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import MetaTrader5 as mt5


def main() -> int:
    print("=" * 60)
    print(" Round 4.6-X MT5 Order Connectivity Test")
    print("=" * 60)

    if not mt5.initialize():
        print(f"FAIL init: {mt5.last_error()}")
        return 1

    acct = mt5.account_info()
    if acct is None:
        print("FAIL account_info: None")
        mt5.shutdown()
        return 1

    print(f"Account:      {acct.login}")
    print(f"Server:       {acct.server}")
    print(f"Balance:      ${acct.balance}")
    print(f"Equity:       ${acct.equity}")

    tick = mt5.symbol_info_tick("XAUUSD")
    if tick is None:
        print("FAIL tick: XAUUSD not available")
        mt5.shutdown()
        return 1

    print(f"XAUUSD tick:  bid=${tick.bid}  ask=${tick.ask}")

    # Build a tiny 0.01 lot SELL order with SL/TP far from price
    # (won't fill anyway because market closed)
    sell_price = tick.bid if tick.bid > 0 else 4834.50
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_SELL,
        "price": sell_price,
        "sl": sell_price + 50.0,  # wide SL
        "tp": sell_price - 50.0,  # wide TP
        "deviation": 20,
        "magic": 19760418,
        "comment": "4.6-X smoke",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print()
    print("Sending test order ...")
    print(f"  symbol:       {request['symbol']}")
    print(f"  volume:       {request['volume']}")
    print(f"  type:         SELL")
    print(f"  price:        ${request['price']}")
    print(f"  sl:           ${request['sl']}")
    print(f"  tp:           ${request['tp']}")
    print(f"  magic:        {request['magic']}")

    try:
        result = mt5.order_send(request)
    except Exception as exc:
        print(f"FAIL order_send exception: {type(exc).__name__}: {exc}")
        mt5.shutdown()
        return 1

    if result is None:
        print(f"FAIL order_send result None: {mt5.last_error()}")
        mt5.shutdown()
        return 1

    print()
    print(f"RESULT retcode:  {result.retcode}")
    print(f"RESULT comment:  {result.comment!r}")
    print(f"RESULT ticket:   {result.order}")

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print()
        print("✓ ORDER EXECUTED (market was open). Closing it immediately ...")
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "position": result.order,
            "deviation": 20,
            "magic": 19760418,
            "comment": "4.6-X smoke close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        close_result = mt5.order_send(close_request)
        if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✓ Close executed. Position {result.order} cleared.")
        else:
            print(f"⚠ Close FAILED retcode={close_result.retcode if close_result else None}")
            print(f"  Manually close ticket {result.order} via MT5 terminal!")
    elif result.retcode == 10018:
        print()
        print("✓ CONNECTIVITY OK — market closed (10018). Order would execute on market open.")
    else:
        print()
        print(f"⚠ retcode={result.retcode} — check MT5 docs for meaning")
        print(f"  Common codes: 10004=REQUOTE, 10018=MARKET_CLOSED, 10030=UNSUPPORTED_FILLING,")
        print(f"                 10027=AUTOTRADING_DISABLED, 10013=INVALID_REQUEST")
        print(f"  Connectivity was reached but request rejected")

    mt5.shutdown()
    print()
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
