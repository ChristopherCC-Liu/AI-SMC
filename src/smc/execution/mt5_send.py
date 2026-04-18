"""MT5 order_send 鲁棒 wrapper with retry, dynamic deviation, circuit breaker."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class OrderSendResult:
    success: bool
    retcode: int
    ticket: Optional[int]
    message: str
    attempts: int
    final_price: Optional[float] = None


class MT5Protocol(Protocol):
    """MT5 依赖的最小 protocol，便于 test mock。"""

    def order_send(self, request: dict) -> Any: ...
    def symbol_info(self, symbol: str) -> Any: ...
    def symbol_info_tick(self, symbol: str) -> Any: ...


TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10013
TRADE_RETCODE_MARKET_CLOSED = 10018

# kept as legacy default for backward-compat; new callers should pass circuit_flag_path explicitly
CIRCUIT_OPEN_PATH = Path("data/execution_circuit_open.flag")


def compute_dynamic_deviation(mt5: MT5Protocol, symbol: str, fallback: int = 100) -> int:
    """基于 symbol 当前 spread × 3 动态 deviation，Monday reopen 宽 spread 自适应。

    Cap [20, 500] guards against extreme spread spikes returning absurd deviations.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None or getattr(info, "spread", 0) <= 0:
            return fallback
        dev = int(info.spread * 3)
        return max(20, min(500, dev))
    except Exception:
        return fallback


def refresh_price(mt5: MT5Protocol, symbol: str, is_buy: bool) -> Optional[float]:
    """获取最新 tick ask (BUY) 或 bid (SELL)。None 表示 tick 不可用。"""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        price = getattr(tick, "ask" if is_buy else "bid", 0.0)
        return price if price > 0 else None
    except Exception:
        return None


def send_with_retry(
    mt5: MT5Protocol,
    request: dict,
    max_attempts: int = 3,
    backoff_ms: tuple[int, ...] = (100, 300, 1000),
    circuit_flag_path: "Path | None" = None,
) -> OrderSendResult:
    """MT5 order_send with retry, price refresh, and circuit breaker.

    Backoff schedule: 100 ms → 300 ms → 1 000 ms (last value repeated if max_attempts > 3).
    MARKET_CLOSED (10018) is not retried — the market is genuinely closed (weekend/holiday)
    and retrying would burn attempts for no reason.
    Three consecutive REQUOTE/REJECT/Exception opens the circuit breaker flag so the caller
    knows to stop sending orders and alert the operator.

    ``circuit_flag_path`` defaults to the module-level ``CIRCUIT_OPEN_PATH`` when ``None``
    (legacy backward-compat). New callers should pass the path explicitly.
    """
    flag_path: Path = circuit_flag_path if circuit_flag_path is not None else CIRCUIT_OPEN_PATH
    if flag_path.exists():
        return OrderSendResult(
            success=False,
            retcode=-1,
            ticket=None,
            message=f"Circuit breaker open ({flag_path})",
            attempts=0,
        )

    symbol = request.get("symbol", "")
    is_buy = request.get("type") == getattr(mt5, "ORDER_TYPE_BUY", 0)
    last_retcode = -1
    last_msg = ""
    attempt = 0

    for attempt in range(1, max_attempts + 1):
        fresh_price = refresh_price(mt5, symbol, is_buy)
        if fresh_price is not None:
            request = {**request, "price": fresh_price}

        try:
            result = mt5.order_send(request)
            if result is None:
                last_retcode = -2
                last_msg = "order_send returned None"
            else:
                last_retcode = getattr(result, "retcode", -1)
                last_msg = getattr(result, "comment", "") or ""
                if last_retcode == TRADE_RETCODE_DONE:
                    return OrderSendResult(
                        success=True,
                        retcode=last_retcode,
                        ticket=getattr(result, "order", None),
                        message=last_msg,
                        attempts=attempt,
                        final_price=fresh_price,
                    )
                if last_retcode == TRADE_RETCODE_MARKET_CLOSED:
                    return OrderSendResult(
                        success=False,
                        retcode=last_retcode,
                        ticket=None,
                        message=last_msg,
                        attempts=attempt,
                        final_price=fresh_price,
                    )
        except Exception as e:
            last_retcode = -99
            last_msg = f"exception: {e!r}"

        if attempt < max_attempts:
            delay_idx = min(attempt - 1, len(backoff_ms) - 1)
            time.sleep(backoff_ms[delay_idx] / 1000.0)

    try:
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text(
            f"opened at attempt {attempt}, last retcode {last_retcode}: {last_msg}\n"
        )
    except Exception:
        pass

    return OrderSendResult(
        success=False,
        retcode=last_retcode,
        ticket=None,
        message=last_msg or "unknown",
        attempts=attempt,
    )


def reset_circuit_breaker(circuit_flag_path: Path = CIRCUIT_OPEN_PATH) -> bool:
    """清除 circuit breaker flag。返回是否有文件被删除。"""
    try:
        if circuit_flag_path.exists():
            circuit_flag_path.unlink()
            return True
    except Exception:
        pass
    return False
