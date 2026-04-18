"""audit-r3 O7: MT5BrokerPort.magic is cfg-driven.

Before: executor.py had two `"magic": 202500` hardcodes (send_order +
close_position) that would cause XAU/BTC position reconciliation to
clash if anyone ever switched the live path from scripts/live_demo.py
(direct mt5.order_send) back to the BrokerPort abstraction.

After: __init__ takes `cfg: InstrumentConfig | None = None` and reads
`cfg.magic`.  None → _DEFAULT_MAGIC (202500) preserves backward compat.

These tests stub `sys.modules['MetaTrader5']` so MT5BrokerPort can be
constructed on macOS/Linux CI without the real package installed.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


def _install_mt5_stub() -> MagicMock:
    """Install a MagicMock as sys.modules['MetaTrader5'].

    Returns the mock so tests can assert calls against it.  The stub
    needs `.initialize()` to return truthy and basic constants so
    MT5BrokerPort.__init__ succeeds without a real terminal.
    """
    stub = MagicMock(name="MetaTrader5")
    stub.initialize.return_value = True
    stub.last_error.return_value = (0, "ok")
    sys.modules["MetaTrader5"] = stub
    return stub


@pytest.fixture
def mt5_stub():
    stub = _install_mt5_stub()
    yield stub
    sys.modules.pop("MetaTrader5", None)


@dataclass
class _FakeCfg:
    magic: int
    mt5_path: str = "XAUUSD"


# ---------------------------------------------------------------------------
# __init__ magic resolution
# ---------------------------------------------------------------------------

class TestMagicResolution:
    def test_no_cfg_falls_back_to_default_magic(self, mt5_stub):
        """Backward compat: legacy callers without cfg get _DEFAULT_MAGIC."""
        from smc.execution.executor import MT5BrokerPort

        broker = MT5BrokerPort(login=1, password="x", server="y")
        assert broker._magic == MT5BrokerPort._DEFAULT_MAGIC
        assert broker._magic == 202500

    def test_cfg_supplies_xauusd_magic(self, mt5_stub):
        from smc.execution.executor import MT5BrokerPort

        cfg = _FakeCfg(magic=19760418)
        broker = MT5BrokerPort(login=1, password="x", server="y", cfg=cfg)
        assert broker._magic == 19760418

    def test_cfg_supplies_btcusd_magic(self, mt5_stub):
        from smc.execution.executor import MT5BrokerPort

        cfg = _FakeCfg(magic=19760419)
        broker = MT5BrokerPort(login=1, password="x", server="y", cfg=cfg)
        assert broker._magic == 19760419

    def test_cfg_none_uses_default(self, mt5_stub):
        """Explicit cfg=None behaves like omitting the kwarg."""
        from smc.execution.executor import MT5BrokerPort

        broker = MT5BrokerPort(login=1, password="x", server="y", cfg=None)
        assert broker._magic == MT5BrokerPort._DEFAULT_MAGIC


# ---------------------------------------------------------------------------
# Magic propagates into order payload
# ---------------------------------------------------------------------------

class TestMagicPropagation:
    def _make_broker(self, mt5_stub, *, cfg) -> "MT5BrokerPort":
        from smc.execution.executor import MT5BrokerPort

        broker = MT5BrokerPort(login=1, password="x", server="y", cfg=cfg)
        # Configure order_send to succeed with a fake result
        result = MagicMock()
        result.retcode = MT5BrokerPort._TRADE_RETCODE_DONE
        result.order = 42
        result.price = 2350.0
        result.volume = 0.05
        mt5_stub.order_send.return_value = result
        return broker

    def test_send_order_uses_cfg_magic(self, mt5_stub):
        from smc.execution.types import OrderRequest

        cfg = _FakeCfg(magic=19760418)
        broker = self._make_broker(mt5_stub, cfg=cfg)
        broker.send_order(
            OrderRequest(
                direction="long", lots=0.05,
                entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
                instrument="XAUUSD",
            )
        )
        assert mt5_stub.order_send.called
        payload = mt5_stub.order_send.call_args[0][0]
        assert payload["magic"] == 19760418

    def test_send_order_without_cfg_uses_default_magic(self, mt5_stub):
        from smc.execution.executor import MT5BrokerPort
        from smc.execution.types import OrderRequest

        broker = self._make_broker(mt5_stub, cfg=None)
        broker.send_order(
            OrderRequest(
                direction="long", lots=0.05,
                entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
                instrument="XAUUSD",
            )
        )
        payload = mt5_stub.order_send.call_args[0][0]
        assert payload["magic"] == MT5BrokerPort._DEFAULT_MAGIC

    def test_close_position_uses_cfg_magic(self, mt5_stub):
        from smc.execution.executor import MT5BrokerPort

        cfg = _FakeCfg(magic=19760419)
        broker = self._make_broker(mt5_stub, cfg=cfg)

        # Stub positions_get so close_position can proceed
        pos = MagicMock()
        pos.symbol = "BTCUSD"
        pos.volume = 0.05
        pos.type = MT5BrokerPort._ORDER_BUY
        mt5_stub.positions_get.return_value = (pos,)

        tick = MagicMock()
        tick.ask = 65_000.0
        tick.bid = 64_998.0
        mt5_stub.symbol_info_tick.return_value = tick

        broker.close_position(ticket=12345)
        # send_order called twice if test_send runs first, so grab last call
        close_payload = mt5_stub.order_send.call_args[0][0]
        assert close_payload["magic"] == 19760419
        assert close_payload["comment"] == "ai-smc-close"


# ---------------------------------------------------------------------------
# XAU/BTC isolation — the primary regression this fix prevents
# ---------------------------------------------------------------------------

class TestXauBtcIsolation:
    def test_xau_and_btc_brokers_have_distinct_magics(self, mt5_stub):
        from smc.execution.executor import MT5BrokerPort

        xau_broker = MT5BrokerPort(
            login=1, password="x", server="y",
            cfg=_FakeCfg(magic=19760418),
        )
        btc_broker = MT5BrokerPort(
            login=1, password="x", server="y",
            cfg=_FakeCfg(magic=19760419),
        )
        assert xau_broker._magic != btc_broker._magic
        # Regression: both used to be 202500, making cross-symbol reconcile
        # ambiguous — one broker couldn't tell its own orders from the other's.
        assert xau_broker._magic != MT5BrokerPort._DEFAULT_MAGIC
        assert btc_broker._magic != MT5BrokerPort._DEFAULT_MAGIC
