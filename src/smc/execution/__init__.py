"""Execution module for the AI-SMC trading system.

Public API
----------
- :class:`OrderRequest` / :class:`OrderResult` / :class:`PositionState` / :class:`AccountInfo` — immutable types
- :class:`BrokerPort` — abstract broker protocol
- :class:`MT5BrokerPort` — real MT5 execution (Windows only)
- :class:`SimBrokerPort` — simulated execution for testing
- :class:`OrderManager` ��� order lifecycle orchestration
- :func:`reconcile` — position reconciliation
"""

from smc.execution.executor import BrokerPort, MT5BrokerPort, SimBrokerPort
from smc.execution.order_manager import OrderManager
from smc.execution.reconciler import reconcile
from smc.execution.types import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    PositionState,
    ReconciliationResult,
)

__all__ = [
    "AccountInfo",
    "BrokerPort",
    "MT5BrokerPort",
    "OrderManager",
    "OrderRequest",
    "OrderResult",
    "PositionState",
    "ReconciliationResult",
    "SimBrokerPort",
    "reconcile",
]
