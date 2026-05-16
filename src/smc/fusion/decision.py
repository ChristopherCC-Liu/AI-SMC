"""DecisionLayer —— 把 PerceptionSnapshot 与 AI direction / Macro bias
合并成最终的 FusedDirection。

设计要点：
1. AI direction 调用是昂贵的（claude CLI / Anthropic API），通过 TTL
   cache + bar_ts 时间戳作为 key 复用结果；
2. Macro bias 调用也是昂贵的（COT/TIPS/DXY），同样 TTL cache；
3. DecisionLayer 自身永不抛异常 —— 任一子组件失败时让 scorer 自然忽略。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Protocol

import polars as pl

from smc.ai.models import AIDirection
from smc.fusion.contracts import FusedDirection, FusionConfig, PerceptionSnapshot
from smc.fusion.scorer import score_fusion

_LOG = logging.getLogger(__name__)

__all__ = [
    "DecisionLayer",
    "DirectionEngineProtocol",
    "MacroLayerProtocol",
]


# ---------------------------------------------------------------------------
# Protocol —— 让我们可以注入 fake 测试（避免硬绑定 ai.direction_engine）
# ---------------------------------------------------------------------------


class DirectionEngineProtocol(Protocol):
    def get_direction(
        self,
        h4_df: pl.DataFrame | None = None,
        bar_ts: datetime | None = None,
    ) -> AIDirection: ...


class MacroLayerProtocol(Protocol):
    def compute_macro_bias(self, instrument: str = "XAUUSD") -> object: ...


# ---------------------------------------------------------------------------
# H4Provider —— 提供 H4 DataFrame 给 DirectionEngine
# ---------------------------------------------------------------------------


class H4DataFrameProvider(Protocol):
    def get_h4_dataframe(self, symbol: str) -> pl.DataFrame | None: ...


# ---------------------------------------------------------------------------
# DecisionLayer
# ---------------------------------------------------------------------------


class DecisionLayer:
    """编排 AI direction + Macro bias + scorer。

    所有外部依赖（direction_engine, macro_layer, h4_provider）都是可
    注入的；缺省时该组件的输出为 None，scorer 会自动忽略其权重。
    """

    def __init__(
        self,
        *,
        config: FusionConfig,
        direction_engine: DirectionEngineProtocol | None = None,
        macro_layer: MacroLayerProtocol | None = None,
        h4_provider: H4DataFrameProvider | None = None,
    ) -> None:
        self._config = config
        self._direction_engine = direction_engine
        self._macro_layer = macro_layer
        self._h4_provider = h4_provider

        # AI direction TTL cache —— key = (symbol, hour bucket)
        self._ai_cache: dict[str, tuple[AIDirection, datetime]] = {}
        self._ai_ttl = timedelta(hours=config.ai_cache_ttl_hours)

        # Macro bias TTL cache —— key = instrument
        self._macro_cache: dict[str, tuple[object, datetime]] = {}
        self._macro_ttl = timedelta(hours=config.macro_cache_ttl_hours)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        snapshot: PerceptionSnapshot,
        now: datetime,
    ) -> FusedDirection:
        ai_direction, ai_conf = self._fetch_ai_direction(
            symbol=snapshot.symbol, now=now,
        )
        macro_direction, macro_total_bias = self._fetch_macro_bias(
            symbol=snapshot.symbol, now=now,
        )
        return score_fusion(
            config=self._config,
            ai_direction=ai_direction,
            ai_confidence=ai_conf,
            macro_direction=macro_direction,
            macro_total_bias=macro_total_bias,
            smc_score=snapshot.smc_score,
            news_intensity=snapshot.news_intensity,
            news_direction=snapshot.news_direction,
            anomaly_level=snapshot.anomaly_level,
        )

    # ------------------------------------------------------------------
    # AI direction —— 永不抛异常
    # ------------------------------------------------------------------

    def _fetch_ai_direction(
        self,
        *,
        symbol: str,
        now: datetime,
    ) -> tuple[str | None, float]:
        if not self._config.enable_ai_direction:
            return None, 0.0
        if self._direction_engine is None:
            return None, 0.0

        key = self._ai_cache_key(symbol, now)
        cached = self._ai_cache.get(key)
        if cached is not None:
            value, cached_at = cached
            if now - cached_at < self._ai_ttl:
                return value.direction, float(value.confidence)

        h4_df: pl.DataFrame | None = None
        if self._h4_provider is not None:
            try:
                h4_df = self._h4_provider.get_h4_dataframe(symbol)
            except Exception:
                _LOG.exception("h4_provider crashed for %s; skipping", symbol)
                h4_df = None

        try:
            direction = self._direction_engine.get_direction(
                h4_df=h4_df, bar_ts=now,
            )
        except Exception:
            _LOG.exception("direction_engine crashed for %s", symbol)
            return None, 0.0

        if direction is None:
            return None, 0.0
        self._ai_cache[key] = (direction, now)
        return direction.direction, float(direction.confidence)

    @staticmethod
    def _ai_cache_key(symbol: str, now: datetime) -> str:
        bucket = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")
        return f"{symbol.upper()}|{bucket}"

    # ------------------------------------------------------------------
    # Macro bias —— 永不抛异常
    # ------------------------------------------------------------------

    def _fetch_macro_bias(
        self,
        *,
        symbol: str,
        now: datetime,
    ) -> tuple[str | None, float]:
        if not self._config.enable_macro_layer:
            return None, 0.0
        if self._macro_layer is None:
            return None, 0.0

        instrument = symbol.upper()
        cached = self._macro_cache.get(instrument)
        if cached is not None:
            value, cached_at = cached
            if now - cached_at < self._macro_ttl:
                return self._extract_macro(value)

        try:
            bias = self._macro_layer.compute_macro_bias(instrument)
        except Exception:
            _LOG.exception("macro_layer crashed for %s", symbol)
            return None, 0.0
        if bias is None:
            return None, 0.0
        self._macro_cache[instrument] = (bias, now)
        return self._extract_macro(bias)

    @staticmethod
    def _extract_macro(bias_obj: object) -> tuple[str | None, float]:
        """对 MacroBias dataclass 取 direction + total_bias —— 鸭子类型。"""
        direction = getattr(bias_obj, "direction", None)
        total_bias = getattr(bias_obj, "total_bias", 0.0)
        try:
            return (direction if direction in ("bullish", "bearish", "neutral") else None), float(total_bias)
        except (TypeError, ValueError):
            return None, 0.0
