# Fusion Architecture — Maximum-Yield Component Wiring

> 中文注释 / 英文代码：把现有 ~120 个 Python 模块全部串成一条完整的信号决策链
> 当前问题：5 天 demo 从 $872 亏到 $314，根因是只跑了最弱执行路径（RegimeCache
> 几个数字 → MT5 EA），SMC 检测、AI 辩论、Gates、SASL 全都闲置。

---

## 1. 总体数据流（端到端）

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 0 ─ DATA INGESTION  (每 tick / 每 bar)                            │
│  MT5 tick + OHLCV(M5/M15/H1/H4/D1)                                     │
│  ForexFactory 经济日历 (NewsEngine)                                    │
│  COT / TIPS / DXY / VIX (ExternalContextFetcher)                       │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1 ─ PERCEPTION  (每 bar close)                                    │
│  ├─ SMCDetectorBundle      ⇒ swings/OB/FVG/BOS/CHoCH/Liquidity         │
│  ├─ classify_regime_v2     ⇒ RegimeAssessmentV2 (range/trend/news/…)   │
│  ├─ AnomalyDetector.detect ⇒ AnomalyState (NORMAL/ELEVATED/…/LOCKDOWN) │
│  └─ NewsEngine + NewsClassifier ⇒ NewsClassification                   │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 2 ─ DECISION  (每 H4 bar 或每 15 min)                             │
│  ├─ DirectionEngine.get_direction  ⇒ AIDirection (bullish/…+confidence)│
│  ├─ MacroLayer.compute_macro_bias  ⇒ MacroBias (COT+yield+DXY)         │
│  └─ FusionScorer                   ⇒ FusedDirection                    │
│       direction = sign(α·AI + β·Macro + γ·SMC + δ·News)                │
│       confidence ∈ [0, 1]                                              │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 3 ─ VALIDATION  (每 candidate signal)                             │
│  ├─ G1–G8 PromotionGates (离线 candidate 评估，运行时取最近通过)        │
│  ├─ ShieldAction (Anomaly → 屏蔽 / 收紧 confidence threshold)          │
│  ├─ AlphaValidation / FilterResult (运行时 hard filter)                │
│  └─ ShadowArtefact 证据链 (审计追溯)                                   │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 4 ─ EXECUTION PARAMS  (每 tick)                                   │
│  RuleEngine.derive_envelope_params(MarketState, prev_envelope)         │
│   → DynamicParams (mode/lot_factor/TP/SL/grid_mult/…/cooldown)         │
│   → build_envelope() → SignalEnvelope                                  │
│   → CacheWriter.write_envelope() → RegimeCache.json                    │
│   → MT5 HedgeRock EA 读取                                              │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 5 ─ EVOLUTION  (每日 / 事件触发)                                  │
│  SASLOrchestrator.run_daily_cycle:                                     │
│    DriftDetector → AutoAdjuster → SASLCircuitBreaker → AutoPurifier    │
│    输出 sasl_cycle_<ts>.json/.md，参数变更进 PolicyRegistry             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 主模块映射

| 已有组件 | 角色 | 调度频率 | Fusion 接入点 |
| -------- | ---- | -------- | ------------- |
| `smc_core/swing.py`, `order_block.py`, `fvg.py`, `structure.py`, `liquidity.py` | 感知 | 每 bar close | `PerceptionLayer.detect_smc()` |
| `ai/regime_classifier.py` + `regime_classifier_v2.py` | 感知 | 每 bar close | `PerceptionLayer.classify_regime()` |
| `evolution/anomaly_shield.py` | 感知 | 每 bar close | `PerceptionLayer.detect_anomaly()` |
| `hedgerock/news_engine.py` + `news_classifier.py` | 感知 | 每 15 min | `PerceptionLayer.fetch_news()` |
| `ai/direction_engine.py` (7-Agent debate) | 决策 | 每 H4 bar (TTL 4h) | `DecisionLayer.assess_direction()` |
| `ai/macro_layer.py` | 决策 | 每日 (TTL 24h) | `DecisionLayer.compute_macro()` |
| `evolution/promotion_gates.py` (G1–G8) | 验证 | 离线 + 启动时加载 | `ValidationLayer.check_gates()` |
| `hedgerock/alpha_validation.py` + `regime_filters.py` | 验证 | 每 tick | `ValidationLayer.runtime_filter()` |
| `evolution/shadow_artefact.py` | 验证 | 每决策 (audit) | `ValidationLayer.record_evidence()` |
| `hedgerock/rule_engine.py` + `market_state.py` | 执行 | 每 tick | `ExecutionLayer.derive_params()` |
| `hedgerock/cache_writer.py` | 执行 | 每 tick | `ExecutionLayer.publish()` |
| `evolution/sasl_orchestrator.py` (Drift/Adjuster/Breaker/Purifier) | 进化 | 每日 | `EvolutionLayer.run_cycle()` (sidecar) |

---

## 3. FusionController 主编排器接口

```python
class FusionController:
    """The single entry point that replaces `decision_server._run_rule_engine`."""

    def __init__(
        self,
        *,
        perception: PerceptionLayer,
        decision: DecisionLayer,
        validation: ValidationLayer,
        execution: ExecutionLayer,
        evolution: EvolutionSidecar | None = None,
        config: FusionConfig,
    ) -> None: ...

    def on_signal_request(
        self,
        symbol: str,
        features: MarketFeatures,
        ea_state_store: EAStateStore,
        prev_envelope: SignalEnvelope | None,
        news_classification: NewsClassification | None,
        now: datetime,
    ) -> FusionOutcome:
        """One-shot pipeline run.  Never raises — each tier can degrade."""
```

`FusionOutcome` = `{envelope, rule_kwargs, regime_override, confidence_override, post_v2_regime, transition_lock_override, evidence_chain, fusion_trace}`

---

## 4. 时间调度

| 周期 | 谁跑 | 内容 |
| ---- | ---- | ---- |
| 每 tick (≈10 s) | `FusionController.on_signal_request` | regime_v2 + rule_engine + cooldown |
| 每 bar close (M15) | `PerceptionLayer.refresh()` (in-memory cache) | SMC detection + anomaly + filters |
| 每 H4 close | `DecisionLayer.refresh_direction()` | AI direction debate (cached) |
| 每日 00:05 UTC | `EvolutionSidecar` | SASL daily cycle |
| 事件触发 | `EvolutionSidecar.on_event()` | drift_spike / breaker_alert |

设计选择：所有"重计算"层做 **TTL cache**，runtime 调用是 O(几个 dict/dataclass 复制 + rule_engine 计算)；
重活全部在后台被 sidecar 异步刷新。

---

## 5. 降级 (Degradation Matrix)

每层都可独立失败，永不影响下游：

| 层 | 失败时 | 行为 |
| -- | ------ | ---- |
| AI direction | 异常/超时/超预算 | 走 SMA fallback；`reasoning_tag="ai_unavailable"` |
| Macro layer | 网络故障 | `MacroBias(direction="neutral", sources_available=0)` |
| SMC detection | DataFrame 空 | 返回空 tuple；融合权重 → 0 |
| Anomaly detector | bars 不足 | `LOCKDOWN_NOT_TRIPPED`，但 `blocking_conditions` 报 insufficient |
| News engine | ForexFactory 不可达 | `NewsClassification=None` |
| Gates | 离线 candidate 不存在 | 用 conservative defaults，写 audit 警告 |
| Rule engine | crash | `_run_rule_engine` 返回 None → 走 legacy `build_envelope` |
| Evolution sidecar | crash | 不影响主线信号；状态写到 ledger |

**总原则**：错配（mode=observe）永远比错放仓位安全。
任何一层异常 → `FusionController` 自动降到 observe 安全模式。

---

## 6. 信号融合公式 (FusionScorer)

```
fused_score = α·AI_direction_score + β·Macro_total_bias + γ·SMC_score + δ·News_score
            ─ω·anomaly_penalty

其中:
  α = 0.35  (AI 7-agent debate - 主导，但 confidence 加权)
  β = 0.25  (Macro - COT+TIPS+DXY 长期偏置)
  γ = 0.30  (SMC - 结构性确认: OB+FVG+BOS方向)
  δ = 0.10  (News - 短期事件冲击)
  ω = 0.50  (Anomaly - LOCKDOWN 时直接扼杀)

最终 direction:
  bullish   if fused_score > +0.15
  bearish   if fused_score < -0.15
  neutral   otherwise

最终 confidence = min(1.0, |fused_score| × component_agreement_factor)
其中 component_agreement_factor = (同向组件数 / 总有效组件数)
```

权重保存在 `FusionConfig`，可通过 `PolicyRegistry` + SASL 自动调参。

---

## 7. 向后兼容 — RegimeCache 字段

`SignalEnvelope` 已经是 v2.0.0，本次融合 **不新增任何字段**；
所有融合信号通过现有字段表达：

- `regime` / `prev_regime` / `confidence` ← 已有
- `mode` / `risk_tier` / `hedgerock_enabled` ← 已有
- `lot_factor`, `grid_multiplier`, `max_next_lot`, `takeprofit_points`, `stoploss_points`, `recovery_multiplier` ← 已有
- `reason` ← 增强为 fusion_trace 摘要 (≤500 chars)

**EA 端零改动。**

---

## 8. 证据链 (Audit Trail)

每次 `on_signal_request` 写一条 `ShadowArtefact`：

```json
{
  "ts": "2026-05-16T08:00:00Z",
  "symbol": "XAUUSD",
  "fusion_trace": {
    "ai_direction": {"direction": "bullish", "confidence": 0.72, "source": "ai_debate"},
    "macro_bias": {"direction": "neutral", "total_bias": 0.04},
    "smc_score": 0.55,
    "news_score": 0.0,
    "anomaly_level": "NORMAL",
    "fused_score": 0.41,
    "component_agreement": 0.75
  },
  "envelope": { "regime": "trend_up", "mode": "hedgerock", "lot_factor": 1.2, ... },
  "degradation_notes": []
}
```

**用途**：drift_detector 用这条数据回看决策质量，sasl_orchestrator 用作自调参输入。

---

## 9. 文件结构

```
src/smc/fusion/
├── __init__.py
├── contracts.py          # FusionOutcome, FusionConfig, FusedDirection 等 dataclass
├── perception.py         # PerceptionLayer - SMC + regime + anomaly + news
├── decision.py           # DecisionLayer - AI + macro + fusion scorer
├── scorer.py             # FusionScorer - 加权求和 + confidence
├── validation.py         # ValidationLayer - Gates + filters + evidence
├── execution.py          # ExecutionLayer - rule_engine wrapper + cache writer
├── evolution_sidecar.py  # SASLOrchestrator 包装 + 后台调度
├── degradation.py        # 降级路径助手
└── fusion_controller.py  # 主编排
```

测试在 `tests/fusion/`，每层一个 unit test 文件 + 一个端到端 integration test。

---

## 10. 接入 decision_server

`create_app(..., fusion_controller=fc)`：
- 当 `fc is not None` 时，`/signal` 用 `fc.on_signal_request(...)`；
- 否则继续走原 `_run_rule_engine` 路径（向后兼容）。

未来 ramp-up：默认开启 fusion，legacy 路径用一两个 sprint 平滑下线。
