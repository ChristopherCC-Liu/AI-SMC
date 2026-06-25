# AI-SMC 系统性改造路线图（2026）

> 方向：**顺大周期方向的超短线高频 scalping + 有界(regime 门控)补仓恢复 + 双时间尺度模型**。
> 本文件是 Round-4 诚实重定基线之后的新主轴；与历史 `docs/HANDOFF.md` / `docs/SPRINT11_REVIEW.md` 冲突时以本文件为准。

## 为什么改（诊断）

项目当前**负收益**，且是范式问题而非调参问题：

- **没有方向性 edge**：生产 v1 走查 56 笔 = PF 1.023 / +$6.05（统计上等于 0，`data/gate_v10_v1.jsonl`）；放宽过滤器 PF 塌到 0.655（`gate_v8_v2.jsonl`）→ 表面 PF>1 是**选择偏差**。
- **结构性抓不到趋势**：mode_router A/B 2021–2024 pooled = −$9 / −$65；**2024 大牛市 PF 0.00 / 胜率 0%**（趋势盘 D1 无 BOS/CHoCH → HTF bias 55% 返回 neutral → 流水线短路）。
- **过拟合 + 验证地基缺失**：`mode_router`/`range_trader` 各 25+ 手调阈值；RangeTrader/mode_router/macro 全是 live-only **零 OOS 回测**；`backtest/fills.py` 只对入场计成本；`walk_forward.py` 无 embargo。

**核心洞察**：HedgeRock 与当前 AI-SMC 都"做空趋势/做多震荡"，而黄金的钱在趋势里。新方案用 HTF 方向过滤器把"逆势找死"改成"顺势捡钱"，用**有界**补仓安全复用 HedgeRock 的恢复引擎。

## 目标架构

```
慢脑 (Python, 小时/日级)                快手 (MT5 EA, tick 级)
DirectionEngine + regime + Macro    →   AISMCScalper.mq5 (fork 自 AISMCReceiver)
 合并为 MarketAssessment 单一契约          缓存 assessment + 每 tick 入场分 + BasketManager
 {direction, regime, allow_averaging,    (镜像 basket_policy: 加层门控/硬篮子止损/分批)
  max_layers, basket_stop_usd, ...}       复用底盘: trailing/throttle/news/panel
```

- **唯一契约** `MarketAssessment`（`ai/models.py`）：慢脑写、server 发、EA 缓存、**回测器同样消费** → live==backtest 的根基。
- **补仓逻辑 = 纯模块** `execution/basket_policy.py`，EA 的 `BasketManager` 是其 MQL5 镜像，parity 测试保证一致。
- **不变量**：`最坏篮子亏损 ≤ K × 典型小赢`（`validate_tail_bound`）。
- 目标从"高胜率"改为"**高胜率 且 含成本正期望 且 尾险有界**"；成本是高频第一决定变量，必须先建模。

## 分阶段（验收门全部含成本、OOS+embargo、样本充分）

| Phase | 交付 | 验收门 |
|---|---|---|
| 0 去冗余 | 归档过拟合/实验模块到 `archive/`，停 `live_demo.py` 单体，落 XAUUSD M1 数据 | pytest 绿、M1 覆盖 ≥99%、无残留 import |
| **1 成本回测地基** ✅(进行中) | `basket_policy.py`、`tick_fills.py`、`tick_engine.py`、`walk_forward` embargo、篮子不变量测试 | always-flat 净=−Σ成本；cost_drag_pct 真实；不变量过 |
| 2 慢脑合并 | `MarketAssessment` + `assessment_service` + `/assessment` + `assessment_replay` | 趋势窗口 100% `allow_averaging=false` |
| 3 EA 快手底盘 | `AISMCScalper.mq5` + `BasketManager` + parity 测试 | demo ≥2 周硬止损不破、趋势日补仓 0 次、parity ≥99% |
| 4 ML 入场+校准 | `ml/{features,entry_model,calibration,export}` | 净成本 OOS PF >1.15 且 > 基线；ECE <0.05 |
| 5 补仓参数标定 | `calibrate_basket.py` 由回测选 `max_layers/spacing/stop` | 有界补仓 OOS 提升净期望且尾界每 fold 成立 |
| 6 实盘 A/B | dual-magic control/treatment，paper→micro-real | demo ≥4 周/≥300 笔，treatment 净成本 PF CI 不含 1.0 |

## 本 PR（Phase 1 地基 · 数据无关部分）

- `src/smc/execution/basket_policy.py` — 有界、regime 门控的补仓纯策略 + 尾险不变量
- `src/smc/backtest/tick_fills.py` — 对称往返成本模型（修 `fills.py` 仅入场计费的问题）
- `src/smc/backtest/types.py` — 新增 `BasketLayerRecord` / `BasketTradeRecord`（含成本与最坏浮亏字段）
- `src/smc/backtest/walk_forward.py` — 新增 `embargo_days`（防泄漏；默认 0 保持旧行为）
- 三个测试文件（成本审计、篮子不变量、embargo 强制）

下一增量：`tick_engine.py`（用上述原语在 M1 上跑 basket 模拟）+ Phase 0 归档 + M1 数据落地。
