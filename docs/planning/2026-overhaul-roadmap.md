# AI-SMC 系统性改造路线图（2026）

> 方向：**自研指标引擎（确定性·无 LLM）+ 顺大周期方向的超短线 + 有界(regime 门控)补仓**。
> Round-4 诚实重定基线之后的新主轴；与 `docs/HANDOFF.md` / `docs/SPRINT11_REVIEW.md` 冲突时以本文件为准。

## 为什么改（诊断）

项目当前**负收益**，是范式问题而非调参问题：

- **没有方向性 edge**：生产 v1 走查 56 笔 = PF 1.023 / +$6.05（统计上≈0，`data/gate_v10_v1.jsonl`）；放宽过滤器 PF 塌到 0.655（`gate_v8_v2.jsonl`）→ 选择偏差。
- **结构性抓不到趋势**：mode_router A/B 2021–2024 pooled = −$9 / −$65；**2024 大牛市 PF 0.00 / 胜率 0%**（趋势盘 D1 无 BOS/CHoCH → HTF bias 55% 返回 neutral → 流水线短路）。
- **过拟合 + 验证地基缺失**：`mode_router`/`range_trader` 各 25+ 手调阈值；RangeTrader/mode_router/macro 全是 live-only **零 OOS 回测**；`fills.py` 只算入场成本；`walk_forward.py` 无 embargo。

## 新方向（已确认）

1. **自研指标引擎为核心，先做指标**（"利用已有的指标"组合出"自研指标"）。
2. **移除 LLM/AI 出实盘决策**（慢、不可靠）——引擎纯确定性规则；轻量 ML 留后期可选，且必须 IC + 净成本跑赢规则基线才接入。
3. **顺大周期方向的超短线高频**：HTF 指标定唯一许可方向，LTF 指标做超短入场；执行 EA 原生 tick 级。
4. **有界、regime 门控补仓**：复用 HedgeRock 执行底盘、挖掉无界马丁；补仓只加在顺势回调上，不做逆势 band fade。
5. **IC 科学选型**：每个候选指标先过 `research/ic.py` 的 IC/OOS+embargo 显著性筛选，只留预测力真实的再组合。
6. 指标族优先级：**① 趋势/方向（MA 斜率级联、ADX/±DI、SMA50 斜率）→ ② 动量/突破（RSI(3) 梯度、NRB、分形）→ ③ SMC 结构**；均值回归带本期暂缓。

**核心洞察**：HedgeRock 与当前 AI-SMC 都"做空趋势/做多震荡"，黄金的钱在趋势里。新方案用 HTF 方向指标把"逆势找死"改成"顺势捡钱"，用 IC 纪律防过拟合，用有界补仓安全复用 HedgeRock 恢复引擎。丢掉 LLM 不浪费 Phase 1——`MarketAssessment` 契约不变，只把生产者从 LLM 换成指标引擎。

## 目标架构

```
smc/indicators/                                   EA 快手 (MT5, tick 级)
  core.py     基础指标 (ATR/SMA/EMA/slope/         AISMCScalper.mq5
              Donchian/VWAP + RSI/ADX-DI/NRB/分形)  · iATR/iRSI/iADX/iMA/iFractals 镜像引擎
  composite.py 自研复合指标 (MA-cascade/ADX-DI 门/  · 每 tick 入场分 + BasketManager
              RSI 梯度/NRB 突破/多周期分形对齐)        (镜像 basket_policy)
  validation.py 用 research/ic.py 给候选打 IC 分
  engine.py   组合 IC 达标指标 → MarketAssessment (无 LLM)
                         │  同一契约 + 同一 basket_policy
              backtest/tick_engine.py (M1 净成本回测, live==backtest)
```

## 分阶段（验收门全部含成本、OOS+embargo、样本充分）

| Phase | 交付 | 验收门 |
|---|---|---|
| **1 成本回测地基** ✅ | `basket_policy.py`、`tick_fills.py`、篮子记录、`walk_forward` embargo | always-flat 净=−Σ成本；不变量过 |
| **2 指标核心库** ✅ | `indicators/core.py`：合并 ATR/SMA/EMA/slope/Donchian/VWAP + 新增 RSI/ADX-DI/NRB/分形/MA-cascade | 无前瞻属性测试 + 与旧实现对拍 + ruff/mypy 绿 |
| 3 复合指标 + IC 选型 | `indicators/{composite,validation}.py` + `scripts/ingest_m1.py`(落 M1) | ≥1 趋势族 + ≥1 动量族 IC 显著、OOS 不塌 |
| 4 指标引擎 → MarketAssessment | `indicators/engine.py` + `ai/models.py` 新增契约 (无 LLM) | 趋势窗口 100% 顺势 direction + allow_averaging=false |
| 5 tick_engine + scalper 回测 | `backtest/tick_engine.py` (指标+basket_policy 跑 M1) | 净成本 OOS PF >1.1、篮子尾界每 fold 成立、趋势日补仓 0 |
| 6 EA 原生 scalper | `AISMCScalper.mq5` 镜像引擎 + BasketManager | parity ≥99%、demo 硬止损不破 |
| 7 补仓标定 + 实盘 A/B | `calibrate_basket.py` + dual-magic control/treatment | 净成本 PF CI 不含 1.0 |
| 可选后期 ML | 指标做特征训 GBT | 仅当 IC + 净成本跑赢规则基线才接入 |

## 已交付

- **Phase 1** `execution/basket_policy.py`（有界补仓+尾险不变量 `最坏亏损 ≤ K×典型小赢`）、`backtest/tick_fills.py`（对称往返成本，修 `fills.py` 仅入场计费）、`backtest/types.py`（篮子记录）、`walk_forward.py`（embargo）。26 测试。
- **Phase 2** `indicators/core.py`：确定性、无前瞻、EA 可镜像的指标库——合并散落 3 处的 ATR/SMA/slope/Donchian/VWAP，新增 RSI、ADX/+DI/−DI、EMA、NRB 突破、Williams 分形、MA-slope-cascade。22 测试（无前瞻属性 + 旧实现对拍 + Wilder 正确性）。

下一增量：Phase 3 复合指标 + 落 M1 数据 + 跑 IC 选型报告。
