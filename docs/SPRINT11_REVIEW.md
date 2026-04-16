# AI-SMC Sprint 11 交叉审核与架构优化报告

**审核日期：** 2026-04-17
**审核范围：** Sprint 10 之后、HANDOFF.md 之外的所有新增工作（dual-mode、AI HARD GATE、dashboard、VPS 脚本）
**审核方式：** T3 三路并行独立审核 + 代码交叉验证
**审核 agent：** Bug Hunter / Architecture Reviewer / Trading Risk Reviewer（各自独立读码，互不通信）

---

## 0. 当前 git 状态 vs 文档状态

HANDOFF.md 停留在 Sprint 10（`41ad087`、`0d8ae03`）。之后有 10 个 commit 未记录：

| commit | 内容 | 对 HANDOFF 的影响 |
|---|---|---|
| `b8e6dae` | Rich 终端 dashboard | 新增模块（dashboard.py） |
| `57fa0cf` | 全局 VPS 命令（smc-dashboard/live/health） | 新增部署入口 |
| `1f252c0` | Claude CLI `.cmd` 路径修复 | AI 在 VPS 可用的前提 |
| `d068b6f` | Dashboard layout 修复 | - |
| `d1bdad0` | AI 分析集成 live loop（BUY/SELL/HOLD） | 回滚了 HANDOFF 所说"AI 未启用" |
| `0019926` | AI filter analysis + action-oriented dashboard | 新增 `ai_filter_analysis.py` |
| `b3cc7aa` | **AI HARD GATE（5-gate：session + direction + confidence sizing）** | 根本性改变生产行为 |
| `0286496` | Dual-mode 设计 | 新设计思路 |
| `e454a40` | **Dual-mode 实现（trending + ranging，系统从不空闲）** | 新增 4 个模块 + live_demo 改写 |

**结论：** 生产行为已与 HANDOFF 描述严重不符。Sprint 11 的第一件事就是更新 HANDOFF，否则下一个接手人会被误导。

---

## 1. Executive Summary

> AI-SMC 在 Sprint 10 之后做了 **3 个 Sprint 级改动**（AI HARD GATE + Dual-mode + 完整 VPS 部署）但没有更新交接文档。代码层面，dual-mode 设计在单元测试层通过（661 tests passing），**但缺少 3 个关键验证**：(1) 没有 `determine_action` 集成测试；(2) 没有 range-mode 的 OOS backtest；(3) 没有 dual-mode 与 v1-only 的 A/B 对比。同时引入了 4 个确定性 bug（其中 1 个与 HANDOFF 已知的 zone cooldown bug 同源）、至少 3 个重大交易风险（过拟合、趋势中做反转、swing 回溯）。
>
> **建议 Sprint 11 不启动任何新功能**，用 9~12 天做：统一 live 路径、补 backtest、加集成测试、配置外部化、修确定性 bug。完成后再考虑 Gate 3（实盘）。

---

## 2. 三视角审核汇总（去重 + 交叉加权）

按「多位审核员独立指出 → 高置信」原则合并。✅=三位都提到；●=两位提到；◯=单人提到但已验证。

### 🔴 S1 — 阻塞级（不修不能进下一个 Sprint）

#### S1-1. ✅ Live 路径分裂，`live_demo.py` 绕过了整个 `LiveLoop`
- **证据：** HANDOFF line 193 自白；`src/smc/monitor/live_loop.py`（有 DI、有 TelegramAlerter、有 HealthMonitor）从未在生产被调用。生产只跑 `scripts/live_demo.py`（裸 `try/except + print`，无告警）。
- **影响：** 任何写进 `LiveLoop` 的修复对生产无效；VPS 崩溃无告警。
- **修法：** 把 `live_demo.py` 的逻辑抽回 `LiveLoop`（现有 DI 协议已经够用），增加 `DualModeStrategyRunner`。
- **工期：** 2–3 天。ROI：高。

#### S1-2. ✅ `determine_action()` 无集成测试，dual-mode 5-gate 全部盲区
- **证据：** `tests/` 下只有 `test_mode_router.py` / `test_range_trader.py` / `test_breakout_detector.py` 三个单元测试；对 `live_demo.determine_action` 的 5-gate 组合（GATE1–5 + 模式切换 + soft override）零覆盖。
- **影响：** mode_router 内部改字段、soft override 阈值变了、session penalty 逻辑改了都不会被 CI 捕获。生产崩溃时才发现。
- **修法：** 写 `tests/smc/integration/test_dual_mode_routing.py`，15 个场景：
  - AI bullish + SMC long → BUY
  - AI bullish + SMC short（conf<0.65）→ soft override BUY
  - AI bullish + SMC short（conf≥0.65）→ HOLD
  - AI neutral + 有 range + regime=ranging → RANGE BUY/SELL
  - ASIAN session → HOLD（确认 router 和 `_determine_trending` 不会 double-block 出 bug）
  - breakout 触发 → HOLD
  - `mode.range_bounds is None` 但 `mode.mode=="ranging"` → fallback 分支
- **工期：** 1 天。ROI：非常高。

#### S1-3. ✅ Range-mode 无 OOS backtest，性能完全未知
- **证据：** `src/smc/backtest/` 里没有 `RangeTrader` 的引用；`FastSMCStrategyAdapter` 只跑 trending。Gate v1–v10 全部只测了 trending。
- **影响：** dual-mode 可能在历史数据上 PF < 1.0 而你不知道。直接进 Gate 3（实盘 PF ≥ 1.3）就是赌博。
- **修法：**
  1. 写 `src/smc/backtest/adapter_range.py`（batch 预算 range bounds，逐 M15 bar 检查 boundary setup）。
  2. 写 `scripts/run_v11_dual_mode_backtest.py`：v1 单模式 vs v1+range 对比。
  3. 硬条件：dual-mode 的 PF ≥ v1 baseline（不伤 v1），且 range-only PF ≥ 1.0。
- **工期：** 3–4 天。ROI：非常高（避免上线 net-negative 策略）。

#### S1-4. ✅ 趋势形成期做反转（Mean-reversion during trend development）
- **证据：** `mode_router.py` line 71：`ai_direction=="neutral" AND ai_confidence<0.5 AND regime!="trending" AND range_bounds is not None` → ranging。但 `regime.py` 的 "transitional" 就是趋势形成期（ATR 1.0–1.4%）。AI 在趋势形成初期经常 neutral（没抓到方向变化）。此时系统会在"阻力位做空"正好进入 breakout。
- **历史数据佐证：** Sprint 9 数据里 2024 ATH 周期 18 trades / 0% WR，绝大多数是 ranging regime——而 2024 本身是强趋势年。
- **影响：** 在趋势转折期连续亏损；mode_router 无 hysteresis，可能 3 根 bar 内 trending↔ranging 反复切换。
- **修法：**
  1. mode 切换加 hysteresis（进入 ranging 后需 2+ bar 连续满足才能重新进 trending，反之亦然）；
  2. Breakout override：若价格已经偏离 range 中心 >20% 则禁止新开 range 单；
  3. **更稳妥：在 paper 阶段整个 disable ranging mode**，等 AI 真正启用 + live 数据过 60 天再开。
- **工期：** 0.5 天。ROI：极高。

#### S1-5. ◯ Range 探测里 `datetime.now()` 再次复现 HANDOFF 已知 bug
- **证据：** `range_trader.py` line 75：`now = datetime.now(tz=timezone.utc)`；`RangeBounds.detected_at` 字段在未来若进 backtest 会记成系统时间而非 `bar_ts`。与 HANDOFF line 215 记录的 zone cooldown bug 完全同源。
- **影响：** 若未来 range 逻辑依赖 `detected_at`（过期、冷却、锚定），backtest 与 live 会分叉。
- **修法：** `detect_range(h1_df, h1_snapshot, *, now: datetime | None = None)`，`now` 可选参数；backtest 传 `bar_ts`，live 传 `datetime.now()`。一并把 `record_zone_loss` 的旧 bug 也改了。
- **工期：** 0.5 天。ROI：中（目前不爆但会爆）。

---

### 🟡 S2 — 应修但不阻塞

#### S2-1. ● 配置漂移 — 新模块的硬编码参数无法从 config 配置
- **证据：**
  - `range_trader.py`：`_OB_BOUNDARY_CONFIDENCE=0.8`、`_SWING_LOOKBACK_BARS=50`、`_MIN_SWING_POINTS_FOR_RANGE=2`、`boundary_pct=0.15`、`min/max_range_width=300/3000` 全部模块级硬编码
  - `breakout_detector.py`：`atr_buffer_mult=0.25` 硬编码
  - `live_demo.py`：session penalty（0.1/0.2）、confidence threshold（0.5/0.6/0.8）、lot sizing 阶梯（0.0025/0.005/0.01）全部内联
  - `SMCConfig` 没有任何 dual-mode 字段
- **影响：** 无法 A/B tuning；paper vs 真仓无法差异化；参数变更必须改代码 + 重新部署 VPS。
- **修法：** `SMCConfig` 增加 `range_*`、`breakout_*`、`session_penalty_*`、`ai_confidence_*`、`lot_sizing_*` 字段；`RangeTrader/BreakoutDetector` 构造器从 config 注入。
- **工期：** 1–2 天。ROI：中高。

#### S2-2. ◯ Range 无 hysteresis / 无最小触碰次数 → 假 range
- **证据：** `_detect_from_ob_boundaries` 只要存在一个 bearish OB + bullish OB 即成 range，不要求价格曾经回测过各 boundary。Classic range-trading 至少要 2–3 次触碰。
- **影响：** 把两个历史孤立 OB 误判为 range，在"非 range"里做 mean reversion 会连续亏损。
- **修法：** 添加 touch-count gate（boundary 附近 ±10 points 在最近 50 H1 bar 内至少触碰 2 次）。
- **工期：** 0.5 天。

#### S2-3. ● AI 直接驱动仓位（confidence-based lot sizing）未做校准
- **证据：** `live_demo.py` line 171：`lot = 0.01 if conf>=0.8 else 0.005 if conf>=0.6 else 0.0025`。Claude debate 的 confidence 是**自报概率**，不是 Brier-calibrated。
- **依据：** LLM 自报概率普遍系统性偏高（Kadavath et al., 2022; Tian et al., 2023）。若真实 WR @ conf=0.8 只有 65%，杠杆倍率 4× 会放大尾部亏损。
- **修法：**
  1. 短期：先固定 lot=0.005，关掉 confidence-based sizing；
  2. 中期：live 跑 100+ 笔后画校准曲线（stated conf 分桶 vs 实际 WR），重新映射倍率；
  3. 或改为基于 portfolio ATR 的仓位而非 AI conf。
- **工期：** 0.5 天（短期固化）。

#### S2-4. ● Spread 未进 RR 计算
- **证据：** `live_demo.py` line 384：`spread = tick.ask - tick.bid` 仅 print。Range trader 的 `_build_setup` 完全不计 spread。XAUUSD 典型 spread $0.30–0.50（3–5 points），range trade 目标利润往往只有 $10–40，spread 占利润 5–15%。
- **影响：** paper PF 比 live PF 系统性高估。Gate 3 的门槛 PF≥1.3 看起来达成了，实盘可能只有 1.15。
- **修法：** 在 `_build_setup` 里 `reward -= spread`、`risk += spread/2`，再过 `rr_ratio>=1.0` 的 gate。
- **工期：** 1 小时。ROI：中高。

#### S2-5. ◯ Swing lookahead 在 range 检测里风险
- **证据：** `smartmoneyconcepts.swing_highs_lows` 需要每个候选 swing 两侧有 `swing_length=10` bar 才能确认。若 H1 dataframe 包含最新那根未完成/刚完成的 bar，最后 10 根 bar 的 swing 未被确认，就被 `_detect_from_swing_extremes` 用作 boundary → 前瞻偏差。
- **修法：** 在 range 检测里只用 `h1_snapshot.swing_points[:-2]`（或过滤 `swing.confirmed_at > bar_ts + 10*H1`）。**backtest 的 FastSMCStrategyAdapter 也应该同样修正**。
- **工期：** 0.5 天。

#### S2-6. ◯ Range 入场 RR 会被"深入 range"腐蚀
- **证据：** `range_trader.py` line 106：`current_price <= bounds.lower + boundary_width`（boundary_width = 15% range）。入场可以在 range 下沿到 15% 之间任意位置。SL = `bounds.lower - sl_buffer`。TP = midpoint。举例：range $100 宽，入场 30% = $30 above lower，SL 在 lower − $5，则 risk $35 / reward $20，RR=0.57。
- **修法：** `_build_setup` 末尾加 `if rr_ratio < 1.0: return None`（或直接要求 ≥ 1.5）。
- **工期：** 10 分钟。ROI：极高（一行代码砍掉大批 trap 入场）。

#### S2-7. ◯ Range setup 从不写 journal
- **证据：** `live_demo.py` line 457 `for s in setups:` 只遍历 trending setups。`best_setup` 若为 `RangeSetup` 则 `save_state()` 会写到 `live_state.json`，但 `live_trades.jsonl` 永远没有 range 记录。
- **影响：** Gate 3 的 PF 计算会漏掉全部 range 交易；审计/回测缺数据。
- **修法：** 在 journal 循环之后加 range 的分支写入（含 range_bounds、trigger、confidence）。
- **工期：** 15 分钟。

#### S2-8. ● AI 层有两个入口做重复工作
- **证据：** `live_demo.run_ai_analysis` 调用 `DirectionEngine.get_direction()`；同一 cycle `classify_regime()` 又触发 `classify_regime_ai()`。两者都在 D1/H4 上抽特征、跑 Claude（若启用）。
- **影响：** 每 cycle 两次 LLM 调用；direction 与 regime 可能给出冲突判断（bullish vs ranging）没有显式裁决。
- **修法：** 合并成单个 `MarketAssessment`，一次特征抽取 + 一次 debate 同时输出 `{direction, regime, confidence}`。
- **工期：** 2–3 天（偏重构）。

#### S2-9. ◯ AI 来源歧义（fallthrough）
- **证据：** `run_ai_analysis` 先写 `direction/confidence`（SMA），再试写 `ai_direction/ai_confidence`（Claude）。若 Claude 失败，只留 SMA 值。下游 `determine_action` 读 `ai_direction or direction`，但日志里没标注"当前走的是哪条路径"。
- **修法：** 无论哪条路径都写 `ai_source` 字段；dashboard 显式展示"AI online/fallback"状态。
- **工期：** 1 小时。

#### S2-10. ◯ 三 JSON 文件无原子写
- **证据：** `live_state.json` / `ai_analysis.json` / `live_trades.jsonl` 都是 `open("w")` + `json.dump` 直接写。dashboard 并发读。若 live_demo 崩在中间，dashboard 读到半截 JSON 会抛异常。
- **修法：** `tempfile.NamedTemporaryFile` + `os.replace()` 原子替换；或 3 个合并成单文件 + atomic write。
- **工期：** 1 小时。

---

### 🟢 S3 — 跟踪即可

- **S3-1.** Session filter 的 "ASIAN PF 0.69" 缺样本量/置信区间细节（HANDOFF 没说，需要回 `worst_indicator_scan.py` 查）。
- **S3-2.** `_SWING_LOOKBACK_BARS = 50` 语义错位（实际是 swing 点数而非 bar 数）——改名 `_SWING_LOOKBACK_COUNT`。
- **S3-3.** `range_trader.detect_range` 的 `h1_df` 参数定义了但没用；不是 bug，是接口冗余。
- **S3-4.** Asian session 双重 block（router + `_determine_trending` 都判断一次）——功能等价但冗余，留 router 一处即可。
- **S3-5.** v1.1 / v2 / v3 实验代码仍在 `src/smc/strategy/` 与生产同目录——ROI 低但建议搬到 `archive/`。
- **S3-6.** Overfitting 风险：3D worst-indicator scan（447 trades × 36 cells）无多重比较修正。v1.1 inversion 未做 hold-out 验证，**当前已经 disable，低优先级**。
- **S3-7.** 回测引擎用 `datetime.now()` 的老 bug（zone cooldown no-op，HANDOFF 已知）。

---

## 3. 架构层面的三个核心观察

### 3.1 单点职责已经崩了
`live_demo.py` 同时做：MT5 数据抓取、AI 调用、regime 分类、range 探测、mode routing、5-gate 决策、状态持久化、日志输出、信号处理。它已经是一个小型 monolith。

**对比：** `src/smc/monitor/live_loop.py` 按协议拆得干净（`DataFetcher`/`BrokerPort`/`StrategyRunner`/`HealthMonitor`/`Alerter`）但被生产绕开了。

**这是系统最大的架构问题。** 所有后续 bug fix、新功能、observability 都会在这个 monolith 里堆叠。

### 3.2 策略目录已经"考古层化"
```
src/smc/strategy/
├── aggregator.py           ← v1 生产
├── aggregator_v1_1.py      ← reference
├── aggregator_v2.py        ← reference
├── aggregator_v3.py        ← reference
├── confluence.py           ← v1
├── confluence_v2.py        ← reference
├── zone_scanner.py         ← v1
├── zone_scanner_v2.py      ← reference
├── entry_trigger.py        ← v1
├── entry_v2.py             ← reference
├── mode_router.py          ← 新（dual-mode）
├── range_trader.py         ← 新
├── range_types.py          ← 新
├── breakout_detector.py    ← 新
├── htf_bias.py / regime.py / types.py
```
7 个 "reference only" 文件和 4 个新文件混在一起。任何人 grep "aggregator" 会看到 4 个候选，无法一眼分辨哪个是生产。

### 3.3 验证路径 vs 生产路径之间有断层
```
回测路径: ForexDataLake → FastSMCStrategyAdapter → BarBacktestEngine → MultiTimeframeAggregator
生产路径: MT5 → live_demo.py → MultiTimeframeAggregator + RangeTrader + mode_router
```
RangeTrader 和 mode_router **不存在于回测路径**。它们 100% 是"live-only 代码"。这意味着：
- 无法用历史数据评估 dual-mode 性能
- 无法 A/B 测试参数
- 无法在上线前发现 range 策略的 edge case

**这违反了项目从 Sprint 1 开始的基本纪律："Gate 系统必须量化通过才能进下一步"**。

---

## 4. 建议的 Sprint 11 计划（9–12 天）

优先级：先修阻塞、再补验证、最后清债。**不加新功能**。

### Phase A — 止血（Day 1–2）
1. **A1** 更新 HANDOFF.md，补 Sprint 11/12/13 内容（AI HARD GATE + dual-mode + VPS 脚本）。把本报告作为单一真相源引用。（0.5 天）
2. **A2** 修 S2-6（range RR ≥ 1.0 gate，一行代码）+ S2-7（range journal）+ S2-4 一半（spread 进 RR）。（0.5 天）
3. **A3** 修 S1-5（`datetime.now()` → `bar_ts`），顺手把 HANDOFF 已知的 zone cooldown 也一起改。（0.5 天）
4. **A4** Paper 阶段暂时 disable ranging mode（只留 trending + AI HARD GATE），同时保留 range detection 仅做显示（S1-4 的最稳妥方案）。（0.5 天）

**止血后状态：** 生产仍是 v1 + AI HARD GATE，但没有 range 交易。代码健康度提升，Gate 3 统计不会被假数据污染。

### Phase B — 统一 live 路径（Day 3–5）
5. **B1** 把 `fetch_mt5_data`、`run_ai_analysis`、`determine_action` 重构成独立类，实现 `LiveLoop` 的现有协议。
6. **B2** 用 `LiveLoop` 跑一周 shadow（并行 `live_demo.py`，比较每 cycle action 是否完全一致）。
7. **B3** 切换生产入口到 `LiveLoop`；保留 `live_demo.py` 30 天做 fallback，之后归档。

### Phase C — 补 backtest + 集成测试（Day 6–9）
8. **C1** 写 `src/smc/backtest/adapter_range.py`：batch 预算 H1 range bounds，逐 M15 bar 判 boundary + M15 CHoCH。（2 天）
9. **C2** 写 `scripts/run_v11_dual_mode_backtest.py`：v1-only vs v1+range 跨 Gate v10 所有窗口对比。（1 天）
10. **C3** 写 `tests/smc/integration/test_dual_mode_routing.py`（S1-2 的 15 个场景）。（1 天）

**Gate v11 通过标准：**
- v1+range 的总 PF ≥ v1-only
- range-only trade 数 ≥ 20 笔，PF ≥ 1.0
- 无 mode 切换 whipsaw（每周 ≤ 1 次切换）
- 所有集成测试绿

### Phase D — 配置外部化（Day 10–11）
11. **D1** `SMCConfig` 加 dual-mode / session / confidence 字段。`RangeTrader/BreakoutDetector` 构造器从 config 注入。（1 天）
12. **D2** 三 JSON 文件合并 + 原子写（S2-10）。（0.5 天）

### Phase E — 可选清债（Day 12+）
13. 归档 v1.1 / v2 / v3。
14. 合并 AI 双入口（`DirectionEngine` + `classify_regime_ai` → `MarketAssessment`）。
15. 补 swing lookahead 修正（S2-5）。

---

## 5. Gate 3（paper → 实盘）通过标准（修订版）

原 HANDOFF 版本：**Live PF ≥ 1.3 over 60+ days**。建议增补：

| 指标 | 原要求 | 修订后 |
|---|---|---|
| Live PF | ≥ 1.3 | ≥ 1.3 **且 spread-adjusted PF ≥ 1.15** |
| Live WR | ≥ 38% | ≥ 38% |
| Max DD | < 5% | < 5% |
| 基础设施 | 无 >1 bar 失败 | 无 >1 bar 失败 **且有告警记录** |
| 设置分布 | （未要求） | 方向/trigger/confluence 分布与 OOS 相差 ±10% |
| Range mode | （未要求） | 若启用：range-only PF ≥ 1.0，range 交易数 ≥ 30 |
| AI 校准 | （未要求） | conf≥0.8 桶的实际 WR 不低于 stated 的 85% |
| Session 过滤 | ASIAN 硬 block | 30+ trades 回测 validation，CI 不包含 PF=1.0 才保留 |
| v1.1 inversion | （未要求） | hold-out 验证，WR hold-out − calibration ≥ −5% 才启用 |

---

## 6. 三份原始审核 agent 报告摘要

| Agent | 关键贡献 | 报告可信度 |
|---|---|---|
| Bug Hunter | 14 个条目，含 4 个 Critical（datetime、unused param、AI fallthrough、journal 漏 range）；精准到行号 | 已 spot-check 3 个，全部属实 |
| Architecture Reviewer | 13 个条目 + 完整 5-phase Sprint 11 plan；最强推论是"两条 live 路径"和"range 无 backtest" | 已 spot-check 2 个，属实 |
| Trading Risk Reviewer | 10 个条目 + APA 引用（LLM 校准文献）；最强推论是"趋势形成期做反转"和"overfitting 风险" | 推理链条严密，但 overfitting 的量化结论（40–60% FP）是保守估算 |

三位审核员**独立一致指出**的问题（✅ 标记）有 7 条，这些几乎可以认为是事实而非观点。

---

## 7. References（APA 7th）

- Benjamini, Y. (2010). Discovering the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 72(4), 405–416. https://doi.org/10.1111/j.1467-9868.2010.00746.x
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language models (mostly) know what they know. *arXiv preprint arXiv:2207.05221*. https://arxiv.org/abs/2207.05221
- Tian, K., Mitchell, E., Zhou, A., et al. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.330/
- Lopez de Prado, M. (2018). *Advances in financial machine learning*. Wiley. [multiple-comparison / backtest overfitting 方法论]

---

*报告生成：Claude Opus，T3 三路并行审核协议。三位审核 agent 独立阅读代码，最终结论由上层 agent 去重与交叉验证。*
