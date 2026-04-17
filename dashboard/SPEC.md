# Dashboard SPEC — Subagent 共享契约

> 这是 4 个 Sonnet subagent 的共享契约，**不要改措辞、不要猜字段**。
> PRD: `docs/dashboard_prd.md`。SPEC 基于 PRD v1.0 + 实际代码字段 verify 后固化。

---

## 1. 路径约定

| 文件 | 路径 |
|---|---|
| 单页 HTML | `/Users/christopher/claudeworkplace/AI-SMC/dashboard/index.html` |
| FastAPI server | `/Users/christopher/claudeworkplace/AI-SMC/scripts/dashboard_server.py` |
| Server 端口 | `8765` (PRD § 7 指定) |

## 2. 数据源字段契约（**verify 过**）

### `data/live_state.json`（每 M15 cycle 重写，由 `live_demo.py:save_state()` 写入）

**顶层字段（一定存在）**:
- `cycle`: int — 第几轮
- `timestamp`: str ISO8601 UTC — 最新 cycle 时间
- `price`: float — 当前价（USD）
- `action`: str — `"BUY"` / `"SELL"` / `"HOLD"` / `"RANGE BUY"` / `"RANGE SELL"`
- `reason`: str — 如 `"price_mid_range"` / `"no_m15_choch_at_lower"` / `"ai_neutral_v1_fallback"` 等
- `regime`: str — `"ranging"` / `"trending"` / `"transition"`
- `ai_direction`: str — `"bullish"` / `"bearish"` / `"neutral"` 或 `"?"`
- `ai_confidence`: float — 0.0–1.0
- `setups_count`: int
- `volatility`: str — `"HIGH"` / `"NORMAL"` / `"LOW"` / `"?"`
- `trading_mode`: str — `"ranging"` / `"trending"` / `"v1_passthrough"`

**可选字段（可能 null）**:
- `range_bounds`: `null` 或 `{upper: float, lower: float, width: float}`
- `range_diagnostic`: `null` 或 `{detect: {...}, guards: {...}, setups: {...}}`
  - `detect.method` / `detect.source`: 取值 `"orderblock"` / `"swing"` / `"donchian"`
  - `guards.width_pass / touches_pass / rr_pass / duration_pass / adx_pass`: bool
  - `guards.rr_value`: float（实际 RR）
  - `guards.rr_threshold`: float
  - `setups.price_at_upper / price_at_lower`: bool
  - `setups.m15_choch_confirmed`: bool
- `best_setup`: `null` 或 `{direction, entry, sl, tp1, trigger, confluence}`

**绑定原则（规范 11-v2）**: 每个嵌套字段 bind 前必须检查 null，否则 display "N/A"。

### `data/journal/live_trades.jsonl`（JSON lines，每笔开仓追加一行）

每行字段:
- `time`: ISO8601 UTC
- `cycle`: int
- `price`: float
- `direction`: `"BUY"` / `"SELL"`
- `entry`: float
- `sl`: float
- `tp1`: float
- `rr_ratio`: float
- `trigger`: str — `"orderblock_bull"` / `"swing_sell"` / `"donchian_breakout"` / ...
- `range_lower` / `range_upper`: float
- `regime`: str
- `ai_direction`: str
- `mode`: `"PAPER"` / `"LIVE"`
- `trading_mode`: str
- `session`: `"asian"` / `"london"` / `"ny"`

### `data/ai_analysis.json`（每 H4 重写，由 `run_ai_analysis()` 写入）

- `source`: `"llm"` / `"technical"`
- `assessed_at`: ISO8601
- `ai_direction`: `"bullish"` / `"bearish"` / `"neutral"`
- `ai_confidence`: float 0–1
- `volatility`: `"HIGH"` / `"NORMAL"` / `"LOW"`
- `atr_pct`: float
- `reasoning`: str（一句话）
- `key_factors`: list[str]（可能为空）

### `data/user_config.json`（**dashboard 设置页写入**，live_demo 下 cycle 读取）

结构 (POST /api/config 的 body 形状):
```json
{
  "updated_at": "2026-04-17T12:00:00+00:00",
  "risk": {
    "boundary_band_pct": 0.30,
    "min_rr_ratio": 1.2,
    "asian_daily_quota": 2
  },
  "sessions": {
    "asian_enabled": true,
    "london_enabled": true,
    "ny_enabled": true,
    "ai_filter_neutral_block": true
  }
}
```

### `data/trading_paused.flag`（KillSwitch，**存在即暂停**）

- 存在（文件存在，内容任意）→ 暂停
- 不存在 → 运行中

---

## 3. 翻译表（**措辞严格契约，不可改**）

来源: PRD § 4 卡片 3。`ReasonCard` 组件根据 `live_state.reason` + `range_diagnostic` 字段查表：

```js
const TRANSLATION_TABLE = [
  {
    match: (s) => s.reason === "price_mid_range",
    text: "价格在区间中段，等靠近上下边界再说"
  },
  {
    match: (s) => s.reason === "no_m15_choch_at_lower",
    text: "价格到了支撑位，但 M15 还没出现反转信号，再等 15 分钟"
  },
  {
    match: (s) => s.reason === "no_m15_choch_at_upper",
    text: "价格到了阻力位，但 M15 还没出现反转信号，再等 15 分钟"
  },
  {
    match: (s) => s.range_diagnostic?.guards?.rr_pass === false,
    text: "找到机会了，但风险回报比 < 1.2:1，划不来"
  },
  {
    match: (s) => s.range_diagnostic?.guards?.width_pass === false,
    text: "区间太窄（< 400 点），等市场扩大波动"
  },
  {
    match: (s) => s.range_diagnostic?.guards?.touches_pass === false,
    text: "上下边界还没被确认碰过 2 次，等再摸一次"
  },
  {
    match: (s) => s.trading_mode === "v1_passthrough",
    text: "AI 看不清方向，交给趋势策略判断"
  },
  {
    match: (s) => s.range_diagnostic?.detect?.method == null,
    text: "找不到可交易区间，市场可能在单边趋势中"
  },
];
// fallback: 显示原始 reason 字符串
```

**匹配顺序**: 按上表从上到下，首个 match 即停。具体字段是否存在要先 null-check。

### 区间来源翻译（卡片 4 + 卡片 5）

```js
const SOURCE_LABEL = {
  "orderblock": "订单块",
  "swing": "摆动高低",
  "donchian": "Donchian 通道",
};
```

### 触发类型翻译（trades 表 trigger 列）

```js
// 提取 trigger 字符串里的 "orderblock"/"swing"/"donchian"，加方向后缀
function translateTrigger(trig) {
  if (!trig) return "未知";
  if (trig.includes("orderblock")) return trig.includes("bull") ? "订单块·多" : "订单块·空";
  if (trig.includes("swing")) return trig.includes("bull") || trig.includes("buy") ? "摆动·多" : "摆动·空";
  if (trig.includes("donchian")) return "Donchian 突破";
  return trig;
}
```

### AI 方向翻译

```js
const AI_DIR_LABEL = {
  "bullish": "看多",
  "bearish": "看空",
  "neutral": "中性",
  "?": "N/A",
};
```

---

## 4. 组件边界（subagent 分工）

### Subagent 1 (HTML skeleton + Hero + 健康条 + Tabs)

- `<html>` 骨架 + `<head>` 引入 Tailwind Play CDN + Alpine.js CDN + 自定义 `<style>` (等宽字体)
- `<body>` 顶部固定 nav: logo + Tab (监控 | 设置) + 全局状态灯
- Alpine.js `x-data="dashboard()"` 顶层 store: `state, journal, config, tab, loading, error`
- fetch 循环: `setInterval` 5 秒拉 `/api/state` + `/api/journal`
- 监控页卡片 1 `StatCard` × 3 (持仓 / P&L / 金价)
- 监控页卡片 2 `StatusBadge` × 3 (MT5 / 数据新鲜 / 倒计时)
- 占位 `<div id="card-3">`, `<div id="card-4">`, `<div id="card-5">`, `<div id="card-6">` 和设置页 `<div id="settings-page">` 给 subagent 2/3 填充

### Subagent 2 (翻译层 + 区间条 + trades 表 + AI 卡)

- 卡片 3 `ReasonCard`: 查表翻译 + 折叠原始 range_diagnostic JSON
- 卡片 4 `RangeBar`: 纯 CSS horizontal bar，dot 位置 = `(price - lower) / width`
- 卡片 5 `TradesTable`: 响应式表格，遍历 journal 今日条目
- 卡片 6 `AI 卡`: 折叠默认 (details/summary)

### Subagent 3 (设置页)

- 🔴 大红 KillSwitch + 双重确认 modal (检查 `/api/status` 里 `trading_paused: true/false`)
- 3 sliders: `boundary_band_pct`, `min_rr_ratio`, `asian_daily_quota`
- 4 toggles: `asian_enabled`, `london_enabled`, `ny_enabled`, `ai_filter_neutral_block`
- 保存按钮 POST `/api/config`

### Subagent 4 (FastAPI server)

- `scripts/dashboard_server.py` ~60 行
- Endpoints:
  - `GET /api/state` → 读 `data/live_state.json` + `data/ai_analysis.json` + 状态灯 (stale check: timestamp > 5min 前视为 stale) + `trading_paused` bool
  - `GET /api/journal?limit=50` → 读 `data/journal/live_trades.jsonl` 尾 N 行
  - `POST /api/config` → 写 `data/user_config.json`（覆盖）
  - `POST /api/toggle_trading` → body `{paused: true/false}` → touch/remove `data/trading_paused.flag`
  - `GET /` → serve `dashboard/index.html`
- CORS allow `localhost:*`
- uvicorn 启动在 `0.0.0.0:8765`

---

## 5. 配色常量

```
bg:       #0F172A (slate-900)
card:     #1E293B (slate-800)
text:     #F1F5F9 (slate-100)
long:     #10B981 (emerald-500)
short:    #EF4444 (red-500)
muted:    #94A3B8 (slate-400)
accent:   #A78BFA (violet-400)
warn:     #F59E0B (amber-500)
```

## 6. 3 种 user state 视觉

- **持仓中**: Hero 左侧大字 "多单 @ $4,780" (emerald) / "空单 @ $4,780" (red)
- **等机会 HOLD**: Hero 左侧大字 "空仓中" (slate-400) + 卡片 3 显示翻译理由
- **系统异常**: 全局状态灯红 + Hero 遮罩 "系统可能已断线" + timestamp 显示 "XX 分钟前，可能已停" + 红色边框

判定逻辑:
- **持仓**: journal 尾部最近一行未平仓（本 MVP 简化: `action != "HOLD"` 且 timestamp 在 30min 内视为 live position）
- **HOLD**: `state.action === "HOLD"`
- **异常**: `state` 为 null / fetch fail / `timestamp` 距离 now > 5min

---

## 7. 依赖版本（CDN）

- Tailwind: `<script src="https://cdn.tailwindcss.com"></script>` (Play CDN v3)
- Alpine: `<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>`

## 8. 关键：禁止事项

- **不允许**修改翻译表措辞
- **不允许**bind `state.xxx` 而不先 null-check
- **不允许**引入 Chart.js / D3 / 其他 lib（用纯 CSS + Alpine）
- **不允许**修改 FastAPI port (必须 8765)
- **不允许**改 PRD 颜色或 Hero 字号约定
