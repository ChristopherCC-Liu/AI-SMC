# AI-SMC Dashboard — PRD

> v1.0 · 2026-04-17 · 作者: UX PM (AI-SMC team)
> 交付对象: design-fe teammate
> 约束: MVP, 单 HTML 文件, 中文界面

---

## 1. 用户画像 + 痛点诊断

### 用户画像
**主用户**: 项目所有者本人（量化交易员 / Boss 视角）
- 已部署 VPS 实盘 ($1000 TMGM demo)，每 M15 cycle 轮询
- **不想**: 看 Textual TUI、JSON、SSH 终端输出
- **想要**: 手机/浏览器扫一眼就知道"今天系统在不在工作、开没开仓、为什么不开"
- 技术背景有，但 dashboard 的目的是"省脑力"而非"秀细节"

### Top 3 疑问（用户打开 dashboard 时脑中的问题）
1. **"现在持仓了吗？今天盈亏多少？"** — 最高优先级，决定要不要紧张
2. **"系统还在工作吗？MT5 连着吗？"** — 第二优先级，系统死没死
3. **"为什么刚才那根 K 没开仓？"** — 第三优先级，理解行为而非盲信

现有 `scripts/dashboard.py` 把三者打散在 9 个 panel 里（account / price / regime / ai / health / positions / journal / log / banner），**用户看完还是不知道答案** —— 这是"用不明白"的根因。

---

## 2. 信息架构 (IA)

把数据源按"用户 care 程度"三档分类。**翻译成人话**，避免 jargon。

### 必看（首屏，不滚动）
| 字段（原始） | 显示文案 | 数据来源 |
|---|---|---|
| action / best_setup.direction | **"现在: 多单 / 空单 / 持币"** | live_state.json |
| 今日 trades 条数 + 未平仓 P&L | **"今日 2 笔 · 浮盈 +$12.4"** | journal + MT5 positions |
| price | **"金价 $4,784.20"** | live_state.price |
| cycle timestamp | **"5 分钟前更新 · 下次 10 分钟后"** | live_state.timestamp |
| MT5 connection | 绿点 / 红点 "系统正常 / 已断线" | health check |
| reason（HOLD 时） | **"为什么不开: 价格在区间中间，等靠近边界"** | translate reason/diagnostic |

### 可看（滚动可见）
- 区间上下轨 + 当前价位置（可视化条）
- 今日 trades 列表（时间 / 方向 / 入场价 / SL / TP / 风险回报比）
- AI 方向判断（看多/看空/中性）+ 1 句理由
- 账户余额 / 保证金占用 / 杠杆

### 折叠（点击展开，debug 用）
- range_diagnostic 三层（detect / guards / setups）细节
- 最近 cycle 日志
- SMC 字段: swing points / order blocks / CHoCH / confluence

---

## 3. 页面结构

**推荐: 单页双 Tab**（顶部切换: "监控 | 设置"）

理由:
- 用户 80% 时间只看监控，设置改完就走
- 分两个 URL 增加心智负担（用户会忘记设置 URL）
- 移动端单页体验更连贯

顶部固定:
- 左: "AI-SMC" logo + 账户名 `TMGM-60078709`
- 中: Tab 切换（监控 / 设置）
- 右: 全局系统状态灯（绿 = 运行中，红 = 断线，黄 = 数据陈旧）

---

## 4. 监控页 — 核心卡片（按从上到下优先级）

### 卡片 1: 「现在怎么样」英雄区 (Hero)
**最重要，占首屏 1/3**。三块：

```
  [  持仓状态 大字  ]   [  今日 P&L 大字  ]   [  金价 + 涨跌  ]
  "多单 @ $4,780"      "+$12.4 · 1 胜 1 败"    "$4,784.20 ▲0.08%"
```
- 颜色: 多单绿 / 空单红 / 持币灰 / P&L 正绿负红
- 无持仓时显示 "空仓中"
- P&L 由 positions.profit + 今日已平 trades 计算

### 卡片 2: 「系统在工作吗」健康条
水平 3 个小徽章，全绿才安心：
- ✅ **MT5 连接** — "已连接 / 账户 $1,000"
- ✅ **数据新鲜** — "最新行情 12 秒前"
- ✅ **下次检查** — "8 分 32 秒后"（倒计时 from timestamp + 15min）

任一红色，点击展开说明该怎么办。

### 卡片 3: 「为什么不开仓」解释卡（HOLD 时显示，开仓时折叠）
核心翻译层。把 range_diagnostic 的三层转成 1-2 句人话：

**翻译规则**（design-fe 实现时查表）:
| reason / diagnostic 字段 | 人话文案 |
|---|---|
| `price_mid_range` | "价格在区间中段（$4,760~$4,800），等靠近上下边界再说" |
| `no_m15_choch_at_lower` | "价格到了支撑位，但 M15 还没出现反转信号，再等 15 分钟" |
| `no_m15_choch_at_upper` | "价格到了阻力位，但 M15 还没出现反转信号，再等 15 分钟" |
| guards.rr_pass=false | "找到机会了，但风险回报比 < 1.2:1，划不来" |
| guards.width_pass=false | "区间太窄（< 400 点），等市场扩大波动" |
| guards.touches_pass=false | "上下边界还没被确认碰过 2 次，等再摸一次" |
| mode=v1_passthrough | "AI 看不清方向，交给趋势策略判断" |
| detect all miss | "找不到可交易区间，市场可能在单边趋势中" |

展开可见原始 JSON（debug 折叠区）。

### 卡片 4: 「价格在区间哪里」可视化条
纯 CSS 水平条（无需 charting 库）：
```
  $4,760 ────●─────────── $4,800
  下轨      现在           上轨
  多单区      中段            空单区
```
- 红绿双色带（两端 30% 染色，中间灰）
- 当前价用圆点标注
- 下方显示 "区间来源: 订单块 / 摆动高低 / Donchian 通道"（bounds.source 翻译）

### 卡片 5: 「今日 trades」列表（表格）
最近 N 条，默认展开今天的：

| 时间 | 方向 | 入场价 | 止损 | 止盈 | 风险回报比 | AI 是否同意 | 区间来源 |
|---|---|---|---|---|---|---|---|
| 03:15 | 多 | $4,772 | $4,762 | $4,795 | 2.3 : 1 | ✅ 同意 | 订单块 |
| 04:45 | 多 | $4,774 | $4,763 | $4,791 | 1.5 : 1 | ⚪ 中性 | 订单块 |

空态: "今天还没开单，继续等"。

### 卡片 6: 「AI 看法」卡（折叠默认）
- 大字: 看多 / 看空 / 中性 + 置信度百分比
- 1 行理由（ai_analysis.reasoning 或 key_factors 第一条）
- "最后更新: 12 分钟前"

---

## 5. 设置页 — 核心可改项

**原则**: 只暴露用户真正会调的，不暴露代码层旋钮。所有改动需要**确认按钮**（危险），修改后写入 `data/user_config.json`，live_demo 下 cycle 读取。

### 最顶部: 🔴 大红色紧急开关
**"暂停交易 / 恢复交易"** — 一键 kill-switch。
- 实现: 写入 `data/trading_paused.flag`，live_demo 每 cycle 检查
- 点击时弹确认: "确定暂停？已开仓位不会自动平仓"

### 风险参数区
| 设置项 | UI 控件 | 默认值 | 说明文案 |
|---|---|---|---|
| 边界触发宽度 | Slider 10%-40% | 30% | "价格离边界多近才算入场信号" |
| 最低风险回报比 | Slider 1.0-2.5 | 1.2 | "低于这个数值不开单" |
| 单日最大开仓次数 | Number input | 2 (Asian quota) | "Asian 时段最多开几笔" |

### 策略开关区
| 开关项 | 默认 |
|---|---|
| Asian 时段交易 | ON |
| London 时段交易 | ON |
| NY 时段交易 | ON |
| AI 方向过滤（中性时暂停） | ON |

### 状态显示（只读）
- 配置文件路径: `C:\AI-SMC\data\user_config.json`
- 最后生效时间: "3 分钟前"
- **重要**: 设置改动不影响已持仓，只影响新信号

---

## 6. User Flow（3 种典型）

1. **盯盘流（占 70%）**: 打开 → Hero 大字看持仓/P&L → 卡片 2 确认绿灯 → 关闭。**10 秒内完成**。
2. **疑问流（占 25%）**: 打开 → 看到"持币" → 点卡片 3 "为什么不开" → 读人话解释 → 恍然大悟。
3. **调参流（占 5%）**: 切到"设置" → 拖动 RR slider 到 1.5 → 点保存 → 回到监控继续盯。

---

## 7. 交付物约束（Technical Handoff）

- **形态**: 单文件 `dashboard/index.html`（自包含，无 build step）
- **依赖**: Tailwind CDN (play or v3) + Alpine.js CDN
- **数据获取**: 两种模式双模运行
  - **模式 A (优先)**: 后台轻量 HTTP server 提供 `/api/state`（读 `data/*.json`）
  - **模式 B (fallback)**: `<input type="file">` 让用户选择 JSON 文件夹（本地浏览器无法访问本地 fs，需 user 手动上传或用 `file://`）
  - 推荐 **模式 A**: 新增 `scripts/dashboard_server.py`（FastAPI 60 行，read-only），部署 `localhost:8765`
- **刷新**: `setInterval` 5 秒拉 `/api/state`
- **响应式**: Tailwind grid, 手机单列 / 桌面双列
- **语言**: 中文 label + 英文原始字段（tooltip hover）
- **保存路径**: `/Users/christopher/claudeworkplace/AI-SMC/dashboard/index.html` + `scripts/dashboard_server.py`

---

## 8. Design Handoff 规格（给 design-fe）

### 配色（交易 + 友好）
- 背景: 深蓝灰 `#0F172A`（slate-900），卡片 `#1E293B`（slate-800）
- 主文字: `#F1F5F9`（slate-100）
- 多单 / 盈利: `#10B981`（emerald-500）
- 空单 / 亏损: `#EF4444`（red-500）
- 持币 / 中性: `#94A3B8`（slate-400）
- AI 高亮 / accent: `#A78BFA`（violet-400）
- 警告: `#F59E0B`（amber-500）

### 字体
- UI 文字: `ui-sans-serif, system-ui, -apple-system, "PingFang SC"`
- 数字（价格/P&L）: `"SF Mono", "JetBrains Mono", monospace` — 等宽对齐关键
- 大小: Hero 数字 `text-4xl md:text-5xl`, 卡片标题 `text-sm uppercase tracking-wider`

### 间距
- 卡片内 padding: `p-5 md:p-6`
- 卡片间 gap: `gap-4 md:gap-6`
- 容器最大宽: `max-w-7xl mx-auto px-4`

### 组件清单（design-fe 需实现）
1. **StatCard** — 大字 + 小 label + 可选趋势箭头（Hero 区三块）
2. **StatusBadge** — 点状灯 + 文字（绿/黄/红），支持 hover 展开说明
3. **ReasonCard** — "为什么不开"卡片，带翻译层 + 折叠原始 JSON
4. **RangeBar** — 纯 CSS 水平条，支持动态定位圆点
5. **TradesTable** — 响应式表格，桌面表格 / 手机卡片堆叠
6. **TabSwitcher** — 顶部监控/设置切换
7. **KillSwitch** — 大红按钮 + 双重确认模态
8. **Slider** + **Toggle** + **NumberInput** — 设置页三原子组件

---

## 9. 最关键的 3 个 Design Decision（交 design-fe）

1. **"为什么不开仓"人话翻译层是 dashboard 的灵魂** — 不只是可视化 JSON，而是把 `range_diagnostic.detect/guards/setups` 三层按优先级查表，输出 1-2 句话。这个翻译表需要 design-fe 严格按 § 4 卡片 3 的规则实现，**不要自行改写措辞**。
2. **Hero 区必须 10 秒内回答 Top 3 疑问** — 如果用户还需要滚动才能看到持仓/P&L/系统状态，设计失败。Hero 占屏比例 > 1/3，字号 > 48px。
3. **数据获取走 HTTP server 而非 file://** — 浏览器 file:// 无法跨域读 `data/*.json`，用户 "打开 HTML 就能用" 的期待需要靠 `dashboard_server.py`（新增 FastAPI 60 行）落地。design-fe 需同时交付 server 骨架（read-only `/api/state` + `/api/config` POST 保存）。

---

**字数**: ~1880 字（正文）
**下一步**: Lead 审核后转 design-fe 实现 `dashboard/index.html` + `scripts/dashboard_server.py`
