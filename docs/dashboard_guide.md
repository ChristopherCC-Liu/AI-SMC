# AI-SMC Dashboard 使用指南

> 面向项目所有者：怎么打开 dashboard、在哪看、怎么调设置。
> 不需要代码知识。

---

## 一、这个 Dashboard 是做什么的

1 秒看懂："现在持仓了吗？今天盈亏多少？系统还在工作吗？"

相比旧的 Textual TUI (`scripts/dashboard.py`)，这是一个**浏览器可以打开的网页**，手机也能看，不用 SSH。

---

## 二、两种使用场景

### 场景 A：在 VPS（Windows 服务器）本地打开

你的量化系统跑在 VPS 上（`43.163.107.158`）。可以直接在 VPS 上远程桌面登录后打开浏览器。

**启动步骤**：

1. 远程桌面登录 VPS（Administrator）
2. 打开文件资源管理器，进入 `C:\AI-SMC\scripts\`
3. **双击** `start_dashboard_web.bat`
4. 会弹出一个命令行黑窗口（不要关）
5. 打开 VPS 上的 Chrome / Edge，访问: **http://localhost:8765**

### 场景 B：从你的本地 Mac / 手机访问 VPS 的 dashboard

把 VPS 的 8765 端口通过 SSH 隧道转到本地，再用本地浏览器访问。

**Mac 终端执行（一次性）**：

```bash
ssh -L 8765:localhost:8765 Administrator@43.163.107.158
# 输入密码后保持这个终端不关
```

然后本地 Mac 的浏览器访问: **http://localhost:8765**

或者直接暴露 VPS 的 8765 端口（需要 Windows 防火墙放行），从任何设备访问: **http://43.163.107.158:8765**。出于安全考虑**不建议暴露到公网**，优先用 SSH 隧道。

### 场景 C：开发时本地预览（Mac）

1. 项目根目录下执行 `python scripts/dashboard_server.py`
2. 浏览器打开 http://localhost:8765

（需要 `data/live_state.json` 存在。如果没跑过实盘，会看到 "系统可能已断线" —— 正常现象，属于第 3 种 user state）

---

## 三、页面结构

### 顶部固定

- 左: **AI-SMC** logo + 账户 `TMGM-60078709`
- 中: **监控 | 设置** Tab 切换
- 右: 全局状态灯
  - 🟢 **运行中** — 一切正常
  - 🟡 **数据陈旧 / 已暂停** — 要留意
  - 🔴 **已断线** — 系统可能挂了

### 监控页（默认）

**6 张卡片，从上到下看重要性递减**：

1. **Hero 三块大字**（最重要）
   - 左：持仓状态（"多单 @ $4,780" 绿 / "空单 @ $4,780" 红 / "空仓中" 灰）
   - 中：今日 P&L
   - 右：金价 + 数据新鲜度
2. **系统健康**（3 个徽章）
   - MT5 连接 / 数据新鲜度 / 下次检查倒计时
3. **为什么不开仓**（HOLD 时出现）
   - 人话解释，例如："价格在区间中段，等靠近上下边界再说"
4. **价格在区间哪里**（可视化水平条）
   - 下轨 / 当前价 / 上轨 + 区间来源（订单块 / 摆动 / Donchian）
5. **今日交易**表格
   - 时间 / 方向 / 入场 / 止损 / 止盈 / 风险回报 / AI 同意 / 来源
6. **AI 方向看法**（折叠默认）
   - 看多 / 看空 / 中性 + 理由 + key factors

### 设置页

- 🔴 **紧急开关**：暂停 / 恢复交易（写 `data/trading_paused.flag`）
- **风险参数 Sliders**
  - 边界触发宽度（10-40%）
  - 最低风险回报比（1.0-2.5）
  - 亚盘每日最多开仓次数（0-5）
- **时段 / AI 开关**
  - 亚盘 / 伦敦 / 纽约 时段
  - AI 中性时暂停
- **保存按钮** → 写入 `data/user_config.json`
- **生效**：下个 cycle（15 分钟内）`live_demo.py` 读取新配置

---

## 四、3 种典型 User State

| 状态 | 顶部灯 | Hero 左侧 | 有没有卡片 3 |
|---|---|---|---|
| **持仓中** | 🟢 绿 | "多单 @ $X" 绿 或 "空单 @ $X" 红 | 不显示 |
| **等机会 HOLD** | 🟢 绿 | "空仓中" 灰 | 显示，带人话理由 |
| **系统异常** | 🔴 红 / 🟡 黄 | 可能是"空仓中"但顶部灯报警 | 可能无数据 |

---

## 五、常见问题

**Q: 打开网页一片空白 / 数据不刷新？**
A: 检查命令行黑窗口是不是关了。只要 bat 窗口还在, server 就在跑。页面每 5 秒自动刷新。

**Q: 全局状态灯红色 "已断线" 是什么意思？**
A: 意味着 dashboard server 联系不到 `data/live_state.json` 了。可能原因：
- `live_demo.py` 挂了 → 去 VPS 看 live_demo 命令行窗口
- 数据文件被误删 → 等下一个 cycle 恢复

**Q: 黄色 "数据陈旧" 是什么意思？**
A: `live_state.json` 的 timestamp 距离现在 >5 分钟了。正常 cycle 应该 15 分钟更新一次，所以 >20 分钟还没更新就有问题。

**Q: 我在设置页改了参数，什么时候生效？**
A: 下一个 M15 cycle（15 分钟内）`live_demo.py` 会读 `data/user_config.json`。
**注意**: 改动不影响已经开仓位，只影响新信号。

**Q: 🔴 紧急暂停后，已开仓位会自动平仓吗？**
A: **不会**。只会阻止开新仓。已开仓位需要你在 MT5 手动处理（或等 SL/TP 触发）。

**Q: 端口 8765 被占用？**
A: 编辑 `scripts/dashboard_server.py` 最后一行 `port=8765` 改成别的（如 8766）。

**Q: 想在手机上看？**
A: 用场景 B 的 SSH 隧道，或者在本地电脑打开 dashboard 时用 iPhone 连同一 WiFi 访问 `http://<mac-ip>:8765`。

---

## 六、文件清单

实施完会得到这些文件:

| 文件 | 作用 |
|---|---|
| `dashboard/index.html` | 单文件前端（Tailwind + Alpine.js） |
| `scripts/dashboard_server.py` | FastAPI 服务器（~60 行，端口 8765） |
| `scripts/start_dashboard_web.bat` | Windows 双击启动器 |
| `docs/dashboard_guide.md` | 本文档 |

运行时生成:

| 文件 | 作用 |
|---|---|
| `data/user_config.json` | 设置页保存的配置（live_demo 会读） |
| `data/trading_paused.flag` | 存在 = 系统已暂停 |

---

## 七、和现有 `scripts/dashboard.py` (Textual TUI) 的关系

- **旧 TUI**（`dashboard.py` / `smc-dashboard.bat`）：在 VPS 命令行跑，panel-heavy，dev 风
- **新 Web Dashboard**（本文档）：浏览器打开，user-first，中文界面

**两者可以同时运行**（端口和资源不冲突）。建议：
- 日常盯盘用新 Web Dashboard
- Debug cycle 细节保留 TUI（输出日志更详细）
