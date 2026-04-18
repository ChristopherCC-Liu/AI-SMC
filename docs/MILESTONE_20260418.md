# AI-SMC 里程碑交接文档 (2026-04-18)

> **版本**: v4.6-Y (34 commits today)  
> **状态**: Paper tested + MT5 live_execute 打通, Monday live 首日待验  
> **作者**: Lead (Opus 4.7) + strategy-lead + skeptic-lead 团队  
> **交付对象**: 未来的 Lead / 接手的工程师 / 用户自查  

---

## 0. 一句话摘要

AI-SMC 是 **XAUUSD (现货黄金) 量化交易系统**：M15 cycle + SMC (Smart Money Concepts) 策略 + AI 辅助方向判断，运行在腾讯云 Windows VPS 上，通过 MT5 Python API 执行 TMGM Demo 账户。今日（2026-04-17 UTC）经 34 commits 从 "不开仓" 调到 "能真实 mt5.order_send"，MT5 AutoTrading 已启用，下周一市场开盘会产生首笔 live trade。

---

## 1. GitHub 仓库

| 字段 | 值 |
|---|---|
| 仓库 URL | https://github.com/ChristopherCC-Liu/AI-SMC.git |
| 默认分支 | `main` |
| Owner | ChristopherCC-Liu |
| 当前 HEAD | `facab7f` fix(Round4.6-Y): clamp position_size_lots to MT5 min 0.01 |
| 今日 commits | 34 (Round 4.6-A 到 4.6-Y) |
| Test 覆盖 | 328 strategy unit tests all pass (ignoring yfinance-dependency tests) |

### 本地 workspace（Lead Mac）
```
/Users/christopher/claudeworkplace/AI-SMC/
```

### 克隆新环境
```bash
git clone https://github.com/ChristopherCC-Liu/AI-SMC.git
cd AI-SMC
python -m venv .venv
.venv/bin/pip install -e .  # 项目 pyproject.toml 设计
pytest tests/smc/unit/strategy/  # 预期 328 passed
```

---

## 2. 服务器部署

### VPS 基本信息

| 字段 | 值 |
|---|---|
| Provider | 腾讯云 |
| OS | Windows Server (TMGM MT5 Terminal bundled) |
| Public IP | `43.163.107.158` |
| Hostname | `10_3_4_3` |
| Timezone | CST (UTC+8) |
| 登录 | `Administrator` / `Lcc5201314^^` (SSH + RDP) |
| Project root | `C:\AI-SMC\` |

### SSH 连接
```bash
SSHPASS='Lcc5201314^^' sshpass -e ssh -o StrictHostKeyChecking=no Administrator@43.163.107.158
```

### 部署关键路径
```
C:\AI-SMC\
├── .venv\Scripts\python.exe         # Python 环境（已装 MetaTrader5, fastapi, polars 等）
├── scripts\
│   ├── live_demo.py                 # 主交易 loop (M15 cycle)
│   ├── dashboard.py                 # Textual TUI dashboard (legacy)
│   ├── dashboard_server.py          # FastAPI Web dashboard (port 8765)
│   ├── start_live.bat               # Task Scheduler AI-SMC-Live 触发
│   ├── start_dashboard.bat          # Task Scheduler AI-SMC-Dashboard
│   ├── start_dashboard_web.bat      # Task Scheduler AI-SMC-DashboardWeb
│   ├── test_mt5_order_connectivity.py  # MT5 connectivity smoke
│   └── check_autotrading.py         # MT5 flags diagnosis
├── src\smc\                         # 核心策略模块
├── data\
│   ├── live_state.json              # 每 cycle 写入, dashboard 读
│   ├── journal\live_trades.jsonl    # 所有 ENTER 记录 (paper + live)
│   ├── live_demo.pid                # 4.6-L/M 单实例 guard
│   ├── asian_range_quota_state.json # Asian 1 trade/day 持久化
│   └── user_config.json             # Dashboard 设置页写入
├── dashboard\index.html             # Web UI (Tailwind + Alpine.js)
└── logs\
    ├── live_stdout.log
    ├── live_stderr.log
    └── dashboard_*.log
```

### MT5 账户
| 字段 | 值 |
|---|---|
| Broker | TradeMaxGlobal (TMGM) |
| Account | `60078709` |
| Server | `TradeMaxGlobal-Demo` |
| Type | Demo (paper money, $1000 start balance) |
| Leverage | 100:1 (typical retail XAUUSD) |
| Min lot | **0.01** (broker-wide, confirmed by user) |
| Magic number | `19760418` (区分 AI-SMC orders) |
| AutoTrading | **Enabled (2026-04-17 UTC 23:xx 用户手动启用)** |
| Python API | Enabled (tradeapi_disabled: False) |

### Windows Task Scheduler 三任务
```
\AI-SMC-Live         → scripts/start_live.bat         → live_demo.py (OnStart trigger)
\AI-SMC-Dashboard    → scripts/start_dashboard.bat    → dashboard.py TUI (OnStart)
\AI-SMC-DashboardWeb → scripts/start_dashboard_web.bat → dashboard_server.py (OnStart)

关键: 
  /sc ONSTART (无 5min repeat)      ← 4.6-N 结构修复 (避免 IgnoreNew race)
  MultipleInstancesPolicy: IgnoreNew
  4.6-L PID guard 防并发
```

### Firewall
```
Port 8765 allow (Windows firewall) — dashboard web server
外网访问: 腾讯云 Security Group 未开 8765 (Round 5 scope)
访问路径:
  A) VPS 远程桌面 → http://localhost:8765
  B) SSH tunnel:    ssh -L 8765:localhost:8765 Administrator@43.163.107.158 
     然后本地 http://localhost:8765
```

### 运行进程
```
6 Python processes (3 logical × 2 venv launcher+interpreter):
  live_demo.py        (每 M15 cycle 开/不开单)
  dashboard.py        (TUI, legacy)
  dashboard_server.py (FastAPI Web, port 8765)
```

---

## 3. 系统架构

### 3.1 Cycle 主循环 (live_demo.py)
```
每 M15 bar close (UTC :00, :15, :30, :45):
  1. Fetch MT5 data (D1/H4/H1/M15 各 500 bars)
  2. AI Direction Engine (direction_engine.py)
     - Cache TTL 1h (4.6-Q)
     - Claude Code CLI (claude -p) debate pipeline
     - Fallback: SMA 20/50 direction
  3. Regime classifier (ATR-based, fallback 简单分类)
  4. SMC detection (OB / FVG / CHoCH / BOS)
  5. HTF bias (D1+H4 Tier 1/2/3)
  6. Range detection (range_trader.py):
     Method A: OB boundaries (上下 OB 对齐)
     Method B: swing extremes (最近 swing high/low)
     Method D: Donchian channel (48-bar) ← 现在主导
  7. Guards (check_bounds_only_guards + check_range_guards):
     Guard 1 width ≥ 400/800 (Asian/London)
     Guard 2 RR ≥ 1.2 (uniform, 4.6-K wire)
     Guard 3 touches ≥ 2
     Guard 4 duration ≥ 8/12 bars
     Guard 5 lot (downstream 4.6-Y clamp)
  8. Mode router (mode_router.py):
     Priority 1: AI strong + non-Asian → trending (v1 5-gate)
     Priority 2: range + guards + session + (regime!=trending OR Asian) + price_in_range → ranging
     Priority 3: fallback → v1_passthrough
  9. Action (determine_action):
     trending → v1 5-gate setups via aggregator
     ranging → range_trader generate_range_setups (w/ soft reversal 4.6-V)
     v1_passthrough → trust v1 HTF bias
  10. MT5 execution (4.6-X):
      if SMC_MT5_EXECUTE=1:
        mt5.order_send(BUY/SELL XAUUSD volume SL TP deviation)
        journal 记录 "mode": "LIVE_EXEC" + ticket
      else:
        journal 记录 "mode": "PAPER" (log only)
  11. Save state → data/live_state.json (dashboard read)
```

### 3.2 策略层次

| 层 | 职责 | 关键文件 |
|---|---|---|
| Data | MT5 polling + OHLCV frames | `scripts/live_demo.py` fetch_mt5_data |
| Detection | SMC patterns (OB/FVG/CHoCH/BOS/Swing) | `src/smc/smc_core/detector.py` |
| HTF Bias | D1+H4 directional bias | `src/smc/strategy/htf_bias.py` |
| Regime | Trending/Ranging/Transition | `src/smc/strategy/regime.py` |
| Range | Mean-reversion range detection + setup | `src/smc/strategy/range_trader.py` |
| Router | Mode selection (trending/ranging/v1) | `src/smc/strategy/mode_router.py` |
| Aggregator | v1 trend-following pipeline | `src/smc/strategy/aggregator.py` |
| Entry | M15 trigger (CHoCH/FVG/OB test) | `src/smc/strategy/entry_trigger.py` |
| AI | Direction + regime classifier | `src/smc/ai/direction_engine.py` |
| Execution | MT5 order_send 真实交易 | `scripts/live_demo.py` L605-640 (4.6-X) |

### 3.3 关键数据结构
```python
RangeBounds       # range_trader 检测的 upper/lower
RangeSetup        # mean-reversion 开仓 spec (direction, entry, SL, TP, TP_ext, rr_ratio)
TradeSetup        # v1 trending 开仓 spec (entry_signal + zone + bias + confluence)
TradingMode       # router decision (mode/reason/ai_dir/ai_conf/regime/range_bounds)
```

---

## 4. Dashboard

### Web UI
```
dashboard/index.html (31 KB, single-file, Tailwind CDN + Alpine.js)
scripts/dashboard_server.py (FastAPI, 137 lines, port 8765)

API endpoints:
  GET  /api/state          # 读 data/live_state.json + freshness
  GET  /api/journal?limit  # 读 journal tail
  GET  /api/config         # 读 user_config.json
  POST /api/config         # 写 user_config.json
  POST /api/toggle_trading # 创建/删除 data/trading_paused.flag
  GET  /                   # serve dashboard/index.html
```

### 用户视角
- Hero: 持仓状态 / 今日 P&L / 金价（大字 text-4xl）
- 健康条: MT5 连接 / 数据新鲜 / 下次 cycle 倒计时
- 翻译层 8 条: "为什么不开仓" → 人话 (`price_mid_range` → "价格在区间中段, 等靠近上下边界")
- 区间可视化水平条 + 今日 trades 表
- 设置 Tab: 🔴 大红 KillSwitch + boundary_pct / RR / quota sliders + session toggles

### 文档
- `docs/dashboard_prd.md` (PRD v1.0, 1880 字)
- `docs/dashboard_guide.md` (用户中文指南, 6 KB)
- `dashboard/SPEC.md` (字段契约归档, 9 KB)

---

## 5. 今日 (2026-04-17) Commits Timeline

所有 34 个 commit 按 session 分组：

### 上午 Guard 体系结构修复 (UTC 06:00-08:30)
```
1dd609f  Round4.5.1  Method A/B duration_bars
cd9555d  Round4.6-A  Donchian lookback 24→48, min_range_width 300→200
ef4f1e7  Round4.6-B  Asian 专用 Guards (width 400 / duration 8)
c32d5ef  Round4.6-C2 measure-first diagnostic for range
c75e46a  Round4.6-C2-ext 3-stage diagnostic
72ba1e8  Round4.6-D  max_range_width 3000→20000
ae6f7d6  Round4.6-E  live_demo guards_passed integration
bcbb0a6  Round4.6-F  Asian boundary_pct 15%→30% (USER)
6e96820  Round4.6-G  zone width 一致 (skeptic catch)
4a4daa8  Round4.6-H  range journal + AsianRangeQuota persist
ed80508  Round4.6-H+ confidence/risk/reward 字段
02d7889  Round4.6-I  London/NY 30% (USER)
4d04966  Round4.6-J  lot size 字段 (USER CATCH)
4a512c4  Round4.6-J-fix margin 语义 + frozenset
4d02c84  Round4.6-J2 tuple→frozenset 一致
```

### 下午 Quality + 双 fix (UTC 08:30-12:00)
```
1e03576  Round4.6-K  Guard 2 RR≥1.2 at _build_setup (strategy)
31c2670  Round4.6-K  wire check_range_guards (Lead, 双层 defense)
d236be5  Round4.6-L  single-instance PID guard
fad7f68  Round4.6-M  realpath PID cwd-independent
4289452  Round4.6-N  track launcher bats + OnStart trigger
```

### Dashboard + AI + 路由 (UTC 12:00-13:00)
```
93f4673  feat(dashboard) HTML + FastAPI MVP
c0c9841  fix(dashboard) start_dashboard_web.bat pause 移除
fb47e6f  Round4.6-O  rr_ratio 用 TP_ext (USER CATCH 永远不开单)
a36a722  Round4.6-Q  AI cache TTL 4h→1h (USER)
5dbd6eb  Round4.6-R  regime=trending non-Asian → v1_passthrough (USER)
```

### V1 诊断链 + 路由精化 + 开仓 (UTC 13:00-17:00)
```
20b8354  Round4.6-S-diag aggregator stage-by-stage reject
d0f489d  Round4.6-T-diag per-zone reject count
bc13cd3  Round4.6-T-v2 zone 坐标 + price 距离
ab75772  Round4.6-T-v3 mode_router price-in-range 精化
456390c  Round4.6-U soft 3-bar reversal fallback
6ba6483  Round4.6-V soft fallback chase mode 保底  ← 首笔开仓
```

### 风控 + MT5 execute (UTC 17:00-23:00)
```
04a2d47  Round4.6-W 30min same-direction cooldown
d5dad00  Round4.6-X MT5 order_send 真实 (USER CATCH PAPER only)
1395ab9  Round4.6-X connectivity smoke test
facab7f  Round4.6-Y position_size_lots min 0.01 clamp (USER)
```

---

## 6. Journal 11 条 (2026-04-17, 全 PAPER mode)

| # | Time UTC | Action | Entry | RR | Grade | Session |
|---|---|---|---|---|---|---|
| 1 | 08:15 | RANGE BUY | $4783.63 | 0.78 | C | LONDON |
| 2 | 08:30 | RANGE BUY | $4784.44 | 0.73 | C | LONDON |
| 3 | 16:45 | RANGE SELL | $4868.45 | 2.31 | A | NEW YORK |
| 4 | 17:00 | RANGE SELL | $4866.97 | 2.19 | A | NEW YORK |
| 5 | 17:15 | RANGE SELL | $4862.78 | 1.89 | B | NEW YORK |
| 6 | 17:30 | RANGE SELL | $4858.54 | 1.62 | B | NEW YORK |
| 7 | 17:45 | RANGE SELL | $4855.01 | 1.45 | C | NEW YORK |
| 8 | 18:00 | RANGE SELL | $4862.47 | 1.86 | B | NEW YORK |
| 9 | 18:15 | RANGE SELL | $4861.14 | 1.79 | B | NEW YORK |
| 10 | 19:00 | RANGE SELL | $4858.14 | 1.60 | B | NEW YORK |
| 11 | 19:45 | RANGE SELL | $4860.23 | 1.73 | B | NEW YORK |

**全部 `"mode": "PAPER"`**。MT5 账户 60078709 实际 0 positions。用户周一 market open 后看 `"mode": "LIVE_EXEC"` 才是真单。

---

## 7. 已知问题 / Round 5 backlog

### P0 (阻塞)
- 无（所有 P0 今日清）

### P1 (下周关注)
1. Monday 周末 gap 风险 — 9 paper shorts 同 zone，若真实 4.6-W cooldown reset restart → 可能 runaway
2. PID guard restart reset — process 死后 PID file 不删, 下次 restart PID 检查 stale
3. reason_if_zero 诊断误报 — 4.6-W cooldown 时标为 "no_m15_choch"（次要 UX）

### P2 (系统改进)
4. 腾讯云 Security Group 开 port 8765 (外网直接访问 dashboard)
5. live_demo.py MT5 position sync → state.json 实时 P/L
6. user_config.json hot-reload → live_demo.py 读取 (当前 settings page 写入但未生效)
7. VPS auto-pull git cron (当前手动 pull)
8. AI cache TTL 的 smart refresh (当前 1h 固定, 可加 regime-change trigger)

### P3 (策略 iteration)
9. Same-zone 多单 concentration cap (4.6-W 每 30min 1 单, 未来可加 "max concurrent per zone")
10. TP_ext vs TP1 execution tradeoff (目前 MT5 只发 TP1, TP_ext 字段仅 journal)
11. Trailing stop 未实施
12. Partial close (分批止盈) 未实施

---

## 8. 团队结构 (Agent Framework)

### Tier 架构
```
Lead (Opus 4.7, 1M context)
├── strategy-lead (Opus teammate)
│   └── Sonnet subagents (code execution)
└── skeptic-lead (Opus teammate)
    └── adversarial review

External:
  ux-pm / design-fe — Dashboard 完成后 shutdown
```

### Lead Cron Loop
```
Schedule: 每 1 小时 :07 minute UTC fire
Prompt: 自主 check VPS + journal + Lead R/H/U 盘点 + 向用户汇报
TTL: Session-only, 7 days auto-expire
Purpose: 用户不在时自主 iterate + 监控 VPS 健康
```

### 规范 (今日 codified)
- 规范 4-updated: 发消息前 tail git log + 读 teammate last 2 messages
- 规范 11-v2: 先 verify 再 RAISE (4 rules: trace entry / math 2x / OBSERVATION vs RAISE / `git log` 前置)
- 规范 12: 小 fix Lead 直接改, 大实施派 Sonnet
- 规范 14: R/H/U 盘点 (Lead R=10 today, strategy H=2, skeptic H=4 F=2)
- 规范 18: 用户代理人模式 (用户明确 directive 时 Lead 自主 execute)
- 规范 19 v4: verify 层级 (grep / math / commit timestamp / 同伴 2-pass)

---

## 9. 操作 runbook

### 启动 / 重启 live_demo
```powershell
# 在 VPS PowerShell 或 RDP 执行
cd C:\AI-SMC
Stop-Process -Id (Get-Content data\live_demo.pid) -Force -ErrorAction SilentlyContinue
Remove-Item data\live_demo.pid -ErrorAction SilentlyContinue
schtasks /run /tn "AI-SMC-Live"
```

### 验证 MT5 AutoTrading
```powershell
cd C:\AI-SMC
.venv\Scripts\python.exe scripts\test_mt5_order_connectivity.py
# 预期 retcode 10018 (周末) 或 10009 (市场开 + 真实 execute)
```

### 查询当前状态
```powershell
Get-Content data\live_state.json | ConvertFrom-Json | Format-List
Get-Content data\journal\live_trades.jsonl -Tail 5
Get-Content data\live_demo.pid
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CreationDate, CommandLine
```

### 关停交易（紧急 kill switch）
```powershell
# 1. 停 Task Scheduler 自动重启
schtasks /change /tn "\AI-SMC-Live" /disable

# 2. 停 live_demo 进程
Stop-Process -Id (Get-Content C:\AI-SMC\data\live_demo.pid) -Force

# 3. 关 MT5 AutoTrading toolbar 按钮 (Ctrl+E)

# 4. 如需全平仓, 在 MT5 terminal 右键 position → Close
```

### Deploy 新 commit
```powershell
cd C:\AI-SMC
git pull
# 如需立即生效 (python 不热加载):
Stop-Process -Id (Get-Content data\live_demo.pid) -Force
Remove-Item data\live_demo.pid
schtasks /run /tn "AI-SMC-Live"
```

---

## 10. 紧急联系 / 凭据

| 类型 | 信息 |
|---|---|
| VPS SSH | Administrator @ 43.163.107.158, password `Lcc5201314^^` |
| MT5 Account | 60078709 @ TradeMaxGlobal-Demo |
| MT5 Password | (saved in MT5 terminal, not hardcoded anywhere) |
| GitHub Owner | ChristopherCC-Liu (wmzmlcc@gmail.com per .env.example) |
| Email | wmzmlcc@gmail.com |

---

## 11. Next Monday Expectations (2026-04-20)

```
UTC 22:00 Sunday  XAUUSD market reopen (46h closed)
UTC 00:00 Monday  Asian session 开始
First LIVE_EXEC trade 预计 UTC 00-08 Asian 或 UTC 08-12 London

需 watch:
  - Journal 新 entry "mode": "LIVE_EXEC" + mt5_ticket: <number>
  - MT5 terminal positions (应见 XAUUSD BUY/SELL 0.01 lot)
  - Balance 变动 (从 $1000 变)
  - Dashboard Hero "今日持仓" 数字变化 (目前未 wire 真实 MT5 positions, Round 5)

如 Monday 首 cycle HOLD → 看 live_state.json smc_diagnostic / range_diagnostic
如 order_send fail → 查 stderr logs + retcode 查 MT5 docs
```

---

## 12. 免责 + 风险披露

- 本系统是**研发项目**，不是金融产品
- **TMGM Demo 账户**，$1000 虚拟资金，亏损不影响真钱
- Lead 不承诺盈利 (见本档第 11 节分析 + 用户问"多久翻倍"的诚实回答)
- 策略 expectancy **尚未经充分 live 验证** (今日 11 paper trades 远不足)
- 周末 gap / black swan / broker 故障等 tail risk 未量化
- 切勿照搬到真钱账户 without 至少 30 天 live profitability + profit factor > 1.5 验证

---

**交接完毕**。下一位 Lead 接手只需：
1. Read this doc end-to-end
2. SSH VPS + run `test_mt5_order_connectivity.py` verify MT5 OK
3. `git log -5` 看最近变动
4. Monday 早起 (UTC 00:00 起) 观察首笔 LIVE_EXEC trade

祝交易顺利。
