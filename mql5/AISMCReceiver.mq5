//+------------------------------------------------------------------+
//|                                            AISMCReceiver.mq5       |
//|                    AI-SMC MQL5 EA Signal Receiver                  |
//|                                                                    |
//|  Polls Python strategy_server at http://127.0.0.1:8080/signal?..   |
//|  every PollIntervalSec; on fresh non-HOLD signal, executes order   |
//|  via MT5 native CTrade (no Python IPC dependency).                 |
//|                                                                    |
//|  audit-r4 v5 Option B (dual-magic):                                |
//|    The /signal response now carries a `signals` array — one entry  |
//|    per leg (control "" and treatment "_macro") with its own magic. |
//|    The EA iterates each leg, dedupes + cools down independently    |
//|    per-magic, and OrderSends with the leg's magic number so broker |
//|    reconcile splits deals per-leg on the same TMGM Demo account.   |
//|                                                                    |
//|  Deployment:                                                       |
//|    1. MT5 → Options → EA Trading → Allow WebRequest for URL        |
//|       http://127.0.0.1:8080                                        |
//|    2. Copy AISMCReceiver.mq5  → <MT5 data folder>/MQL5/Experts/    |
//|    3. Copy include/aismc_panel.mqh → <MT5 data folder>/MQL5/Include|
//|    4. In MetaEditor, compile → attach to XAUUSD chart              |
//|    5. Attach a 2nd instance to BTCUSD chart                        |
//+------------------------------------------------------------------+
#property copyright "AI-SMC"
#property version   "2.10"
#property strict
#property description "Polls Python strategy server, executes SMC signals (dual-magic)."

#include <Trade/Trade.mqh>
#include <aismc_panel.mqh>

input string   SignalURL            = "http://127.0.0.1:8080/signal";
input int      PollIntervalSec      = 15;
input double   MinConfidence        = 0.3;
input int      MagicXAU             = 19760418;
input int      MagicBTC             = 19760419;
input int      DeviationPoints      = 100;
input double   DefaultLot           = 0.01;
input bool     EnableTrading        = true;
input bool     VerboseLog           = true;
// audit-r3 R5: cooldown is now **mode-aware** based on /signal.trading_mode:
//   "ranging"  → RangingCooldownSec  (high-frequency reversals)
//   "trending" → TrendingCooldownSec (low-frequency continuation)
//   other/missing → TrendingCooldownSec (conservative fallback)
input int      RangingCooldownSec   = 300;   // 5 min — ranging: quick re-entry OK
input int      TrendingCooldownSec  = 1800;  // 30 min — trending: avoid chasing
input int      DirectionCooldownSec = 1800;  // DEPRECATED — kept for rollback only

// Round 4 v5: trailing SL — backtest 2020-2024 showed
// trail_points=50, trail_activate_r=0.5 gave PF 2.13 vs baseline 0.95.
// Round 4 v5 HOTFIX (task #56 deep-dive): 2024 PF collapsed to 1.00
// because fixed 50pt trail was 1/27 of the SL (median SL 1343 pts in
// 2024). Winners locked in $0.50 profit vs avg loss $19.20 → net flat.
// New default trails as a FRACTION of initial risk, so scales with SL.
// Set TrailDistanceR > 0 for R-proportional (recommended).
// Set TrailDistanceR = 0 to fall back to fixed TrailDistancePoints.
input bool     EnableTrailing       = true;
input double   TrailActivateR       = 0.5;  // Start trailing at N x initial_risk profit
input double   TrailDistanceR       = 0.5;  // Trail N x initial_risk behind current (0 = use points)
input int      TrailDistancePoints  = 50;   // Fallback fixed trail distance when TrailDistanceR == 0
input int      TrailingMinIntervalSec = 5;  // Debounce PositionModify calls per ticket

// Round 6 B5: on-chart live status panel (render-only, reads EA globals).
input bool     ShowPanel            = true;
input int      PanelCorner          = 0;    // ENUM_BASE_CORNER; 0 = top-left
input int      PanelFontSize        = 9;

// audit-r4 v5 Option B: max legs we track (control + treatment + headroom).
#define AISMC_MAX_LEGS 8

// Round 4 v5: per-ticket initial SL memo for R-based trailing activation.
// 128 is generous — in practice mt5 positions count is low single digits.
#define AISMC_MAX_TRACKED 128
ulong    g_track_ticket     [AISMC_MAX_TRACKED];
double   g_track_initial_sl [AISMC_MAX_TRACKED];
datetime g_track_last_trail [AISMC_MAX_TRACKED];
// Round 5 A-track Task #8: per-ticket regime-aware trailing SL.
// When 0.0 → fall back to the EA-level TrailActivateR/TrailDistanceR inputs
// (preserves legacy behaviour on pre-R5 deploys where /signal lacks the
// trail_activate_r / trail_distance_r fields).  Populated at position
// open from the signal's regime preset (TREND → 0.3/0.5, CONSOLIDATION
// → 0.5/0.3, TRANSITION → 0.5/0.5, ATH_BREAKOUT → 0.8/0.7).
double   g_track_activate_r [AISMC_MAX_TRACKED];
double   g_track_distance_r [AISMC_MAX_TRACKED];
int      g_track_count = 0;

CTrade         g_trade;
datetime       g_last_poll_ts    = 0;

// Per-leg state — indexed 0..AISMC_MAX_LEGS-1 by order of appearance in the
// signals array.  signal_id dedupe, buy/sell cooldown, and magic are all
// tracked separately so control + treatment never interfere with each other.
int            g_leg_magic       [AISMC_MAX_LEGS];
string         g_leg_last_sig_id [AISMC_MAX_LEGS];
datetime       g_leg_last_buy_ts [AISMC_MAX_LEGS];
datetime       g_leg_last_sell_ts[AISMC_MAX_LEGS];
// Last observed action label per leg ("BUY"/"SELL"/"HOLD"/"RANGE BUY"/...).
// Panel reads this to render the Control/Treat rows.  Updated in ExecuteLeg.
string         g_leg_last_action [AISMC_MAX_LEGS];
// Last observed mode per leg — used together with last-buy/sell_ts to
// compute cooldown remaining for the panel.  Populated in ExecuteLeg.
int            g_leg_last_cooldown[AISMC_MAX_LEGS];

// Round 6 B5: panel-only snapshot of last /signal response.  Parsed once
// per poll; panel reads these every tick (render-only, never drives exec).
string         g_last_regime              = "";
double         g_last_confidence          = -1.0;  // -1 == unknown
double         g_last_trail_activate_r    = 0.0;
double         g_last_trail_distance_r    = 0.0;
// Daily P&L baseline — captured on first tick of the UTC day.
double         g_day_start_balance = 0.0;
int            g_day_start_doy     = -1;
datetime       g_last_panel_update = 0;


int OnInit()
{
   // Default magic for unknown symbols; refreshed per-leg at execute time.
   int default_magic = ResolveDefaultMagic();
   g_trade.SetExpertMagicNumber(default_magic);
   g_trade.SetDeviationInPoints(DeviationPoints);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   for (int i = 0; i < AISMC_MAX_LEGS; i++)
   {
      g_leg_magic[i]         = 0;
      g_leg_last_sig_id[i]   = "";
      g_leg_last_buy_ts[i]   = 0;
      g_leg_last_sell_ts[i]  = 0;
      g_leg_last_action[i]   = "";
      g_leg_last_cooldown[i] = 0;
   }

   // Round 6 B5: create chart panel — render-only, safe to skip via input.
   if (ShowPanel) PanelInit(PanelCorner, PanelFontSize);

   PrintFormat("AISMCReceiver v2 init on %s (default_magic=%d, poll=%ds, dual-magic)",
               _Symbol, default_magic, PollIntervalSec);
   return(INIT_SUCCEEDED);
}


void OnDeinit(const int reason)
{
   // Round 6 B5: wipe every AISMC_PANEL_* object we may have created.
   // Safe to call even if PanelInit never ran.
   PanelDeinit();
   PrintFormat("AISMCReceiver deinit on %s (reason=%d)", _Symbol, reason);
}


void OnTick()
{
   // Round 6 B5: panel updates run every tick (throttled 1s) even when
   // EnableTrading is false, so the user can still see state on a chart
   // with trading paused.  The render path is read-only.
   datetime now = TimeCurrent();
   if (ShowPanel && (now - g_last_panel_update) >= 1)
   {
      g_last_panel_update = now;
      RefreshPanel(now);
   }

   if (!EnableTrading) return;

   // Round 4 v5: trailing runs every tick, NOT gated by PollIntervalSec,
   // so intra-bar price moves lock profits promptly. The debounce inside
   // ManageTrailingStops prevents flooding PositionModify on the broker.
   if (EnableTrailing) ManageTrailingStops();

   if (now - g_last_poll_ts < PollIntervalSec) return;
   g_last_poll_ts = now;
   PollAndExecute();
}


//+------------------------------------------------------------------+
//| Round 6 B5: compute PanelState from live EA globals + account     |
//| info, then hand off to PanelUpdate.  Pure read — never writes.    |
//+------------------------------------------------------------------+
void RefreshPanel(const datetime now)
{
   // Daily P&L baseline — reset when UTC day-of-year changes.
   MqlDateTime d;
   TimeToStruct(now, d);
   if (d.day_of_year != g_day_start_doy)
   {
      g_day_start_doy     = d.day_of_year;
      g_day_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   }

   // Accumulate floating P&L across all AI-SMC magics on THIS symbol.
   double floating = 0.0;
   int    tracked  = 0;
   int    total    = PositionsTotal();
   for (int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if (ticket == 0) continue;
      if (!PositionSelectByTicket(ticket)) continue;
      if (PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      long mg = PositionGetInteger(POSITION_MAGIC);
      if (mg != MagicXAU && mg != MagicBTC
          && mg != MagicXAU + 10 && mg != MagicBTC + 10
          && mg != 19760428) continue;
      floating += PositionGetDouble(POSITION_PROFIT);
      tracked++;
   }

   PanelState s;
   s.symbol           = _Symbol;
   s.bid              = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   s.ask              = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   s.balance          = AccountInfoDouble(ACCOUNT_BALANCE);
   s.equity           = AccountInfoDouble(ACCOUNT_EQUITY);
   s.floating_pnl     = floating;
   s.daily_pnl        = s.balance - g_day_start_balance + floating;
   s.regime           = g_last_regime;
   s.confidence       = g_last_confidence;
   s.trail_activate_r = g_last_trail_activate_r;
   s.trail_distance_r = g_last_trail_distance_r;

   // Control = leg 0, Treatment = leg 1 (convention from /signal array).
   s.control_action   = g_leg_last_action[0];
   s.control_cooldown = LegCooldownRemaining(0, now);
   s.treat_action     = g_leg_last_action[1];
   s.treat_cooldown   = LegCooldownRemaining(1, now);

   s.tickets_tracked  = tracked;
   s.last_poll_ts     = g_last_poll_ts;
   s.now              = now;

   PanelUpdate(s);
}


//+------------------------------------------------------------------+
//| Round 6 B5: best-effort cooldown remaining in seconds for a leg.  |
//| Returns the maximum of (remaining buy cd, remaining sell cd) so   |
//| the panel shows the most restrictive lockout.  -1 when none.      |
//+------------------------------------------------------------------+
int LegCooldownRemaining(const int leg_idx, const datetime now)
{
   if (leg_idx < 0 || leg_idx >= AISMC_MAX_LEGS) return -1;
   int cd = g_leg_last_cooldown[leg_idx];
   if (cd <= 0) return -1;
   int best = 0;
   if (g_leg_last_buy_ts[leg_idx] > 0)
   {
      int rem = cd - (int)(now - g_leg_last_buy_ts[leg_idx]);
      if (rem > best) best = rem;
   }
   if (g_leg_last_sell_ts[leg_idx] > 0)
   {
      int rem = cd - (int)(now - g_leg_last_sell_ts[leg_idx]);
      if (rem > best) best = rem;
   }
   return best > 0 ? best : -1;
}


// ---------------------------------------------------------------------
// Round 4 v5: per-ticket state helpers for R-based trailing activation.
// ---------------------------------------------------------------------

int TrackedIndex(ulong ticket)
{
   for (int i = 0; i < g_track_count; i++)
      if (g_track_ticket[i] == ticket) return i;
   return -1;
}


double GetInitialSL(ulong ticket, double fallback_sl)
{
   int idx = TrackedIndex(ticket);
   if (idx >= 0) return g_track_initial_sl[idx];
   if (g_track_count >= AISMC_MAX_TRACKED) return fallback_sl;
   g_track_ticket     [g_track_count] = ticket;
   g_track_initial_sl [g_track_count] = fallback_sl;
   g_track_last_trail [g_track_count] = 0;
   // Round 5 A-track Task #8: trail params default to 0.0 → signals
   // "use EA-level input default" in ManageTrailingStops.  RecordTrailParams
   // overwrites these when the signal carries regime-specific values.
   g_track_activate_r [g_track_count] = 0.0;
   g_track_distance_r [g_track_count] = 0.0;
   g_track_count++;
   return fallback_sl;
}


// Round 5 A-track Task #8: store per-ticket regime-derived trail params
// (called right after a successful Buy()/Sell()).  When values are <=0
// the per-ticket slot stays 0.0 and ManageTrailingStops falls back to
// the EA-level TrailActivateR/TrailDistanceR inputs.
void RecordTrailParams(ulong ticket, double activate_r, double distance_r)
{
   if (ticket == 0) return;
   if (activate_r <= 0.0 && distance_r <= 0.0) return;  // nothing to record
   int idx = TrackedIndex(ticket);
   if (idx < 0)
   {
      // Seed the slot so subsequent ManageTrailingStops sees the params
      // even before the first GetInitialSL call.
      if (g_track_count >= AISMC_MAX_TRACKED) return;
      g_track_ticket     [g_track_count] = ticket;
      g_track_initial_sl [g_track_count] = 0.0;  // will be filled by GetInitialSL
      g_track_last_trail [g_track_count] = 0;
      g_track_activate_r [g_track_count] = 0.0;
      g_track_distance_r [g_track_count] = 0.0;
      idx = g_track_count;
      g_track_count++;
   }
   if (activate_r > 0.0) g_track_activate_r[idx] = activate_r;
   if (distance_r > 0.0) g_track_distance_r[idx] = distance_r;
}


// Round 5 A-track Task #8: resolve the effective activate_r / distance_r
// for a ticket — per-ticket override when recorded, EA input otherwise.
double GetEffectiveActivateR(ulong ticket)
{
   int idx = TrackedIndex(ticket);
   if (idx >= 0 && g_track_activate_r[idx] > 0.0)
      return g_track_activate_r[idx];
   return TrailActivateR;
}


double GetEffectiveDistanceR(ulong ticket)
{
   int idx = TrackedIndex(ticket);
   if (idx >= 0 && g_track_distance_r[idx] > 0.0)
      return g_track_distance_r[idx];
   return TrailDistanceR;
}


bool ShouldDebounce(ulong ticket, datetime now)
{
   int idx = TrackedIndex(ticket);
   if (idx < 0) return false;
   return (now - g_track_last_trail[idx]) < TrailingMinIntervalSec;
}


void MarkTrailed(ulong ticket, datetime now)
{
   int idx = TrackedIndex(ticket);
   if (idx >= 0) g_track_last_trail[idx] = now;
}


void ManageTrailingStops()
{
   datetime now = TimeCurrent();
   int total = PositionsTotal();
   for (int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if (ticket == 0) continue;
      if (!PositionSelectByTicket(ticket)) continue;

      // Filter to AI-SMC-owned positions on this chart's symbol.
      long magic = PositionGetInteger(POSITION_MAGIC);
      if (magic != MagicXAU && magic != MagicBTC
          && magic != MagicXAU + 10 && magic != MagicBTC + 10
          && magic != 19760428)   // XAU treatment magic
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      if (sym != _Symbol) continue;

      long   type       = PositionGetInteger(POSITION_TYPE);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      double point      = SymbolInfoDouble(sym, SYMBOL_POINT);
      if (point <= 0) continue;

      double initial_sl = GetInitialSL(ticket, current_sl);

      MqlTick last_tick;
      if (!SymbolInfoTick(sym, last_tick)) continue;

      double current_price = (type == POSITION_TYPE_BUY) ? last_tick.bid : last_tick.ask;
      double profit_pts    = 0.0;
      double initial_risk  = 0.0;

      if (type == POSITION_TYPE_BUY)
      {
         profit_pts   = (current_price - open_price) / point;
         initial_risk = (open_price   - initial_sl) / point;
      }
      else
      {
         profit_pts   = (open_price   - current_price) / point;
         initial_risk = (initial_sl   - open_price)   / point;
      }
      if (initial_risk <= 0) continue;  // malformed — skip

      // Round 5 A-track Task #8: per-ticket regime-aware activate/distance.
      // Falls back to EA-level TrailActivateR/TrailDistanceR when not
      // recorded (pre-R5 /signal response without trail_activate_r/_r).
      double eff_activate_r = GetEffectiveActivateR(ticket);
      double eff_distance_r = GetEffectiveDistanceR(ticket);

      double activate_pts = eff_activate_r * initial_risk;
      if (profit_pts < activate_pts) continue;

      // Round 4 v5 hotfix (task #56): R-proportional trail distance so
      // a wide-SL trade (e.g. 1343 pts SL) trails by 672 pts — locking
      // meaningful profit — instead of the old fixed 50 pts that
      // would exit at 3.7% of the intended RR.
      double trail_dist_pts = (eff_distance_r > 0.0)
                                ? eff_distance_r * initial_risk
                                : (double)TrailDistancePoints;
      double candidate_sl;
      if (type == POSITION_TYPE_BUY)
         candidate_sl = NormalizeDouble(current_price - trail_dist_pts * point, _Digits);
      else
         candidate_sl = NormalizeDouble(current_price + trail_dist_pts * point, _Digits);

      // Monotonic — SL never moves against the position.
      bool progress;
      if (type == POSITION_TYPE_BUY)
         progress = (candidate_sl > current_sl);
      else
         progress = (current_sl <= 0.0 || candidate_sl < current_sl);
      if (!progress) continue;

      if (ShouldDebounce(ticket, now)) continue;

      g_trade.SetExpertMagicNumber((int)magic);
      if (g_trade.PositionModify(ticket, candidate_sl, current_tp))
      {
         MarkTrailed(ticket, now);
         if (VerboseLog)
            PrintFormat("Trail ticket=%I64u mag=%d new_sl=%.2f profit=%.0fp (%.2fR) trail_dist=%.0fp",
                        ticket, magic, candidate_sl, profit_pts,
                        profit_pts / initial_risk, trail_dist_pts);
      }
      else if (VerboseLog)
      {
         PrintFormat("Trail FAIL ticket=%I64u rc=%d msg=%s",
                     ticket, g_trade.ResultRetcode(), g_trade.ResultComment());
      }
   }
}


int ResolveDefaultMagic()
{
   string sym = _Symbol;
   StringToUpper(sym);
   if (StringFind(sym, "BTC") >= 0) return MagicBTC;
   return MagicXAU;
}


void PollAndExecute()
{
   string url = SignalURL + "?symbol=" + _Symbol;
   char   post[];
   char   result[];
   string headers = "";
   string response_headers = "";
   int    timeout_ms = 5000;

   ResetLastError();
   int status = WebRequest("GET", url, headers, timeout_ms,
                           post, result, response_headers);
   if (status != 200)
   {
      int err = GetLastError();
      if (VerboseLog)
         PrintFormat("WebRequest failed status=%d err=%d url=%s", status, err, url);
      return;
   }

   string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   if (VerboseLog) PrintFormat("Signal (%s): %s", _Symbol, body);

   // Round 6 B5: snapshot fields for the panel.  JsonStr is a flat
   // first-match extractor — for keys that appear in both signals[0]
   // and the top-level flat mirror it returns the control leg's copy.
   // Non-breaking on pre-R5 servers that omit these keys (JsonStr
   // returns "" → we keep the prior value, not overwrite with junk).
   string regime = JsonStr(body, "regime_label");
   if (StringLen(regime) > 0 && regime != "null") g_last_regime = regime;
   string conf_s = JsonStr(body, "confidence");
   if (StringLen(conf_s) > 0 && conf_s != "null")
      g_last_confidence = StringToDouble(conf_s);
   string act_r  = JsonStr(body, "trail_activate_r");
   if (StringLen(act_r) > 0 && act_r != "null")
      g_last_trail_activate_r = StringToDouble(act_r);
   string dist_r = JsonStr(body, "trail_distance_r");
   if (StringLen(dist_r) > 0 && dist_r != "null")
      g_last_trail_distance_r = StringToDouble(dist_r);

   // audit-r4 v5 Option B: iterate the signals array.  The response may
   // still carry flat top-level fields for backward compat but we prefer
   // the array — if it's missing (unlikely after server upgrade), fall
   // back to the legacy flat path with default magic.
   int leg_count = ExtractSignalsAndExecute(body);
   if (leg_count == 0)
   {
      if (VerboseLog) Print("No signals[] array found; falling back to legacy flat signal");
      ExecuteLegacyFlat(body);
   }
}


//+------------------------------------------------------------------+
//| Extract signals[] entries and execute each.  Returns number of   |
//| entries processed (0 if array missing / empty).                  |
//+------------------------------------------------------------------+
int ExtractSignalsAndExecute(const string &body)
{
   int arr_start = StringFind(body, "\"signals\"");
   if (arr_start < 0) return 0;
   // Find the opening bracket of the array
   int bracket_open = StringFind(body, "[", arr_start);
   if (bracket_open < 0) return 0;
   // Match closing bracket (accounting for nested braces)
   int depth = 0;
   int len = StringLen(body);
   int bracket_close = -1;
   for (int i = bracket_open; i < len; i++)
   {
      ushort c = StringGetCharacter(body, i);
      if (c == '[') depth++;
      else if (c == ']')
      {
         depth--;
         if (depth == 0) { bracket_close = i; break; }
      }
   }
   if (bracket_close < 0) return 0;

   int leg_idx = 0;
   int cursor = bracket_open + 1;
   while (cursor < bracket_close && leg_idx < AISMC_MAX_LEGS)
   {
      // Find the next object opening '{'
      int obj_start = StringFind(body, "{", cursor);
      if (obj_start < 0 || obj_start > bracket_close) break;
      int obj_end = MatchBrace(body, obj_start, bracket_close);
      if (obj_end < 0) break;
      string leg_body = StringSubstr(body, obj_start, obj_end - obj_start + 1);
      ExecuteLeg(leg_body, leg_idx);
      leg_idx++;
      cursor = obj_end + 1;
   }
   return leg_idx;
}


//+------------------------------------------------------------------+
//| Find the matching '}' starting at a '{', respecting nesting.     |
//+------------------------------------------------------------------+
int MatchBrace(const string &body, int open_pos, int max_pos)
{
   int depth = 0;
   int len = StringLen(body);
   int end = max_pos > 0 && max_pos < len ? max_pos : len - 1;
   for (int i = open_pos; i <= end; i++)
   {
      ushort c = StringGetCharacter(body, i);
      if (c == '{') depth++;
      else if (c == '}')
      {
         depth--;
         if (depth == 0) return i;
      }
   }
   return -1;
}


//+------------------------------------------------------------------+
//| Execute a single leg object (extracted from signals[] entry).    |
//| Records per-leg dedupe + cooldown against the leg's magic.       |
//+------------------------------------------------------------------+
void ExecuteLeg(const string &leg_body, int leg_idx)
{
   string action    = JsonStr(leg_body, "action");
   string signal_id = JsonStr(leg_body, "signal_id");
   string fresh     = JsonStr(leg_body, "fresh");
   int    magic     = (int)StringToInteger(JsonStr(leg_body, "magic"));
   string leg_label = JsonStr(leg_body, "leg");

   if (action == "" || signal_id == "") return;
   if (magic <= 0)
   {
      if (VerboseLog)
         PrintFormat("Leg %d: missing magic, skipping", leg_idx);
      return;
   }

   // Record magic for this leg slot (used for logs; dedupe table is the
   // persistent surface).
   g_leg_magic[leg_idx] = magic;

   if (signal_id == g_leg_last_sig_id[leg_idx])
   {
      if (VerboseLog)
         PrintFormat("Leg %d (magic=%d): duplicate signal_id; skipping", leg_idx, magic);
      return;
   }
   if (fresh != "true")
   {
      if (VerboseLog)
         PrintFormat("Leg %d (magic=%d): signal stale; skipping", leg_idx, magic);
      // Do NOT update g_leg_last_sig_id — a stale signal can become fresh
      // next cycle after live_demo writes a new state.
      return;
   }

   // Round 6 B5: remember the action label for the panel (HOLD / BUY / SELL
   // / RANGE BUY / RANGE SELL).  Overwritten even on skip so the panel
   // reflects the current signal, not an older placed trade.
   g_leg_last_action[leg_idx] = action;

   bool placed = false;
   if (action == "BUY" || action == "RANGE BUY")
      placed = ExecuteDirection(true, leg_body, leg_idx, magic);
   else if (action == "SELL" || action == "RANGE SELL")
      placed = ExecuteDirection(false, leg_body, leg_idx, magic);

   // Remember signal_id regardless of placed — prevents retry on same cycle.
   g_leg_last_sig_id[leg_idx] = signal_id;

   if (VerboseLog)
      PrintFormat("Leg %d (magic=%d, label='%s'): action=%s placed=%s",
                  leg_idx, magic, leg_label, action, placed ? "yes" : "no");
}


//+------------------------------------------------------------------+
//| Fallback for legacy flat response (no signals[] array).          |
//+------------------------------------------------------------------+
void ExecuteLegacyFlat(const string &body)
{
   string action    = JsonStr(body, "action");
   string signal_id = JsonStr(body, "signal_id");
   string fresh     = JsonStr(body, "fresh");
   int    magic     = ResolveDefaultMagic();

   if (action == "" || signal_id == "") return;
   if (signal_id == g_leg_last_sig_id[0]) return;
   if (fresh != "true")
   {
      if (VerboseLog) Print("Legacy flat signal stale; skipping");
      return;
   }

   // Round 6 B5: mirror leg-0 action for the panel on legacy-flat responses.
   g_leg_last_action[0] = action;

   bool placed = false;
   if (action == "BUY" || action == "RANGE BUY")
      placed = ExecuteDirection(true, body, 0, magic);
   else if (action == "SELL" || action == "RANGE SELL")
      placed = ExecuteDirection(false, body, 0, magic);

   g_leg_last_sig_id[0] = signal_id;
}


int ResolveCooldownSec(const string &body)
{
   // audit-r3 R5: mode-aware direction cooldown.
   string mode = JsonStr(body, "trading_mode");
   StringToLower(mode);
   if (mode == "ranging") return RangingCooldownSec;
   if (mode == "trending") return TrendingCooldownSec;
   return TrendingCooldownSec;
}


bool ExecuteDirection(bool is_buy, const string &body, int leg_idx, int magic)
{
   // audit-r3 R5: cooldown resolved from /signal.trading_mode (ranging=300,
   // trending=1800, fallback=1800).
   int cooldown = ResolveCooldownSec(body);
   // Round 6 B5: remember the active cooldown window for this leg so the
   // panel can show "CD MM:SS" remaining without re-parsing the body.
   if (leg_idx >= 0 && leg_idx < AISMC_MAX_LEGS)
      g_leg_last_cooldown[leg_idx] = cooldown;
   datetime now = TimeCurrent();
   if (is_buy && now - g_leg_last_buy_ts[leg_idx] < cooldown)
   {
      PrintFormat("Leg %d (magic=%d): skip BUY cooldown %ds remaining (mode=%d)",
                  leg_idx, magic,
                  cooldown - (int)(now - g_leg_last_buy_ts[leg_idx]), cooldown);
      return false;
   }
   if (!is_buy && now - g_leg_last_sell_ts[leg_idx] < cooldown)
   {
      PrintFormat("Leg %d (magic=%d): skip SELL cooldown %ds remaining (mode=%d)",
                  leg_idx, magic,
                  cooldown - (int)(now - g_leg_last_sell_ts[leg_idx]), cooldown);
      return false;
   }

   double entry = StringToDouble(JsonStr(body, "entry"));
   double sl    = StringToDouble(JsonStr(body, "sl"));
   double tp    = StringToDouble(JsonStr(body, "tp"));
   double conf  = StringToDouble(JsonStr(body, "confidence"));

   if (conf < MinConfidence)
   {
      PrintFormat("Leg %d (magic=%d): skip confidence %.2f < %.2f",
                  leg_idx, magic, conf, MinConfidence);
      return false;
   }

   // Use server-provided lot; fall back to DefaultLot if missing/zero.
   double lot = StringToDouble(JsonStr(body, "lot"));
   if (lot <= 0.0) lot = DefaultLot;

   double price = is_buy
                  ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // CRITICAL: set magic immediately before the OrderSend so concurrent
   // leg executions cannot race on g_trade's magic property.  OnTick is
   // single-threaded per EA instance, so legs are processed sequentially,
   // but this keeps the invariant explicit for readers.
   g_trade.SetExpertMagicNumber(magic);

   bool ok = is_buy
             ? g_trade.Buy(lot, _Symbol, price, sl, tp, "AI-SMC EA")
             : g_trade.Sell(lot, _Symbol, price, sl, tp, "AI-SMC EA");

   // Round 5 A-track Task #8: record the regime-derived trailing SL
   // parameters against the freshly-opened ticket.  Missing fields in
   // the signal body parse to "" → StringToDouble returns 0.0 → the
   // RecordTrailParams no-ops and ManageTrailingStops uses the EA
   // input defaults (legacy behaviour preserved).
   double signal_activate_r = StringToDouble(JsonStr(body, "trail_activate_r"));
   double signal_distance_r = StringToDouble(JsonStr(body, "trail_distance_r"));
   ulong  new_ticket        = g_trade.ResultDeal();

   PrintFormat("Leg %d (magic=%d) %s %.2f lot %s @ %.2f SL=%.2f TP=%.2f conf=%.2f trailR=(%.2f/%.2f) → retcode=%u %s",
               leg_idx, magic,
               is_buy ? "BUY" : "SELL", lot, _Symbol, price, sl, tp, conf,
               signal_activate_r, signal_distance_r,
               g_trade.ResultRetcode(),
               ok ? "OK" : g_trade.ResultComment());

   // Record successful order timestamp for per-leg cooldown tracking.
   if (ok)
   {
      if (is_buy) g_leg_last_buy_ts[leg_idx]  = TimeCurrent();
      else        g_leg_last_sell_ts[leg_idx] = TimeCurrent();
      // Round 5 A-track Task #8: attach trail params to the new ticket
      // (when present).  ResultDeal returns the deal ticket — which is
      // what PositionSelectByTicket uses in ManageTrailingStops.
      if (new_ticket != 0)
         RecordTrailParams(new_ticket, signal_activate_r, signal_distance_r);
   }
   return ok;
}


//+------------------------------------------------------------------+
//| Minimal JSON value extractor for flat key:value bodies.          |
//| Handles "key":"string", "key":number, "key":true/false/null.     |
//+------------------------------------------------------------------+
string JsonStr(const string &body, const string key)
{
   string needle = "\"" + key + "\":";
   int pos = StringFind(body, needle);
   if (pos < 0) return "";
   pos += StringLen(needle);

   // skip whitespace
   int len = StringLen(body);
   while (pos < len)
   {
      ushort c = StringGetCharacter(body, pos);
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      pos++;
   }
   if (pos >= len) return "";

   ushort first = StringGetCharacter(body, pos);
   if (first == '"')
   {
      int end = StringFind(body, "\"", pos + 1);
      if (end < 0) return "";
      return StringSubstr(body, pos + 1, end - pos - 1);
   }

   // number / bool / null — read until , or } or whitespace
   int start = pos;
   while (pos < len)
   {
      ushort c = StringGetCharacter(body, pos);
      if (c == ',' || c == '}' || c == ' ' || c == '\n' || c == '\r') break;
      pos++;
   }
   return StringSubstr(body, start, pos - start);
}
//+------------------------------------------------------------------+
