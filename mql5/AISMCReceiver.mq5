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
//|    2. Copy this file to <MT5 data folder>/MQL5/Experts/            |
//|    3. In MetaEditor, compile → attach to XAUUSD chart              |
//|    4. Attach a 2nd instance to BTCUSD chart                        |
//+------------------------------------------------------------------+
#property copyright "AI-SMC"
#property version   "2.00"
#property strict
#property description "Polls Python strategy server, executes SMC signals (dual-magic)."

#include <Trade/Trade.mqh>

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

// audit-r4 v5 Option B: max legs we track (control + treatment + headroom).
#define AISMC_MAX_LEGS 8

CTrade         g_trade;
datetime       g_last_poll_ts    = 0;

// Per-leg state — indexed 0..AISMC_MAX_LEGS-1 by order of appearance in the
// signals array.  signal_id dedupe, buy/sell cooldown, and magic are all
// tracked separately so control + treatment never interfere with each other.
int            g_leg_magic       [AISMC_MAX_LEGS];
string         g_leg_last_sig_id [AISMC_MAX_LEGS];
datetime       g_leg_last_buy_ts [AISMC_MAX_LEGS];
datetime       g_leg_last_sell_ts[AISMC_MAX_LEGS];


int OnInit()
{
   // Default magic for unknown symbols; refreshed per-leg at execute time.
   int default_magic = ResolveDefaultMagic();
   g_trade.SetExpertMagicNumber(default_magic);
   g_trade.SetDeviationInPoints(DeviationPoints);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   for (int i = 0; i < AISMC_MAX_LEGS; i++)
   {
      g_leg_magic[i]        = 0;
      g_leg_last_sig_id[i]  = "";
      g_leg_last_buy_ts[i]  = 0;
      g_leg_last_sell_ts[i] = 0;
   }

   PrintFormat("AISMCReceiver v2 init on %s (default_magic=%d, poll=%ds, dual-magic)",
               _Symbol, default_magic, PollIntervalSec);
   return(INIT_SUCCEEDED);
}


void OnDeinit(const int reason)
{
   PrintFormat("AISMCReceiver deinit on %s (reason=%d)", _Symbol, reason);
}


void OnTick()
{
   if (!EnableTrading) return;
   datetime now = TimeCurrent();
   if (now - g_last_poll_ts < PollIntervalSec) return;
   g_last_poll_ts = now;
   PollAndExecute();
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

   PrintFormat("Leg %d (magic=%d) %s %.2f lot %s @ %.2f SL=%.2f TP=%.2f conf=%.2f → retcode=%u %s",
               leg_idx, magic,
               is_buy ? "BUY" : "SELL", lot, _Symbol, price, sl, tp, conf,
               g_trade.ResultRetcode(),
               ok ? "OK" : g_trade.ResultComment());

   // Record successful order timestamp for per-leg cooldown tracking.
   if (ok)
   {
      if (is_buy) g_leg_last_buy_ts[leg_idx]  = TimeCurrent();
      else        g_leg_last_sell_ts[leg_idx] = TimeCurrent();
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
