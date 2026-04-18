//+------------------------------------------------------------------+
//|                                            AISMCReceiver.mq5       |
//|                    AI-SMC MQL5 EA Signal Receiver                  |
//|                                                                    |
//|  Polls Python strategy_server at http://127.0.0.1:8080/signal?..   |
//|  every PollIntervalSec; on fresh non-HOLD signal, executes order   |
//|  via MT5 native CTrade (no Python IPC dependency).                 |
//|                                                                    |
//|  Deployment:                                                       |
//|    1. MT5 → Options → EA Trading → Allow WebRequest for URL        |
//|       http://127.0.0.1:8080                                        |
//|    2. Copy this file to <MT5 data folder>/MQL5/Experts/            |
//|    3. In MetaEditor, compile → attach to XAUUSD chart              |
//|    4. Attach a 2nd instance to BTCUSD chart                        |
//|                                                                    |
//|  Magic number is resolved per-symbol so XAU / BTC positions are    |
//|  tracked independently by the Python reconcile loop.               |
//+------------------------------------------------------------------+
#property copyright "AI-SMC"
#property version   "1.00"
#property strict
#property description "Polls Python strategy server, executes SMC signals."

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
// Old single DirectionCooldownSec input kept as deprecated fallback for
// backward-compatible deployment; if both are present we prefer the
// mode-specific inputs and ignore the legacy one.
input int      RangingCooldownSec   = 300;   // 5 min — ranging: quick re-entry OK
input int      TrendingCooldownSec  = 1800;  // 30 min — trending: avoid chasing
input int      DirectionCooldownSec = 1800;  // DEPRECATED — kept for rollback only

CTrade         g_trade;
datetime       g_last_poll_ts    = 0;
string         g_last_signal_id  = "";
datetime       g_last_buy_ts     = 0;
datetime       g_last_sell_ts    = 0;


int OnInit()
{
   int magic = ResolveMagic();
   g_trade.SetExpertMagicNumber(magic);
   g_trade.SetDeviationInPoints(DeviationPoints);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   PrintFormat("AISMCReceiver init on %s (magic=%d, poll=%ds)",
               _Symbol, magic, PollIntervalSec);
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


int ResolveMagic()
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

   string action    = JsonStr(body, "action");
   string signal_id = JsonStr(body, "signal_id");
   string fresh     = JsonStr(body, "fresh");

   if (action == "" || signal_id == "") return;
   if (signal_id == g_last_signal_id) return;
   if (fresh != "true")
   {
      if (VerboseLog) Print("Signal stale; skipping");
      return;
   }

   bool placed = false;
   if (action == "BUY" || action == "RANGE BUY")
      placed = ExecuteDirection(true, body);
   else if (action == "SELL" || action == "RANGE SELL")
      placed = ExecuteDirection(false, body);

   // Remember signal_id regardless of placed — prevents retry on same cycle.
   g_last_signal_id = signal_id;
}


int ResolveCooldownSec(const string &body)
{
   // audit-r3 R5: mode-aware direction cooldown.
   // Ranging sessions (Asian) reverse quickly → short cooldown to not miss
   // the next leg.  Trending sessions (London/NY) want fewer re-entries
   // to avoid chasing.  Unknown mode → conservative trending value.
   string mode = JsonStr(body, "trading_mode");
   StringToLower(mode);
   if (mode == "ranging") return RangingCooldownSec;
   if (mode == "trending") return TrendingCooldownSec;
   return TrendingCooldownSec;
}


bool ExecuteDirection(bool is_buy, const string &body)
{
   // audit-r3 R5: cooldown resolved from /signal.trading_mode (ranging=300,
   // trending=1800, fallback=1800).  Replaces the previous fixed
   // DirectionCooldownSec input-level constant.
   int cooldown = ResolveCooldownSec(body);
   datetime now = TimeCurrent();
   if (is_buy && now - g_last_buy_ts < cooldown)
   {
      PrintFormat("Skip BUY: cooldown %ds remaining (mode cooldown=%d)",
                  cooldown - (int)(now - g_last_buy_ts), cooldown);
      return false;
   }
   if (!is_buy && now - g_last_sell_ts < cooldown)
   {
      PrintFormat("Skip SELL: cooldown %ds remaining (mode cooldown=%d)",
                  cooldown - (int)(now - g_last_sell_ts), cooldown);
      return false;
   }

   double entry = StringToDouble(JsonStr(body, "entry"));
   double sl    = StringToDouble(JsonStr(body, "sl"));
   double tp    = StringToDouble(JsonStr(body, "tp"));
   double conf  = StringToDouble(JsonStr(body, "confidence"));

   if (conf < MinConfidence)
   {
      PrintFormat("Skip: confidence %.2f < %.2f", conf, MinConfidence);
      return false;
   }

   // Fix 2: use server-provided lot; fall back to DefaultLot if missing/zero.
   double lot = StringToDouble(JsonStr(body, "lot"));
   if (lot <= 0.0) lot = DefaultLot;

   double price = is_buy
                  ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   bool ok = is_buy
             ? g_trade.Buy(lot, _Symbol, price, sl, tp, "AI-SMC EA")
             : g_trade.Sell(lot, _Symbol, price, sl, tp, "AI-SMC EA");

   PrintFormat("%s %.2f lot %s @ %.2f SL=%.2f TP=%.2f conf=%.2f → retcode=%u %s",
               is_buy ? "BUY" : "SELL", lot, _Symbol, price, sl, tp, conf,
               g_trade.ResultRetcode(),
               ok ? "OK" : g_trade.ResultComment());

   // Fix 3: record successful order timestamp for cooldown tracking.
   if (ok)
   {
      if (is_buy) g_last_buy_ts  = TimeCurrent();
      else        g_last_sell_ts = TimeCurrent();
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
