//+------------------------------------------------------------------+
//|                                            HedgeRock_Lite.mq5    |
//|                                                                  |
//|  Minimal EMA-crossover EA — AB-test control against              |
//|  HedgeRock_v2_patched on the same XAUUSD demo account.           |
//|                                                                  |
//|  DESIGN PRINCIPLES                                               |
//|  -----------------                                               |
//|  *  One position at a time, fixed 0.01 lot, no martingale,       |
//|     no grid, no hedge-pair, no COA, no trailing.                 |
//|  *  Entry only fires on a freshly-closed M15 bar.                |
//|  *  Direction: EMA5 vs EMA13. Optionally overridden by a         |
//|     RegimeCache.json file written by the off-board pipeline.     |
//|  *  Exit only via broker-side TP / SL (no on-bar exit code).     |
//|  *  Magic 30333333 keeps deals separable from HedgeRock_v2's     |
//|     20222222 so /scripts/ab_test_monitor.py can split histories. |
//+------------------------------------------------------------------+
#property copyright "AI-SMC"
#property version   "1.00"
#property strict
#property description "Minimal EMA5/13 EA — control arm for HedgeRock AB test."

#include <Trade/Trade.mqh>

input double          LotSize         = 0.01;            // Fixed lot size
input int             TP_Points       = 400;             // Take-profit distance (points)
input int             SL_Points       = 400;             // Stop-loss distance (points)
input int             EMA_Fast        = 5;               // Fast EMA period
input int             EMA_Slow        = 13;              // Slow EMA period
input long            Magic           = 30333333;        // Magic number (must differ from HedgeRock_v2)
input bool            EnableTrading   = true;            // Master switch
input bool            UseRegimeCache  = true;            // Read RegimeCache.json if present
input string          RegimeCachePath = "RegimeCache.json";
input int             DeviationPoints = 100;             // Max slippage (points)
input ENUM_TIMEFRAMES TF              = PERIOD_M15;      // Signal timeframe
input bool            VerboseLog      = true;            // Print signal + order diagnostics

// ---- internal state -------------------------------------------------------
int      g_h_fast  = INVALID_HANDLE;
int      g_h_slow  = INVALID_HANDLE;
datetime g_last_bar = 0;
CTrade   g_trade;
string   g_sym;

//+------------------------------------------------------------------+
int OnInit()
{
   g_sym = _Symbol;
   g_h_fast = iMA(g_sym, TF, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   g_h_slow = iMA(g_sym, TF, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   if(g_h_fast == INVALID_HANDLE || g_h_slow == INVALID_HANDLE)
   {
      Print("[HedgeRock_Lite] iMA handle creation failed");
      return INIT_FAILED;
   }
   g_trade.SetExpertMagicNumber((ulong)Magic);
   g_trade.SetDeviationInPoints((ulong)DeviationPoints);
   g_trade.SetTypeFillingBySymbol(g_sym);
   PrintFormat("[HedgeRock_Lite] init OK magic=%I64d lot=%.2f tp=%d sl=%d tf=%s",
               Magic, LotSize, TP_Points, SL_Points, EnumToString(TF));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_h_fast != INVALID_HANDLE) IndicatorRelease(g_h_fast);
   if(g_h_slow != INVALID_HANDLE) IndicatorRelease(g_h_slow);
}

//+------------------------------------------------------------------+
//| Returns +1 (buy), -1 (sell), 0 (no override).                    |
//| Parses a tiny key:value pair — refuses to fail loudly so the     |
//| EA degrades to EMA logic if the cache is missing or malformed.   |
//+------------------------------------------------------------------+
int ReadRegimeCacheDirection()
{
   if(!UseRegimeCache) return 0;
   int h = FileOpen(RegimeCachePath, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return 0;
   string content = "";
   while(!FileIsEnding(h)) content += FileReadString(h);
   FileClose(h);
   int p = StringFind(content, "\"direction\"");
   if(p < 0) return 0;
   int q = StringFind(content, ":", p);
   if(q < 0) return 0;
   string tail = StringSubstr(content, q + 1, 32);
   StringToLower(tail);
   if(StringFind(tail, "buy")  >= 0) return +1;
   if(StringFind(tail, "sell") >= 0) return -1;
   return 0;
}

//+------------------------------------------------------------------+
int EMADirection()
{
   double fast[], slow[];
   ArraySetAsSeries(fast, true);
   ArraySetAsSeries(slow, true);
   if(CopyBuffer(g_h_fast, 0, 0, 2, fast) < 2) return 0;
   if(CopyBuffer(g_h_slow, 0, 0, 2, slow) < 2) return 0;
   if(fast[0] > slow[0]) return +1;
   if(fast[0] < slow[0]) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| Returns +1 (long), -1 (short), 0 (flat) for our magic+symbol.    |
//+------------------------------------------------------------------+
int OurDirection()
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      long type = PositionGetInteger(POSITION_TYPE);
      return (type == POSITION_TYPE_BUY) ? +1 : -1;
   }
   return 0;
}

bool CloseOurPositions()
{
   bool ok = true;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      if(!g_trade.PositionClose(t)) ok = false;
   }
   return ok;
}

bool OpenPosition(int dir)
{
   if(dir == 0) return false;
   double pt = SymbolInfoDouble(g_sym, SYMBOL_POINT);
   MqlTick tk;
   if(!SymbolInfoTick(g_sym, tk)) return false;
   double price, sl, tp;
   if(dir > 0)
   {
      price = tk.ask;
      sl    = NormalizeDouble(price - SL_Points * pt, _Digits);
      tp    = NormalizeDouble(price + TP_Points * pt, _Digits);
      return g_trade.Buy(LotSize, g_sym, price, sl, tp, "lite");
   }
   price = tk.bid;
   sl    = NormalizeDouble(price + SL_Points * pt, _Digits);
   tp    = NormalizeDouble(price - TP_Points * pt, _Digits);
   return g_trade.Sell(LotSize, g_sym, price, sl, tp, "lite");
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!EnableTrading) return;
   datetime bars[];
   if(CopyTime(g_sym, TF, 0, 1, bars) < 1) return;
   datetime bar0 = bars[0];
   if(bar0 == g_last_bar) return;
   g_last_bar = bar0;

   int dir = ReadRegimeCacheDirection();
   string src = (dir != 0) ? "regime" : "ema";
   if(dir == 0) dir = EMADirection();
   if(dir == 0) return;

   int cur = OurDirection();
   if(cur == dir)
   {
      if(VerboseLog) PrintFormat("[HedgeRock_Lite] bar %s — already %s, hold", TimeToString(bar0), (dir>0?"BUY":"SELL"));
      return;
   }
   if(cur != 0)
   {
      if(!CloseOurPositions())
      {
         PrintFormat("[HedgeRock_Lite] close failed rc=%u", g_trade.ResultRetcode());
         return;
      }
   }
   if(!OpenPosition(dir))
      PrintFormat("[HedgeRock_Lite] open %s failed rc=%u", (dir>0?"BUY":"SELL"), g_trade.ResultRetcode());
   else if(VerboseLog)
      PrintFormat("[HedgeRock_Lite] opened %s lot=%.2f src=%s bar=%s",
                  (dir>0?"BUY":"SELL"), LotSize, src, TimeToString(bar0));
}
//+------------------------------------------------------------------+
