//+------------------------------------------------------------------+
//|                                              ab_dump_deals.mq5   |
//|                                                                  |
//|  Read-only MT5 script: dumps deal history for HedgeRock vs       |
//|  HedgeRock_Lite into two CSV files, ready to be pulled off the   |
//|  VPS by scripts/ab_test_monitor.py.                              |
//|                                                                  |
//|  Output (in <Terminal>\MQL5\Files\):                             |
//|    ab_deals_20222222.csv   ← HedgeRock_v2_patched                |
//|    ab_deals_30333333.csv   ← HedgeRock_Lite                      |
//|                                                                  |
//|  Each row:                                                       |
//|    ticket,time_utc,symbol,type,entry,volume,price,profit,        |
//|    swap,commission,magic,position_id,deal_reason                 |
//+------------------------------------------------------------------+
#property copyright "AI-SMC"
#property version   "1.00"
#property strict
#property script_show_inputs

input long     MagicA       = 20222222;   // HedgeRock_v2_patched
input long     MagicB       = 30333333;   // HedgeRock_Lite
input string   FilePrefix   = "ab_deals_";
input int      LookbackDays = 30;         // how far back to dump

//+------------------------------------------------------------------+
int DumpForMagic(const long magic, const datetime from, const datetime to)
{
   if(!HistorySelect(from, to))
   {
      PrintFormat("[ab_dump] HistorySelect failed for magic=%I64d", magic);
      return -1;
   }
   string path = FilePrefix + IntegerToString(magic) + ".csv";
   int h = FileOpen(path, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
   {
      PrintFormat("[ab_dump] FileOpen %s failed err=%d", path, GetLastError());
      return -1;
   }
   FileWrite(h, "ticket", "time_utc", "symbol", "type", "entry", "volume",
             "price", "profit", "swap", "commission", "magic",
             "position_id", "deal_reason");

   int total = HistoryDealsTotal();
   int written = 0;
   for(int i = 0; i < total; i++)
   {
      ulong tk = HistoryDealGetTicket(i);
      if(tk == 0) continue;
      long  dmagic = HistoryDealGetInteger(tk, DEAL_MAGIC);
      if(dmagic != magic) continue;
      datetime t   = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
      string sym   = HistoryDealGetString (tk, DEAL_SYMBOL);
      long type    = HistoryDealGetInteger(tk, DEAL_TYPE);
      long entry   = HistoryDealGetInteger(tk, DEAL_ENTRY);
      double vol   = HistoryDealGetDouble (tk, DEAL_VOLUME);
      double price = HistoryDealGetDouble (tk, DEAL_PRICE);
      double profit= HistoryDealGetDouble (tk, DEAL_PROFIT);
      double swap  = HistoryDealGetDouble (tk, DEAL_SWAP);
      double comm  = HistoryDealGetDouble (tk, DEAL_COMMISSION);
      long pos_id  = HistoryDealGetInteger(tk, DEAL_POSITION_ID);
      long reason  = HistoryDealGetInteger(tk, DEAL_REASON);
      FileWrite(h,
                IntegerToString((long)tk),
                TimeToString(t, TIME_DATE | TIME_SECONDS),
                sym,
                IntegerToString(type),
                IntegerToString(entry),
                DoubleToString(vol, 2),
                DoubleToString(price, 5),
                DoubleToString(profit, 2),
                DoubleToString(swap, 2),
                DoubleToString(comm, 2),
                IntegerToString(dmagic),
                IntegerToString(pos_id),
                IntegerToString(reason));
      written++;
   }
   FileClose(h);
   PrintFormat("[ab_dump] magic=%I64d rows=%d → %s", magic, written, path);
   return written;
}

void OnStart()
{
   datetime to   = TimeCurrent();
   datetime from = to - (datetime)LookbackDays * 86400;
   PrintFormat("[ab_dump] range %s → %s", TimeToString(from), TimeToString(to));
   int a = DumpForMagic(MagicA, from, to);
   int b = DumpForMagic(MagicB, from, to);
   PrintFormat("[ab_dump] done A=%d B=%d", a, b);
}
//+------------------------------------------------------------------+
