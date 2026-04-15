//+------------------------------------------------------------------+
//| export_mt5_csv.mq5                                               |
//| AI-SMC project — MT5 history CSV exporter                        |
//|                                                                  |
//| USAGE                                                            |
//| -----                                                            |
//| 1. Copy this file into your MT5 terminal's Scripts folder:       |
//|    <MT5 data folder>\MQL5\Scripts\                               |
//|    (Menu: File → Open Data Folder → MQL5\Scripts)                |
//|                                                                  |
//| 2. Compile in MetaEditor (F7) — there should be no errors.       |
//|                                                                  |
//| 3. In MT5, open the Navigator panel (Ctrl+N), expand Scripts,    |
//|    then double-click "export_mt5_csv" to run it.                 |
//|    A dialog box will appear to let you set input parameters.     |
//|                                                                  |
//| 4. The script exports D1, H4, H1, and M15 bars for the           |
//|    configured symbol to separate CSV files.                      |
//|                                                                  |
//| OUTPUT FORMAT                                                    |
//| -------------                                                    |
//| Each CSV file uses a TAB separator with the header:              |
//|   Date\tTime\tOpen\tHigh\tLow\tClose\tTickvol\tVol\tSpread      |
//|                                                                  |
//| This matches the format expected by CSVAdapter in:               |
//|   src/smc/data/adapters/csv_adapter.py                           |
//|                                                                  |
//| The files are written to ExportDir, organised as:                |
//|   {ExportDir}\{Timeframe}\{Symbol}.csv                           |
//|                                                                  |
//| Example output for XAUUSD:                                       |
//|   C:\mt5_export\D1\XAUUSD.csv                                    |
//|   C:\mt5_export\H4\XAUUSD.csv                                    |
//|   C:\mt5_export\H1\XAUUSD.csv                                    |
//|   C:\mt5_export\M15\XAUUSD.csv                                   |
//|                                                                  |
//| IMPORTING INTO THE DATA LAKE                                     |
//| ----------------------------                                      |
//| After export, run the CSV ingest pipeline from the project root: |
//|                                                                  |
//|   python -c "                                                     |
//|   from pathlib import Path                                       |
//|   from smc.data.ingest import ingest_csv_files                   |
//|   from smc.data.schemas import Timeframe                         |
//|   csv_dir = Path(r'C:\\mt5_export')                              |
//|   for tf in [Timeframe.D1, Timeframe.H4, Timeframe.H1,          |
//|               Timeframe.M15]:                                    |
//|       ingest_csv_files(csv_dir=csv_dir, instrument='XAUUSD',     |
//|                        timeframe=tf, data_dir=Path('data/lake')) |
//|   "                                                               |
//|                                                                  |
//| Or just run the backfill script which handles everything:        |
//|   python scripts/backfill_data.py --timeframes D1,H4,H1,M15     |
//|                                                                  |
//| NOTES                                                            |
//| -----                                                            |
//| - Bars are written in ascending chronological order (oldest →    |
//|   newest), matching the lake's natural sort order.               |
//| - All timestamps use the MT5 terminal's server time.  MT5 forex  |
//|   brokers typically use UTC+2 (winter) or UTC+3 (summer).        |
//|   The CSVAdapter assumes the exported timestamps are in UTC;     |
//|   verify your broker's server time offset if timestamps look     |
//|   shifted.                                                       |
//| - The script requests enough bars to cover the export range.     |
//|   MT5 must have the full history downloaded for the symbol:      |
//|   Tools → Options → Charts → Max bars in history (set to max).   |
//| - ExportDir must exist or the script will create it (and         |
//|   sub-folders for each timeframe).                               |
//+------------------------------------------------------------------+

#property copyright "AI-SMC Project"
#property version   "1.00"
#property strict
#property script_show_inputs

//--- Input parameters (editable in the dialog before each run)
input string Symbol     = "XAUUSD";          // Symbol to export
input string ExportDir  = "C:\\mt5_export";  // Output directory (Windows path)
input string StartDate  = "2020.01.01";      // Export start date (YYYY.MM.DD)
input string EndDate    = "2025.01.01";      // Export end date   (YYYY.MM.DD)
input bool   ExportD1   = true;              // Export Daily (D1)
input bool   ExportH4   = true;              // Export 4-Hour (H4)
input bool   ExportH1   = true;              // Export Hourly (H1)
input bool   ExportM15  = true;              // Export 15-Minute (M15)

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   datetime dtStart = StringToTime(StartDate);
   datetime dtEnd   = StringToTime(EndDate);

   if(dtStart == 0 || dtEnd == 0 || dtStart >= dtEnd)
   {
      Alert("Invalid date range. Please check StartDate and EndDate inputs.");
      return;
   }

   Print("=== AI-SMC CSV Exporter starting ===");
   Print("Symbol: ", Symbol);
   Print("Range: ", TimeToString(dtStart), " → ", TimeToString(dtEnd));
   Print("Output: ", ExportDir);

   int exported = 0;

   if(ExportD1)  { if(ExportTimeframe(Symbol, PERIOD_D1,  "D1",  dtStart, dtEnd)) exported++; }
   if(ExportH4)  { if(ExportTimeframe(Symbol, PERIOD_H4,  "H4",  dtStart, dtEnd)) exported++; }
   if(ExportH1)  { if(ExportTimeframe(Symbol, PERIOD_H1,  "H1",  dtStart, dtEnd)) exported++; }
   if(ExportM15) { if(ExportTimeframe(Symbol, PERIOD_M15, "M15", dtStart, dtEnd)) exported++; }

   string msg = StringFormat("Export complete: %d timeframe(s) written to %s", exported, ExportDir);
   Print(msg);
   Alert(msg);
}

//+------------------------------------------------------------------+
//| Export a single timeframe to CSV                                  |
//|                                                                  |
//| Writes bars in the half-open interval [dtStart, dtEnd) to:       |
//|   {ExportDir}\{tfName}\{symbol}.csv                              |
//|                                                                  |
//| Returns true on success, false on any failure.                   |
//+------------------------------------------------------------------+
bool ExportTimeframe(
   const string symbol,
   const ENUM_TIMEFRAMES period,
   const string tfName,
   const datetime dtStart,
   const datetime dtEnd)
{
   //--- Build output path
   string outDir  = ExportDir + "\\" + tfName;
   string outFile = outDir + "\\" + symbol + ".csv";

   //--- Ensure the output sub-directory exists
   if(!FolderCreate(outDir, 0))
   {
      // FolderCreate returns false if the folder already exists — that is OK
      if(GetLastError() != 0 && GetLastError() != 5018)  // 5018 = ERR_WRONG_DIRECTORY
      {
         PrintFormat("ERROR: Cannot create directory %s (error %d)", outDir, GetLastError());
         return false;
      }
   }

   //--- Calculate how many bars are needed
   //    CopyRates by time range is cleaner than by count
   MqlRates rates[];
   int copied = CopyRates(symbol, period, dtStart, dtEnd, rates);

   if(copied <= 0)
   {
      PrintFormat("WARNING: No bars returned for %s/%s in [%s, %s). "
                  "Make sure the history is downloaded in MT5.",
                  symbol, tfName,
                  TimeToString(dtStart), TimeToString(dtEnd));
      return false;
   }

   PrintFormat("  %s/%s: %d bars retrieved", symbol, tfName, copied);

   //--- Open file for writing (overwrites any existing file)
   int handle = FileOpen(
      outFile,
      FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON,
      '\t'   // TAB separator — matches CSVAdapter expectations
   );

   if(handle == INVALID_HANDLE)
   {
      PrintFormat("ERROR: Cannot open file %s for writing (error %d)", outFile, GetLastError());
      return false;
   }

   //--- Write header row
   //    Columns match the CSVAdapter's _ALIAS_MAP exactly:
   //    Date, Time, Open, High, Low, Close, Tickvol, Vol, Spread
   FileWrite(handle, "Date", "Time", "Open", "High", "Low", "Close", "Tickvol", "Vol", "Spread");

   //--- Write data rows (oldest bar first — ascending chronological order)
   int written = 0;
   for(int i = 0; i < copied; i++)
   {
      datetime barTime = rates[i].time;

      //--- Skip bars outside [dtStart, dtEnd)
      if(barTime < dtStart || barTime >= dtEnd)
         continue;

      string dateStr = TimeToString(barTime, TIME_DATE);   // e.g. "2024.01.02"
      string timeStr = TimeToString(barTime, TIME_MINUTES); // e.g. "00:00"

      // Strip the date portion from timeStr if MT5 returns "2024.01.02 00:00"
      int spacePos = StringFind(timeStr, " ");
      if(spacePos >= 0)
         timeStr = StringSubstr(timeStr, spacePos + 1);

      FileWrite(
         handle,
         dateStr,
         timeStr,
         DoubleToString(rates[i].open,  5),
         DoubleToString(rates[i].high,  5),
         DoubleToString(rates[i].low,   5),
         DoubleToString(rates[i].close, 5),
         IntegerToString(rates[i].tick_volume),
         IntegerToString(rates[i].real_volume),
         IntegerToString(rates[i].spread)
      );
      written++;
   }

   FileClose(handle);

   PrintFormat("  %s/%s: %d bars written → %s", symbol, tfName, written, outFile);
   return true;
}
