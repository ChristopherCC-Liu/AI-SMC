//+------------------------------------------------------------------+
//|                                                 aismc_panel.mqh   |
//|                    AI-SMC EA on-chart live status panel (Round 6) |
//|                                                                   |
//|  Pure render layer — reads EA globals, never writes them. All     |
//|  chart objects share prefix AISMC_PANEL_ so ObjectsDeleteAll on   |
//|  OnDeinit cleans up without touching foreign objects.             |
//|                                                                   |
//|  Entry points:                                                    |
//|    PanelInit(corner, font_size)  — create labels + rectangles     |
//|    PanelUpdate(state)             — refresh text/colors (throttle  |
//|                                     by caller, typically 1s)      |
//|    PanelDeinit()                  — delete every AISMC_PANEL_ obj |
//|                                                                   |
//|  Row layout (y-offsets from panel origin, top-left):              |
//|     0- 22  header      "AI-SMC                      HH:MM:SS"     |
//|    25- 45  row1        symbol + price + balance                   |
//|    45- 65  row2        daily P&L + floating P&L                   |
//|    70- 90  row3        regime + confidence                        |
//|    90-110  row4        trail params                               |
//|   115-135  row5        control leg status                         |
//|   135-155  row6        treatment leg status                       |
//|   160-180  row7        tickets tracked                            |
//|   180-200  row8        last poll age                              |
//|   205-225  row9        heartbeat                                  |
//+------------------------------------------------------------------+
#ifndef AISMC_PANEL_MQH
#define AISMC_PANEL_MQH

#property strict

//--- Namespacing — every object starts with this prefix
#define PANEL_PREFIX          "AISMC_PANEL_"

//--- Geometry (pixels)
#define PANEL_WIDTH           220
#define PANEL_HEIGHT          240
#define PANEL_MARGIN_X        10
#define PANEL_MARGIN_Y        10
#define PANEL_HEADER_H        22
#define PANEL_ROW_H           18
#define PANEL_DIVIDER_H       1
#define PANEL_PAD_LEFT        6
#define PANEL_PAD_RIGHT       6

//--- Colors (MT5 clr_* constants; alpha not supported on label bg —
//    we approximate "alpha 220" by stacking body rect under labels.)
#define PANEL_BG_HEADER       clrDarkSlateBlue
#define PANEL_BG_BODY         clrBlack
#define PANEL_COLOR_BORDER    clrDimGray
#define PANEL_COLOR_DIVIDER   clrDimGray
#define PANEL_COLOR_TEXT      clrLightGray
#define PANEL_COLOR_TEXT_DIM  clrGray
#define PANEL_COLOR_HEADER    clrWhite
#define PANEL_COLOR_POS       clrLime
#define PANEL_COLOR_NEG       clrRed
#define PANEL_COLOR_ZERO      clrLightGray
#define PANEL_COLOR_HEART_ON  clrLime
#define PANEL_COLOR_HEART_OFF clrDarkGreen

//--- Internal state (module-local)
int    g_panel_corner      = 0;   // ENUM_BASE_CORNER; 0 = upper-left
int    g_panel_font_size   = 9;
string g_panel_font        = "Consolas";
bool   g_panel_initialized = false;

//+------------------------------------------------------------------+
//| Render state — a pure data carrier passed from the EA into      |
//| PanelUpdate().  Decouples the panel from EA globals so this     |
//| include file can be reused elsewhere (e.g. BTC EA) later.       |
//+------------------------------------------------------------------+
struct PanelState
{
   // Symbol + account
   string symbol;
   double bid;
   double ask;
   double balance;
   double equity;
   double floating_pnl;
   double daily_pnl;         // balance - start-of-day-balance; 0 if unknown

   // Regime / signal state (from /signal parse)
   string regime;            // e.g. "TRANSITION"
   double confidence;        // 0..1; negative means "unknown"
   double trail_activate_r;  // last signal's trail activate R
   double trail_distance_r;  // last signal's trail distance R

   // Per-leg status
   string control_action;    // last action (BUY/SELL/HOLD)
   int    control_cooldown;  // seconds remaining; -1 if none
   string treat_action;
   int    treat_cooldown;

   // Bookkeeping
   int      tickets_tracked;
   datetime last_poll_ts;
   datetime now;
};

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
string PanelObjName(const string suffix)
{
   return PANEL_PREFIX + suffix;
}

void PanelSetLabel(const string suffix,
                   const int x,
                   const int y,
                   const string text,
                   const color clr,
                   const int font_size = 0,
                   const string font = "")
{
   string name = PanelObjName(suffix);
   if (ObjectFind(0, name) < 0)
   {
      if (!ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0)) return;
      ObjectSetInteger(0, name, OBJPROP_CORNER,     g_panel_corner);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR,     ANCHOR_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
      ObjectSetInteger(0, name, OBJPROP_BACK,       false);
      ObjectSetString (0, name, OBJPROP_FONT,
                       StringLen(font) > 0 ? font : g_panel_font);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE,
                       font_size > 0 ? font_size : g_panel_font_size);
   }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString (0, name, OBJPROP_TEXT,      text);
   ObjectSetInteger(0, name, OBJPROP_COLOR,     clr);
}

void PanelSetRect(const string suffix,
                  const int x,
                  const int y,
                  const int w,
                  const int h,
                  const color bg,
                  const color border = CLR_NONE)
{
   string name = PanelObjName(suffix);
   if (ObjectFind(0, name) < 0)
   {
      if (!ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0)) return;
      ObjectSetInteger(0, name, OBJPROP_CORNER,     g_panel_corner);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
      ObjectSetInteger(0, name, OBJPROP_BACK,       true);
      ObjectSetInteger(0, name, OBJPROP_STYLE,      STYLE_SOLID);
      ObjectSetInteger(0, name, OBJPROP_WIDTH,      1);
   }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE,    x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE,    y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE,        w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE,        h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR,      bg);
   if (border == CLR_NONE)
   {
      ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, name, OBJPROP_COLOR, bg);
   }
   else
   {
      ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, name, OBJPROP_COLOR, border);
   }
}

//+------------------------------------------------------------------+
//| Formatting helpers                                                |
//+------------------------------------------------------------------+
color PanelColorForPnL(const double v)
{
   if (v > 0.005)  return PANEL_COLOR_POS;
   if (v < -0.005) return PANEL_COLOR_NEG;
   return PANEL_COLOR_ZERO;
}

string PanelFormatPnL(const double v)
{
   // Preserve leading sign so the UI reads "+$23.03" / "-$23.03".
   string sign = (v >= 0.0) ? "+" : "-";
   return sign + "$" + DoubleToString(MathAbs(v), 2);
}

color PanelColorForRegime(const string regime)
{
   if (regime == "TREND_UP")      return clrLime;
   if (regime == "TREND_DOWN")    return clrRed;
   if (regime == "TRANSITION")    return clrGold;
   if (regime == "CONSOLIDATION") return clrDodgerBlue;
   if (regime == "ATH_BREAKOUT")  return clrMagenta;
   return PANEL_COLOR_TEXT;
}

string PanelFormatConfidence(const double c)
{
   if (c < 0.0) return "—";
   string tag;
   if      (c < 0.4) tag = " weak";
   else if (c < 0.7) tag = " neutral";
   else              tag = " strong";
   return DoubleToString(c, 2) + tag;
}

string PanelFormatDuration(const int total_sec)
{
   if (total_sec < 0) return "—";
   int m = total_sec / 60;
   int s = total_sec % 60;
   string mm = (m < 10) ? "0" + IntegerToString(m) : IntegerToString(m);
   string ss = (s < 10) ? "0" + IntegerToString(s) : IntegerToString(s);
   return mm + ":" + ss;
}

string PanelFormatAge(const datetime ts, const datetime now)
{
   if (ts <= 0) return "never";
   int age = (int)(now - ts);
   if (age < 0) age = 0;
   if (age < 60) return IntegerToString(age) + "s ago";
   if (age < 3600)
      return IntegerToString(age / 60) + "m"
             + IntegerToString(age % 60) + "s ago";
   return IntegerToString(age / 3600) + "h ago";
}

string PanelFormatTime(const datetime ts)
{
   MqlDateTime dt;
   TimeToStruct(ts, dt);
   string hh = (dt.hour < 10)   ? "0" + IntegerToString(dt.hour)
                                : IntegerToString(dt.hour);
   string mm = (dt.min  < 10)   ? "0" + IntegerToString(dt.min)
                                : IntegerToString(dt.min);
   string ss = (dt.sec  < 10)   ? "0" + IntegerToString(dt.sec)
                                : IntegerToString(dt.sec);
   return hh + ":" + mm + ":" + ss;
}

//+------------------------------------------------------------------+
//| PanelInit — creates all objects.  Safe to call multiple times    |
//| (idempotent due to ObjectFind guards in PanelSet*).              |
//+------------------------------------------------------------------+
void PanelInit(const int corner = 0, const int font_size = 9)
{
   g_panel_corner    = corner;
   g_panel_font_size = font_size > 0 ? font_size : 9;

   int ox = PANEL_MARGIN_X;
   int oy = PANEL_MARGIN_Y;

   // --- Body background (behind everything else) -----------------
   PanelSetRect("bg",
                ox, oy, PANEL_WIDTH, PANEL_HEIGHT,
                PANEL_BG_BODY, PANEL_COLOR_BORDER);

   // --- Header bar ----------------------------------------------
   PanelSetRect("bg_header",
                ox, oy, PANEL_WIDTH, PANEL_HEADER_H,
                PANEL_BG_HEADER);
   PanelSetLabel("title",
                 ox + PANEL_PAD_LEFT,
                 oy + 4,
                 "AI-SMC",
                 PANEL_COLOR_HEADER,
                 g_panel_font_size + 1, "Consolas");
   PanelSetLabel("clock",
                 ox + PANEL_WIDTH - 60,
                 oy + 4,
                 "00:00:00",
                 PANEL_COLOR_HEADER);

   // --- Row 1 : symbol + price + balance ------------------------
   int y1 = oy + PANEL_HEADER_H + 6;
   PanelSetLabel("row1_sym",
                 ox + PANEL_PAD_LEFT, y1, "—", PANEL_COLOR_TEXT);
   PanelSetLabel("row1_px",
                 ox + 70, y1, "—", PANEL_COLOR_TEXT);
   PanelSetLabel("row1_bal",
                 ox + 130, y1, "—", PANEL_COLOR_TEXT);

   // --- Row 2 : daily P&L + floating P&L ------------------------
   int y2 = y1 + PANEL_ROW_H;
   PanelSetLabel("row2_daily_lbl",
                 ox + PANEL_PAD_LEFT, y2, "Daily", PANEL_COLOR_TEXT_DIM);
   PanelSetLabel("row2_daily",
                 ox + 50, y2, "—", PANEL_COLOR_TEXT);
   PanelSetLabel("row2_float_lbl",
                 ox + 115, y2, "Float", PANEL_COLOR_TEXT_DIM);
   PanelSetLabel("row2_float",
                 ox + 155, y2, "—", PANEL_COLOR_TEXT);

   PanelSetRect("div1",
                ox + PANEL_PAD_LEFT,
                y2 + PANEL_ROW_H - 2,
                PANEL_WIDTH - PANEL_PAD_LEFT - PANEL_PAD_RIGHT,
                PANEL_DIVIDER_H,
                PANEL_COLOR_DIVIDER);

   // --- Row 3 : regime + confidence -----------------------------
   int y3 = y2 + PANEL_ROW_H + 4;
   PanelSetLabel("row3_lbl",
                 ox + PANEL_PAD_LEFT, y3, "Regime", PANEL_COLOR_TEXT_DIM);
   PanelSetLabel("row3_val",
                 ox + 60, y3, "—", PANEL_COLOR_TEXT);
   PanelSetLabel("row3_conf",
                 ox + 145, y3, "—", PANEL_COLOR_TEXT);

   // --- Row 4 : trail params ------------------------------------
   int y4 = y3 + PANEL_ROW_H;
   PanelSetLabel("row4_lbl",
                 ox + PANEL_PAD_LEFT, y4, "Trail", PANEL_COLOR_TEXT_DIM);
   PanelSetLabel("row4_val",
                 ox + 60, y4, "—", PANEL_COLOR_TEXT);

   PanelSetRect("div2",
                ox + PANEL_PAD_LEFT,
                y4 + PANEL_ROW_H - 2,
                PANEL_WIDTH - PANEL_PAD_LEFT - PANEL_PAD_RIGHT,
                PANEL_DIVIDER_H,
                PANEL_COLOR_DIVIDER);

   // --- Row 5 : control leg -------------------------------------
   int y5 = y4 + PANEL_ROW_H + 4;
   PanelSetLabel("row5_lbl",
                 ox + PANEL_PAD_LEFT, y5, "Control",
                 PANEL_COLOR_TEXT_DIM);
   PanelSetLabel("row5_act",
                 ox + 70, y5, "—", PANEL_COLOR_TEXT);
   PanelSetLabel("row5_cd",
                 ox + 135, y5, "—", PANEL_COLOR_TEXT_DIM);

   // --- Row 6 : treatment leg -----------------------------------
   int y6 = y5 + PANEL_ROW_H;
   PanelSetLabel("row6_lbl",
                 ox + PANEL_PAD_LEFT, y6, "Treat",
                 PANEL_COLOR_TEXT_DIM);
   PanelSetLabel("row6_act",
                 ox + 70, y6, "—", PANEL_COLOR_TEXT);
   PanelSetLabel("row6_cd",
                 ox + 135, y6, "—", PANEL_COLOR_TEXT_DIM);

   PanelSetRect("div3",
                ox + PANEL_PAD_LEFT,
                y6 + PANEL_ROW_H - 2,
                PANEL_WIDTH - PANEL_PAD_LEFT - PANEL_PAD_RIGHT,
                PANEL_DIVIDER_H,
                PANEL_COLOR_DIVIDER);

   // --- Row 7 : tickets tracked ---------------------------------
   int y7 = y6 + PANEL_ROW_H + 4;
   PanelSetLabel("row7",
                 ox + PANEL_PAD_LEFT, y7,
                 "Tickets tracked: 0",
                 PANEL_COLOR_TEXT);

   // --- Row 8 : last poll age -----------------------------------
   int y8 = y7 + PANEL_ROW_H;
   PanelSetLabel("row8",
                 ox + PANEL_PAD_LEFT, y8,
                 "Last poll —",
                 PANEL_COLOR_TEXT_DIM);

   PanelSetRect("div4",
                ox + PANEL_PAD_LEFT,
                y8 + PANEL_ROW_H - 2,
                PANEL_WIDTH - PANEL_PAD_LEFT - PANEL_PAD_RIGHT,
                PANEL_DIVIDER_H,
                PANEL_COLOR_DIVIDER);

   // --- Row 9 : heartbeat ---------------------------------------
   int y9 = y8 + PANEL_ROW_H + 4;
   PanelSetLabel("heart",
                 ox + PANEL_PAD_LEFT, y9,
                 "● Running",
                 PANEL_COLOR_HEART_ON);

   g_panel_initialized = true;
   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| PanelUpdate — refresh text + color from a PanelState snapshot.   |
//| Caller should throttle (1 Hz is plenty; the chart is fine with   |
//| more but flicker is wasteful).                                   |
//+------------------------------------------------------------------+
void PanelUpdate(const PanelState &s)
{
   if (!g_panel_initialized) return;

   int ox = PANEL_MARGIN_X;
   int oy = PANEL_MARGIN_Y;

   // Header clock — use terminal time (what the user sees on chart).
   PanelSetLabel("clock",
                 ox + PANEL_WIDTH - 60,
                 oy + 4,
                 PanelFormatTime(s.now),
                 PANEL_COLOR_HEADER);

   // Row 1 : symbol / price / balance
   double mid = (s.bid > 0 && s.ask > 0) ? (s.bid + s.ask) * 0.5 : s.bid;
   PanelSetLabel("row1_sym",
                 ox + PANEL_PAD_LEFT,
                 oy + PANEL_HEADER_H + 6,
                 s.symbol, PANEL_COLOR_TEXT);
   PanelSetLabel("row1_px",
                 ox + 70,
                 oy + PANEL_HEADER_H + 6,
                 DoubleToString(mid, 2), PANEL_COLOR_TEXT);
   PanelSetLabel("row1_bal",
                 ox + 130,
                 oy + PANEL_HEADER_H + 6,
                 "Bal $" + DoubleToString(s.balance, 2),
                 PANEL_COLOR_TEXT);

   // Row 2 : daily / floating P&L
   int y2 = oy + PANEL_HEADER_H + 6 + PANEL_ROW_H;
   PanelSetLabel("row2_daily",
                 ox + 50, y2,
                 PanelFormatPnL(s.daily_pnl),
                 PanelColorForPnL(s.daily_pnl));
   PanelSetLabel("row2_float",
                 ox + 155, y2,
                 PanelFormatPnL(s.floating_pnl),
                 PanelColorForPnL(s.floating_pnl));

   // Row 3 : regime + confidence
   int y3 = y2 + PANEL_ROW_H + 4;
   string regime_text = (StringLen(s.regime) > 0) ? s.regime : "—";
   PanelSetLabel("row3_val",
                 ox + 60, y3,
                 regime_text, PanelColorForRegime(s.regime));
   PanelSetLabel("row3_conf",
                 ox + 145, y3,
                 PanelFormatConfidence(s.confidence),
                 PANEL_COLOR_TEXT);

   // Row 4 : trail params — e.g. "0.5R @ 0.5R"
   int y4 = y3 + PANEL_ROW_H;
   string trail_text;
   if (s.trail_activate_r > 0 && s.trail_distance_r > 0)
      trail_text = DoubleToString(s.trail_activate_r, 2) + "R @ "
                 + DoubleToString(s.trail_distance_r, 2) + "R";
   else
      trail_text = "—";
   PanelSetLabel("row4_val",
                 ox + 60, y4,
                 trail_text, PANEL_COLOR_TEXT);

   // Row 5 : control leg
   int y5 = y4 + PANEL_ROW_H + 4;
   string ctrl_act = (StringLen(s.control_action) > 0)
                        ? s.control_action : "HOLD";
   color  ctrl_clr = (ctrl_act == "BUY"  || ctrl_act == "RANGE BUY")
                        ? PANEL_COLOR_POS
                     : (ctrl_act == "SELL" || ctrl_act == "RANGE SELL")
                        ? PANEL_COLOR_NEG
                        : PANEL_COLOR_TEXT;
   PanelSetLabel("row5_act",
                 ox + 70, y5, ctrl_act, ctrl_clr);
   string ctrl_cd_text = (s.control_cooldown > 0)
                           ? "CD " + PanelFormatDuration(s.control_cooldown)
                           : "—";
   PanelSetLabel("row5_cd",
                 ox + 135, y5, ctrl_cd_text, PANEL_COLOR_TEXT_DIM);

   // Row 6 : treatment leg
   int y6 = y5 + PANEL_ROW_H;
   string tr_act = (StringLen(s.treat_action) > 0)
                        ? s.treat_action : "HOLD";
   color  tr_clr = (tr_act == "BUY"  || tr_act == "RANGE BUY")
                        ? PANEL_COLOR_POS
                     : (tr_act == "SELL" || tr_act == "RANGE SELL")
                        ? PANEL_COLOR_NEG
                        : PANEL_COLOR_TEXT;
   PanelSetLabel("row6_act",
                 ox + 70, y6, tr_act, tr_clr);
   string tr_cd_text = (s.treat_cooldown > 0)
                           ? "CD " + PanelFormatDuration(s.treat_cooldown)
                           : "—";
   PanelSetLabel("row6_cd",
                 ox + 135, y6, tr_cd_text, PANEL_COLOR_TEXT_DIM);

   // Row 7 : tickets tracked
   int y7 = y6 + PANEL_ROW_H + 4;
   PanelSetLabel("row7",
                 ox + PANEL_PAD_LEFT, y7,
                 "Tickets tracked: " + IntegerToString(s.tickets_tracked),
                 PANEL_COLOR_TEXT);

   // Row 8 : last poll age
   int y8 = y7 + PANEL_ROW_H;
   PanelSetLabel("row8",
                 ox + PANEL_PAD_LEFT, y8,
                 "Last poll " + PanelFormatAge(s.last_poll_ts, s.now),
                 PANEL_COLOR_TEXT_DIM);

   // Row 9 : heartbeat — blink dot every second
   int y9 = y8 + PANEL_ROW_H + 4;
   bool on = ((int)s.now % 2) == 0;
   PanelSetLabel("heart",
                 ox + PANEL_PAD_LEFT, y9,
                 "● Running",
                 on ? PANEL_COLOR_HEART_ON : PANEL_COLOR_HEART_OFF);

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| PanelDeinit — wipe every AISMC_PANEL_* object we created.         |
//| Safe to call even if PanelInit was never called.                  |
//+------------------------------------------------------------------+
void PanelDeinit()
{
   ObjectsDeleteAll(0, PANEL_PREFIX);
   g_panel_initialized = false;
   ChartRedraw(0);
}

#endif // AISMC_PANEL_MQH
