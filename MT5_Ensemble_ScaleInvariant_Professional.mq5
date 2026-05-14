//+------------------------------------------------------------------+
//| Ensemble Scale-Invariant Professional             Copyright 2026 |
//+------------------------------------------------------------------+
#property strict
#property version "1.41"
#property description "MT5 EA: professional scale-invariant ONNX ensemble (MLP + LightGBM + HGB + ExtraTrees + Ridge + NaiveBayes)"
#property description "Crypto-aware adaptive spread guard, support/resistance skipping, trend confirmation"

#include <Trade/Trade.mqh>

#resource "mlp.onnx" as uchar MlpModel[]
#resource "lightgbm.onnx" as uchar LgbmModel[]
#resource "hgb.onnx" as uchar HgbModel[]
#resource "extratrees.onnx" as uchar ExtraTreesModel[]
#resource "ridge.onnx"      as uchar RidgeModel[]
#resource "naivebayes.onnx" as uchar NaiveBayesModel[]

input double InpBaseLots = 0.10;
input bool InpUseConfidenceSizing = true;
input double InpMinLotMultiplier = 0.50;
input double InpMaxLotMultiplier = 1.50;

input double InpEntryProbThreshold = 0.60;
input double InpMinProbGap = 0.15;
input bool InpUseAtrStops = true;
input double InpStopAtrMultiple = 1.00;
input double InpTakeAtrMultiple = 3.00;
input int InpMaxBarsInTrade = 8;
input bool InpCloseOnOppositeSignal = false;
input bool InpAllowLong = true;
input bool InpAllowShort = true;

input double InpMlpWeight  = 0.20;
input double InpLgbmWeight = 0.20;
input double InpHgbWeight  = 0.20;
input double InpExtraTreesWeight = 0.20;
input double InpRidgeWeight      = 0.20;
input double InpNaiveBayesWeight = 0.20;

input bool InpUseHourFilter = false;
input int InpHourStart = 0;
input int InpHourEnd = 23;
input string InpAllowedHoursCsv = "";
input string InpDeniedHoursCsv = "";
input bool InpUseHourSoftBias = false;
input int InpSoftHourStart = 4;
input int InpSoftHourEnd = 18;
input double InpPreferredHourMultiplier = 1.08;
input double InpOffHourMultiplier = 0.94;

input bool InpUseSpreadGuard = false;
input bool InpUseAdaptiveSpreadGuard = true;
input double InpMaxSpreadPoints = 800.0;
input double InpMaxSpreadPctOfPrice = 0.0015;
input double InpMaxSpreadAtrFraction = 0.20;
input double InpAdaptiveSpreadSlack = 1.15;

input bool InpUseCooldownAfterClose = true;
input int InpCooldownBars = 2;

input bool InpUseDailyLossGuard = true;
input double InpDailyLossLimitMoney = 25.0;
input bool InpDailyLossFlatOnTrigger = true;

input bool InpUseTrend = true;
input bool InpUseSupportAndResistance = true;
input double InpSupportAndResistanceBuffer = 0.002; // InpSupportAndResistanceBuffer=0.2%
input bool InpUseBreakout = true;
input bool InpUseStrongBreakout = true;

input bool InpUseVolumeFilter = false;
input double InpMinVolume = 0.001;
input bool InpFilterMinVolume = false;
input double InpMaxVolume = 0.95;
input bool InpFilterMaxVolume = false;

input long InpMagic = 26042026;
input bool InpLog = false;
input bool InpDebugLog = false;

const int FEATURE_COUNT = 17;
const int CLASS_COUNT = 3;
const long EXT_INPUT_SHAPE[] = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[] = {1};
const long EXT_PROBA_SHAPE[] = {1, CLASS_COUNT};

enum SignalDirection { SIGNAL_SELL = -1, SIGNAL_FLAT = 0, SIGNAL_BUY = 1 };

struct SignalInfo
  {
   SignalDirection   signal;
   double            pSell;
   double            pFlat;
   double            pBuy;
   double            bestDirectionProb;
   double            probGap;
  };

CTrade trade;

long g_mlp_handle        = INVALID_HANDLE;
long g_lgbm_handle       = INVALID_HANDLE;
long g_hgb_handle        = INVALID_HANDLE;
long g_extratrees_handle = INVALID_HANDLE;
long g_ridge_handle      = INVALID_HANDLE;
long g_naivebayes_handle = INVALID_HANDLE;

datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;

double g_w_mlp        = 0.0;
double g_w_lgbm       = 0.0;
double g_w_hgb        = 0.0;
double g_w_extratrees = 0.0;
double g_w_ridge      = 0.0;
double g_w_naivebayes = 0.0;

int g_cooldown_remaining = 0;
int g_last_history_deals_total = 0;
int g_guard_day_key = -1;
double g_guard_day_closed_pnl = 0.0;
bool g_daily_loss_guard_active = false;
bool g_allowed_hours_mask[24];
bool g_denied_hours_mask[24];
bool g_has_allowed_hours_csv = false;
bool g_has_denied_hours_csv = false;

//+------------------------------------------------------------------+
//| LogInfo                                                          |
//+------------------------------------------------------------------+
void LogInfo(string message)
  {
   if(InpLog)
      Print(message);
  }

//+------------------------------------------------------------------+
//| LogDebug                                                         |
//+------------------------------------------------------------------+
void LogDebug(string message)
  {
   if(InpLog && InpDebugLog)
      Print(message);
  }

//+------------------------------------------------------------------+
//| DayKey                                                           |
//+------------------------------------------------------------------+
int DayKey(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
  }

//+------------------------------------------------------------------+
//| IsNewBar                                                         |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == 0)
     {
      LogDebug("IsNewBar: iTime returned 0, skipping tick.");
      return false;
     }

   if(g_last_bar_time == 0)
     {
      g_last_bar_time = current_bar_time;
      LogDebug("IsNewBar: initialized last bar timestamp, waiting for next bar.");
      return false;
     }

   if(current_bar_time != g_last_bar_time)
     {
      g_last_bar_time = current_bar_time;
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Mean                                                             |
//+------------------------------------------------------------------+
double Mean(const double &arr[], int start_shift, int count)
  {
   double sum = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
      sum += arr[i];
   return sum / count;
  }

//+------------------------------------------------------------------+
//| StdDev                                                           |
//+------------------------------------------------------------------+
double StdDev(const double &arr[], int start_shift, int count)
  {
   double m = Mean(arr, start_shift, count);
   double s = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
     {
      double d = arr[i] - m;
      s += d * d;
     }
   return MathSqrt(s / MathMax(count - 1, 1));
  }

//+------------------------------------------------------------------+
//| CalcATR                                                          |
//+------------------------------------------------------------------+
double CalcATR(const MqlRates &rates[], int start_shift, int period)
  {
   double sum_tr = 0.0;
   for(int i = start_shift; i < start_shift + period; i++)
     {
      double high = rates[i].high;
      double low = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr1 = high - low;
      double tr2 = MathAbs(high - prev_close);
      double tr3 = MathAbs(low - prev_close);
      double tr = MathMax(tr1, MathMax(tr2, tr3));
      sum_tr += tr;
     }
   return sum_tr / period;
  }

//+------------------------------------------------------------------+
//| NormalizeWeights                                                 |
//+------------------------------------------------------------------+
bool NormalizeWeights()
  {
   double a = MathMax(0.0, InpMlpWeight);
   double b = MathMax(0.0, InpLgbmWeight);
   double c = MathMax(0.0, InpHgbWeight);
   double d = MathMax(0.0, InpExtraTreesWeight);
   double e = MathMax(0.0, InpRidgeWeight);
   double f = MathMax(0.0, InpNaiveBayesWeight);

   double s = a + b + c + d + e + f;
   if(s <= 0.0)
     {
      LogInfo(
         "NormalizeWeights failed: sum of ensemble weights is <= 0. Check "
         "InpMlpWeight/InpLgbmWeight/InpHgbWeight/InpExtraTreesWeight/InpRidgeWeight/InpNaiveBayesWeight.");
      return false;
     }

   g_w_mlp        = a / s;
   g_w_lgbm       = b / s;
   g_w_hgb        = c / s;
   g_w_extratrees = d / s;
   g_w_ridge      = e / s;
   g_w_naivebayes = f / s;
   return true;
  }

//+------------------------------------------------------------------+
//| IsHourInRange                                                    |
//+------------------------------------------------------------------+
bool IsHourInRange(int hour_value, int start_hour, int end_hour)
  {
   if(start_hour < 0 || start_hour > 23 || end_hour < 0 || end_hour > 23)
      return false;

   if(start_hour <= end_hour)
      return (hour_value >= start_hour && hour_value <= end_hour);

   return (hour_value >= start_hour || hour_value <= end_hour);
  }

//+------------------------------------------------------------------+
//| IsDigitsOnly                                                     |
//+------------------------------------------------------------------+
bool IsDigitsOnly(const string value)
  {
   int n = StringLen(value);
   if(n <= 0)
      return false;

   for(int i = 0; i < n; i++)
     {
      ushort ch = StringGetCharacter(value, i);
      if(ch < '0' || ch > '9')
         return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| ParseHoursCsv                                                    |
//+------------------------------------------------------------------+
bool ParseHoursCsv(const string csv, bool &hours_mask[], bool &has_any_values,
                   const string label)
  {
   has_any_values = false;
   ArrayInitialize(hours_mask, false);

   string normalized = csv;
   StringReplace(normalized, ";", ",");
   StringReplace(normalized, " ", "");
   StringReplace(normalized, "\t", "");

   if(StringLen(normalized) == 0)
      return true;

   string tokens[];
   int count = StringSplit(normalized, ',', tokens);
   if(count <= 0)
      return true;

   for(int i = 0; i < count; i++)
     {
      string token = tokens[i];
      if(StringLen(token) == 0)
         continue;

      if(!IsDigitsOnly(token))
        {
         LogInfo(
            StringFormat("ParseHoursCsv: invalid token '%s' in %s. Expected CSV "
                         "of integers 0..23.",
                         token, label));
         return false;
        }

      int hour = (int)StringToInteger(token);
      if(hour < 0 || hour > 23)
        {
         LogInfo(StringFormat(
                    "ParseHoursCsv: hour %d out of range in %s. Allowed range is 0..23.",
                    hour, label));
         return false;
        }

      hours_mask[hour] = true;
      has_any_values = true;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| IsHourAllowedByMasks                                             |
//+------------------------------------------------------------------+
bool IsHourAllowedByMasks(int hour_value)
  {
   if(hour_value < 0 || hour_value > 23)
      return false;

   if(g_has_allowed_hours_csv && !g_allowed_hours_mask[hour_value])
      return false;

   if(g_denied_hours_mask[hour_value])
      return false;

   return true;
  }

//+------------------------------------------------------------------+
//| CurrentServerHour                                                |
//+------------------------------------------------------------------+
int CurrentServerHour()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return dt.hour;
  }

//+------------------------------------------------------------------+
//| NormalizeVolumeToSymbol                                          |
//+------------------------------------------------------------------+
double NormalizeVolumeToSymbol(double requested_lots)
  {
   double vol_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vol_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(vol_step <= 0.0)
      vol_step = vol_min;

   double lots = MathMax(vol_min, MathMin(vol_max, requested_lots));
   lots = MathFloor(lots / vol_step) * vol_step;

   int digits = 2;
   if(vol_step > 0.0)
     {
      double tmp = vol_step;
      digits = 0;
      while(digits < 8 && MathRound(tmp) != tmp)
        {
         tmp *= 10.0;
         digits++;
        }
     }

   lots = NormalizeDouble(lots, digits);
   return MathMax(vol_min, MathMin(vol_max, lots));
  }

//+------------------------------------------------------------------+
//| GetCurrentSpreadPoints                                           |
//+------------------------------------------------------------------+
double GetCurrentSpreadPoints()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      return 0.0;
   return (ask - bid) / point;
  }

//+------------------------------------------------------------------+
//| GetCurrentSpreadPrice                                            |
//+------------------------------------------------------------------+
double GetCurrentSpreadPrice()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return (ask - bid);
  }

//+------------------------------------------------------------------+
//| SpreadAllows                                                     |
//+------------------------------------------------------------------+
bool SpreadAllows(double atr14_raw)
  {
   if(!InpUseSpreadGuard)
      return true;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double mid = (ask + bid) * 0.5;
   double spread_points = GetCurrentSpreadPoints();
   double spread_price = GetCurrentSpreadPrice();

   if(mid <= 0.0)
     {
      LogInfo("Spread guard blocked entry: mid price <= 0.");
      return false;
     }

   if(!InpUseAdaptiveSpreadGuard)
     {
      bool ok_points = (spread_points <= InpMaxSpreadPoints);
      if(!ok_points)
         LogInfo(StringFormat(
                    "SpreadAllows: blocked by fixed spread guard %.2f > %.2f points.",
                    spread_points, InpMaxSpreadPoints));
      return ok_points;
     }

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double max_spread_price_from_points = InpMaxSpreadPoints * point;
   double max_spread_price_from_pct = mid * MathMax(0.0, InpMaxSpreadPctOfPrice);
   double max_spread_price_from_atr =
      atr14_raw * MathMax(0.0, InpMaxSpreadAtrFraction);

   double adaptive_limit_price =
      MathMin(max_spread_price_from_points,
              MathMin(max_spread_price_from_pct, max_spread_price_from_atr));
   adaptive_limit_price *= MathMax(1.0, InpAdaptiveSpreadSlack);

   if(InpLog && InpDebugLog)
      PrintFormat(
         "Spread check: pts=%.2f spread=%.5f mid=%.5f atr=%.5f limPts=%.5f "
         "limPct=%.5f limAtr=%.5f final=%.5f",
         spread_points, spread_price, mid, atr14_raw,
         max_spread_price_from_points, max_spread_price_from_pct,
         max_spread_price_from_atr, adaptive_limit_price);

   bool allow = (spread_price <= adaptive_limit_price);
   if(!allow)
      LogInfo(
         "Spread guard blocked entry: current spread is above adaptive limit.");
   return allow;
  }

//+------------------------------------------------------------------+
//| BuildFeatureVector                                               |
//+------------------------------------------------------------------+
bool BuildFeatureVector(matrixf &features, double &atr14_raw)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(_Symbol, _Period, 0, 230, rates);
   if(copied < 220)
     {
      LogInfo("BuildFeatureVector failed: not enough bars from CopyRates (need >= 220).");
      return false;
     }

   double closes[], opens[];
   ArrayResize(closes, ArraySize(rates));
   ArrayResize(opens, ArraySize(rates));
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(opens, true);

   for(int i = 0; i < ArraySize(rates); i++)
     {
      closes[i] = rates[i].close;
      opens[i] = rates[i].open;
     }

   int s = 1;
   double eps = 1e-12;
   double c = closes[s];
   double o = opens[s];
   double h = rates[s].high;
   double l = rates[s].low;

   double ret_1 = (closes[s] / (closes[s + 1] + eps)) - 1.0;
   double ret_3 = (closes[s] / (closes[s + 3] + eps)) - 1.0;
   double ret_5 = (closes[s] / (closes[s + 5] + eps)) - 1.0;
   double ret_10 = (closes[s] / (closes[s + 10] + eps)) - 1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns, 30);
   for(int i = 0; i < 30; i++)
      one_bar_returns[i] = (closes[s + i] / (closes[s + i + 1] + eps)) - 1.0;

   double vol_10 = StdDev(one_bar_returns, 0, 10);
   double vol_20 = StdDev(one_bar_returns, 0, 20);
   double vol_ratio_10_20 = (vol_10 / (vol_20 + eps)) - 1.0;

   double sma_10 = Mean(closes, s, 10);
   double sma_20 = Mean(closes, s, 20);
   if(sma_10 == 0.0 || sma_20 == 0.0)
     {
      LogInfo(
         "BuildFeatureVector failed: SMA10 or SMA20 equals zero, cannot compute "
         "normalized features.");
      return false;
     }

   double dist_sma_10 = (c / (sma_10 + eps)) - 1.0;
   double dist_sma_20 = (c / (sma_20 + eps)) - 1.0;

   double mean_20 = Mean(closes, s, 20);
   double std_20 = StdDev(closes, s, 20);
   double zscore_20 = 0.0;
   if(std_20 > 0.0)
      zscore_20 = (c - mean_20) / std_20;

   atr14_raw = CalcATR(rates, s, 14);
   double atr_pct_14 = atr14_raw / (c + eps);
   double range_pct_1 = (h - l) / (c + eps);
   double body_pct_1 = (c - o) / (o + eps);

// === RSI 14 ===
   double gain=0, loss=0;
   for(int i=1;i<=14;i++)
     {
      double diff = closes[i] - closes[i + 1];
      if(diff > 0)
         gain += diff;
      else
         loss -= diff;
     }

   double avg_gain = gain / 14.0;
   double avg_loss = loss / 14.0;
   double rs = avg_gain / (avg_loss + eps);
   double rsi_14 = 100.0 - (100.0 / (1.0 + rs));


// === SMA 50 / 200 ===

   double sma50 = Mean(closes, s, 50);
   double sma200 = Mean(closes, s, 200);

   double sma_ratio_50_200 = (sma50 / (sma200 + eps)) - 1.0;
   double dist_sma_50 = (c / (sma50 + eps)) - 1.0;
   double dist_sma_200 = (c / (sma200 + eps)) - 1.0;


   features.Resize(1, FEATURE_COUNT);
   features[0][0] = (float)ret_1;
   features[0][1] = (float)ret_3;
   features[0][2] = (float)ret_5;
   features[0][3] = (float)ret_10;
   features[0][4] = (float)vol_10;
   features[0][5] = (float)vol_20;
   features[0][6] = (float)vol_ratio_10_20;
   features[0][7] = (float)dist_sma_10;
   features[0][8] = (float)dist_sma_20;
   features[0][9] = (float)zscore_20;
   features[0][10] = (float)atr_pct_14;
   features[0][11] = (float)range_pct_1;
   features[0][12] = (float)body_pct_1;
   features[0][13] = (float)rsi_14;
   features[0][14] = (float)sma_ratio_50_200;
   features[0][15] = (float)dist_sma_50;
   features[0][16] = (float)dist_sma_200;

   return true;
  }

//+------------------------------------------------------------------+
//| RunSingleModel                                                   |
//+------------------------------------------------------------------+
bool RunSingleModel(long model_handle, const matrixf &x, double &pSell,
                    double &pFlat, double &pBuy)
  {
   long predicted_label[1];
   matrixf probs;
   probs.Resize(1, CLASS_COUNT);

   if(!OnnxRun(model_handle, 0, x, predicted_label, probs))
     {
      LogInfo("RunSingleModel: OnnxRun failed.");
      return false;
     }

   pSell = probs[0][0];
   pFlat = probs[0][1];
   pBuy = probs[0][2];
   return true;
  }

//+------------------------------------------------------------------+
//| RunRidgeModel                                                    |
//| Run a RidgeClassifier model exported to ONNX.  The ONNX output   |
//| provides raw decision scores for each class rather than          |
//| calibrated probabilities.                                        |
//| Convert these scores into a probability distribution using a     |
//| softmax.                                                         |
//+------------------------------------------------------------------+
bool RunRidgeModel(long model_handle, const matrixf &x, double &pSell,
                   double &pFlat, double &pBuy)
  {
   long predicted_label[1];
   matrixf scores;
   scores.Resize(1, CLASS_COUNT);
   if(!OnnxRun(model_handle, 0, x, predicted_label, scores))
     {
      LogInfo("RunRidgeModel: OnnxRun failed.");
      return false;
     }
   double e0 = MathExp(scores[0][0]);
   double e1 = MathExp(scores[0][1]);
   double e2 = MathExp(scores[0][2]);
   double sum = e0 + e1 + e2;
   if(sum <= 0.0)
      sum = 1e-9;
   pSell = e0 / sum;
   pFlat = e1 / sum;
   pBuy  = e2 / sum;
   return true;
  }

//+------------------------------------------------------------------+
//| PredictEnsembleProbabilities                                     |
//+------------------------------------------------------------------+
bool PredictEnsembleProbabilities(double &pSell, double &pFlat, double &pBuy,
                                  double &atr14_raw)
  {
   matrixf x;
   if(!BuildFeatureVector(x, atr14_raw))
     {
      LogInfo("PredictEnsembleProbabilities: feature vector build failed.");
      return false;
     }

   double s1, f1, b1; // MLP
   double s2, f2, b2; // LightGBM
   double s3, f3, b3; // HGB
   double s4, f4, b4; // ExtraTrees
   double s5, f5, b5; // Ridge
   double s6, f6, b6; // NaiveBayes

   if(!RunSingleModel(g_mlp_handle, x, s1, f1, b1))
     {
      LogInfo("PredictEnsembleProbabilities: MLP model inference failed.");
      return false;
     }
   if(!RunSingleModel(g_lgbm_handle, x, s2, f2, b2))
     {
      LogInfo("PredictEnsembleProbabilities: LightGBM model inference failed.");
      return false;
     }
   if(!RunSingleModel(g_hgb_handle, x, s3, f3, b3))
     {
      LogInfo("PredictEnsembleProbabilities: HistGradientBoosting model inference failed.");
      return false;
     }
   if(!RunSingleModel(g_extratrees_handle, x, s4, f4, b4))
     {
      LogInfo("PredictEnsembleProbabilities: ExtraTrees model inference failed.");
      return false;
     }
   if(!RunRidgeModel(g_ridge_handle, x, s5, f5, b5))
     {
      LogInfo("PredictEnsembleProbabilities: Ridge model inference failed.");
      return false;
     }
   if(!RunSingleModel(g_naivebayes_handle, x, s6, f6, b6))
     {
      LogInfo("PredictEnsembleProbabilities: NaiveBayes model inference failed.");
      return false;
     }

// Combine probabilities using normalized weights
   pSell = g_w_mlp * s1 + g_w_lgbm * s2 + g_w_hgb * s3 + g_w_extratrees * s4 + g_w_ridge * s5 + g_w_naivebayes * s6;
   pFlat = g_w_mlp * f1 + g_w_lgbm * f2 + g_w_hgb * f3 + g_w_extratrees * f4 + g_w_ridge * f5 + g_w_naivebayes * f6;
   pBuy  = g_w_mlp * b1 + g_w_lgbm * b2 + g_w_hgb * b3 + g_w_extratrees * b4 + g_w_ridge * b5 + g_w_naivebayes * b6;
   LogDebug(StringFormat("Ensemble probabilities: pSell=%.5f pFlat=%.5f pBuy=%.5f",
                         pSell, pFlat, pBuy));

   return true;
  }

//+------------------------------------------------------------------+
//| ApplyHourSoftBias                                                |
//+------------------------------------------------------------------+
void ApplyHourSoftBias(double &pSell, double &pBuy)
  {
   if(!InpUseHourSoftBias)
      return;

   int hour_now = CurrentServerHour();
   bool preferred = IsHourInRange(hour_now, InpSoftHourStart, InpSoftHourEnd);
   double mult = (preferred ? InpPreferredHourMultiplier : InpOffHourMultiplier);

   pSell *= mult;
   pBuy *= mult;

   double maxv = MathMax(pSell, pBuy);
   if(maxv > 1.0)
     {
      pSell /= maxv;
      pBuy /= maxv;
     }
   LogDebug(StringFormat(
               "Hour soft bias applied (hour=%d preferred=%s): pSell=%.5f pBuy=%.5f",
               hour_now, (preferred ? "true" : "false"), pSell, pBuy));
  }

//+------------------------------------------------------------------+
//| BuildSignalInfo                                                  |
//+------------------------------------------------------------------+
SignalInfo BuildSignalInfo(double pSell, double pFlat, double pBuy)
  {
   SignalInfo info;
   info.signal = SIGNAL_FLAT;
   info.pSell = pSell;
   info.pFlat = pFlat;
   info.pBuy = pBuy;
   info.bestDirectionProb = MathMax(pSell, pBuy);
   info.probGap = 0.0;

   double best = pFlat;
   double second = -1.0;
   SignalDirection signal = SIGNAL_FLAT;

   if(pBuy >= pSell && pBuy > best)
     {
      second = MathMax(best, pSell);
      best = pBuy;
      signal = SIGNAL_BUY;
     }
   else
      if(pSell > pBuy && pSell > best)
        {
         second = MathMax(best, pBuy);
         best = pSell;
         signal = SIGNAL_SELL;
        }
      else
        {
         second = MathMax(pBuy, pSell);
         signal = SIGNAL_FLAT;
        }

   info.bestDirectionProb = best;
   info.probGap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!InpAllowLong || pBuy < InpEntryProbThreshold ||
         info.probGap < InpMinProbGap)
        {
         LogInfo(
            StringFormat("BuildSignalInfo: BUY rejected allowLong=%d pBuy=%.4f "
                         "threshold=%.4f gap=%.4f minGap=%.4f",
                         (int)InpAllowLong, pBuy, InpEntryProbThreshold,
                         info.probGap, InpMinProbGap));
         info.signal = SIGNAL_FLAT;
        }
      else
         info.signal = SIGNAL_BUY;
      return info;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!InpAllowShort || pSell < InpEntryProbThreshold ||
         info.probGap < InpMinProbGap)
        {
         LogInfo(
            StringFormat("BuildSignalInfo: SELL rejected allowShort=%d "
                         "pSell=%.4f threshold=%.4f gap=%.4f minGap=%.4f",
                         (int)InpAllowShort, pSell, InpEntryProbThreshold,
                         info.probGap, InpMinProbGap));
         info.signal = SIGNAL_FLAT;
        }
      else
         info.signal = SIGNAL_SELL;
      return info;
     }

   info.signal = SIGNAL_FLAT;
   return info;
  }

//+------------------------------------------------------------------+
//| HasOpenPosition                                                  |
//+------------------------------------------------------------------+
bool HasOpenPosition(long &pos_type, double &pos_price)
  {
   if(!PositionSelect(_Symbol))
      return false;

   if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic)
      return false;

   pos_type = (long)PositionGetInteger(POSITION_TYPE);
   pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
   return true;
  }

//+------------------------------------------------------------------+
//| CloseOpenPosition                                                |
//+------------------------------------------------------------------+
void CloseOpenPosition()
  {
   if(PositionSelect(_Symbol) &&
      (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
     {
      if(!trade.PositionClose(_Symbol))
         LogInfo(
            StringFormat("CloseOpenPosition: PositionClose failed. retcode=%d",
                         trade.ResultRetcode()));
      else
         LogInfo("CloseOpenPosition: position closed.");
     }
  }

//+------------------------------------------------------------------+
//| ComputeLotSize                                                   |
//+------------------------------------------------------------------+
double ComputeLotSize(const SignalInfo &info)
  {
   double lots = InpBaseLots;
   if(!InpUseConfidenceSizing)
      return NormalizeVolumeToSymbol(lots);

   double strength_prob =
      MathMax(0.0, info.bestDirectionProb - InpEntryProbThreshold);
   double span_prob = MathMax(1e-8, 1.0 - InpEntryProbThreshold);
   double prob_score = MathMin(1.0, strength_prob / span_prob);

   double gap_score = 0.0;
   if(InpMinProbGap <= 0.0)
      gap_score = MathMin(1.0, info.probGap / 0.25);
   else
      gap_score = MathMin(1.0, MathMax(0.0, info.probGap - InpMinProbGap) /
                          MathMax(1e-8, 0.30 - InpMinProbGap));

   double blended = 0.70 * prob_score + 0.30 * gap_score;
   blended = MathMax(0.0, MathMin(1.0, blended));

   double mult = InpMinLotMultiplier +
                 (InpMaxLotMultiplier - InpMinLotMultiplier) * blended;
   lots *= mult;

   return NormalizeVolumeToSymbol(lots);
  }

//+------------------------------------------------------------------+
//| OpenTrade                                                        |
//+------------------------------------------------------------------+
void OpenTrade(const SignalInfo &info, double atr14_raw)
  {
   double lots = ComputeLotSize(info);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   double min_stop =
      (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sl_dist = MathMax(atr14_raw * InpStopAtrMultiple, min_stop);
   double tp_dist = MathMax(atr14_raw * InpTakeAtrMultiple, min_stop);

   double sl = 0.0;
   double tp = 0.0;

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   if(info.signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      if(trade.Buy(lots, _Symbol, ask, sl, tp, "Pro ensemble buy"))
        {
         g_bars_in_trade = 0;
         LogInfo(StringFormat("Opened BUY: lots=%.2f ask=%.5f sl=%.5f tp=%.5f",
                              lots, ask, sl, tp));
        }
      else
         LogInfo(StringFormat("BUY order failed. retcode=%d lots=%.2f",
                              trade.ResultRetcode(), lots));
     }
   else
      if(info.signal == SIGNAL_SELL)
        {
         if(InpUseAtrStops)
           {
            sl = bid + sl_dist;
            tp = bid - tp_dist;
           }
         if(trade.Sell(lots, _Symbol, bid, sl, tp, "Pro ensemble sell"))
           {
            g_bars_in_trade = 0;
            LogInfo(StringFormat("Opened SELL: lots=%.2f bid=%.5f sl=%.5f tp=%.5f",
                                 lots, bid, sl, tp));
           }
         else
            LogInfo(StringFormat("SELL order failed. retcode=%d lots=%.2f",
                                 trade.ResultRetcode(), lots));
        }
  }

//+------------------------------------------------------------------+
//| ManageExistingPosition                                           |
//+------------------------------------------------------------------+
void ManageExistingPosition(const SignalInfo &info)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;
   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY && info.signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && info.signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
     {
      should_close = true;
      LogDebug(
         StringFormat("ManageExistingPosition: max bars in trade reached (%d).",
                      g_bars_in_trade));
     }

   if(should_close)
     {
      LogInfo(StringFormat(
                 "ManageExistingPosition: closing existing position. bars_in_trade=%d",
                 g_bars_in_trade));
      CloseOpenPosition();
     }
  }

//+------------------------------------------------------------------+
//| ResetDailyLossStateIfNeeded                                      |
//+------------------------------------------------------------------+
void ResetDailyLossStateIfNeeded()
  {
   int today_key = DayKey(TimeCurrent());
   if(g_guard_day_key != today_key)
     {
      g_guard_day_key = today_key;
      g_guard_day_closed_pnl = 0.0;
      g_daily_loss_guard_active = false;
     }
  }

//+------------------------------------------------------------------+
//| RefreshClosedDealState                                           |
//+------------------------------------------------------------------+
void RefreshClosedDealState()
  {
   ResetDailyLossStateIfNeeded();

   if(!HistorySelect(0, TimeCurrent()))
     {
      LogInfo("RefreshClosedDealState: HistorySelect failed.");
      return;
     }

   int total = HistoryDealsTotal();
   if(total <= g_last_history_deals_total)
      return;

   for(int i = g_last_history_deals_total; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      long magic = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      long entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(symbol != _Symbol || magic != InpMagic || entry != DEAL_ENTRY_OUT)
         continue;

      double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double swap = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      double net = profit + swap + commission;

      datetime deal_time =
         (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      int deal_day_key = DayKey(deal_time);
      if(deal_day_key == g_guard_day_key)
         g_guard_day_closed_pnl += net;

      if(InpUseCooldownAfterClose)
        {
         g_cooldown_remaining = InpCooldownBars;
         LogInfo(StringFormat("Cooldown activated after close: %d bars.",
                              g_cooldown_remaining));
        }
     }

   g_last_history_deals_total = total;

   if(InpUseDailyLossGuard && !g_daily_loss_guard_active &&
      g_guard_day_closed_pnl <= -MathAbs(InpDailyLossLimitMoney))
     {
      g_daily_loss_guard_active = true;
      LogInfo(StringFormat("Daily loss guard activated: dayPnL=%.2f limit=%.2f",
                           g_guard_day_closed_pnl, InpDailyLossLimitMoney));
      if(InpDailyLossFlatOnTrigger)
         CloseOpenPosition();
     }
  }

//+------------------------------------------------------------------+
//| DecrementCooldown                                                |
//+------------------------------------------------------------------+
void DecrementCooldown()
  {
   if(g_cooldown_remaining > 0)
      g_cooldown_remaining--;
  }

//+------------------------------------------------------------------+
//| EntryGuardsAllow                                                 |
//+------------------------------------------------------------------+
bool EntryGuardsAllow(double atr14_raw)
  {
   if(InpUseDailyLossGuard && g_daily_loss_guard_active)
     {
      LogInfo(
         StringFormat("EntryGuardsAllow: blocked by daily loss guard (closed "
                      "pnl %.2f, limit %.2f).",
                      g_guard_day_closed_pnl, -MathAbs(InpDailyLossLimitMoney)));
      return false;
     }

   if(InpUseCooldownAfterClose && g_cooldown_remaining > 0)
     {
      LogInfo(StringFormat(
                 "EntryGuardsAllow: blocked by cooldown, bars remaining=%d.",
                 g_cooldown_remaining));
      return false;
     }

   if(InpUseHourFilter)
     {
      int h = CurrentServerHour();
      bool allowed_by_csv = IsHourAllowedByMasks(h);
      bool allowed_by_range = IsHourInRange(h, InpHourStart, InpHourEnd);
      bool has_csv_rules = (g_has_allowed_hours_csv || g_has_denied_hours_csv);

      bool allowed = (has_csv_rules ? allowed_by_csv : allowed_by_range);
      if(!allowed)
        {
         if(has_csv_rules)
            LogInfo(
               StringFormat("EntryGuardsAllow: blocked by hour CSV filter, "
                            "current=%d allowedCsv='%s' deniedCsv='%s'.",
                            h, InpAllowedHoursCsv, InpDeniedHoursCsv));
         else
            LogInfo(
               StringFormat("EntryGuardsAllow: blocked by hour range filter, "
                            "current=%d allowed=%d..%d.",
                            h, InpHourStart, InpHourEnd));
         return false;
        }
     }

   if(!SpreadAllows(atr14_raw))
     {
      LogInfo("EntryGuardsAllow: blocked by spread guard.");
      return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| GetTrend                                                         |
//+------------------------------------------------------------------+
SignalDirection GetTrend()
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(_Symbol, _Period, 0, 230, rates);
   if(copied < 220)
      return SIGNAL_FLAT;

   double closes[];
   ArrayResize(closes, copied);
   ArraySetAsSeries(closes, true);

   for(int i = 0; i < copied; i++)
      closes[i] = rates[i].close;

   int s = 1;
   double sma50 = Mean(closes, s, 50);
   double sma200 = Mean(closes, s, 200);

   if(sma50 > sma200)
      return SIGNAL_BUY;

   if(sma50 < sma200)
      return SIGNAL_SELL;

   return SIGNAL_FLAT;
  }

//+------------------------------------------------------------------+
//| GetResistance - shifted to exclude last bar (could be breakout)  |
//+------------------------------------------------------------------+
double GetResistance(int period=50)
  {
   int index = iHighest(_Symbol, _Period, MODE_HIGH, period, 1);
   return iHigh(_Symbol, _Period, index);
  }

//+------------------------------------------------------------------+
//| GetSupport - shifted to exclude last bar (could be breakout)     |
//+------------------------------------------------------------------+
double GetSupport(int period=50)
  {
   int index = iLowest(_Symbol, _Period, MODE_LOW, period, 1);
   return iLow(_Symbol, _Period, index);
  }

//+------------------------------------------------------------------+
//| GetATR                                                           |
//+------------------------------------------------------------------+
double GetATR()
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(_Symbol, _Period, 0, 30, rates);
   if(copied < 20)
      return 0.0;

   return CalcATR(rates, 1, 14);
  }

//+------------------------------------------------------------------+
//| IsBreakoutBuy                                                    |
//+------------------------------------------------------------------+
bool IsBreakoutBuy(double resistance)
  {
   double close0 = iClose(_Symbol, _Period, 0);
   return close0 > resistance;
  }

//+------------------------------------------------------------------+
//| IsBreakoutSell                                                   |
//+------------------------------------------------------------------+
bool IsBreakoutSell(double support)
  {
   double close0 = iClose(_Symbol, _Period, 0);
   return close0 < support;
  }

//+------------------------------------------------------------------+
//| IsStrongBreakoutBuy                                              |
//+------------------------------------------------------------------+
bool IsStrongBreakoutBuy(double atr, double resistance)
  {
   double close0 = iClose(_Symbol, _Period, 0);
   return close0 > resistance + 0.5 * atr;
  }

//+------------------------------------------------------------------+
//| IsStrongBreakoutSell                                             |
//+------------------------------------------------------------------+
bool IsStrongBreakoutSell(double atr, double support)
  {
   double close0 = iClose(_Symbol, _Period, 0);
   return close0 < support - 0.5 * atr;
  }

//+------------------------------------------------------------------+
//| InitSingleModel                                                  |
//+------------------------------------------------------------------+
bool InitSingleModel(long &handle_ref, const uchar &buffer[])
  {
   handle_ref = OnnxCreateFromBuffer(buffer, ONNX_DEFAULT);
   if(handle_ref == INVALID_HANDLE)
     {
      LogInfo(
         "InitSingleModel failed: OnnxCreateFromBuffer returned "
         "INVALID_HANDLE.");
      return false;
     }

   if(!OnnxSetInputShape(handle_ref, 0, EXT_INPUT_SHAPE))
     {
      LogInfo("InitSingleModel failed: OnnxSetInputShape failed.");
      return false;
     }

   if(!OnnxSetOutputShape(handle_ref, 0, EXT_LABEL_SHAPE))
     {
      LogInfo("InitSingleModel failed: OnnxSetOutputShape label failed.");
      return false;
     }

   if(!OnnxSetOutputShape(handle_ref, 1, EXT_PROBA_SHAPE))
     {
      LogInfo("InitSingleModel failed: OnnxSetOutputShape probabilities failed.");
      return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   LogInfo("OnInit started.");

   if(!NormalizeWeights())
      return INIT_PARAMETERS_INCORRECT;

   if(!ParseHoursCsv(InpAllowedHoursCsv, g_allowed_hours_mask,
                     g_has_allowed_hours_csv, "InpAllowedHoursCsv"))
      return INIT_PARAMETERS_INCORRECT;

   if(!ParseHoursCsv(InpDeniedHoursCsv, g_denied_hours_mask,
                     g_has_denied_hours_csv, "InpDeniedHoursCsv"))
      return INIT_PARAMETERS_INCORRECT;

   if(!InitSingleModel(g_mlp_handle, MlpModel))
      return INIT_FAILED;

   if(!InitSingleModel(g_lgbm_handle, LgbmModel))
      return INIT_FAILED;

   if(!InitSingleModel(g_hgb_handle, HgbModel))
      return INIT_FAILED;

   if(!InitSingleModel(g_extratrees_handle, ExtraTreesModel))
      return INIT_FAILED;

   if(!InitSingleModel(g_ridge_handle, RidgeModel))
      return INIT_FAILED;

   if(!InitSingleModel(g_naivebayes_handle, NaiveBayesModel))
      return INIT_FAILED;

   ResetDailyLossStateIfNeeded();

   if(HistorySelect(0, TimeCurrent()))
      g_last_history_deals_total = HistoryDealsTotal();
   else
      g_last_history_deals_total = 0;

   LogInfo(StringFormat(
              "OnInit complete. weights=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f) historyDeals=%d",
              g_w_mlp, g_w_lgbm, g_w_hgb, g_w_extratrees, g_w_ridge, g_w_naivebayes,
              g_last_history_deals_total));

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_mlp_handle != INVALID_HANDLE)
      OnnxRelease(g_mlp_handle);
   if(g_lgbm_handle != INVALID_HANDLE)
      OnnxRelease(g_lgbm_handle);
   if(g_hgb_handle != INVALID_HANDLE)
      OnnxRelease(g_hgb_handle);
   if(g_extratrees_handle != INVALID_HANDLE)
      OnnxRelease(g_extratrees_handle);
   if(g_ridge_handle      != INVALID_HANDLE)
      OnnxRelease(g_ridge_handle);
   if(g_naivebayes_handle != INVALID_HANDLE)
      OnnxRelease(g_naivebayes_handle);

   g_mlp_handle = INVALID_HANDLE;
   g_lgbm_handle = INVALID_HANDLE;
   g_hgb_handle = INVALID_HANDLE;
   g_extratrees_handle = INVALID_HANDLE;
   g_ridge_handle      = INVALID_HANDLE;
   g_naivebayes_handle = INVALID_HANDLE;
  }

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar())
      return;

   RefreshClosedDealState();
   DecrementCooldown();

   double pSell = 0.0;
   double pFlat = 0.0;
   double pBuy = 0.0;
   double atr14_raw = 0.0;

   if(!PredictEnsembleProbabilities(pSell, pFlat, pBuy, atr14_raw))
      return;

   LogDebug(
      StringFormat("OnTick: raw probs sell=%.4f flat=%.4f buy=%.4f atr=%.5f",
                   pSell, pFlat, pBuy, atr14_raw));

   ApplyHourSoftBias(pSell, pBuy);
   LogDebug(StringFormat("OnTick: post-bias probs sell=%.4f flat=%.4f buy=%.4f",
                         pSell, pFlat, pBuy));

   SignalInfo info = BuildSignalInfo(pSell, pFlat, pBuy);
   LogDebug(StringFormat("Signal evaluated: Probabilities signal=%d pSell=%.5f pFlat=%.5f pBuy=%.5f gap=%.5f entry_prob = %.5f min_gap = %.5f",
                         info.signal, info.pSell, info.pFlat, info.pBuy, info.probGap, InpEntryProbThreshold, InpMinProbGap));

   ManageExistingPosition(info);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
     {
      LogDebug("OnTick: skipping entry, position already open.");
      return;
     }

   if(info.signal == SIGNAL_FLAT)
     {
      LogDebug("OnTick: no entry, signal is FLAT after thresholds/filters.");
      return;
     }

   if(!EntryGuardsAllow(atr14_raw))
     {
      LogDebug("OnTick: no entry, entry guards stop.");
      return;
     }

   if(InpUseTrend)
     {
      int trend = GetTrend();

      if(info.signal == SIGNAL_BUY && trend != SIGNAL_BUY)
        {
         LogDebug("OnTick: skipping entry, trend not uptrend.");
         return;
        }

      if(info.signal == SIGNAL_SELL && trend != SIGNAL_SELL)
        {
         LogDebug("OnTick: skipping entry, trend not downtrend.");
         return;
        }
     }

   double atr = GetATR();

   if(InpUseSupportAndResistance)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      double resistance = GetResistance(50);
      double support = GetSupport(50);
      bool isBuyBreakout = InpUseBreakout && IsBreakoutBuy(resistance);
      bool isSellBreakout = InpUseBreakout && IsBreakoutSell(support);
      bool isBuyStrongBreakout = InpUseStrongBreakout && IsStrongBreakoutBuy(atr, resistance);
      bool isSellStrongBreakout = InpUseStrongBreakout && IsStrongBreakoutSell(atr, support);

      // block BUY near resistance (except BUY breakout & strong BUY breakout)
      if(info.signal == SIGNAL_BUY && price > resistance * (1 - InpSupportAndResistanceBuffer) && !isBuyBreakout && !isBuyStrongBreakout)
        {
         LogDebug("OnTick: skipping entry, skipping buy near resistance.");
         return;
        }

      // block SELL near support (except SELL breakout & strong SELL breakout)
      if(info.signal == SIGNAL_SELL && price < support * (1 + InpSupportAndResistanceBuffer) && !isSellBreakout && !isSellStrongBreakout)
        {
         LogDebug("OnTick: skipping entry, skipping sell near support.");
         return;
        }
     }

   if(InpUseVolumeFilter)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double volumeRatio = atr/price;

      if(InpFilterMinVolume && volumeRatio < InpMinVolume)
        {
         LogDebug("OnTick: skipping entry, market too quiet (volume too low).");
         return;
        }

      if(InpFilterMaxVolume && volumeRatio > InpMaxVolume)
        {
         LogDebug("OnTick: skipping entry, market too loud (volume too high).");
         return;
        }
     }

   LogInfo(
      StringFormat("OnTick: opening trade for signal=%d.", (int)info.signal));
   OpenTrade(info, atr14_raw);
  }

//+------------------------------------------------------------------+
//| OnTester                                                         |
//+------------------------------------------------------------------+
double OnTester()
  {
   double profit = TesterStatistics(STAT_PROFIT);
   double pf = TesterStatistics(STAT_PROFIT_FACTOR);
   double recovery = TesterStatistics(STAT_RECOVERY_FACTOR);
   double dd_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   double trades = TesterStatistics(STAT_TRADES);

// Penalty if there are too few transactions
   double trade_penalty = 1.0;
   if(trades < 20)
      trade_penalty = 0.25;
   else
      if(trades < 50)
         trade_penalty = 0.60;

// Robust score, not only brut profit
   double score = 0.0;

   if(dd_percent >= 0.0)
      score =
         (profit * MathMax(pf, 0.01) * MathMax(recovery, 0.01) * trade_penalty) /
         (1.0 + dd_percent);

   return score;
  }
//+------------------------------------------------------------------+
