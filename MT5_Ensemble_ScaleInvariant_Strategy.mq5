//+------------------------------------------------------------------+
//| Ensemble Scale-Invariant Strategy                 Copyright 2026 |
//+------------------------------------------------------------------+
#property strict
#property version "1.41"
#property description "MT5 EA: scale-invariant ONNX ensemble (MLP + LightGBM + HGB + ExtraTrees + Ridge + NaiveBayes)"

#include <Trade/Trade.mqh>

#resource "mlp.onnx" as uchar MlpModel[]
#resource "lightgbm.onnx" as uchar LgbmModel[]
#resource "hgb.onnx" as uchar HgbModel[]
#resource "extratrees.onnx" as uchar ExtraTreesModel[]
#resource "ridge.onnx"      as uchar RidgeModel[]
#resource "naivebayes.onnx" as uchar NaiveBayesModel[]

input double InpLots = 0.10;
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

input long InpMagic = 26042026;
input bool InpLog = false;
input bool InpDebugLog = false;

const int FEATURE_COUNT = 17;
const int CLASS_COUNT = 3;
const long EXT_INPUT_SHAPE[] = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[] = {1};
const long EXT_PROBA_SHAPE[] = {1, CLASS_COUNT};

CTrade trade;
long g_mlp_handle = INVALID_HANDLE;
long g_lgbm_handle = INVALID_HANDLE;
long g_hgb_handle = INVALID_HANDLE;
long g_extratrees_handle = INVALID_HANDLE;
long g_ridge_handle      = INVALID_HANDLE;
long g_naivebayes_handle = INVALID_HANDLE;

datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;

double g_w_mlp = 0.0;
double g_w_lgbm = 0.0;
double g_w_hgb = 0.0;
double g_w_extratrees = 0.0;
double g_w_ridge      = 0.0;
double g_w_naivebayes = 0.0;

enum SignalDirection { SIGNAL_SELL = -1, SIGNAL_FLAT = 0, SIGNAL_BUY = 1 };

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
      LogInfo("NormalizeWeights failed: sum of ensemble weights is <= 0.");
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
//| IsNewBar                                                         |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == 0)
      return false;
   if(g_last_bar_time == 0)
     {
      g_last_bar_time = current_bar_time;
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
      LogInfo("RunSingleModel failed: OnnxRun returned false.");
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
      LogInfo("RunRidgeModel failed: OnnxRun returned false.");
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
      LogInfo(
         "PredictEnsembleProbabilities aborted: feature vector build failed.");
      return false;
     }

   double s1, f1, b1; // MLP
   double s2, f2, b2; // LightGBM
   double s3, f3, b3; // HGB
   double s4, f4, b4; // ExtraTrees
   double s5, f5, b5; // Ridge
   double s6, f6, b6; // NaiveBayes

// Run individual models.  Stop early on failure to conserve resources.
   if(!RunSingleModel(g_mlp_handle, x, s1, f1, b1))
      return false;
   if(!RunSingleModel(g_lgbm_handle, x, s2, f2, b2))
      return false;
   if(!RunSingleModel(g_hgb_handle, x, s3, f3, b3))
      return false;
   if(!RunSingleModel(g_extratrees_handle, x, s4, f4, b4))
      return false;
   if(!RunRidgeModel(g_ridge_handle, x, s5, f5, b5))
      return false;
   if(!RunSingleModel(g_naivebayes_handle, x, s6, f6, b6))
      return false;

// Combine probabilities using normalized weights.
   pSell = g_w_mlp * s1 + g_w_lgbm * s2 + g_w_hgb * s3 + g_w_extratrees * s4 + g_w_ridge * s5 + g_w_naivebayes * s6;
   pFlat = g_w_mlp * f1 + g_w_lgbm * f2 + g_w_hgb * f3 + g_w_extratrees * f4 + g_w_ridge * f5 + g_w_naivebayes * f6;
   pBuy  = g_w_mlp * b1 + g_w_lgbm * b2 + g_w_hgb * b3 + g_w_extratrees * b4 + g_w_ridge * b5 + g_w_naivebayes * b6;
   LogDebug(StringFormat("Ensemble probabilities: pSell=%.5f pFlat=%.5f pBuy=%.5f",
                         pSell, pFlat, pBuy));
   return true;
  }

//+------------------------------------------------------------------+
//| SignalFromProbabilities                                          |
//+------------------------------------------------------------------+
SignalDirection SignalFromProbabilities(double pSell, double pFlat,
                                        double pBuy)
  {
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

   double gap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!InpAllowLong)
        {
         LogInfo("Signal BUY filtered to FLAT: long entries disabled.");
         return SIGNAL_FLAT;
        }
      if(pBuy < InpEntryProbThreshold || gap < InpMinProbGap)
        {
         LogInfo(
            StringFormat("Signal BUY filtered to FLAT: pBuy=%.5f threshold=%.5f "
                         "gap=%.5f minGap=%.5f",
                         pBuy, InpEntryProbThreshold, gap, InpMinProbGap));
         return SIGNAL_FLAT;
        }
      return SIGNAL_BUY;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!InpAllowShort)
        {
         LogInfo("Signal SELL filtered to FLAT: short entries disabled.");
         return SIGNAL_FLAT;
        }
      if(pSell < InpEntryProbThreshold || gap < InpMinProbGap)
        {
         LogInfo(
            StringFormat("Signal SELL filtered to FLAT: pSell=%.5f "
                         "threshold=%.5f gap=%.5f minGap=%.5f",
                         pSell, InpEntryProbThreshold, gap, InpMinProbGap));
         return SIGNAL_FLAT;
        }
      return SIGNAL_SELL;
     }

   return SIGNAL_FLAT;
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
      if(trade.PositionClose(_Symbol))
         LogInfo("Position close request succeeded.");
      else
         LogInfo(StringFormat("Position close request failed. retcode=%d",
                              trade.ResultRetcode()));
     }
  }

//+------------------------------------------------------------------+
//| OpenTrade                                                        |
//+------------------------------------------------------------------+
void OpenTrade(SignalDirection signal, double atr14_raw)
  {
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

   if(signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      if(trade.Buy(InpLots, _Symbol, ask, sl, tp,
                   "ScaleInvariant ensemble buy"))
        {
         g_bars_in_trade = 0;
         LogInfo(StringFormat("Opened BUY: lots=%.2f ask=%.5f sl=%.5f tp=%.5f",
                              InpLots, ask, sl, tp));
        }
      else
         LogInfo(StringFormat("BUY order failed. retcode=%d lots=%.2f",
                              trade.ResultRetcode(), InpLots));
     }
   else
      if(signal == SIGNAL_SELL)
        {
         if(InpUseAtrStops)
           {
            sl = bid + sl_dist;
            tp = bid - tp_dist;
           }
         if(trade.Sell(InpLots, _Symbol, bid, sl, tp,
                       "ScaleInvariant ensemble sell"))
           {
            g_bars_in_trade = 0;
            LogInfo(StringFormat("Opened SELL: lots=%.2f bid=%.5f sl=%.5f tp=%.5f",
                                 InpLots, bid, sl, tp));
           }
         else
            LogInfo(StringFormat("SELL order failed. retcode=%d lots=%.2f",
                                 trade.ResultRetcode(), InpLots));
        }
  }

//+------------------------------------------------------------------+
//| ManageExistingPosition                                           |
//+------------------------------------------------------------------+
void ManageExistingPosition(SignalDirection signal)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;
   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY && signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
      should_close = true;
   if(should_close)
     {
      LogInfo(StringFormat(
                 "ManageExistingPosition: closing existing position. bars_in_trade=%d",
                 g_bars_in_trade));
      CloseOpenPosition();
     }
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
   LogInfo(StringFormat("OnInit complete. weights=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)",
                        g_w_mlp, g_w_lgbm, g_w_hgb,
                        g_w_extratrees, g_w_ridge, g_w_naivebayes));
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

   double pSell = 0.0, pFlat = 0.0, pBuy = 0.0, atr14_raw = 0.0;
   if(!PredictEnsembleProbabilities(pSell, pFlat, pBuy, atr14_raw))
      return;

   SignalDirection signal = SignalFromProbabilities(pSell, pFlat, pBuy);
   LogInfo(StringFormat(
              "Signal evaluated: signal=%d pSell=%.5f pFlat=%.5f pBuy=%.5f", signal,
              pSell, pFlat, pBuy));
   ManageExistingPosition(signal);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
     {
      LogInfo("No new entry: existing position is already open.");
      return;
     }

   if(signal == SIGNAL_BUY || signal == SIGNAL_SELL)
      OpenTrade(signal, atr14_raw);
   else
      LogInfo("No entry: signal is FLAT.");
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
