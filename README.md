# MT5 Scale-Invariant Ensemble Package

This package rewrites the original GitHub-ready ensemble so that Python training and MT5 live trading use the exact same normalized feature engineering.

## Why this rewrite exists

The earlier ensemble could become fragile when price levels changed a lot, especially across instruments such as silver vs crypto.
This package makes the feature set more scale-invariant by replacing raw-price-sensitive inputs with relative features.

## Models
The ensemble uses:
- MLP – a multilayer perceptron with two hidden layers, trained on standardized features.
- LightGBM – a gradient boosting tree model that handles complex non‑linear patterns.
- HistGradientBoosting – a histogram‑based gradient boosting classifier optimized for speed.
- ExtraTrees – an ensemble of extremely randomized trees offering robust performance and diversity.
- Ridge – a linear classifier with L2 regularization whose decision scores are converted to probabilities via softmax.
- Naive Bayes – Gaussian Naive Bayes classifier with standardized inputs.

Each model is exported to ONNX and loaded by the MT5 EA.

## Features used by both Python and MT5

The package uses 13 features:

1. ret_1
2. ret_3
3. ret_5
4. ret_10
5. vol_10
6. vol_20
7. vol_ratio_10_20
8. dist_sma_10
9. dist_sma_20
10. zscore_20
11. atr_pct_14
12. range_pct_1
13. body_pct_1

### Feature definitions

- ret_n = close_t / close_t-n - 1
- vol_10 = stddev(ret_1 over 10 bars)
- vol_20 = stddev(ret_1 over 20 bars)
- vol_ratio_10_20 = vol_10 / vol_20 - 1
- dist_sma_10 = close / sma_10 - 1
- dist_sma_20 = close / sma_20 - 1
- zscore_20 = (close - mean_20) / std_20
- atr_pct_14 = ATR(14) / close
- range_pct_1 = (high - low) / close
- body_pct_1 = (close - open) / open

These are all relative or standardized features, which makes the model more robust when price levels move into ranges never seen before.

### Python Prerequisites

```text
python -m venv .venv
source .venv/bin/activate
pip install MetaTrader5 pandas numpy scikit-learn lightgbm skl2onnx onnxmltools onnx
```

### Training of ONNX models (at once)

```text
train_mt5_ensemble_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.82 --output-dir output_ensemble_XAGUSD_M15_h8_82_scale_invariant
```
