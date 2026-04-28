from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx.helper as onnx_helper
import pandas as pd
from lightgbm import LGBMClassifier
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxToolsFloatTensorType
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


SELL_CLASS = -1
FLAT_CLASS = 0
BUY_CLASS = 1
CLASS_ORDER = [SELL_CLASS, FLAT_CLASS, BUY_CLASS]
CLASS_TO_ENC = {SELL_CLASS: 0, FLAT_CLASS: 1, BUY_CLASS: 2}
ENC_TO_CLASS = {v: k for k, v in CLASS_TO_ENC.items()}

FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "vol_ratio_10_20",
    "dist_sma_10",
    "dist_sma_20",
    "zscore_20",
    "atr_pct_14",
    "range_pct_1",
    "body_pct_1",
]


def _coerce_bool_attributes_for_onnx():
    original_make_attribute = onnx_helper.make_attribute

    def patched_make_attribute(key, value, *args, **kwargs):
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        elif isinstance(value, (list, tuple)):
            value = [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
        elif isinstance(value, np.ndarray) and value.dtype == np.bool_:
            value = value.astype(np.int64)
        return original_make_attribute(key, value, *args, **kwargs)

    return original_make_attribute, patched_make_attribute


def fetch_rates_from_mt5(symbol: str, timeframe_name: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 Python package is not installed.")

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe_name not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe_name}")

    if not mt5.initialize():
        raise RuntimeError(f"initialize() failed: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe_name], 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Could not read data for {symbol} {timeframe_name}. last_error={mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["time", "open", "high", "low", "close", "volume"]].copy()
    finally:
        mt5.shutdown()


def load_rates_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"time", "open", "high", "low", "close"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    return df[["time", "open", "high", "low", "close", "volume"]]


def build_features(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    eps = 1e-12

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_ratio_10_20"] = (df["vol_10"] / (df["vol_20"] + eps)) - 1.0

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["dist_sma_10"] = (df["close"] / (df["sma_10"] + eps)) - 1.0
    df["dist_sma_20"] = (df["close"] / (df["sma_20"] + eps)) - 1.0

    roll_mean_20 = df["close"].rolling(20).mean()
    roll_std_20 = df["close"].rolling(20).std()
    df["zscore_20"] = (df["close"] - roll_mean_20) / (roll_std_20 + eps)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()
    df["atr_pct_14"] = df["atr_14"] / (df["close"] + eps)

    df["range_pct_1"] = (df["high"] - df["low"]) / (df["close"] + eps)
    df["body_pct_1"] = (df["close"] - df["open"]) / (df["open"] + eps)

    df["fwd_ret_h"] = df["close"].shift(-horizon_bars) / (df["close"] + eps) - 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["fwd_ret_h"]).copy()
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if len(train_df) < 1500 or len(test_df) < 250:
        raise ValueError("Too few rows after split.")
    return train_df, test_df


def compute_return_barrier(train_df: pd.DataFrame, label_quantile: float) -> float:
    barrier = float(train_df["fwd_ret_h"].abs().quantile(label_quantile))
    return max(barrier, 1e-6)


def label_targets(df: pd.DataFrame, return_barrier: float) -> pd.DataFrame:
    out = df.copy()
    out["target_class"] = FLAT_CLASS
    out.loc[out["fwd_ret_h"] > return_barrier, "target_class"] = BUY_CLASS
    out.loc[out["fwd_ret_h"] < -return_barrier, "target_class"] = SELL_CLASS
    out["target_class"] = out["target_class"].astype(np.int64)
    out["target_class_enc"] = out["target_class"].map(CLASS_TO_ENC).astype(np.int64)
    return out


def make_mlp(random_state: int = 42) -> Pipeline:
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=False,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def make_lgbm(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=350,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_hgb(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_iter=400,
        max_leaf_nodes=31,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=random_state,
    )


# -----------------------------------------------------------------------------
# Additional model factories for Extra Trees and Ridge classifiers
#
# The ensemble originally included MLP, LightGBM and HGB models.  This patch
# adds support for Extra Trees and Ridge classifiers to broaden the ensemble.
# Both of these models are trained on the same set of 13 scale‑invariant
# features defined in FEATURE_COLS.  Hyperparameters are borrowed from the
# standalone training scripts for Extra Trees and Ridge, ensuring comparable
# behaviour when ported to MT5 via ONNX.

def make_extratrees(random_state: int = 42) -> ExtraTreesClassifier:
    """
    Construct an ExtraTreesClassifier with sensible defaults.

    The parameters mirror those used in the separate Extra Trees training
    script.  In particular, a fairly large forest (600 trees) is built with
    moderate depth and leaf constraints to reduce overfitting.  Feature
    subsampling is set to "sqrt" to encourage diversity across trees.
    """
    return ExtraTreesClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=12,
        min_samples_split=40,
        max_features="sqrt",
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )


def make_ridge(random_state: int = 42) -> Pipeline:
    """
    Build a pipeline of StandardScaler followed by a RidgeClassifier.

    The Ridge classifier does not natively expose probability estimates, but
    decision function scores can be converted to probabilities via the softmax
    transformation when forming the ensemble.  Class weights are balanced to
    account for class imbalance in the training data.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ridge",
                RidgeClassifier(
                    alpha=1.0,
                    class_weight="balanced",
                    fit_intercept=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_models(train_df: pd.DataFrame, random_state: int = 42):
    """
    Train all constituent models on the supplied training data.

    Each model is fitted on the 13‑dimensional feature space defined by
    FEATURE_COLS.  Categorical targets are provided as encoded integers
    (0=sell, 1=flat, 2=buy) via the "target_class_enc" column.  Models
    requiring specific input formats (e.g. LightGBM expecting a DataFrame)
    are handled accordingly.
    """
    X_df = train_df[FEATURE_COLS].astype(np.float32)
    X_np = X_df.to_numpy(dtype=np.float32)
    y = train_df["target_class_enc"].to_numpy(dtype=np.int64)

    # MLP
    mlp = make_mlp(random_state)
    mlp.fit(X_np, y)

    # LightGBM
    lgbm = make_lgbm(random_state)
    lgbm.fit(X_df, y)

    # HistGradientBoosting
    hgb = make_hgb(random_state)
    hgb.fit(X_np, y)

    # Extra Trees
    extratrees = make_extratrees(random_state)
    extratrees.fit(X_np, y)

    # Ridge (with StandardScaler)
    ridge = make_ridge(random_state)
    ridge.fit(X_np, y)

    return {
        "mlp": mlp,
        "lgbm": lgbm,
        "hgb": hgb,
        "extratrees": extratrees,
        "ridge": ridge,
    }


def normalize_weights(
    mlp_weight: float,
    lgbm_weight: float,
    hgb_weight: float,
    extratrees_weight: float,
    ridge_weight: float,
) -> Dict[str, float]:
    """
    Normalize ensemble weights across all models.

    The ensemble may contain up to five model types.  All weights are
    coerced to be non‑negative.  At least one weight must be positive, else
    a ValueError is raised.  Returned weights sum to one across the set
    {mlp, lgbm, hgb, extratrees, ridge}.
    """
    raw = {
        "mlp": float(max(0.0, mlp_weight)),
        "lgbm": float(max(0.0, lgbm_weight)),
        "hgb": float(max(0.0, hgb_weight)),
        "extratrees": float(max(0.0, extratrees_weight)),
        "ridge": float(max(0.0, ridge_weight)),
    }
    s = sum(raw.values())
    if s <= 0.0:
        raise ValueError("At least one weight must be > 0.")
    return {k: v / s for k, v in raw.items()}


def weighted_probabilities(models, X: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    """
    Compute a weighted sum of class probabilities across all models.

    For models exposing ``predict_proba`` (e.g. MLP, LightGBM, HGB and
    Extra Trees) probabilities are used directly.  For models lacking
    ``predict_proba`` but exposing ``decision_function`` (e.g. the Ridge
    classifier pipeline), softmax is applied to the decision scores to
    generate probability distributions.  Each model's class order is
    aligned to the global class constants SELL_CLASS, FLAT_CLASS and
    BUY_CLASS via ENC_TO_CLASS before weighting.
    """
    out = np.zeros((len(X), 3), dtype=np.float64)
    # For LightGBM we need a DataFrame; other models work with numpy arrays
    X_df = pd.DataFrame(X, columns=FEATURE_COLS, dtype=np.float32)
    for name, model in models.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue

        # Determine how to obtain probabilities
        if isinstance(model, LGBMClassifier):
            # LightGBM expects a DataFrame for predict_proba
            p = model.predict_proba(X_df)
            classes = model.classes_
        elif hasattr(model, "predict_proba"):
            # Pipelines (e.g. MLP) and ExtraTrees provide predict_proba directly
            p = model.predict_proba(X)
            if hasattr(model, "classes_"):
                classes = model.classes_
            elif hasattr(model, "named_steps") and "mlp" in model.named_steps:
                classes = model.named_steps["mlp"].classes_
            else:
                raise ValueError(f"Cannot determine classes for model {name}")
        elif hasattr(model, "decision_function"):
            # RidgeClassifier pipeline exposes decision_function; convert to probabilities
            scores = model.decision_function(X)
            if scores.ndim == 1 or scores.shape[1] != len(CLASS_ORDER):
                raise ValueError(
                    f"decision_function returned unexpected shape for model {name}: {scores.shape}"
                )
            # Apply row‑wise softmax to scores
            scores_max = scores.max(axis=1, keepdims=True)
            exps = np.exp(scores - scores_max)
            p = exps / exps.sum(axis=1, keepdims=True)
            # Extract classes from the underlying estimator
            if hasattr(model, "classes_"):
                classes = model.classes_
            elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
                classes = model.named_steps["ridge"].classes_
            else:
                raise ValueError(f"Cannot determine classes for model {name}")
        else:
            raise ValueError(f"Model {name} does not support predict_proba or decision_function")

        # Align classes to SELL/FLAT/BUY order using ENC_TO_CLASS
        class_to_idx = {ENC_TO_CLASS[int(c)]: i for i, c in enumerate(classes)}
        aligned = np.column_stack(
            [
                p[:, class_to_idx[SELL_CLASS]],
                p[:, class_to_idx[FLAT_CLASS]],
                p[:, class_to_idx[BUY_CLASS]],
            ]
        )
        out += weight * aligned

    return out


def derive_decision_thresholds(models, train_df: pd.DataFrame, weights: Dict[str, float], prob_quantile: float, margin_quantile: float):
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    proba = weighted_probabilities(models, X_train, weights)

    p_sell = proba[:, 0]
    p_flat = proba[:, 1]
    p_buy = proba[:, 2]

    best_direction_prob = np.maximum(p_buy, p_sell)
    direction_is_buy = p_buy >= p_sell
    opposite_prob = np.where(direction_is_buy, p_sell, p_buy)
    best_vs_next = best_direction_prob - np.maximum(p_flat, opposite_prob)

    candidates = best_direction_prob > p_flat
    if candidates.any():
        entry_prob_threshold = float(np.quantile(best_direction_prob[candidates], prob_quantile))
        min_prob_gap = float(np.quantile(best_vs_next[candidates], margin_quantile))
    else:
        entry_prob_threshold, min_prob_gap = 0.55, 0.05

    entry_prob_threshold = float(np.clip(entry_prob_threshold, 0.34, 0.95))
    min_prob_gap = float(np.clip(min_prob_gap, 0.00, 0.50))
    return entry_prob_threshold, min_prob_gap


def classify_with_thresholds(models, df: pd.DataFrame, weights: Dict[str, float], entry_prob_threshold: float, min_prob_gap: float) -> pd.DataFrame:
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    proba = weighted_probabilities(models, X, weights)

    p_sell = proba[:, 0]
    p_flat = proba[:, 1]
    p_buy = proba[:, 2]

    best_direction_prob = np.maximum(p_buy, p_sell)
    direction = np.where(p_buy >= p_sell, BUY_CLASS, SELL_CLASS)
    second_best = np.maximum(p_flat, np.where(direction == BUY_CLASS, p_sell, p_buy))
    prob_gap = best_direction_prob - second_best

    pred = np.full(len(df), FLAT_CLASS, dtype=np.int32)
    take_trade = (
        (best_direction_prob >= entry_prob_threshold)
        & (prob_gap >= min_prob_gap)
        & (best_direction_prob > p_flat)
    )
    pred[take_trade] = direction[take_trade]

    out = df.copy()
    out["p_sell"] = p_sell
    out["p_flat"] = p_flat
    out["p_buy"] = p_buy
    out["best_direction_prob"] = best_direction_prob
    out["prob_gap"] = prob_gap
    out["pred_class"] = pred
    out["trade_taken"] = out["pred_class"] != FLAT_CLASS
    out["direction_correct"] = (
        ((out["pred_class"] == BUY_CLASS) & (out["target_class"] == BUY_CLASS))
        | ((out["pred_class"] == SELL_CLASS) & (out["target_class"] == SELL_CLASS))
    )
    return out


def summarize_predictions(pred_df: pd.DataFrame) -> Dict[str, float]:
    y_true = pred_df["target_class"].to_numpy()
    y_pred = pred_df["pred_class"].to_numpy()
    trade_mask = pred_df["trade_taken"].to_numpy()

    if trade_mask.any():
        directional_precision = float(pred_df.loc[trade_mask, "direction_correct"].mean())
        signed_trade_ret = np.where(
            pred_df.loc[trade_mask, "pred_class"].to_numpy() == BUY_CLASS,
            pred_df.loc[trade_mask, "fwd_ret_h"].to_numpy(),
            -pred_df.loc[trade_mask, "fwd_ret_h"].to_numpy(),
        )
        mean_trade_return = float(signed_trade_ret.mean())
        accepted_trades = int(trade_mask.sum())
        accepted_rate = float(trade_mask.mean())

        gross_profit = float(signed_trade_ret[signed_trade_ret > 0].sum()) if np.any(signed_trade_ret > 0) else 0.0
        gross_loss = float(-signed_trade_ret[signed_trade_ret < 0].sum()) if np.any(signed_trade_ret < 0) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    else:
        directional_precision = 0.0
        mean_trade_return = 0.0
        accepted_trades = 0
        accepted_rate = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = 0.0

    return {
        "rows": int(len(pred_df)),
        "accepted_trades": accepted_trades,
        "accepted_rate": accepted_rate,
        "directional_precision_on_trades": directional_precision,
        "mean_signed_fwd_return_on_trades": mean_trade_return,
        "gross_profit_return_units": gross_profit,
        "gross_loss_return_units": gross_loss,
        "profit_factor_return_units": profit_factor,
        "ternary_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix_sell_flat_buy": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }


def walk_forward_report(train_df: pd.DataFrame, weights: Dict[str, float], label_quantile: float, prob_quantile: float, margin_quantile: float, n_splits: int) -> Dict[str, float]:
    X = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accepted_rates: List[float] = []
    directional_precisions: List[float] = []
    mean_trade_returns: List[float] = []
    ternary_accs: List[float] = []
    bal_accs: List[float] = []
    entry_thresholds: List[float] = []
    gap_thresholds: List[float] = []

    for fold, (fit_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        fit_df = train_df.iloc[fit_idx].copy()
        valid_df = train_df.iloc[valid_idx].copy()

        barrier = compute_return_barrier(fit_df, label_quantile)
        fit_df = label_targets(fit_df, barrier)
        valid_df = label_targets(valid_df, barrier)

        models = fit_models(fit_df, random_state=42 + fold)
        entry_prob_threshold, min_prob_gap = derive_decision_thresholds(models, fit_df, weights, prob_quantile, margin_quantile)
        pred_df = classify_with_thresholds(models, valid_df, weights, entry_prob_threshold, min_prob_gap)
        summary = summarize_predictions(pred_df)

        accepted_rates.append(summary["accepted_rate"])
        directional_precisions.append(summary["directional_precision_on_trades"])
        mean_trade_returns.append(summary["mean_signed_fwd_return_on_trades"])
        ternary_accs.append(summary["ternary_accuracy"])
        bal_accs.append(summary["balanced_accuracy"])
        entry_thresholds.append(entry_prob_threshold)
        gap_thresholds.append(min_prob_gap)

        print(
            f"Fold {fold}: barrier={barrier:.6f} entry_prob={entry_prob_threshold:.4f} "
            f"gap={min_prob_gap:.4f} accepted_rate={summary['accepted_rate']:.3f} "
            f"precision={summary['directional_precision_on_trades']:.3f} "
            f"mean_trade_ret={summary['mean_signed_fwd_return_on_trades']:.6f} "
            f"bal_acc={summary['balanced_accuracy']:.3f}"
        )

    return {
        "accepted_rate_mean": float(np.mean(accepted_rates)),
        "directional_precision_mean": float(np.mean(directional_precisions)),
        "mean_signed_fwd_return_on_trades_mean": float(np.mean(mean_trade_returns)),
        "ternary_accuracy_mean": float(np.mean(ternary_accs)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "entry_prob_threshold_mean": float(np.mean(entry_thresholds)),
        "prob_gap_threshold_mean": float(np.mean(gap_thresholds)),
    }


def export_model_to_onnx(model, output_path: Path, feature_count: int) -> None:
    if isinstance(model, LGBMClassifier):
        initial_types = [("float_input", OnnxToolsFloatTensorType([1, feature_count]))]
        onx = convert_lightgbm(model, initial_types=initial_types, target_opset=15, zipmap=False)
    elif isinstance(model, HistGradientBoostingClassifier):
        # HistGradientBoostingClassifier requires special handling of boolean
        # attributes for ONNX; reuse the existing workaround.
        initial_types = [("float_input", SklFloatTensorType([1, feature_count]))]
        options = {id(model): {"zipmap": False}}
        original_make_attribute, patched_make_attribute = _coerce_bool_attributes_for_onnx()
        onnx_helper.make_attribute = patched_make_attribute
        try:
            onx = convert_sklearn(model, initial_types=initial_types, options=options, target_opset=15)
        finally:
            onnx_helper.make_attribute = original_make_attribute
    else:
        # Default conversion for other sklearn models (including ExtraTrees and Ridge pipelines)
        initial_types = [("float_input", SklFloatTensorType([1, feature_count]))]
        options = {id(model): {"zipmap": False}}
        onx = convert_sklearn(
            model,
            initial_types=initial_types,
            options=options,
            target_opset=15,
        )

    output_path.write_bytes(onx.SerializeToString())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a scale-invariant MLP + LightGBM + HGB ensemble for MT5.")
    p.add_argument("--symbol", default="XAGUSD")
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--output-dir", default="output_scale_invariant_ensemble")
    p.add_argument("--horizon-bars", type=int, default=8)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--label-quantile", type=float, default=0.67)
    p.add_argument("--prob-quantile", type=float, default=0.80)
    p.add_argument("--margin-quantile", type=float, default=0.65)
    p.add_argument("--walk-forward-splits", type=int, default=5)
    # Ensemble weight arguments.  Five models are available: MLP, LightGBM,
    # HistGradientBoosting, ExtraTrees and Ridge.  Default weights are evenly
    # split between the five models.  At least one weight must be non‑zero.
    p.add_argument("--mlp-weight", type=float, default=0.20)
    p.add_argument("--lgbm-weight", type=float, default=0.20)
    p.add_argument("--hgb-weight", type=float, default=0.20)
    p.add_argument("--extratrees-weight", type=float, default=0.20)
    p.add_argument("--ridge-weight", type=float, default=0.20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize all five model weights.  ExtraTrees and Ridge weights are
    # included alongside MLP, LightGBM and HGB.  The returned dictionary
    # contains keys for each model with values summing to one.
    weights = normalize_weights(
        args.mlp_weight,
        args.lgbm_weight,
        args.hgb_weight,
        args.extratrees_weight,
        args.ridge_weight,
    )

    raw = load_rates_from_csv(Path(args.csv)) if args.csv else fetch_rates_from_mt5(args.symbol, args.timeframe, args.bars)
    raw.to_csv(output_dir / "training_rates_snapshot.csv", index=False)

    feat_df = build_features(raw, args.horizon_bars)
    feat_df.to_csv(output_dir / "training_features_snapshot.csv", index=False)

    print(f"Total feature rows: {len(feat_df)}")
    train_df, test_df = split_train_test(feat_df, args.train_ratio)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Train window: {train_df['time'].iloc[0]} -> {train_df['time'].iloc[-1]}")
    print(f"Test window : {test_df['time'].iloc[0]} -> {test_df['time'].iloc[-1]}")
    print(f"Normalized weights: {weights}")

    print("\\nWalk-forward on train:")
    walk_forward = walk_forward_report(
        train_df=train_df,
        weights=weights,
        label_quantile=args.label_quantile,
        prob_quantile=args.prob_quantile,
        margin_quantile=args.margin_quantile,
        n_splits=args.walk_forward_splits,
    )
    print("\\nWalk-forward summary:")
    print(json.dumps(walk_forward, indent=2))

    barrier = compute_return_barrier(train_df, args.label_quantile)
    train_lab = label_targets(train_df, barrier)
    test_lab = label_targets(test_df, barrier)

    models = fit_models(train_lab, random_state=42)
    entry_prob_threshold, min_prob_gap = derive_decision_thresholds(
        models, train_lab, weights, args.prob_quantile, args.margin_quantile
    )

    train_pred = classify_with_thresholds(models, train_lab, weights, entry_prob_threshold, min_prob_gap)
    test_pred = classify_with_thresholds(models, test_lab, weights, entry_prob_threshold, min_prob_gap)

    train_summary = summarize_predictions(train_pred)
    test_summary = summarize_predictions(test_pred)

    print(f"\\nLabel barrier abs(fwd_ret_h): {barrier:.8f}")
    print(f"Recommended InpEntryProbThreshold: {entry_prob_threshold:.6f}")
    print(f"Recommended InpMinProbGap:        {min_prob_gap:.6f}")

    print("\\nTrain summary:")
    print(json.dumps(train_summary, indent=2))
    print("\\nTest summary:")
    print(json.dumps(test_summary, indent=2))

    train_pred.to_csv(output_dir / "train_predictions_snapshot.csv", index=False)
    test_pred.to_csv(output_dir / "test_predictions_snapshot.csv", index=False)

    # Export all models to ONNX format.  Each model uses the same feature
    # count and therefore the same input shape.  ZipMap is disabled so
    # probabilities/scores are returned as a simple tensor.
    export_model_to_onnx(models["mlp"], output_dir / "mlp.onnx", len(FEATURE_COLS))
    export_model_to_onnx(models["lgbm"], output_dir / "lightgbm.onnx", len(FEATURE_COLS))
    export_model_to_onnx(models["hgb"], output_dir / "hgb.onnx", len(FEATURE_COLS))
    export_model_to_onnx(models["extratrees"], output_dir / "extratrees.onnx", len(FEATURE_COLS))
    export_model_to_onnx(models["ridge"], output_dir / "ridge.onnx", len(FEATURE_COLS))

    meta = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "bars": args.bars,
        "horizon_bars": args.horizon_bars,
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "train_ratio": args.train_ratio,
        "weights": weights,
        # quantiles and thresholds controlling label assignment and entry/exit rules
        "label_quantile": args.label_quantile,
        "prob_quantile": args.prob_quantile,
        "margin_quantile": args.margin_quantile,
        "barrier_abs_fwd_ret_h": barrier,
        "entry_prob_threshold": entry_prob_threshold,
        "min_prob_gap": min_prob_gap,
        # diagnostic results
        "walk_forward": walk_forward,
        "train_summary": train_summary,
        "test_summary": test_summary,
    }
    (output_dir / "ensemble_scale_invariant_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    run_txt = f"""MODEL: Scale-invariant MLP + LightGBM + HGB + ExtraTrees + Ridge ensemble
SYMBOL: {args.symbol}
TIMEFRAME: {args.timeframe}
HORIZON BARS: {args.horizon_bars}
FEATURE_COUNT: {len(FEATURE_COLS)}

FEATURES:
- ret_1
- ret_3
- ret_5
- ret_10
- vol_10
- vol_20
- vol_ratio_10_20
- dist_sma_10
- dist_sma_20
- zscore_20
- atr_pct_14
- range_pct_1
- body_pct_1

TRAIN UTC:
  start: {train_df["time"].iloc[0]}
  end  : {train_df["time"].iloc[-1]}

TEST UTC:
  start: {test_df["time"].iloc[0]}
  end  : {test_df["time"].iloc[-1]}

NORMALIZED WEIGHTS:
  InpMlpWeight       = {weights['mlp']:.6f}
  InpLgbmWeight      = {weights['lgbm']:.6f}
  InpHgbWeight       = {weights['hgb']:.6f}
  InpExtraTreesWeight = {weights['extratrees']:.6f}
  InpRidgeWeight      = {weights['ridge']:.6f}

RECOMMENDED EA INPUTS:
  InpEntryProbThreshold = {entry_prob_threshold:.6f}
  InpMinProbGap        = {min_prob_gap:.6f}
  InpMaxBarsInTrade    = {args.horizon_bars}

FILES:
  - mlp.onnx
  - lightgbm.onnx
  - hgb.onnx
  - extratrees.onnx
  - ridge.onnx

IMPORTANT:
- Python and MT5 must use exactly the same feature engineering.
- atr_pct_14 = ATR(14) / close
- range_pct_1 = (high - low) / close
- body_pct_1  = (close - open) / open
"""
    (output_dir / "run_in_mt5.txt").write_text(run_txt, encoding="utf-8")

    print(f"\\nONNX models saved to: {output_dir}")
    print("  - mlp.onnx")
    print("  - lightgbm.onnx")
    print("  - hgb.onnx")
    print("  - extratrees.onnx")
    print("  - ridge.onnx")
    print("Read: run_in_mt5.txt")


if __name__ == "__main__":
    main()
