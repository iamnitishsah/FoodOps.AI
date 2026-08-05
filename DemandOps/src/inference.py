"""Inference pipeline for FoodOps.AI - DemandOps module.

Serves the LightGBM model trained in ``notebooks/model_training.ipynb``.

The model's inputs are mostly history features (lags, rolling stats), so serving
means rebuilding them from a (center, meal) history with exactly the definitions
used in ``notebooks/feature_engineering.ipynb``. ``check_feature_parity`` proves
that against the saved parquet; the app runs it at start-up.

Forecast modes
--------------
* ``predict_next_week`` / scenario helpers: one week ahead, built from true history
  (this is the setting the validation scores in results.json describe).
* ``forecast_pair``: multi-week, recursive. Week t's prediction is fed back in as the
  lag for week t+1, so errors compound and later weeks are less reliable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = MODULE_ROOT / "models"

# Preferred first: a model refit on all weeks. Falls back to the validated (weeks <= 131) model.
BUNDLE_CANDIDATES = [
    MODELS_DIR / "lgb_final_bundle.joblib",
    MODELS_DIR / "lgb_bundle.joblib",
]

# Same order as ``cat_cols`` in model_training.ipynb.
DEFAULT_CAT_COLS = ["center_id", "meal_id", "city_code", "region_code",
                    "center_type", "category", "cuisine"]
STATIC_COLS = ["city_code", "region_code", "center_type", "op_area", "category", "cuisine"]

# price_change_pct = (checkout - base) / base. Flip to -1 only if check_feature_parity says so.
PRICE_CHANGE_SIGN = 1


# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------
def load_model_bundle(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the exported LightGBM bundle and normalise it for serving.

    Works with both bundle layouts:
      * {'model', 'config': {'features', 'params', 'train_week', ...}, 'metrics'}
      * the same, with 'cat_cols' also stored in config.
    The category lists are read from the fitted booster itself, so inference always
    uses exactly the categories the model was trained with.
    """
    candidates = [Path(path)] if path else BUNDLE_CANDIDATES
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            "No LightGBM bundle found. Looked for: " + ", ".join(str(p) for p in candidates)
        )

    raw = joblib.load(found)
    model, cfg = raw["model"], raw["config"]
    features = list(cfg["features"])
    cat_set = set(cfg.get("cat_cols", DEFAULT_CAT_COLS))
    cat_cols = [f for f in features if f in cat_set]

    categories = model.booster_.pandas_categorical
    if not categories or len(categories) != len(cat_cols):
        raise ValueError(
            f"Categorical layout mismatch: booster has {0 if not categories else len(categories)} "
            f"categorical columns, config implies {len(cat_cols)} ({cat_cols})."
        )

    return {
        "model": model,
        "features": features,
        "cat_cols": cat_cols,
        "categories": [pd.Index(c) for c in categories],
        "train_week": int(cfg.get("train_week", -1)),
        "params": cfg.get("params", {}),
        "metrics": raw.get("metrics", {}),
        "source": found.name,
    }


def _to_matrix(bundle: Dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    """Select model features and apply the training-time categorical dtypes."""
    X = rows[bundle["features"]].copy()
    for col, cats in zip(bundle["cat_cols"], bundle["categories"]):
        X[col] = pd.Categorical(X[col].astype(cats.dtype), categories=cats)
    unseen = [c for c in bundle["cat_cols"] if X[c].isna().any()]
    if unseen:
        raise ValueError(f"Values never seen during training in: {unseen}")
    return X


def predict_log(bundle: Dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    """Predict log1p(orders) for one or many fully-built feature rows."""
    return np.asarray(bundle["model"].predict(_to_matrix(bundle, rows)), dtype=float)


# ------------------------------------------------------------------------------
# Feature rebuilding (must mirror feature_engineering.ipynb)
# ------------------------------------------------------------------------------
def week_of_year(week: int) -> int:
    return ((int(week) - 1) % 52) + 1


def price_change(checkout_price: float, base_price: float) -> float:
    return PRICE_CHANGE_SIGN * (float(checkout_price) - float(base_price)) / float(base_price)


def lag_features(log_hist: np.ndarray) -> Dict[str, float]:
    """History features for the NEXT week, from log1p(orders) history (oldest -> newest).

    lag_k            = shift(k) of log1p(orders)
    rolling_mean_4   = mean of the last 4 values (shift(1) first, then roll)
    rolling_std_4    = sample std (ddof=1) of the last 4 values
    new_centre_meal  = 0 for any week >= 5 (the grid starts at week 1 for every pair)
    """
    h = np.asarray(log_hist, dtype=float)
    if len(h) < 4:
        raise ValueError("Need at least 4 weeks of history.")
    last4 = h[-4:]
    return {
        "lag_1": float(h[-1]),
        "lag_2": float(h[-2]),
        "lag_4": float(h[-4]),
        "rolling_mean_4": float(last4.mean()),
        "rolling_std_4": float(last4.std(ddof=1)),
        "new_centre_meal": 0,
    }


def get_pair_state(panel: pd.DataFrame, center_id: int, meal_id: int) -> Dict[str, Any]:
    """Everything the model needs to know about one (center, meal) pair's past."""
    sub = panel[(panel["center_id"] == center_id) & (panel["meal_id"] == meal_id)].sort_values("week")
    if sub.empty:
        raise KeyError(f"No history for center {center_id}, meal {meal_id}")
    last = sub.iloc[-1]
    observed = sub[sub["num_orders"] > 0]
    return {
        "center_id": int(center_id),
        "meal_id": int(meal_id),
        "weeks": sub["week"].to_numpy(),
        "log_hist": sub["log_num_orders"].to_numpy(dtype=float),
        "orders_hist": sub["num_orders"].to_numpy(dtype=float),
        "last_week": int(last["week"]),
        "last_observed_week": int(observed["week"].iloc[-1]) if len(observed) else None,
        "last_checkout_price": float(last["checkout_price"]),
        "last_base_price": float(last["base_price"]),
        "static": {c: last[c] for c in STATIC_COLS},
    }


def build_feature_row(
    state: Dict[str, Any],
    week: int,
    checkout_price: float,
    base_price: float,
    emailer: int,
    homepage: int,
    log_hist: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """One complete model-input row for ``week``, given the history available at that time."""
    hist = state["log_hist"] if log_hist is None else log_hist
    row = {
        **state["static"],
        "center_id": state["center_id"],
        "meal_id": state["meal_id"],
        "week": int(week),
        "checkout_price": float(checkout_price),
        "base_price": float(base_price),
        "price_change_pct": price_change(checkout_price, base_price),
        "emailer_for_promotion": int(emailer),
        "homepage_featured": int(homepage),
        "week_of_year": week_of_year(week),
    }
    row.update(lag_features(hist))
    return row


# ------------------------------------------------------------------------------
# Forecasting
# ------------------------------------------------------------------------------
def _orders_from_log(pred_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(pred_log, 0.0, None))


def safety_stock(predicted_orders: float, buffer_pct: float = 0.15) -> float:
    """Prep quantity = forecast plus a planning buffer (a policy choice, not a statistic)."""
    return float(np.ceil(predicted_orders * (1.0 + buffer_pct)))


def predict_next_week(
    bundle: Dict[str, Any],
    state: Dict[str, Any],
    checkout_price: float,
    base_price: float,
    emailer: int,
    homepage: int,
    week: Optional[int] = None,
) -> Dict[str, float]:
    """One-week-ahead forecast from true history."""
    week = state["last_week"] + 1 if week is None else week
    row = build_feature_row(state, week, checkout_price, base_price, emailer, homepage)
    log = float(np.clip(predict_log(bundle, pd.DataFrame([row]))[0], 0.0, None))
    return {"week": int(week), "pred_log": log, "predicted_orders": float(np.expm1(log))}


def forecast_pair(bundle: Dict[str, Any], state: Dict[str, Any], plan: pd.DataFrame) -> pd.DataFrame:
    """Recursive multi-week forecast for one pair.

    ``plan`` has one row per future week with: week, checkout_price, base_price,
    emailer_for_promotion, homepage_featured. Weeks must follow on from the history.
    """
    plan = plan.sort_values("week").reset_index(drop=True)
    expected = state["last_week"] + 1 + np.arange(len(plan))
    if not np.array_equal(plan["week"].to_numpy(), expected):
        raise ValueError(f"Plan weeks must be {expected[0]}..{expected[-1]} without gaps.")

    hist = list(state["log_hist"])
    out = []
    for step, r in enumerate(plan.itertuples(index=False), start=1):
        row = build_feature_row(state, r.week, r.checkout_price, r.base_price,
                                r.emailer_for_promotion, r.homepage_featured, log_hist=np.array(hist))
        log = float(np.clip(predict_log(bundle, pd.DataFrame([row]))[0], 0.0, None))
        hist.append(log)  # the prediction becomes next week's lag_1
        out.append({
            "step": step,
            "week": int(r.week),
            "checkout_price": float(r.checkout_price),
            "base_price": float(r.base_price),
            "price_change_pct": row["price_change_pct"],
            "emailer_for_promotion": int(r.emailer_for_promotion),
            "homepage_featured": int(r.homepage_featured),
            "predicted_orders": float(np.expm1(log)),
        })
    return pd.DataFrame(out)


def simulate_price_sensitivity(
    bundle: Dict[str, Any],
    state: Dict[str, Any],
    base_price: float,
    emailer: int,
    homepage: int,
    week: Optional[int] = None,
    pct_range: tuple = (-0.30, 0.30),
    steps: int = 21,
) -> pd.DataFrame:
    """Next-week demand and revenue across a range of checkout prices (batched)."""
    week = state["last_week"] + 1 if week is None else week
    pcts = np.linspace(pct_range[0], pct_range[1], steps)
    prices = np.round(base_price * (1.0 + pcts), 2)
    rows = pd.DataFrame([
        build_feature_row(state, week, p, base_price, emailer, homepage) for p in prices
    ])
    orders = _orders_from_log(predict_log(bundle, rows))
    return pd.DataFrame({
        "checkout_price": prices,
        "pct_from_base": np.round(pcts * 100, 1),
        "predicted_orders": np.round(orders, 1),
        "expected_revenue": np.round(prices * orders, 2),
    })


def simulate_promo_scenarios(
    bundle: Dict[str, Any],
    state: Dict[str, Any],
    base_price: float,
    checkout_price: float,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """Next-week demand across the four email / homepage promotion combinations."""
    week = state["last_week"] + 1 if week is None else week
    scenarios = [
        ("No Promotion", 0, 0),
        ("Emailer Only", 1, 0),
        ("Homepage Only", 0, 1),
        ("Email + Homepage", 1, 1),
    ]
    rows = pd.DataFrame([
        build_feature_row(state, week, checkout_price, base_price, e, h) for _, e, h in scenarios
    ])
    orders = _orders_from_log(predict_log(bundle, rows))
    baseline = orders[0]
    return pd.DataFrame({
        "Scenario": [s[0] for s in scenarios],
        "Email Promo": ["Yes" if s[1] else "No" for s in scenarios],
        "Homepage": ["Yes" if s[2] else "No" for s in scenarios],
        "Predicted Orders": np.round(orders, 1),
        "Uplift vs Baseline (%)": np.round((orders / baseline - 1.0) * 100 if baseline > 0 else 0.0, 1),
    })


# ------------------------------------------------------------------------------
# Serving-parity check
# ------------------------------------------------------------------------------
def check_feature_parity(
    panel: pd.DataFrame,
    n_pairs: int = 150,
    weeks: tuple = (5, 40, 90, 132, 145),
    seed: int = 0,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Rebuild features from history and compare with the saved feature-engineered rows.

    For each sampled (pair, week) the history is cut at week-1, exactly what the app
    sees when forecasting that week. Raises AssertionError on any mismatch, so a
    silent training/serving skew cannot reach the dashboard.
    """
    pairs = panel[["center_id", "meal_id"]].drop_duplicates()
    pairs = pairs.sample(n=min(n_pairs, len(pairs)), random_state=seed)
    sample = panel.merge(pairs, on=["center_id", "meal_id"])

    worst = {k: 0.0 for k in ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4",
                              "new_centre_meal", "week_of_year", "price_change_pct"]}
    for _, sub in sample.groupby(["center_id", "meal_id"]):
        sub = sub.sort_values("week").reset_index(drop=True)
        log = sub["log_num_orders"].to_numpy(dtype=float)
        for w in weeks:
            if w > len(sub):
                continue
            idx = w - 1
            assert int(sub.loc[idx, "week"]) == w, "panel is not a complete weekly grid"
            built = lag_features(log[:idx])
            built["week_of_year"] = week_of_year(w)
            built["price_change_pct"] = price_change(sub.loc[idx, "checkout_price"],
                                                     sub.loc[idx, "base_price"])
            for k in worst:
                worst[k] = max(worst[k], abs(float(built[k]) - float(sub.loc[idx, k])))

    bad = {k: v for k, v in worst.items() if v > tol}
    if bad:
        hint = ""
        if "price_change_pct" in bad:
            hint = " (if only price_change_pct differs, try PRICE_CHANGE_SIGN = -1 in inference.py)"
        raise AssertionError(f"Training/serving feature mismatch: {bad}{hint}")
    return worst