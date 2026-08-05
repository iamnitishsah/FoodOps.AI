"""Data loader utilities for FoodOps.AI - DemandOps module.

Everything the app shows is derived from three artifacts produced by the notebooks:

* dataset/processed/feature_engineered_data.parquet  (full weekly panel + features)
* models/results.json                                (scoreboard written by model_training)
* models/lgb_val_pred.parquet                        (optional: validation predictions,
                                                      used for an empirical error range)

dataset/raw/test.csv is optional: it pre-fills the planning table with the official
planned prices and promotions for weeks 146-155.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = MODULE_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
MODELS_DIR = MODULE_ROOT / "models"

PANEL_PATH = PROCESSED_DIR / "feature_engineered_data.parquet"


def load_panel() -> pd.DataFrame:
    """Full (center, meal, week) grid with engineered features.

    Rows with num_orders == 0 are weeks with no recorded order for that pair
    (zero-filled during feature engineering). Treat them as 'not recorded', not
    as demand that was measured to be zero.
    """
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PANEL_PATH}. Run notebooks/feature_engineering.ipynb first.")
    return pd.read_parquet(PANEL_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)


def load_planned_inputs() -> pd.DataFrame:
    """Official planned price/promo inputs for the forecast weeks (Kaggle test.csv), if present."""
    path = RAW_DIR / "test.csv"
    cols = ["week", "center_id", "meal_id", "checkout_price", "base_price",
            "emailer_for_promotion", "homepage_featured"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, usecols=cols)


def load_results() -> Dict[str, Dict[str, float]]:
    """Scoreboard written by the model_training notebook."""
    path = MODELS_DIR / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


_MODEL_INFO = {
    "lgb": ("LightGBM (L1 objective)", "Gradient Boosted Trees", "Deployed",
            "Log1p target, L1 loss, native categoricals, lags and rolling stats"),
    "lstm": ("LSTM (PyTorch)", "Deep Learning (sequential)", "Challenger",
             "Weekly history window plus ID embeddings; trained on Colab, not served here"),
    "lgb+lstm": ("LightGBM + LSTM blend", "Ensemble", "Experiment",
                 "Fixed 50/50 average of the two models' predictions"),
    "ridge": ("Ridge Regression", "Regularized Linear Model", "Linear baseline",
              "One-hot IDs, sin/cos week-of-year, log1p target"),
    "naive": ("Naive Lag-1", "Heuristic", "Lower bound",
              "Repeats last week's orders (true previous-week value)"),
}


def load_benchmark_metrics() -> pd.DataFrame:
    """Scoreboard as a tidy table, best WAPE first."""
    results = load_results()
    rows = []
    for key, m in results.items():
        name, family, status, note = _MODEL_INFO.get(key, (key, "-", "-", ""))
        rows.append({
            "Model Architecture": name,
            "Family": family,
            "Validation WAPE (%)": round(float(m["wape"]), 2),
            "Validation MAPE (%)": round(float(m["mape"]), 2),
            "Bias (%)": round(float(m["bias"]), 2),
            "Status": status,
            "Notes": note,
        })
    if not rows:
        return pd.DataFrame(columns=["Rank", "Model Architecture", "Family", "Validation WAPE (%)",
                                     "Validation MAPE (%)", "Bias (%)", "Status", "Notes"])
    table = pd.DataFrame(rows).sort_values("Validation WAPE (%)").reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table


def load_error_band(panel: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Range multipliers applied to a forecast to show typical error.

    Preferred: empirical 10th/90th percentile of actual/predicted on the validation weeks
    (needs models/lgb_val_pred.parquet). Fallback: +/- validation WAPE, which is indicative
    only (WAPE is an average error, not a coverage interval).
    """
    path = MODELS_DIR / "lgb_val_pred.parquet"
    if path.exists():
        preds = pd.read_parquet(path)
        merged = preds.merge(panel[["week", "center_id", "meal_id", "num_orders"]],
                             on=["week", "center_id", "meal_id"], how="left")
        merged = merged[(merged["num_orders"] > 0) & (merged["lgb_pred"] > 0)]
        if len(merged) >= 500:
            ratio = (merged["num_orders"] / merged["lgb_pred"]).to_numpy()
            lo, hi = np.quantile(ratio, [0.10, 0.90])
            return {"lo": float(lo), "hi": float(hi), "kind": "empirical",
                    "label": "80% empirical range from validation weeks (1-week-ahead errors)"}
    wape = load_results().get("lgb", {}).get("wape")
    if wape is not None:
        w = float(wape) / 100.0
        return {"lo": max(0.0, 1.0 - w), "hi": 1.0 + w, "kind": "wape",
                "label": f"Indicative range: ±{wape:.0f}% (validation WAPE)"}
    return None


def build_pair_catalog(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per real (center, meal) pair with static attributes and recording stats."""
    static = panel.drop_duplicates(["center_id", "meal_id"])[
        ["center_id", "meal_id", "center_type", "city_code", "region_code", "op_area", "category", "cuisine"]
    ]
    stats = (panel[panel["num_orders"] > 0]
             .groupby(["center_id", "meal_id"])["week"]
             .agg(last_observed_week="max", observed_weeks="count")
             .reset_index())
    cat = static.merge(stats, on=["center_id", "meal_id"], how="left")
    cat["observed_weeks"] = cat["observed_weeks"].fillna(0).astype(int)
    return cat.sort_values(["center_id", "meal_id"]).reset_index(drop=True)


def build_analytics(panel: pd.DataFrame) -> Dict[str, Any]:
    """Portfolio-level aggregates for the analytics tab."""
    active = panel.assign(active=(panel["num_orders"] > 0))
    weekly = (active.groupby("week")
              .agg(num_orders=("num_orders", "sum"), active_pairs=("active", "sum"))
              .reset_index())
    weekly["coverage_pct"] = weekly["active_pairs"] / panel["center_id"].groupby(panel["week"]).size().to_numpy() * 100
    by_category = (panel.groupby("category")["num_orders"].sum()
                   .rename("total_orders").reset_index().sort_values("total_orders", ascending=False))
    by_cuisine = (panel.groupby("cuisine")["num_orders"].sum()
                  .rename("total_orders").reset_index().sort_values("total_orders", ascending=False))
    return {
        "weekly": weekly,
        "by_category": by_category,
        "by_cuisine": by_cuisine,
        "kpis": {
            "hubs": int(panel["center_id"].nunique()),
            "dishes": int(panel["meal_id"].nunique()),
            "pairs": int(panel[["center_id", "meal_id"]].drop_duplicates().shape[0]),
            "cuisines": int(panel["cuisine"].nunique()),
            "categories": int(panel["category"].nunique()),
            "first_week": int(panel["week"].min()),
            "last_week": int(panel["week"].max()),
            "total_orders": float(panel["num_orders"].sum()),
        },
    }