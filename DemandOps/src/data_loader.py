"""Data loader utilities for FoodOps.AI - DemandOps module.

Loads center/meal metadata, model benchmarks, cached operational states,
and pre-computed analytics.
"""

import os
import json
from typing import Dict, Any, Tuple
import pandas as pd

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(MODULE_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(MODULE_ROOT, "data", "processed")


def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load fulfillment center and meal metadata."""
    centers_path = os.path.join(RAW_DIR, "fulfilment_center_info.csv")
    meals_path = os.path.join(RAW_DIR, "meal_info.csv")

    centers_df = pd.read_csv(centers_path)
    meals_df = pd.read_csv(meals_path)
    return centers_df, meals_df


def load_latest_state() -> pd.DataFrame:
    """Load week 145 baseline state for quick (center_id, meal_id) lookups."""
    state_path = os.path.join(PROCESSED_DIR, "latest_known_state.csv")
    if os.path.exists(state_path):
        return pd.read_csv(state_path)
    return pd.DataFrame()


def load_operational_summary() -> Dict[str, Any]:
    """Load pre-computed category, cuisine, and weekly platform summaries."""
    summary_path = os.path.join(PROCESSED_DIR, "operational_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}


def load_benchmark_metrics() -> pd.DataFrame:
    """Return model evaluation scoreboard metrics from validation phase."""
    return pd.DataFrame([
        {
            "Rank": 1,
            "Model Architecture": "LightGBM Regressor (Champion)",
            "Family": "Gradient Boosted Trees",
            "Validation WAPE (%)": 28.75,
            "Validation MAPE (%)": 44.14,
            "Status": "Champion (Deployed)",
            "Notes": "Optimal tree splits on price elasticity & multi-week lags"
        },
        {
            "Rank": 2,
            "Model Architecture": "Simple LSTM Neural Network",
            "Family": "Deep Learning (Sequential)",
            "Validation WAPE (%)": 29.98,
            "Validation MAPE (%)": 44.02,
            "Status": "Challenger",
            "Notes": "PyTorch 1-layer LSTM on 8-week history vector"
        },
        {
            "Rank": 3,
            "Model Architecture": "Ridge Regression",
            "Family": "Regularized Linear Model",
            "Validation WAPE (%)": 35.72,
            "Validation MAPE (%)": 46.98,
            "Status": "Linear Baseline",
            "Notes": "L2 regularization (alpha=10) with one-hot encoding"
        },
        {
            "Rank": 4,
            "Model Architecture": "Naive Lag-1 Baseline",
            "Family": "Heuristic Benchmark",
            "Validation WAPE (%)": 41.96,
            "Validation MAPE (%)": 67.23,
            "Status": "Lower Bound",
            "Notes": "Carries forward prior week demand (no ML)"
        }
    ])
