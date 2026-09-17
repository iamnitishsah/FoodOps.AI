"""Data loader utilities for FoodOps.AI - DeliveryOps module.
Loads model benchmarks and pre-computed operational analytics.
"""

import os
import json
import pandas as pd

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(MODULE_ROOT, "data", "processed")

def load_operational_summary() -> dict:
    """Load pre-computed category, market, and daily dispatch summaries."""
    summary_path = os.path.join(PROCESSED_DIR, "operational_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}

def load_benchmark_metrics() -> pd.DataFrame:
    """Return model evaluation scoreboard metrics from the validation phase."""
    return pd.DataFrame([
        {
            "Rank": 1,
            "Model Architecture": "Tuned LightGBM (P10/P50/P90)",
            "Family": "Gradient Boosted Trees",
            "Test MAE (min)": 10.29,
            "P90 Coverage (%)": 86.70,
            "Status": "Champion (Deployed)",
            "Notes": "Optuna tuned (547 trees). Heavy regularization protects against holiday concept drift."
        },
        {
            "Rank": 2,
            "Model Architecture": "Baseline LightGBM",
            "Family": "Gradient Boosted Trees",
            "Test MAE (min)": 10.31,
            "P90 Coverage (%)": 87.49,
            "Status": "Challenger",
            "Notes": "Default hyperparameters. Slightly overfit to normal operational weeks."
        },
        {
            "Rank": 3,
            "Model Architecture": "Ridge Regression",
            "Family": "Regularized Linear Model",
            "Test MAE (min)": 10.89,
            "P90 Coverage (%)": None,
            "Status": "Linear Baseline",
            "Notes": "L2 regularization with StandardScaling and One-Hot Encoding."
        },
        {
            "Rank": 4,
            "Model Architecture": "Naive Median Baseline",
            "Family": "Heuristic Benchmark",
            "Test MAE (min)": 13.08,
            "P90 Coverage (%)": None,
            "Status": "Lower Bound",
            "Notes": "Predicts global median (44 min) for all orders regardless of features."
        }
    ])