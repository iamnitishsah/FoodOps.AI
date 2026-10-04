"""Data loader utilities for FoodOps.AI - DeliveryOps.

Reads the model card written by notebooks/model_training.ipynb and the processed feature table written by
notebooks/feature_engineering.ipynb, and turns them into the tables the dashboard displays.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = MODULE_ROOT / "models"
PROCESSED_DIR = MODULE_ROOT / "dataset" / "processed"

CARD_PATH = MODELS_DIR / "delivery_model_card.json"
FEATURE_TABLE_PATH = PROCESSED_DIR / "processed_data.parquet"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Plausible bounds for the simulator inputs when the feature table is not available
FALLBACK_RANGES = {
    "total_items": {"lo": 1, "median": 3, "hi": 25},
    "num_distinct_items": {"lo": 1, "median": 3, "hi": 25},
    "subtotal": {"lo": 100, "median": 2500, "hi": 30000},
    "min_item_price": {"lo": 0, "median": 500, "hi": 10000},
    "max_item_price": {"lo": 0, "median": 1200, "hi": 10000},
    "estimated_store_to_consumer_driving_duration": {"lo": 60, "median": 600, "hi": 2000},
    "total_onshift_dashers": {"lo": 0, "median": 30, "hi": 200},
    "total_busy_dashers": {"lo": 0, "median": 28, "hi": 200},
    "total_outstanding_orders": {"lo": 0, "median": 35, "hi": 300},
}

# Feature ablation recorded in notebooks/model_training.ipynb (LightGBM, two forward validation folds).
# Delta MAE is in minutes against the 21-feature reference; positive means worse. Noise floor: 0.026 (A) / 0.018 (B).
ABLATION_NOISE_FLOOR = {"A": 0.026, "B": 0.018}
ABLATION = pd.DataFrame([
    ("Drop zero_onshift flag", -0.008, 0.006, 6, "Adopted (tie, smaller set)"),
    ("Capped busy ratio instead of uncapped", 0.002, -0.016, 6, "Kept reference (tie)"),
    ("Hour as sine / cosine", 0.010, 0.006, 6, "Kept reference (tie)"),
    ("Weekday as one-hot dummies", 0.046, -0.032, 7, "Kept integer weekday (folds disagree)"),
    ("Drop order-place estimate", 0.021, 0.025, 4, "Kept feature (small, consistent loss)"),
    ("Drop store category", 0.041, 0.054, 0, "Kept feature (hurts)"),
    ("Hour as meal-period buckets", 0.094, 0.108, 1, "Kept integer hour (hurts)"),
    ("Drop store target encoding", 0.183, 0.250, 1, "Kept feature (largest loss)"),
], columns=["Change from the reference", "Delta MAE, Fold A (min)", "Delta MAE, Fold B (min)",
            "Days better (of 14)", "Outcome"])


def dig(data, *keys, default=None):
    """Safely read nested dictionary keys."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


# ---------------------------------------------------------------------------------------------------- artifacts
def load_model_card() -> dict:
    """Model card written by the training notebook (results, segments, limitations, versions)."""
    if not CARD_PATH.exists():
        raise FileNotFoundError(f"Model card not found at {CARD_PATH}. Run notebooks/model_training.ipynb to create it.")
    with open(CARD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_table(columns: Optional[list] = None) -> Optional[pd.DataFrame]:
    """Processed feature table, or None when it is not shipped with the app."""
    if not FEATURE_TABLE_PATH.exists():
        return None
    return pd.read_parquet(FEATURE_TABLE_PATH, columns=columns)


def version_mismatches(card: dict) -> list:
    """Libraries whose installed version differs from the one the models were trained with."""
    trained = card.get("versions", {})
    mismatches = []
    for package in ("lightgbm", "scikit-learn", "pandas", "numpy"):
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = "not installed"
        if package in trained and trained[package] != installed:
            mismatches.append((package, trained[package], installed))
    return mismatches


# --------------------------------------------------------------------------------------------- model card tables
_MODEL_SPECS = [
    # (test key, validation name prefix, display name, family, role)
    ("lightgbm P50", "lightgbm final", "LightGBM (P10/P50/P90)", "Gradient Boosted Trees", "Champion (Deployed)"),
    ("ridge", "ridge", "Ridge Regression", "L2-Regularized Linear Model", "Linear Baseline"),
    ("market x hour median", "market x hour", "Market × Hour Median", "Heuristic Lookup", "Lookup Baseline"),
    ("naive median", "naive", "Naive Median", "Heuristic Benchmark", "Lower Bound"),
]


def card_scoreboard(card: dict) -> pd.DataFrame:
    """Model scoreboard: test-week MAE and bias plus validation MAE per fold."""
    test = dig(card, "test", "point_models", default={})
    validation = {r["model"]: r for r in dig(card, "validation", "point_models", default=[])}
    rows = []
    for test_key, prefix, name, family, role in _MODEL_SPECS:
        if test_key not in test:
            continue
        val = next((r for model, r in validation.items() if model.startswith(prefix)), {})
        rows.append({
            "Model Architecture": name, "Family": family,
            "Test MAE (min)": test[test_key]["MAE (min)"], "Test bias (min)": test[test_key]["bias (min)"],
            "Validation MAE A (min)": val.get("MAE A"), "Validation MAE B (min)": val.get("MAE B"), "Role": role,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("Test MAE (min)").reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def card_intervals(card: dict) -> pd.DataFrame:
    """Interval coverage, width and pinball loss for each validation fold and the test week."""
    folds = dig(card, "validation", "intervals", default={})
    test = dig(card, "test", "intervals", default={})
    specs = [
        ("P10 coverage (%)", "P10 coverage %", "P10 coverage % (nominal 10)", 10),
        ("P90 coverage (%)", "P90 coverage %", "P90 coverage % (nominal 90)", 90),
        ("80% interval coverage (%)", "80% interval coverage %", "80% interval coverage % (nominal 80)", 80),
        ("Mean P10–P90 width (min)", "mean width (min)", "mean width (min)", None),
        ("Pinball loss P10", "pinball P10", "pinball P10", None),
        ("Pinball loss P90", "pinball P90", "pinball P90", None),
    ]
    rows = [{"Metric": label, "Nominal": nominal,
             "Fold A": dig(folds, "A", fold_key), "Fold B": dig(folds, "B", fold_key),
             "Test week": test.get(test_key)} for label, fold_key, test_key, nominal in specs]
    return pd.DataFrame(rows)


def card_segments(card: dict) -> dict:
    """Test-week results per segment type: market, hour, day, telemetry, store history."""
    renames = {"orders": "Orders", "share %": "Share (%)", "MAE_lgbm": "LightGBM MAE (min)",
               "MAE_ridge": "Ridge MAE (min)", "bias": "Bias (min)", "P10_cov": "P10 coverage (%)",
               "P90_cov": "P90 coverage (%)", "width": "Mean width (min)"}
    out = {}
    for name, rows in (card.get("segments") or {}).items():
        table = pd.DataFrame.from_dict(rows, orient="index").rename(columns=renames)
        table.index.name = "Segment"
        out[name] = table.reset_index()
    return out


# ---------------------------------------------------------------------------------------- feature-table products
def input_ranges(table: pd.DataFrame) -> dict:
    """Min / median / max of the numeric simulator inputs, from the feature table."""
    ranges = {}
    for col in FALLBACK_RANGES:
        series = table[col].dropna()
        if series.empty:
            ranges[col] = FALLBACK_RANGES[col]
            continue
        ranges[col] = {"lo": max(int(np.floor(series.min())), 0), "median": int(round(series.median())),
                       "hi": int(np.ceil(series.max()))}
    return ranges


def build_store_catalog(table: pd.DataFrame, store_table: pd.DataFrame) -> pd.DataFrame:
    """Stores with training history: category, most common market, order count and average delivery time."""
    def most_common(column: str) -> pd.Series:
        counts = table.groupby(["store_id", column], observed=True).size().reset_index(name="n")
        counts = counts.sort_values(["store_id", "n"], ascending=[True, False]).drop_duplicates("store_id")
        return counts.set_index("store_id")[column]

    catalog = store_table[["k_n", "k_sum"]].join(most_common("store_primary_category").astype(str).rename("category"))
    catalog = catalog.join(most_common("market_id").astype(int).rename("market"))
    catalog = catalog.dropna(subset=["category"])
    catalog["avg_min"] = catalog["k_sum"] / catalog["k_n"] / 60
    catalog = catalog.sort_values("k_n", ascending=False)
    catalog["label"] = [f"Store {sid} · {cat} · {int(n)} orders · avg {avg:.0f} min"
                        for sid, cat, n, avg in zip(catalog.index, catalog["category"], catalog["k_n"], catalog["avg_min"])]
    catalog.index.name = "store_id"
    return catalog


def build_analytics(table: pd.DataFrame) -> dict:
    """Descriptive aggregates behind the analytics tab (all cleaned orders)."""
    t = table[["total_delivery_duration", "order_hour", "order_day_of_week", "market_id", "store_id",
               "outstanding_order_ratio", "dasher_telemetry_missing", "zero_onshift"]].copy()
    t["minutes"] = t["total_delivery_duration"] / 60
    t["market"] = t["market_id"].astype(int)

    def summarize(by: str) -> pd.DataFrame:
        g = t.groupby(by)["minutes"]
        return pd.DataFrame({"orders": g.size(), "median": g.median(), "p90": g.quantile(0.9)}).reset_index()

    by_hour, by_market = summarize("order_hour"), summarize("market")
    by_weekday = summarize("order_day_of_week")
    by_weekday["weekday"] = by_weekday["order_day_of_week"].map(lambda d: WEEKDAYS[int(d)][:3])

    flags = t.groupby("market")[["dasher_telemetry_missing", "zero_onshift"]].mean() * 100
    telemetry = flags.rename(columns={"dasher_telemetry_missing": "Telemetry missing (%)",
                                      "zero_onshift": "Zero on-shift (%)"}).reset_index()

    # Delivery time against dispatch pressure: deciles of outstanding orders per on-shift dasher
    with_ratio = t.dropna(subset=["outstanding_order_ratio"]).copy()
    with_ratio["load_bin"] = pd.qcut(with_ratio["outstanding_order_ratio"], 10, duplicates="drop")
    g = with_ratio.groupby("load_bin", observed=True)
    load = pd.DataFrame({
        "orders": g.size(),
        "median": g["minutes"].median(),
        "p90": g["minutes"].quantile(0.9),
        "ratio_mid": g["outstanding_order_ratio"].median(),
    }).reset_index(drop=True)
    load["load"] = [f"{r:.2f}" for r in load["ratio_mid"]]

    special = pd.DataFrame({
        "State": ["Telemetry present", "Zero on-shift", "Telemetry missing"],
        "Orders": [int(((t["zero_onshift"] == 0) & (~t["dasher_telemetry_missing"].astype(bool))).sum()),
                   int((t["zero_onshift"] == 1).sum()), int(t["dasher_telemetry_missing"].astype(bool).sum())],
        "Median (min)": [
            t.loc[(t["zero_onshift"] == 0) & (~t["dasher_telemetry_missing"].astype(bool)), "minutes"].median(),
            t.loc[t["zero_onshift"] == 1, "minutes"].median(),
            t.loc[t["dasher_telemetry_missing"].astype(bool), "minutes"].median()],
    })

    return {
        "kpis": {"orders": int(len(t)), "markets": int(t["market"].nunique()), "stores": int(t["store_id"].nunique()),
                 "median_min": float(t["minutes"].median()), "p90_min": float(t["minutes"].quantile(0.9))},
        "by_hour": by_hour, "by_weekday": by_weekday, "by_market": by_market,
        "telemetry": telemetry, "load": load, "special": special,
    }