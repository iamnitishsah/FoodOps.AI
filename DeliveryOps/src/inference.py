"""Inference engine for FoodOps.AI - DeliveryOps.

Loads the deployed model bundle (P10 / P50 / P90 LightGBM models, the feature contract, the categorical levels and the
store-encoding table) and rebuilds the 20 model features from raw order inputs with the same definitions used in
training (see notebooks/feature_engineering.ipynb and notebooks/model_training.ipynb).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = MODULE_ROOT / "models" / "delivery_model_bundle.joblib"

QUANTILE_KEYS = ("p10", "p50", "p90")

# Raw numeric inputs passed to the models unchanged
NUMERIC_INPUTS = [
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
    "order_hour",
    "order_day_of_week",
]
DASHER_COLUMNS = ["total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]

# Store history bands used in the evaluation (training orders behind a store)
THIN_STORE_MAX_ORDERS = 24

FEATURE_LABELS = {
    "market_id": "Market",
    "order_protocol": "Order protocol",
    "store_primary_category": "Store category",
    "total_items": "Items in cart",
    "subtotal": "Subtotal (cents)",
    "num_distinct_items": "Distinct items",
    "min_item_price": "Cheapest item (cents)",
    "max_item_price": "Priciest item (cents)",
    "total_onshift_dashers": "Dashers on shift",
    "total_busy_dashers": "Busy dashers",
    "total_outstanding_orders": "Outstanding orders",
    "estimated_order_place_duration": "Order-place estimate (s)",
    "estimated_store_to_consumer_driving_duration": "Drive estimate (s)",
    "order_hour": "Order hour",
    "order_day_of_week": "Weekday (0 = Mon)",
    "busy_dasher_ratio": "Busy dashers per on-shift dasher",
    "outstanding_order_ratio": "Outstanding orders per on-shift dasher",
    "store_target_enc": "Store average delivery time",
    "store_category_imputed": "Category filled in",
    "dasher_telemetry_missing": "Dispatch telemetry missing",
}


def format_feature_value(feature: str, value) -> str:
    """Readable value for display next to a feature name."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if feature == "store_target_enc":
        return f"{float(value) / 60:.1f} min"
    if feature in ("store_category_imputed", "dasher_telemetry_missing"):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)


class DeliveryOpsEngine:
    """Predicts the P10 / P50 / P90 delivery time of an order from raw operational inputs."""

    def __init__(self, bundle_path: Optional[Path] = None):
        self.bundle = joblib.load(Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH)

        self.models = self.bundle["models"]
        self.features = list(self.bundle["features"])
        self.levels = self.bundle["categorical"]
        self.serving_notes = list(self.bundle.get("serving_notes", []))

        encoding = self.bundle["store_encoding"]
        self.store_table = encoding["table"]                    # per store: train seconds, train orders, smoothed mean
        self.store_prior = float(encoding["prior"])             # training mean (seconds), used for unseen stores
        self.store_smoothing = encoding["smoothing"]
        self._store_enc = self.store_table["enc"].to_dict()

        self._explainer = None

    # ------------------------------------------------------------------ features
    def build_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Rebuild the model's feature matrix from raw order columns.

        Required columns: market_id, order_protocol, store_primary_category and the numeric inputs in NUMERIC_INPUTS
        (the three dasher columns may be NaN). Optional: store_id (a store not in the encoding table gets the
        training mean) and store_category_imputed (defaults to category == 'unknown').
        """
        X = pd.DataFrame(index=orders.index)

        for col in ("market_id", "order_protocol"):
            values = pd.to_numeric(orders[col]).astype(int)
            unknown = sorted(set(values) - set(self.levels[col]))
            if unknown:
                raise ValueError(f"{col} {unknown} was not seen in training (valid: {self.levels[col]}).")
            X[col] = pd.Categorical(values, categories=self.levels[col])

        category = orders["store_primary_category"].astype(str)
        category = category.where(category.isin(self.levels["store_primary_category"]), "unknown")
        X["store_primary_category"] = pd.Categorical(category, categories=self.levels["store_primary_category"])

        for col in NUMERIC_INPUTS:
            X[col] = pd.to_numeric(orders[col], errors="coerce").astype(float)

        # Load ratios: per on-shift dasher, undefined (missing) when nobody is on shift or telemetry is missing
        denominator = X["total_onshift_dashers"].where(X["total_onshift_dashers"] > 0)
        X["busy_dasher_ratio"] = X["total_busy_dashers"] / denominator
        X["outstanding_order_ratio"] = X["total_outstanding_orders"] / denominator

        # Store encoding from the training-period table; a store without history gets the training mean
        if "store_id" in orders.columns:
            X["store_target_enc"] = orders["store_id"].map(self._store_enc).fillna(self.store_prior).astype(float)
        else:
            X["store_target_enc"] = self.store_prior

        if "store_category_imputed" in orders.columns:
            X["store_category_imputed"] = orders["store_category_imputed"].astype(bool)
        else:
            X["store_category_imputed"] = (category == "unknown")
        X["dasher_telemetry_missing"] = X[DASHER_COLUMNS].isna().any(axis=1)

        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"Feature contract mismatch, missing: {missing}")
        return X[self.features]

    def _cast_stored_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Cast a stored feature table (int-coded categoricals) to the dtypes the models were trained on."""
        X = frame[self.features].copy()
        for col in ("market_id", "order_protocol"):
            X[col] = pd.Categorical(pd.to_numeric(frame[col]).astype(int), categories=self.levels[col])
        X["store_primary_category"] = pd.Categorical(
            frame["store_primary_category"].astype(str), categories=self.levels["store_primary_category"])
        return X

    # --------------------------------------------------------------- predictions
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """P10 / P50 / P90 in seconds, each row sorted so that P10 <= P50 <= P90."""
        raw = np.column_stack([self.models[k].predict(X) for k in QUANTILE_KEYS])
        p10, p50, p90 = np.sort(raw, axis=1).T
        return pd.DataFrame({"p10_s": p10, "p50_s": p50, "p90_s": p90}, index=X.index)

    def predict_minutes(self, orders: pd.DataFrame) -> pd.DataFrame:
        pred = self.predict(self.build_features(orders)) / 60
        return pred.rename(columns={"p10_s": "p10_min", "p50_s": "p50_min", "p90_s": "p90_min"})

    def simulate_eta(self, order: dict) -> dict:
        """Estimate one order given a dict of raw inputs. Returns minutes."""
        pred = self.predict_minutes(pd.DataFrame([order])).iloc[0]
        return {
            "p10_min": round(float(pred["p10_min"]), 1),
            "p50_min": round(float(pred["p50_min"]), 1),
            "p90_min": round(float(pred["p90_min"]), 1),
            "spread_min": round(float(pred["p90_min"] - pred["p10_min"]), 1),
        }

    def sweep(self, order: dict, field: str, values: Iterable) -> pd.DataFrame:
        """What-if: the same order with one raw input varied. Returns P10 / P50 / P90 in minutes per value."""
        values = list(values)
        orders = pd.DataFrame([order] * len(values))
        orders[field] = values
        out = self.predict_minutes(orders)
        out.insert(0, field, values)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------- stores
    def store_summary(self, store_id) -> Optional[dict]:
        """Training history of a store, or None when the store has no history."""
        if store_id is None or store_id not in self.store_table.index:
            return None
        row = self.store_table.loc[store_id]
        n_orders = int(row["k_n"])
        return {
            "orders": n_orders,
            "avg_min": float(row["k_sum"]) / n_orders / 60,
            "encoding_min": float(row["enc"]) / 60,
            "tier": store_history_tier(n_orders),
        }

    # ------------------------------------------------------------- explanations
    def explain(self, order: dict, top_n: int = 8) -> dict:
        """SHAP breakdown of the P50 estimate for one order, in minutes (requires the `shap` package)."""
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.models["p50"])
        X = self.build_features(pd.DataFrame([order]))
        values = self._explainer(X)
        shap_min = np.asarray(values.values)[0] / 60
        base_min = float(np.ravel(values.base_values)[0]) / 60

        rows = pd.DataFrame({
            "feature": self.features,
            "label": [FEATURE_LABELS.get(f, f) for f in self.features],
            "value": [format_feature_value(f, X[f].iloc[0]) for f in self.features],
            "shap_min": shap_min,
        })
        rows["abs"] = rows["shap_min"].abs()
        rows = rows.sort_values("abs", ascending=False)
        top, rest = rows.head(top_n), rows.iloc[top_n:]
        if len(rest):
            top = pd.concat([top, pd.DataFrame([{
                "feature": "other", "label": f"{len(rest)} other features", "value": "",
                "shap_min": rest["shap_min"].sum(), "abs": abs(rest["shap_min"].sum()),
            }])], ignore_index=True)
        return {"base_min": base_min, "prediction_min": base_min + float(shap_min.sum()),
                "contributions": top.drop(columns="abs").reset_index(drop=True)}

    # ------------------------------------------------------------------- parity
    def parity_check(self, table: pd.DataFrame, n: int = 500, seed: int = 42, tol: float = 1e-6) -> dict:
        """Rebuild the features of test-week orders from raw columns and compare them with the saved feature table.

        Features built at serving time must equal the ones the models were trained and evaluated on, and the models
        must give the same predictions on both. Test-week rows are used because their stored store encoding is the
        same full-training lookup the app uses (training rows carry a leave-one-day-out value).
        """
        test = table[table["split"] == "test"]
        sample = test.sample(min(n, len(test)), random_state=seed)

        rebuilt = self.build_features(sample)
        stored = self._cast_stored_features(sample)

        mismatched = []
        for col in self.features:
            a, b = rebuilt[col], stored[col]
            if col in self.levels:
                same = bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
            else:
                same = bool(np.allclose(a.astype(float), b.astype(float), atol=tol, rtol=1e-6, equal_nan=True))
            if not same:
                mismatched.append(col)

        max_diff_s = float(np.abs(self.predict(rebuilt).to_numpy() - self.predict(stored).to_numpy()).max())
        return {"ok": (not mismatched) and max_diff_s < 1e-3, "n_rows": int(len(sample)),
                "mismatched_features": mismatched, "max_prediction_diff_s": max_diff_s}


def store_history_tier(n_orders: int) -> str:
    """Segment label used in the model card for the number of training orders behind a store."""
    if n_orders <= 0:
        return "unseen (0)"
    if n_orders <= THIN_STORE_MAX_ORDERS:
        return "thin (1-24)"
    return "established (25+)"


if __name__ == "__main__":
    engine = DeliveryOpsEngine()
    example = {
        "market_id": 1, "store_id": None, "store_primary_category": "mexican", "order_protocol": 1,
        "total_items": 3, "subtotal": 2500, "num_distinct_items": 3, "min_item_price": 500, "max_item_price": 1200,
        "total_onshift_dashers": 50, "total_busy_dashers": 45, "total_outstanding_orders": 60,
        "estimated_order_place_duration": 446, "estimated_store_to_consumer_driving_duration": 600,
        "order_hour": 18, "order_day_of_week": 4,
    }
    result = engine.simulate_eta(example)
    print(f"ETA window: {result['p10_min']} to {result['p90_min']} min (expected {result['p50_min']} min)")