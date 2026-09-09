import os
import joblib
import numpy as np
import pandas as pd

MODEL_BUNDLE_PATH = "models/delivery_model_bundle.joblib"

class DeliveryOpsEngine:
    """
    Inference engine for DeliveryOps.
    Loads the serialized model bundle and handles real-time feature transformations.
    """

    def __init__(self, bundle_path=MODEL_BUNDLE_PATH):
        # Resolve path dynamically depending on where the script is called from
        if not os.path.exists(bundle_path):
            bundle_path = os.path.join(os.path.dirname(__file__), "../../DeliveryOps/models/delivery_model_bundle.joblib")

        self.bundle = joblib.load(bundle_path)

        # Unpack the bundle
        self.model_p10 = self.bundle["model_p10"]
        self.model_p50 = self.bundle["model_p50"]
        self.model_p90 = self.bundle["model_p90"]
        self.category_map = self.bundle["store_category_map"]
        self.global_mean = self.bundle["global_category_mean"]
        self.feature_names = self.bundle["feature_names"]

    def simulate_eta(self, input_data: dict) -> dict:
        """
        Accepts a dictionary of raw operational variables, applies feature engineering,
        and returns the P10, P50, and P90 ETA predictions in minutes.
        """
        # 1. Convert raw input to DataFrame
        df = pd.DataFrame([input_data])

        # 2. Marketplace Telemetry & Congestion Features
        onshift = df["total_onshift_dashers"].iloc[0]
        if onshift == 0:
            # Fallback caps derived from EDA 99th percentiles for onshift=0 cases
            df["busy_dasher_ratio"] = 3.929
            df["outstanding_order_ratio"] = 4.571
        else:
            df["busy_dasher_ratio"] = df["total_busy_dashers"] / onshift
            df["outstanding_order_ratio"] = df["total_outstanding_orders"] / onshift

        # 3. Target Encoding for Category
        cat = df["store_primary_category"].iloc[0]
        df["store_category_target_enc"] = self.category_map.get(cat, self.global_mean)

        # 4. Cyclical Temporal Encodings
        hour = df["order_hour"].iloc[0]
        dow = df["order_day_of_week"].iloc[0]

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # 5. Default Imputation Flags (Since UI inputs are explicit, nothing is missing)
        df["total_onshift_dashers_imputed"] = False
        df["total_busy_dashers_imputed"] = False
        df["total_outstanding_orders_imputed"] = False

        # 6. Categorical Formatting for LightGBM
        for col in ["market_id", "order_protocol"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # 7. Strict Feature Contract Alignment
        # This prevents crashes by ensuring the UI columns match the training columns exactly
        X = df[self.feature_names]

        # 8. Predict & Reverse Log Transform (converting log seconds -> minutes)
        p10_sec = np.expm1(self.model_p10.predict(X)[0])
        p50_sec = np.expm1(self.model_p50.predict(X)[0])
        p90_sec = np.expm1(self.model_p90.predict(X)[0])

        return {
            "p10_min": round(p10_sec / 60, 1),
            "eta_min": round(p50_sec / 60, 1),
            "p90_min": round(p90_sec / 60, 1),
            "spread_min": round((p90_sec - p10_sec) / 60, 1)  # Uncertainty width
        }


if __name__ == "__main__":
    # Quick local test to verify the engine compiles and runs
    engine = DeliveryOpsEngine()
    test_input = {
        "market_id": 1.0,
        "store_primary_category": "mexican",
        "order_protocol": 1.0,
        "total_items": 3,
        "subtotal": 2500,
        "num_distinct_items": 3,
        "min_item_price": 500,
        "max_item_price": 1200,
        "total_onshift_dashers": 50,
        "total_busy_dashers": 45,
        "total_outstanding_orders": 60,
        "estimated_order_place_duration": 251,
        "estimated_store_to_consumer_driving_duration": 600,
        "order_hour": 19,
        "order_day_of_week": 5
    }

    res = engine.simulate_eta(test_input)
    print(f"Test Inference Success! ETA Window: {res['p10_min']} - {res['p90_min']} minutes")