"""Inference pipeline for FoodOps.AI - DemandOps module.

Provides model loading, input validation, and single/batch demand predictions
using the trained LightGBM model bundle.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

DEFAULT_BUNDLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "lgb_model_bundle.joblib"
)


def load_model_bundle(bundle_path: str = DEFAULT_BUNDLE_PATH) -> Dict[str, Any]:
    """Load the trained LightGBM model bundle artifact."""
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found at: {bundle_path}")
    bundle = joblib.load(bundle_path)
    return bundle


def predict_single(
    bundle: Dict[str, Any],
    input_data: Dict[str, Any]
) -> Dict[str, float]:
    """Run inference for a single meal-center-week record.

    Parameters
    ----------
    bundle : dict
        Loaded artifact containing model, feature list, and categorical columns.
    input_data : dict
        Dictionary containing all required feature values.

    Returns
    -------
    dict
        Dictionary containing predicted log demand, predicted raw orders,
        and estimated operational safety buffer.
    """
    model = bundle["model"]
    feature_cols = bundle["features"]
    cat_cols = bundle["cat_cols"]

    # Construct single-row DataFrame
    row_df = pd.DataFrame([input_data])

    # Ensure all required features are present
    missing = set(feature_cols) - set(row_df.columns)
    if missing:
        raise ValueError(f"Missing required features for inference: {missing}")

    X = row_df[feature_cols].copy()

    # Enforce categorical dtypes
    for col in cat_cols:
        X[col] = X[col].astype("category")

    pred_log = float(model.predict(X)[0])
    pred_orders = float(np.clip(np.expm1(pred_log), 0, None))

    # Calculate operational inventory recommendations (+15% buffer)
    safety_buffer = float(np.ceil(pred_orders * 1.15))

    return {
        "pred_log": round(pred_log, 4),
        "predicted_orders": round(pred_orders, 1),
        "safety_stock_prep": safety_buffer,
        "lower_bound_80": round(max(0.0, pred_orders * 0.85), 1),
        "upper_bound_80": round(pred_orders * 1.15, 1),
    }


def simulate_price_sensitivity(
    bundle: Dict[str, Any],
    base_input: Dict[str, Any],
    price_pct_range: Tuple[float, float] = (-0.30, 0.30),
    steps: int = 21
) -> pd.DataFrame:
    """Generate a demand elasticity curve across varying checkout prices."""
    base_price = float(base_input["base_price"])
    pct_changes = np.linspace(price_pct_range[0], price_pct_range[1], steps)

    records = []
    for pct in pct_changes:
        sim_price = round(base_price * (1.0 + pct), 2)
        sim_input = base_input.copy()
        sim_input["checkout_price"] = sim_price
        sim_input["price_change_pct"] = (sim_price - base_price) / base_price

        pred = predict_single(bundle, sim_input)
        records.append({
            "checkout_price": sim_price,
            "pct_from_base": round(pct * 100, 1),
            "predicted_orders": pred["predicted_orders"],
            "expected_revenue": round(sim_price * pred["predicted_orders"], 2)
        })

    return pd.DataFrame(records)


def simulate_promo_scenarios(
    bundle: Dict[str, Any],
    base_input: Dict[str, Any]
) -> pd.DataFrame:
    """Simulate order impact across promotional channels (Email vs Homepage)."""
    scenarios = [
        {"name": "No Promotion", "email": 0, "home": 0},
        {"name": "Emailer Only", "email": 1, "home": 0},
        {"name": "Homepage Featured Only", "email": 0, "home": 1},
        {"name": "Both Email & Homepage", "email": 1, "home": 1},
    ]

    results = []
    base_pred_orders = None

    for sc in scenarios:
        sim_input = base_input.copy()
        sim_input["emailer_for_promotion"] = sc["email"]
        sim_input["homepage_featured"] = sc["home"]

        pred = predict_single(bundle, sim_input)
        orders = pred["predicted_orders"]

        if base_pred_orders is None:
            base_pred_orders = orders

        uplift_pct = ((orders - base_pred_orders) / base_pred_orders * 100) if base_pred_orders > 0 else 0.0

        results.append({
            "Scenario": sc["name"],
            "Email Promo": "Yes" if sc["email"] else "No",
            "Homepage": "Yes" if sc["home"] else "No",
            "Predicted Orders": orders,
            "Uplift vs Baseline (%)": round(uplift_pct, 1)
        })

    return pd.DataFrame(results)
