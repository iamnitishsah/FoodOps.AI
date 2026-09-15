# DemandOps — Fulfillment Demand Forecasting

**DemandOps** is the demand intelligence module of FoodOps.AI. It forecasts weekly meal demand across a decentralized fulfillment center network, enabling kitchen managers and operations teams to optimize inventory procurement, minimize food spoilage, and balance kitchen labor.

---

## 1. Problem Formulation

- **Goal:** Predict weekly order demand (`num_orders`) for each `(center_id, meal_id)` pair for upcoming operational weeks.
- **Horizon:** 10 weeks forward out-of-sample (weeks 146 to 155).
- **Core Challenge:** Over-predicting causes perishable food waste and excessive storage overhead; under-predicting leads to kitchen stockouts, order cancellations, and delayed dispatch.
- **Target Transformation:** $\log(1 + \texttt{num\_orders})$ to stabilize right-skewed demand variance and heavy-tail spikes. Predictions are inverted back to raw order counts via $\max(0, \exp(\hat{y}_{\log}) - 1)$.

---

## 2. Model Architecture & Benchmark Scoreboard

Models were trained strictly on **historical weeks 1 to 131** and evaluated against an unseen **temporal hold-out validation set (weeks 132 to 145)** to simulate production deployment without lookahead bias.

### Final Validation Scoreboard:

| Model Architecture | Model Family | Validation WAPE (%) | Validation MAPE (%) | Role / Status |
| :--- | :--- | :---: | :---: | :--- |
| **LightGBM Regressor** | Gradient Boosted Trees | **28.75%** | **44.14%** | **Champion (Production Deployed)** |
| **Simple LSTM** | Deep Learning (PyTorch) | **29.98%** | **44.02%** | Challenger Model |
| **Ridge Regression** | L2-Regularized Linear | **35.72%** | **46.98%** | Linear Baseline |
| **Naive Lag-1 Baseline** | Heuristic | **41.96%** | **67.23%** | Lower Bound Benchmark |

### Why WAPE is the Primary Supply Chain KPI:
$$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100$$
Standard MAPE divides each error by actual orders ($|y - \hat{y}| / y$), which artificially blows up when order counts are very small (e.g. 1 or 2 orders). WAPE weights errors by true order volume, reflecting the actual operational and financial impact on the kitchen.

---

## 3. Autoregressive Multi-Step Roll-Forward Engine

The raw test dataset (`data/raw/test.csv`) contains 32,573 rows across weeks 146 to 155. Because future lags (`lag_1`, `lag_2`, `lag_4`, `rolling_mean_4`) are unknown for weeks beyond week 146:

1. **State Maintenance:** An in-memory state dictionary maintains the latest demand counts per `(center_id, meal_id, week)`.
2. **Sequential Roll-Forward:** For each week $w \in [146, 155]$:
   - Dynamic lags and 4-week rolling statistics are computed from the active state.
   - LightGBM predicts $\hat{y}_{\log}$, which is un-scaled and clipped at 0.
   - $\hat{y}_{\log}$ is fed back into the state dictionary for week $w$, allowing week $w+1$ to use it as its `lag_1`.
3. **Audit Results:**
   - 32,573 rows predicted with 0 missing or negative values.
   - Exported to [`data/processed/submission_lgb.csv`](./data/processed/submission_lgb.csv).

---

## 4. Interactive Streamlit Dashboard

The interactive operations dashboard is located at [`app/streamlit_app.py`](app/streamlit_app.py).

### Core Features:
1. **🔮 Live Demand Predictor & What-If Simulator:**
   - Select any of the 77 fulfillment centers and 51 meal catalog items.
   - Automatically loads the latest known historical baseline for that center-meal pair.
   - Pricing controls: Base price, checkout price, and dynamic discount/markup calculation.
   - Promotional toggles: Email campaign (`emailer_for_promotion`) and App homepage banner (`homepage_featured`).
   - Calibrated inventory safety buffer recommendation (+15% prep stock).
   - Real-time **Price Elasticity Curve** sweeping -30% to +30% price variations.
   - **Promotional Channel Lift** simulator comparing standalone and combined campaign uplifts.
2. **📊 Model Benchmarks & Metrics:**
   - Interactive comparison of all 4 architectures on WAPE and MAPE.
   - Feature importance breakdown (splits and gain) from the champion LightGBM model.
   - Metric methodology deep dive.
3. **🍲 Product & Operational Analytics:**
   - Portfolio breakdown across 119.5M historical orders, 14 categories, and 4 cuisines.
   - Category demand distribution and cuisine share.
   - Multi-year weekly platform order trends.
4. **📋 Out-of-Sample Test Forecasts:**
   - Summary statistics of the 10-week forward test predictions.
   - Interactive data table and one-click download for `submission_lgb.csv`.

---

## 5. How to Run the App

From the root repository directory:

```bash
# Activate your environment
source .venv/bin/activate

# Launch the Streamlit application
streamlit run DemandOps/app/streamlit_app.py
```

The application will open automatically at `http://localhost:8501`.

---

## 6. Directory Layout

```
DemandOps/
├── README.md                      ← this file
├── app/
│   ├── streamlit_app.py           ← Streamlit operations application
│   └── theme.py                   ← Streamlit theme configuration
├── data/
│   ├── raw/                       ← fulfilment_center_info.csv, meal_info.csv, train.csv, test.csv
│   └── processed/                 ← processed_data.csv, latest_known_state.csv, submission_lgb.csv
├── models/
│   ├── lgb_model.joblib           ← trained LightGBM model
│   └── lgb_model_bundle.joblib    ← full model bundle (model, feature contract, categoricals, metrics)
├── notebooks/
│   ├── EDA.ipynb                  ← exploratory data analysis
│   ├── feature_engineering.ipynb  ← full panel generation & temporal feature engineering
│   ├── model_training.ipynb       ← 4-model benchmark training (Naive, Ridge, LightGBM, LSTM)
│   └── test_evaluation.ipynb      ← autoregressive roll-forward test evaluation
└── src/
    ├── __init__.py
    ├── data_loader.py             ← cached loaders for metadata, states & summaries
    └── inference.py               ← single-row inference, price elasticity & promo simulations
```
