# FoodOps.AI

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-189B4C?logoColor=white)](https://lightgbm.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**FoodOps.AI** is an end-to-end food delivery machine learning platform built around four independent, self-contained operational intelligence modules: **demand forecasting**, **delivery time (ETA) prediction**, **personalized menu recommendation**, and **experimentation / causal inference**.

Each module solves a high-impact operational decision problem using real-world or realistically simulated data, progressing from exploratory data analysis and simple baselines to stronger machine learning models and interactive dashboards.

### 📊 DemandOps

> 🚀 **Live Cloud Application**  
> [https://foodops-demand.streamlit.app/](https://foodops-demand.streamlit.app/)

---

### 🛵 DeliveryOps

> 🚀 **Live Cloud Application**  
> [https://foodops-delivery.streamlit.app/](https://foodops-delivery.streamlit.app/)

---

## Module Status Overview

| Module | Operational Domain | Primary Modeling Paradigm | Benchmark KPI | Status | Live Console / Documentation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Module 1: DemandOps** | Weekly Fulfillment Demand Forecasting | LightGBM GBDT, PyTorch LSTM, Ridge | **28.61% WAPE** | 🟢 **Complete & Deployed** | [Live App](https://foodops-demand.streamlit.app/) · [DemandOps README](./DemandOps/README.md) |
| **Module 2: DeliveryOps** | Real-Time Order Delivery ETA Prediction | LightGBM (L1 + Quantile / Pinball Loss), Ridge | MAE: **9.97 min** · P90 Coverage: **88.2%** | 🟢 **Complete & Deployed** | [Live App](https://foodops-delivery.streamlit.app/) · [DeliveryOps README](./DeliveryOps/README.md) |
| **Module 3: PersonalizeOps** | Implicit Feedback Dish & Restaurant Ranking | Matrix Factorization & Two-Tower Embeddings | NDCG@10 / Recall@10 | ⚪ Queued | [PersonalizeOps README](./PersonalizeOps/README.md) |
| **Module 4: Experimentation** | A/B Test Harness & Causal Inference | Frequentist/Bayesian Testing, DiD, Matching | SRM / Uplift / Power | ⚪ Queued | [Experimentation README](./Experimentation/README.md) |

---

## Table of Contents

- [What This Repository Is](#what-this-repository-is)
- [Module 1: DemandOps (Demand Forecasting)](#module-1-demandops-demand-forecasting)
- [Module 2: DeliveryOps (ETA Prediction)](#module-2-deliveryops-eta-prediction)
- [Module 3: PersonalizeOps (Recommendation Engine)](#module-3-personalizeops-recommendation-engine)
- [Module 4: Experimentation (Experimentation Framework)](#module-4-experimentation-experimentation-framework)
- [Repository Structure](#repository-structure)
- [Why Modules Are Independent](#why-modules-are-independent)
- [Build Order](#build-order)
- [Tech Stack](#tech-stack)

---

## What This Repository Is

A food-delivery platform (riders, restaurants, orders, zones) generates distinct operational decisions that data science can inform:
1. **How much food demand to expect** at each fulfillment center to prevent spoilage and kitchen stockouts (*Time Series Forecasting*).
2. **How long a delivery will take** to set realistic customer expectations and dispatch couriers efficiently (*Tabular Regression & Uncertainty Estimation*).
3. **What dishes and restaurants to recommend** to optimize customer conversion (*Implicit Recommendation & Deep Ranking*).
4. **Whether a commercial or product change actually worked** without being misled by confounders or sample ratio mismatch (*Causal Inference & A/B Testing*).

Because these four decision types belong to fundamentally different problem families, FoodOps.AI structures them as independent modules rather than forcing an artificial unified abstraction.

Each module:
- Formulates a concrete operational problem with domain-specific KPIs.
- Uses public benchmark datasets or rigorous simulations reflecting real operational scale.
- Builds an incremental modeling progression (Heuristic $\to$ Linear $\to$ Deep Learning $\to$ Gradient Boosted Trees).
- Deploys an interactive Streamlit operations application allowing non-technical stakeholders to simulate scenarios and audit benchmarks.

---

## Module 1: DemandOps (Demand Forecasting)

**Status:** 🟢 **Complete & Live Deployed**  
**Detailed Documentation:** [`DemandOps/README.md`](./DemandOps/README.md)  
**Interactive Live Console:** [https://foodops-demand.streamlit.app/](https://foodops-demand.streamlit.app/)

### Operational Problem
Decentralized fulfillment centers face a severe tradeoff: over-predicting weekly dish demand leads to perishable food waste and cold-chain storage overload; under-predicting causes kitchen stockouts, canceled orders, and delayed deliveries. DemandOps forecasts weekly order volume (`num_orders`) for each `(center_id, meal_id)` combination across a decentralized fulfillment center network (77 hubs, 51 dishes, 3,597 active hub–dish pairs, 14 categories, 4 cuisines, ≈119.6M historical orders).

- **Primary Dataset Source:** [Food Demand Forecasting (Kaggle)](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting)

### Technical Highlights
- **Chronological Validation:** Trained on weeks 1 to 131 and scored on held-out weeks 132 to 145 (14 continuous weeks), so no model is evaluated on weeks it could have learned from.
- **Full Weekly Grid:** The panel is expanded to a complete (hub, dish, week) grid of 3,597 pairs × 145 weeks so every lag refers to the previous *calendar* week. The 12.5% of cells with no recorded order are zero-filled, a documented modeling assumption.
- **Supply Chain Feature Engineering (20 features):** Autoregressive lags (`lag_1`, `lag_2`, `lag_4`), 4-week rolling mean and standard deviation (shifted one week before rolling), signed price change (`price_change_pct`), promotional flags (`emailer_for_promotion`, `homepage_featured`), `week_of_year`, and hub/dish metadata.
- **Target Transformation:** Trained on $\log(1 + \mathrm{num\_orders})$ with an L1 objective, with inverted non-negative predictions $\max(0, \exp(\hat{y}) - 1)$.
- **Beyond WAPE:** MAPE and total-volume **bias** are tracked for every model, since two models with similar WAPE can carry very different stockout risk.
- **Training/Serving Parity Check:** The dashboard rebuilds lag, rolling, calendar and price features from history at start-up, verifies them against the saved feature table, and refuses to run on a mismatch.
- **10-Week Planning Dashboard:** Editable price and promotion plan, recursive 10-week forecast with prep quantities, next-week price and promotion what-if analysis, and portfolio analytics.

### Model Benchmark Scoreboard

| Rank | Model Architecture | Model Family | Validation WAPE (%) | Validation MAPE (%) | Bias (%) | Operational Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **LightGBM (L1 objective)** | Gradient Boosted Decision Trees | **28.61%** | **43.84%** | −6.37% | **Champion (Deployed)** |
| 🥈 | **LSTM** | Deep Learning (PyTorch) | 29.85% | 50.79% | −1.70% | Challenger Model |
| 🥉 | **Ridge Regression** | L2-Regularized Linear Model | 35.42% | 48.91% | −3.19% | Linear Baseline |
| 4 | **Naive Lag-1 Baseline** | Heuristic Benchmark | 41.96% | 67.23% | +2.08% | Lower Bound Benchmark |

*Primary KPI: Volume-Weighted Absolute Percentage Error (WAPE) — weights prediction errors by order volume so high-demand kitchen staples are prioritized over low-volume outliers. Bias is total forecast versus total actual (negative = under-forecast). All scores are one-week-ahead on validation weeks 132–145; the dashboard's multi-week plan is recursive, and its accuracy beyond week 1 is not yet backtested. See [Evaluation Notes & Limitations](./DemandOps/README.md#7-evaluation-notes--limitations).*

---

## Module 2: DeliveryOps (ETA Prediction)

**Status:** 🟢 **Complete & Live Deployed**  
**Detailed Documentation / Folder:** [`DeliveryOps/README.md`](./DeliveryOps/README.md)  
**Interactive Live Console:** [https://foodops-delivery.streamlit.app/](https://foodops-delivery.streamlit.app/)

### Operational Problem
Predict total order delivery duration (order placed → customer doorstep) from the order, the store, the market, live dispatch load (on-shift, busy and outstanding dashers), the platform's drive-time estimate and the local time of day. DeliveryOps emphasizes uncertainty-aware promise windows: it predicts the P10, P50 and P90 so operations can show customers an expected time with an honest range, and it reports the coverage those ranges actually deliver. Scope: 192,730 cleaned orders across 6 markets and 6,725 stores, Jan 21 – Feb 17, 2015.

- **Primary Dataset Source:** [DoorDash ETA Prediction (Kaggle)](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction)

### Technical Highlights
- **Chronological Validation:** Trained on Jan 21 to Feb 10 and scored once on the held-out Feb 11 to 17 test week (51,542 orders, including Valentine's weekend). Every model choice was made on two forward 7-day folds inside the training period, so the test week never influenced selection.
- **Scoped, Audited Cleaning:** Deliveries restricted to 15 min to 2 h; 4,698 of 197,428 rows removed (2.38%), each step logged. Dasher-telemetry gaps (8.25% of rows) are kept as missing with an explicit flag rather than imputed.
- **Leakage-Safe Store Target Encoding:** A smoothed store average delivery time (weight 5), built leave-one-day-out for training rows and recomputed inside each validation fold; stores never seen in training fall back to the training mean.
- **Dispatch-Load Ratios:** `busy_dasher_ratio` and `outstanding_order_ratio` (per on-shift dasher) are left missing when undefined and handled natively by LightGBM.
- **Ablation-Driven Feature Set:** 32 candidate features reduced to 20 through one-at-a-time swaps judged against a 3-seed noise floor and per-day paired comparisons.
- **P10/P50/P90 Quantile Models:** An L1 median model plus pinball-loss P10 and P90 models, with boosting rounds read from validation curves. Intervals are not calibrated; raw coverage is reported.
- **Explainability:** SHAP on the P50 model (top drivers: `outstanding_order_ratio`, the drive-time estimate and `store_target_enc`).

### Model Benchmark Scoreboard

| Rank | Model Architecture | Family | Test MAE (min) | P90 SLA Coverage | Status | Key Characteristics |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **LightGBM (P10/P50/P90)** | Gradient Boosted Trees | **9.97** | **88.19%** | **Champion (Deployed)** | L1 median plus pinball quantiles; 20 features, raw-seconds target, native categoricals and missing values; no hyperparameter search. |
| 🥈 | **Ridge Regression** | L2-Regularized Linear Model | 10.28 | N/A | Linear Baseline | Log target, one-hot hour, α = 1000 picked on the forward folds. |
| 🥉 | **Market × Hour Median** | Heuristic Lookup | 12.26 | N/A | Lookup Baseline | Median delivery time per (market, order hour). |
| 4 | **Naive Median Baseline** | Heuristic Benchmark | 13.05 | N/A | Lower Bound | Predicts the training median (43.9 min). |

*All scores come from the single unseen Feb 11–17 test week; the champion is 23.6% below the naive baseline. Intervals are raw (not calibrated): P10 coverage 9.1%, P90 coverage 88.2%, 80% interval coverage 79.1%, mean P10–P90 width 30.6 min. Validation scores, the feature ablation and segment results are in the [DeliveryOps README](./DeliveryOps/README.md#6-models--benchmark-scoreboard).*


---

## Module 3: PersonalizeOps (Recommendation Engine)

**Status:** ⚪ Queued  
**Folder:** [`PersonalizeOps/`](./PersonalizeOps/README.md)

### Operational Problem
Rank dishes and restaurants for users based on sparse, implicit interaction histories (orders, re-orders, impressions).

### What's Involved:
- Implicit interaction matrix construction and popularity baselines.
- Classical collaborative filtering (Matrix Factorization / Implicit ALS).
- Deep two-tower neural network in PyTorch (User Tower + Dish/Restaurant Tower).
- Ranking evaluation: NDCG@k, Precision@k, Recall@k, and coverage.
- Cold-start handling for new consumers and newly onboarded kitchens.

---

## Module 4: Experimentation (Experimentation Framework)

**Status:** ⚪ Queued  
**Folder:** [`Experimentation/`](./Experimentation/README.md)

### Operational Problem
Evaluate whether product interventions (pricing changes, algorithm updates, delivery fee adjustments) drive statistically significant and causal business improvements.

### What's Involved:
- Minimum detectable effect (MDE) and sample size calculations.
- Rigorous hypothesis testing with guardrails against common experimentation pitfalls (peeking, Sample Ratio Mismatch, multiple comparisons).
- Observational causal inference (Difference-in-Differences, Propensity Score Matching) for network-wide policy changes where randomized A/B splits cannot be isolated.
- Interactive experiment diagnostics dashboard in Streamlit.

---

## Repository Structure

```
FoodOps.AI/
├── README.md                          ← Main platform overview (this file)
├── requirements.txt                   ← Pinned project dependencies
├── LICENSE                            ← MIT license
│
├── DemandOps/                         ← Module 1: Demand Forecasting (Complete & Live)
│   ├── README.md
│   ├── app/
│   ├── dataset/{raw,processed}/
│   ├── models/
│   ├── notebooks/
│   └── src/
│
├── DeliveryOps/                       ← Module 2: ETA Prediction (Complete & Live)
│   ├── README.md
│   ├── app/
│   ├── dataset/{raw,processed}/
│   ├── models/
│   ├── notebooks/
│   └── src/
│
├── PersonalizeOps/                    ← Module 3: Recommendation Engine (Queued)
│   ├── README.md
│   ├── app/
│   ├── data/{raw,processed}/
│   ├── models/
│   ├── notebooks/
│   └── src/
│
└── Experimentation/                   ← Module 4: Experimentation Framework (Queued)
    ├── README.md
    ├── app/
    ├── data/{raw,processed}/
    ├── notebooks/
    └── src/
```

---

## Why Modules Are Independent

Each module maintains its own dedicated dataset folder, `models/`, `notebooks/`, `src/`, and `app/` subdirectories. This architecture ensures:
- **Zero Coupling:** Each operational domain can be explored, run, trained, and audited in complete isolation.
- **No Premature Abstraction:** Machine learning problems in time series, spatial regression, collaborative filtering, and causal inference require distinct schemas, preprocessing pipelines, and evaluation harnesses.
- **Standalone Deployment:** Every module’s Streamlit app runs independently and is deployed on its own; the landing page simply links to each one.

---

## Build Order

Development proceeds across four sequential operational stages:

1. **DemandOps (Demand Forecasting)** — **Complete & Live Deployed**: Core supply chain foundation. Built a full weekly (hub, dish, week) grid from 145 weeks of fulfillment transactions, engineered lag, rolling and price features, benchmarked 4 models on a chronological hold-out (**28.61% WAPE**, a 31.8% relative reduction over the naive baseline), and deployed a 10-week planning dashboard.
2. **DeliveryOps (ETA Prediction)** — **Complete & Live Deployed**: Cleaned 197,428 raw orders to 192,730, engineered 20 leakage-safe features (including a leave-one-day-out store encoding), benchmarked four models on forward-in-time folds and a held-out week, and deployed LightGBM P10/P50/P90 estimates (Test MAE **9.97 min**, 23.6% below the naive median; raw P90 coverage **88.2%**).
3. **PersonalizeOps (Recommendation Engine)** — Rank dishes and restaurants for users based on sparse, implicit interaction histories using matrix factorization and deep two-tower architectures.
4. **Experimentation (A/B Testing & Causal Inference)** — Establish the statistical experimentation harness to evaluate changes (recommendations, pricing, dispatch rules) with guardrail checks against sample ratio mismatch and novelty effects.


---

## Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Core Language** | Python 3.12 |
| **Data Processing & Storage** | pandas, numpy, pyarrow, joblib |
| **Classical ML & Gradient Boosting** | LightGBM, scikit-learn, SHAP (explainability) |
| **Deep Learning** | PyTorch (LSTM sequence modeling) |
| **Interactive Applications** | Streamlit, Plotly Express, Plotly Graph Objects |
| **Experimentation & Statistics** | SciPy, statsmodels |
| **Deployment & Hosting** | Streamlit Community Cloud, GitHub |

---