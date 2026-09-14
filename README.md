# FoodOps.AI

A food-delivery data platform built around four independent data-science modules: demand forecasting, delivery time (ETA) prediction, restaurant/menu recommendation, and A/B test analysis. Each module solves one self-contained problem using public or simulated data, with its own dataset, models, and evaluation.

This document describes what the project is, what problem each module solves, and how the repository is organized. Implementation details (exact dataset, features, models, metrics, results) for each module live in that module's own `README.md`, filled in as the module is built.

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
- [Planned Integration Layer](#planned-integration-layer)
- [Tech Stack](#tech-stack)
- [Status](#status)

---

## What This Repository Is

A food-delivery platform (riders, restaurants, orders, zones) generates several distinct types of decisions that data can inform: how much demand to expect, how long a delivery will take, what to recommend to a given user, and whether a given change actually improved something. These four decision types don't share a common model or method — they're different problem families (time series, regression, recommendation, causal inference) — so this repo treats them as four separate modules rather than forcing one unified approach.

Each module:
- States a concrete problem (input → output, and what "good" means for that output)
- Uses a public dataset or a simulated dataset built to resemble the real data shape
- Is built up from a simple baseline to a more complete solution, with the reasoning behind each step documented
- Exposes its results through an interactive Streamlit operations application so models, what-if scenarios, and benchmarks can be explored directly, not just read as static notebooks

---

## Module 1: DemandOps (Demand Forecasting)

**Folder:** [`DemandOps/`](./DemandOps/README.md)

**Problem:** Given historical order counts for a location and item category, predict order volume for a future period (e.g., the next several weeks). Underestimating demand leads to understaffing and slow deliveries; overestimating leads to wasted rider/inventory cost — so both the forecast and its likely error matter.

**What's involved:**
- Time-series feature engineering: lag features, rolling averages, calendar effects, price/promotion effects
- A time-based (not random) train/validation/test split — random splitting leaks future information into training for time series and produces misleadingly good validation scores
- A classical baseline (naive lag-1 forecast, regularized Ridge regression) to establish what "no real model" and linear baselines achieve
- A tree-based model (LightGBM) using engineered features, capturing non-linear pricing elasticity and promotional uplifts
- A PyTorch LSTM as a deep sequential comparison, evaluated honestly against the tree-based model
- Evaluation using volume-weighted metrics (WAPE) rather than raw RMSE or unweighted MAPE
- An autoregressive multi-step roll-forward forecasting engine for out-of-sample horizons (`test.csv`)
- An interactive Streamlit dashboard featuring live meal-center demand forecasting, price elasticity simulation, promotional channel lift, and model benchmarks

**Details (dataset, exact features, results, how it's exposed):** in the [DemandOps README](./DemandOps/README.md).

---

## Module 2: DeliveryOps (ETA Prediction)

**Folder:** [`DeliveryOps/`](./DeliveryOps/README.md)

**Problem:** Given order-level information available at the time an order is placed (distance, time of day, traffic/weather conditions, restaurant load), predict how long the delivery will take. This estimate is shown to the customer directly, so both average accuracy and where the model tends to be wrong (e.g., specific weather conditions or times of day) matter.

**What's involved:**
- Feature engineering from raw signals (e.g., computing distance from coordinates, encoding time of day, joining weather/traffic data if available)
- A regression model progression from a simple baseline to a tuned gradient-boosted model
- Error analysis by segment — checking whether errors are randomly distributed or systematically worse under specific conditions
- Calibrated prediction intervals rather than only a single point estimate, providing operational confidence bounds

**Details (dataset, features, results, exposure):** in the [DeliveryOps README](./DeliveryOps/README.md) once built.

---

## Module 3: PersonalizeOps (Recommendation Engine)

**Folder:** [`PersonalizeOps/`](./PersonalizeOps/README.md)

**Problem:** Given a history of user interactions with restaurants/items (orders, clicks, or ratings), rank restaurants/items for a given user. Interaction data here is implicit (an order signals interest but there's no explicit rating scale in most of it) and sparse (most users have not interacted with most restaurants), which shapes the whole approach.

**What's involved:**
- Building an interaction matrix from implicit feedback
- A classical collaborative-filtering / matrix-factorization approach, implemented directly to understand the mechanics
- A two-tower embedding model in PyTorch as a deep-learning approach, compared against the classical method
- Ranking-appropriate evaluation metrics (precision@k, recall@k, NDCG) rather than classification accuracy
- Handling cold-start scenarios for new users and newly onboarded restaurants

**Details (dataset, interaction data construction, results, exposure):** in the [PersonalizeOps README](./PersonalizeOps/README.md) once built.

---

## Module 4: Experimentation (Experimentation Framework)

**Folder:** [`Experimentation/`](./Experimentation/README.md)

**Problem:** Given two variants of something (e.g., a promo strategy, a UI change, a pricing change) and outcome data from users exposed to each, determine whether the observed difference reflects a real effect or noise, and by how much. Most product decisions are made this way rather than by assumption, so getting the statistical analysis right — not just running a test but understanding what can invalidate it — is the actual skill involved.

**What's involved:**
- Defining a hypothesis, primary metric, and required sample size before generating or looking at outcome data
- Correctly applying statistical tests, checking assumptions rather than applying tests blindly
- Demonstrating and diagnosing common experiment pitfalls: peeking at results early, novelty effects, sample ratio mismatch (SRM), multiple comparisons
- Observational causal inference methods for scenarios where randomized A/B splits cannot be executed

**Details (scenario simulated, tools used, methodology, results, exposure):** in the [Experimentation README](./Experimentation/README.md) once built.

---

## Repository Structure

Single repository, one folder per module, each module self-contained:

```
FoodOps.AI/
├── README.md                          ← this file
├── requirements.txt                   ← pinned project dependencies
├── LICENSE                            ← MIT license
│
├── DemandOps/                         ← Module 1: Demand Forecasting
│   ├── README.md                      ← module architecture, results & user guide
│   ├── app/
│   │   └── streamlit_app.py           ← interactive Streamlit operations dashboard
│   ├── data/
│   │   ├── raw/                       ← fulfillment center, meal, train & test data
│   │   └── processed/                 ← engineered panels, state caches & submission
│   ├── models/                        ← lgb_model_bundle.joblib
│   ├── notebooks/                     ← EDA, feature_engineering, model_training, test_evaluation
│   └── src/                           ← inference.py, data_loader.py
│
├── DeliveryOps/                       ← Module 2: ETA Prediction
│   ├── README.md
│   ├── app/                           ← interactive ETA prediction dashboard
│   ├── data/{raw,processed}/
│   ├── models/
│   ├── notebooks/
│   └── src/
│
├── PersonalizeOps/                    ← Module 3: Recommendation Engine
│   ├── README.md
│   ├── app/                           ← interactive ranking & discovery dashboard
│   ├── data/{raw,processed}/
│   ├── models/
│   ├── notebooks/
│   └── src/
│
├── Experimentation/                   ← Module 4: Experimentation Framework
│   ├── README.md
│   ├── app/                           ← experiment diagnostics & reporting dashboard
│   ├── data/{raw,processed}/
│   ├── notebooks/
│   └── src/
│
└── .gitignore
```

Each module exposes its results through an interactive **Streamlit** dashboard that embeds real-time inference, scenario testing, and diagnostic metrics directly with no external serving dependencies.

---

## Why Modules Are Independent

Each module has its own `data/`, `src/`, `app/`, and `notebooks/`, with no shared top-level `common/` code at this stage. This keeps each module understandable and runnable on its own — reading `DeliveryOps` doesn't require first understanding an abstraction used across all four — and avoids designing an artificial shared layer before it's clear the modules actually need one.

---

## Build Order

Development proceeds in four sequential stages, matching the operational hierarchy of a food delivery platform:

1. **DemandOps (Demand Forecasting)** — *Completed*: Foundational operational forecasting. Ingested historical fulfillment center and meal order streams, engineered time-series lag/rolling features, benchmarked 4 models (Naive, Ridge, LSTM, LightGBM), generated out-of-sample multi-step forecasts for `test.csv`, and deployed an interactive Streamlit operations dashboard.
2. **DeliveryOps (ETA Prediction)** — Predict customer delivery duration based on order timestamp, route distance, weather, and fulfillment/restaurant load with calibrated uncertainty intervals.
3. **PersonalizeOps (Recommendation Engine)** — Rank dishes and restaurants for users based on sparse, implicit interaction histories using matrix factorization and deep two-tower embedding architectures.
4. **Experimentation (A/B Testing & Causal Inference)** — Establish the statistical experimentation harness to evaluate changes (e.g., recommendation algorithms, delivery fee adjustments, dispatch rules) with guardrail checks (sample ratio mismatch, novelty effects, and observational causal inference).

---

## Planned Integration Layer

While each module is strictly independent during development, an optional integration layer can be introduced after all four modules mature:

- **Unified Flow:** A customer browses personalized dish rankings (`PersonalizeOps`), places an order under an active variant in an experiment (`Experimentation`), receives a dynamic delivery ETA (`DeliveryOps`), while batch orders feed downstream fulfillment and staffing models (`DemandOps`).
- **End-to-End Orchestration:** A central demo application or pipeline connecting module outputs to simulate the operational lifecycle of a live food delivery network.

---

## Tech Stack

| Category | Tools |
|---|---|
| Core language | Python (3.12) |
| Data wrangling & storage | pandas, numpy, pyarrow |
| Classical ML & Forecasting | LightGBM, scikit-learn, statsmodels, pmdarima |
| Deep learning | PyTorch |
| Dashboards & Interactive UI | Streamlit, Plotly Express & Graph Objects |
| Experimentation / stats | SciPy, statsmodels, causal inference packages |
| Version control | Git, single repository |

---

## Status

Current development status by module:

- **DemandOps:** **Complete & Operational.** 
  - Exploratory Data Analysis ([`EDA.ipynb`](./DemandOps/notebooks/EDA.ipynb))
  - Panel feature engineering ([`feature_engineering.ipynb`](./DemandOps/notebooks/feature_engineering.ipynb))
  - Multi-model training and benchmarking ([`model_training.ipynb`](./DemandOps/notebooks/model_training.ipynb)): LightGBM achieved **28.75% WAPE**
  - Autoregressive out-of-sample test forecasting ([`test_evaluation.ipynb`](./DemandOps/notebooks/test_evaluation.ipynb)) producing `submission_lgb.csv`
  - Interactive operations dashboard ([`streamlit_app.py`](DemandOps/app/streamlit_app.py)) with price elasticity curves, promotional lift, and benchmark scoreboard
- **DeliveryOps:** Queued (Next module).
- **PersonalizeOps:** Queued.
- **Experimentation:** Queued.
