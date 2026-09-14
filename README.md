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
- Exposes its result through a small API + dashboard so it can be interacted with, not just read as a notebook

---

## Module 1: DemandOps (Demand Forecasting)

**Folder:** [`DemandOps/`](./DemandOps/README.md)

**Problem:** Given historical order counts for a location and item category, predict order volume for a future period (e.g., the next several weeks). Underestimating demand leads to understaffing and slow deliveries; overestimating leads to wasted rider/inventory cost — so both the forecast and its likely error matter.

**What's involved:**
- Time-series feature engineering: lag features, rolling averages, calendar effects, price/promotion effects
- A time-based (not random) train/validation/test split — random splitting leaks future information into training for time series and produces misleadingly good validation scores
- A classical baseline (naive forecast, exponential smoothing, or SARIMA) to establish what "no real model" and "a properly specified classical model" achieve
- A tree-based model (e.g., LightGBM) using the engineered features, compared against the classical baseline
- A small LSTM as a second point of comparison, evaluated honestly against the tree-based model rather than assumed to be better
- Evaluation using WAPE/MAPE rather than raw RMSE, since RMSE is dominated by high-volume locations and can hide poor performance on smaller ones

**Details (dataset, exact features, results, how it's exposed):** in the [DemandOps README](./DemandOps/README.md).

---

## Module 2: DeliveryOps (ETA Prediction)

**Folder:** [`DeliveryOps/`](./DeliveryOps/README.md)

**Problem:** Given order-level information available at the time an order is placed (distance, time of day, traffic/weather conditions, restaurant load), predict how long the delivery will take. This estimate is shown to the customer directly, so both average accuracy and where the model tends to be wrong (e.g., specific weather conditions or times of day) matter.

**What's involved:**
- Feature engineering from raw signals (e.g., computing distance from coordinates, encoding time of day, joining weather/traffic data if available)
- A regression model progression from a simple baseline to a tuned gradient-boosted model
- Error analysis by segment — checking whether errors are randomly distributed or systematically worse under specific conditions
- Optionally, prediction intervals rather than only a single point estimate, since a wrong ETA with no sense of its own uncertainty is less useful than one with a stated confidence range

**Details (dataset, features, results, exposure):** in the [DeliveryOps README](./DeliveryOps/README.md) once built.

---

## Module 3: PersonalizeOps (Recommendation Engine)

**Folder:** [`PersonalizeOps/`](./PersonalizeOps/README.md)

**Problem:** Given a history of user interactions with restaurants/items (orders, clicks, or ratings), rank restaurants/items for a given user. Interaction data here is implicit (an order signals interest but there's no explicit rating scale in most of it) and sparse (most users have not interacted with most restaurants), which shapes the whole approach.

**What's involved:**
- Building an interaction matrix from implicit feedback
- A classical collaborative-filtering / matrix-factorization approach, implemented directly (not only via a library call) to understand the mechanics
- A two-tower embedding model in PyTorch as a deep-learning approach, compared against the classical method
- Ranking-appropriate evaluation metrics (precision@k, recall@k, NDCG) rather than classification accuracy, which doesn't fit a ranking task
- Discussion of the cold-start problem (new users or new restaurants with no interaction history) and how each approach handles or fails to handle it

**Details (dataset, interaction data construction, results, exposure):** in the [PersonalizeOps README](./PersonalizeOps/README.md) once built.

---

## Module 4: Experimentation (Experimentation Framework)

**Folder:** [`Experimentation/`](./Experimentation/README.md)

**Problem:** Given two variants of something (e.g., a promo strategy, a UI change, a pricing change) and outcome data from users exposed to each, determine whether the observed difference reflects a real effect or noise, and by how much. Most product decisions are made this way rather than by assumption, so getting the statistical analysis right — not just running a test but understanding what can invalidate it — is the actual skill involved.

**What's involved:**
- Defining a hypothesis, primary metric, and required sample size before generating or looking at outcome data
- Correctly applying a statistical test, checking its assumptions rather than applying it by default
- Demonstrating and explaining common problems that invalidate naive A/B analysis: peeking at results early, novelty effects, sample ratio mismatch, multiple comparisons
- A stretch extension into causal inference methods for non-randomized/observational scenarios, where a clean A/B split isn't available

**Details (scenario simulated, tools used, methodology, results, exposure):** in the [Experimentation README](./Experimentation/README.md) once built.

---

## Repository Structure

Single repository, one folder per module, each module self-contained:

```
FoodOps.AI/
├── README.md                          ← this file
├── requirements.txt                   ← project dependencies
├── LICENSE                            ← MIT license
│
├── DemandOps/                         ← Module 1: Demand Forecasting
│   ├── README.md                      ← dataset, approach, results, how it's exposed
│   ├── data/
│   │   ├── raw/                       ← fulfillment center, meal, and order data
│   │   └── processed/                 ← engineered and merged panel datasets
│   ├── notebooks/                     ← EDA.ipynb, feature_engineering.ipynb
│   ├── src/                           ← feature_engineering.py, train.py, evaluate.py
│   ├── api/                           ← FastAPI service serving the trained model
│   ├── app/                           ← dashboard consuming the API
│   ├── models/                        ← saved model artifacts
│   └── requirements.txt
│
├── DeliveryOps/                       ← Module 2: ETA Prediction
│   ├── README.md
│   ├── data/{raw,processed}/
│   ├── notebooks/
│   ├── src/
│   ├── api/
│   ├── app/
│   ├── models/
│   └── requirements.txt
│
├── PersonalizeOps/                    ← Module 3: Recommendation Engine
│   ├── README.md
│   ├── data/{raw,processed}/
│   ├── notebooks/
│   ├── src/
│   ├── api/
│   ├── app/
│   ├── models/
│   └── requirements.txt
│
├── Experimentation/                   ← Module 4: Experimentation Framework
│   ├── README.md
│   ├── data/{raw,processed}/
│   ├── notebooks/
│   ├── src/
│   ├── analysis/                      ← statistical test scripts, causal inference notebooks
│   ├── app/                           ← report/dashboard exposing the analysis
│   └── requirements.txt
│
└── .gitignore
```

Each module exposes its result through its own small FastAPI service plus a lightweight dashboard that calls it.
The Experimentation module differs slightly. It produces an analysis/report rather than a servable model, so its `app/` presents results rather than calling a prediction endpoint.

This top-level README is not updated with per-module implementation detail (dataset names, metrics, results) as modules are built — that goes into each module's own README. This file stays a stable map of what the project contains.

---

## Why Modules Are Independent

Each module has its own `data/`, `src/`, `api/`, `app/`, and `requirements.txt`, with no shared top-level `common/` code at this stage. This keeps each module understandable and runnable on its own — reading `DeliveryOps` doesn't require first understanding a shared abstraction used across all four — and avoids designing a shared layer before it's clear the modules actually need one.

---

## Build Order

Development proceeds in four sequential stages, matching the operational hierarchy of a food delivery platform:

1. **DemandOps (Demand Forecasting)** — *In Progress*: Foundational operational forecasting. Ingest historical fulfillment center and meal order streams, formulate time-series lag/rolling features, and benchmark classical models against tree-based architectures (LightGBM) to forecast center-level demand volume.
2. **DeliveryOps (ETA Prediction)** — Predict customer delivery duration based on order timestamp, route distance, weather, and fulfillment/restaurant load with calibrated uncertainty intervals.
3. **PersonalizeOps (Recommendation Engine)** — Rank dishes and restaurants for users based on sparse, implicit interaction histories using matrix factorization and deep two-tower embedding architectures.
4. **Experimentation (A/B Testing & Causal Inference)** — Establish the statistical experimentation harness to evaluate changes (e.g., recommendation algorithms, delivery fee adjustments, dispatch rules) with guardrail checks (sample ratio mismatch, novelty effects, and observational causal inference).

---

## Planned Integration Layer

While each module is strictly independent during development, an optional integration layer can be introduced after all four modules mature:

- **Unified Flow:** A customer browses personalized dish rankings (`PersonalizeOps`), places an order under an active variant in an experiment (`Experimentation`), receives a dynamic delivery ETA (`DeliveryOps`), while batch orders feed downstream fulfillment and staffing models (`DemandOps`).
- **Orchestration / Gateway:** A lightweight API gateway or event pipeline to simulate an end-to-end food delivery platform without compromising the modularity and standalone reproducibility of each submodule.

---

## Tech Stack

| Category | Tools |
|---|---|
| Core language | Python |
| Data wrangling / storage | SQL (Postgres/MySQL), pandas, numpy |
| Classical ML | scikit-learn, LightGBM/XGBoost, statsmodels, pmdarima |
| Deep learning | PyTorch |
| Serving | FastAPI (per module) |
| Dashboards | Streamlit (per module) |
| Experimentation / stats | Scipy, statsmodels, causal inference packages |
| Version control | Git, single repository |

---

## Status

All four module directories are initialized in the repository. Current development status by module:

- **DemandOps:** Active development. Exploratory Data Analysis ([`EDA.ipynb`](./DemandOps/notebooks/EDA.ipynb)) and panel feature engineering ([`feature_engineering.ipynb`](./DemandOps/notebooks/feature_engineering.ipynb)) completed on meal delivery historical data. Model baseline and training pipeline underway.
- **DeliveryOps:** Scaffolding complete; dataset curation and baseline setup planned next.
- **PersonalizeOps:** Scaffolding complete; interaction data schema and matrix modeling scheduled.
- **Experimentation:** Scaffolding complete; synthetic A/B test simulation scenarios and causal analysis scheduled.

Each module's own `README.md` is the source of truth for what has actually been built in it.
