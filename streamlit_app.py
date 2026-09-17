"""
FoodOps.AI — Root Landing Console

Each module (DemandOps, DeliveryOps, PersonalizeOps, Experimentation) is
deployed as its own independent Streamlit app with its own URL, data,
models, and requirements. This root app does not embed those apps —
embedding another Streamlit app in an iframe breaks due to websocket /
CORS restrictions — instead it presents an executive console of module
cards, styled with the same design system (theme.py) as every submodule,
and routes out to each one's live deployment in a new tab.
"""

import streamlit as st

import theme

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodOps.AI",
    page_icon="😋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.apply_theme()

# --------------------------------------------------------------------------
# Module registry — single source of truth for the console grid
# --------------------------------------------------------------------------
MODULES = [
    {
        "icon": "📊",
        "name": "DemandOps",
        "tagline": "Weekly Fulfillment Demand Forecasting",
        "description": (
            "Forecasts weekly order volume per (center, meal) across 77 "
            "fulfillment hubs and 51 dishes using autoregressive lags, "
            "rolling statistics, and gradient boosted trees."
        ),
        "status": "live",
        "metrics": ["28.75% WAPE", "119.5M records", "77 centers"],
        "tags": ["Time Series", "LightGBM", "PyTorch LSTM", "Ridge"],
        "url": "https://foodops-demand.streamlit.app/",
    },
    {
        "icon": "🛵",
        "name": "DeliveryOps",
        "tagline": "Real-Time Delivery ETA Prediction & Logistics SLAs",
        "description": (
            "Predicts delivery duration as calibrated P10/P50/P90 quantiles "
            "via Pinball Loss, enabling SLA-compliant customer-facing "
            "delivery windows under peak congestion."
        ),
        "status": "live",
        "metrics": ["10.29 min MAE", "86.70% P90 coverage", "196K deliveries"],
        "tags": ["Quantile Regression", "LightGBM", "Optuna", "SHAP"],
        "url": "https://foodops-delivery.streamlit.app/",
    },
    {
        "icon": "🍽️",
        "name": "PersonalizeOps",
        "tagline": "Implicit Feedback Dish & Restaurant Ranking",
        "description": (
            "Ranks dishes and restaurants from sparse, implicit interaction "
            "histories using classical matrix factorization and a deep "
            "two-tower neural ranking architecture."
        ),
        "status": "queued",
        "metrics": ["NDCG@10", "Recall@10"],
        "tags": ["Recommenders", "Matrix Factorization", "Two-Tower", "PyTorch"],
        "url": None,
    },
    {
        "icon": "🧪",
        "name": "Experimentation",
        "tagline": "A/B Test Harness & Causal Inference",
        "description": (
            "Evaluates whether product and pricing interventions drive "
            "real causal impact, with guardrails against sample ratio "
            "mismatch, peeking, and multiple comparisons."
        ),
        "status": "queued",
        "metrics": ["SRM Guardrails", "DiD", "PSM"],
        "tags": ["A/B Testing", "Bayesian Stats", "Causal Inference"],
        "url": None,
    },
]

GITHUB_URL = "https://github.com/iamnitishsah/FoodOps.AI"

# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------
live_count = sum(1 for m in MODULES if m["status"] == "live")
queued_count = len(MODULES) - live_count

theme.masthead(
    week_label="4 Independent Modules · Kaggle & Simulated Operational Data",
    model_label=f"Platform Status: {live_count} Live · {queued_count} Queued",
)

col_intro, col_gh = st.columns([4, 1])
with col_intro:
    st.markdown(
        '''FoodOps.AI is an enterprise-grade food delivery machine learning platform built around four independent, self-contained operational intelligence modules: demand forecasting, delivery time (ETA) prediction, personalized menu recommendation, and experimentation / causal inference. Each module solves a high-impact operational decision problem using real-world or realistically simulated data, progressing from exploratory data analysis and simple baselines to production-grade ML architectures and interactive operations consoles.'''
    )
with col_gh:
    st.link_button("⭐ GitHub Repo", GITHUB_URL, use_container_width=True)

# --------------------------------------------------------------------------
# Platform-level stat strip
# --------------------------------------------------------------------------
stat_cols = st.columns(4)
stat_values = [
    ("4", "Operational Modules"),
    ("28.75%", "DemandOps WAPE"),
    ("10.29 min", "DeliveryOps MAE"),
    (f"{live_count}/4", "Modules Live"),
]
for col, (value, label) in zip(stat_cols, stat_values):
    with col:
        st.markdown(theme.stat(value, label), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Module console grid
# --------------------------------------------------------------------------
theme.section("01", "Operational Modules", "Select a console to open its live application in a new tab.")

cols_per_row = 2
rows = [MODULES[i : i + cols_per_row] for i in range(0, len(MODULES), cols_per_row)]

for row in rows:
    grid = st.columns(cols_per_row)
    for col, module in zip(grid, row):
        with col:
            theme.module_card(
                icon=module["icon"],
                name=module["name"],
                tagline=module["tagline"],
                description=module["description"],
                status=module["status"],
                metrics=module["metrics"],
                tags=module["tags"],
            )
            st.write("")
            if module["url"]:
                st.link_button(
                    f"Open {module['name']} Console →",
                    module["url"],
                    use_container_width=True,
                )
            else:
                st.button(
                    "Coming Soon",
                    disabled=True,
                    use_container_width=True,
                    key=f"disabled_{module['name']}",
                )
    st.write("")

# --------------------------------------------------------------------------
# Build order / roadmap
# --------------------------------------------------------------------------
theme.section("02", "Build Order & Roadmap", "Development proceeds across four sequential operational stages.")

roadmap_cols = st.columns(4)
roadmap = [
    ("1", "DemandOps", "Complete & Live", "live"),
    ("2", "DeliveryOps", "Complete & Live", "live"),
    ("3", "PersonalizeOps", "Queued", "queued"),
    ("4", "Experimentation", "Queued", "queued"),
]
for col, (step, name, label, status) in zip(roadmap_cols, roadmap):
    with col:
        st.markdown(
            f'<div class="fo-card" style="align-items:center; text-align:center;">'
            f'<div class="fo-card-name">{step}. {name}</div>'
            f'{theme.status_badge(status)}'
            f'<div class="fo-card-tagline">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.write("")
st.divider()
st.caption(
    "Each module is deployed and versioned independently. This console only routes you to the right application."
)