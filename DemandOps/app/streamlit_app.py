"""FoodOps.AI — DemandOps Streamlit Application.

Interactive operations dashboard for demand forecasting, price elasticity
simulation, promotional lift analysis, model benchmarking, and operational analytics.

Theme Architecture:
  Harmonizes dynamically with Streamlit's built-in Light and Dark themes.
  All layout elements and Plotly visualizations inherit system theme tokens
  without hardcoding low-contrast colors or causing washed-out text.
"""

import sys
import os

# Guarantee absolute and relative import resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMANDOPS_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(DEMANDOPS_DIR)
for p in [PROJECT_ROOT, DEMANDOPS_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from DemandOps.src.inference import (
    load_model_bundle,
    predict_single,
    simulate_price_sensitivity,
    simulate_promo_scenarios,
)
from DemandOps.src.data_loader import (
    load_metadata,
    load_latest_state,
    load_operational_summary,
    load_benchmark_metrics,
)

try:
    from DemandOps.app.theme import (
        apply_theme,
        masthead,
        section,
        price_badge,
        callout,
        chart_layout,
        ACCENT_COLOR,
    )
except ImportError:
    from theme import (
        apply_theme,
        masthead,
        section,
        price_badge,
        callout,
        chart_layout,
        ACCENT_COLOR,
    )

# ------------------------------------------------------------------------------
# APP CONFIGURATION & THEMING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodOps.AI — DemandOps",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()


@st.cache_resource(show_spinner="Loading trained model bundle...")
def get_model():
    """Load model bundle artifact once into memory."""
    return load_model_bundle()


@st.cache_data(show_spinner="Loading operational metadata & precomputed caches...")
def get_data():
    """Load static metadata and precomputed summary caches."""
    centers_df, meals_df = load_metadata()
    latest_state = load_latest_state()
    summary = load_operational_summary()
    benchmarks = load_benchmark_metrics()
    return centers_df, meals_df, latest_state, summary, benchmarks


# Initialize Core Resources
try:
    bundle = get_model()
    centers_df, meals_df, latest_state, summary_data, benchmarks_df = get_data()
    model_loaded = True
except Exception as e:
    st.error(f"Initialization Error: Unable to load model or dataset artifacts ({e})")
    model_loaded = False

# Render Executive Masthead
masthead(
    week_label="Historical panel: Weeks 1–145 · 77 Hubs · 51 Catalog Dishes",
    model_label="LightGBM Champion (28.75% WAPE)",
)

if not model_loaded:
    st.stop()

# Primary Navigation Tabs
tab_pred, tab_benchmarks, tab_analytics = st.tabs([
    "🔮 Forecast & What-If Simulator",
    "📊 Model Benchmarks & Evaluation",
    "🍲 Portfolio & Kitchen Analytics"
])


# ==============================================================================
# TAB 1: DEMAND PREDICTOR & WHAT-IF SIMULATOR
# ==============================================================================
with tab_pred:
    col_input, col_output = st.columns([1.05, 1.35], gap="large")

    # --------------------------------------------------------------------------
    # LEFT COLUMN: INPUT CONTROLS & CONSTRAINTS
    # --------------------------------------------------------------------------
    with col_input:
        section("01", "Facility & Dish Selection", "Pick a fulfillment hub and a catalog dish to evaluate.")

        # Center Selector with descriptive label
        center_options = centers_df.sort_values("center_id").apply(
            lambda r: f"Hub {r['center_id']} — {r['center_type']} (City {r['city_code']}, {r['op_area']} km²)",
            axis=1,
        ).tolist()
        center_id_map = {opt: cid for opt, cid in zip(center_options, centers_df.sort_values("center_id")["center_id"])}

        selected_center_str = st.selectbox("Fulfillment Hub", center_options, index=0)
        selected_center_id = center_id_map[selected_center_str]
        center_row = centers_df[centers_df["center_id"] == selected_center_id].iloc[0]

        # Meal Selector with descriptive label
        meal_options = meals_df.sort_values("meal_id").apply(
            lambda r: f"Meal {r['meal_id']} — {r['category']} ({r['cuisine']})",
            axis=1,
        ).tolist()
        meal_id_map = {opt: mid for opt, mid in zip(meal_options, meals_df.sort_values("meal_id")["meal_id"])}

        selected_meal_str = st.selectbox("Catalog Dish", meal_options, index=0)
        selected_meal_id = meal_id_map[selected_meal_str]
        meal_row = meals_df[meals_df["meal_id"] == selected_meal_id].iloc[0]

        # Check latest historical baseline from cache
        match = latest_state[
            (latest_state["center_id"] == selected_center_id)
            & (latest_state["meal_id"] == selected_meal_id)
        ]

        has_history = len(match) > 0
        if has_history:
            hist_row = match.iloc[0]
            default_base_price = float(hist_row["base_price"])
            default_checkout_price = float(hist_row["checkout_price"])
            default_lag_1 = float(hist_row["lag_1"])
            default_lag_2 = float(hist_row["lag_2"])
            default_lag_4 = float(hist_row["lag_4"])
            default_rm4 = float(hist_row["rolling_mean_4"])
            default_rs4 = float(hist_row["rolling_std_4"])
            prior_actual_orders = int(hist_row["num_orders"])
            st.caption(f"✓ Baseline loaded — Prior week actual: **{prior_actual_orders:,} orders**")
        else:
            default_base_price = 280.0
            default_checkout_price = 280.0
            default_lag_1 = 4.6
            default_lag_2 = 4.6
            default_lag_4 = 4.6
            default_rm4 = 4.6
            default_rs4 = 0.25
            prior_actual_orders = int(np.expm1(4.6))
            st.caption("ℹ️ No historical orders found for this exact pair; using estimated regional priors.")

        section("02", "Pricing & Commercial Strategy", "Set base list price and customer checkout price.")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            base_price = st.number_input(
                "Base List Price (₹ / $)",
                min_value=10.0,
                max_value=1200.0,
                value=round(default_base_price, 2),
                step=5.0,
                help="Catalog list price prior to promotional discounting.",
            )
        with p_col2:
            checkout_price = st.number_input(
                "Customer Checkout Price (₹ / $)",
                min_value=5.0,
                max_value=1200.0,
                value=round(default_checkout_price, 2),
                step=5.0,
                help="Net price charged to the consumer at checkout.",
            )

        # Calculate Price Delta
        price_diff = checkout_price - base_price
        price_change_pct = price_diff / base_price if base_price > 0 else 0.0
        st.markdown(price_badge(price_change_pct), unsafe_allow_html=True)

        section("03", "Promotions & Seasonality", "Configure marketing push and seasonal timing.")

        pr_col1, pr_col2 = st.columns(2)
        with pr_col1:
            emailer_promo = st.checkbox(
                "📧 Email Campaign",
                value=False,
                help="Featured in targeted email promotions sent to customer inbox.",
            )
        with pr_col2:
            homepage_featured = st.checkbox(
                "🏠 App Homepage Banner",
                value=False,
                help="Prominently featured on the delivery app homepage carousel.",
            )

        week_of_year = st.slider(
            "Target Operational Week of Year",
            min_value=1,
            max_value=52,
            value=25,
            help="Captures calendar seasonality across the 52-week annual cycle.",
        )

        with st.expander("⚙️ Advanced: Historical Lag Context (Log Scale)", expanded=False):
            st.caption("Auto-populated from recent orders. Adjust to simulate demand shifts or stockout recovery.")
            lag_1 = st.number_input(
                "Lag 1 (1 week prior, log orders)",
                min_value=0.0, max_value=12.0, value=round(default_lag_1, 2), step=0.1
            )
            lag_2 = st.number_input(
                "Lag 2 (2 weeks prior, log orders)",
                min_value=0.0, max_value=12.0, value=round(default_lag_2, 2), step=0.1
            )
            lag_4 = st.number_input(
                "Lag 4 (4 weeks prior, log orders)",
                min_value=0.0, max_value=12.0, value=round(default_lag_4, 2), step=0.1
            )
            rolling_mean_4 = st.number_input(
                "4-Week Rolling Mean (log)",
                min_value=0.0, max_value=12.0, value=round(default_rm4, 2), step=0.1
            )
            rolling_std_4 = st.number_input(
                "4-Week Rolling Std (log)",
                min_value=0.0, max_value=5.0, value=round(default_rs4, 2), step=0.05
            )

    # --------------------------------------------------------------------------
    # RIGHT COLUMN: PREDICTION OUTPUTS & STRATEGIC CHARTS
    # --------------------------------------------------------------------------
    with col_output:
        section("04", "Forecast Output & Inventory Buffer", "Actionable demand forecast and kitchen prep guidance.")

        # Build Inference Payload
        sample_input = {
            "checkout_price": float(checkout_price),
            "base_price": float(base_price),
            "price_change_pct": float(price_change_pct),
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "lag_4": float(lag_4),
            "rolling_mean_4": float(rolling_mean_4),
            "rolling_std_4": float(rolling_std_4),
            "op_area": float(center_row["op_area"]),
            "week_of_year": int(week_of_year),
            "emailer_for_promotion": int(emailer_promo),
            "homepage_featured": int(homepage_featured),
            "category": str(meal_row["category"]),
            "cuisine": str(meal_row["cuisine"]),
            "center_type": str(center_row["center_type"]),
            "center_id": int(selected_center_id),
            "meal_id": int(selected_meal_id),
        }

        # Predict with champion bundle
        pred_res = predict_single(bundle, sample_input)
        pred_orders = pred_res["predicted_orders"]
        safety_stock = pred_res["safety_stock_prep"]
        est_revenue = checkout_price * pred_orders

        # Calculate week-over-week velocity
        prior_orders = int(np.expm1(lag_1)) if lag_1 > 0 else 0
        delta_pct = ((pred_orders - prior_orders) / prior_orders * 100) if prior_orders > 0 else 0.0

        delta_str = None
        if prior_orders > 0:
            delta_str = f"{delta_pct:+.1f}% vs. prior week"

        # Native Streamlit Bordered Metric Cards (Immune to HTML markdown code block bugs)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            with st.container(border=True):
                st.metric(
                    label="FORECASTED ORDERS",
                    value=f"{pred_orders:,.0f}",
                    delta=delta_str,
                    help="Predicted customer demand volume based on model inference.",
                )
        with m_col2:
            with st.container(border=True):
                st.metric(
                    label="KITCHEN PREP BUFFER",
                    value=f"{safety_stock:,.0f}",
                    delta="+15% Safety Buffer",
                    delta_color="normal",
                    help="Recommended raw inventory & preparation batch size to avoid stockouts.",
                )
        with m_col3:
            with st.container(border=True):
                st.metric(
                    label="EST. GROSS REVENUE",
                    value=f"${est_revenue:,.0f}",
                    delta=f"${checkout_price:.2f} / unit",
                    delta_color="off",
                    help="Gross order value = checkout price × forecasted volume.",
                )

        st.caption(
            f"80% Expected Demand Range: **{pred_res['lower_bound_80']:.0f} to {pred_res['upper_bound_80']:.0f} orders** "
            f"(Model: LightGBM Regressor)"
        )

        section("05", "Price Elasticity & Revenue Sensitivity", "Simulate demand and revenue across ±30% price variations.")

        price_df = simulate_price_sensitivity(bundle, sample_input)

        # Identify revenue-maximizing price point
        opt_row = price_df.loc[price_df["expected_revenue"].idxmax()]
        opt_price = opt_row["checkout_price"]
        opt_orders = opt_row["predicted_orders"]
        opt_rev = opt_row["expected_revenue"]

        fig_price = go.Figure()

        # Demand Curve (Primary Y)
        fig_price.add_trace(go.Scatter(
            x=price_df["checkout_price"],
            y=price_df["predicted_orders"],
            mode="lines+markers",
            name="Demand Curve (Orders)",
            line=dict(color=ACCENT_COLOR, width=2.8),
            marker=dict(size=5, color=ACCENT_COLOR),
        ))

        # Current Operational Price Marker
        fig_price.add_trace(go.Scatter(
            x=[checkout_price],
            y=[pred_orders],
            mode="markers+text",
            name="Current Price",
            text=[f"Current: ${checkout_price:.2f}"],
            textposition="top center",
            marker=dict(size=11, symbol="star", color="#3b82f6", line=dict(color="#1d4ed8", width=1.5)),
        ))

        # Optimal Revenue Price Marker
        fig_price.add_trace(go.Scatter(
            x=[opt_price],
            y=[opt_orders],
            mode="markers+text",
            name="Revenue Optimal Price",
            text=[f"Max Rev: ${opt_price:.2f}"],
            textposition="bottom center",
            marker=dict(size=10, symbol="diamond", color="#10b981", line=dict(color="#047857", width=1.5)),
        ))

        chart_layout(
            fig_price,
            title=f"Price Elasticity for Meal {selected_meal_id} ({meal_row['category']})",
            xaxis_title="Checkout Price (₹ / $)",
            yaxis_title="Forecasted Weekly Orders",
            height=310,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_price, use_container_width=True, theme="streamlit")

        section("06", "Promotional Channel Lift Analysis", "Marginal impact of individual and combined marketing channels.")

        promo_df = simulate_promo_scenarios(bundle, sample_input)

        fig_promo = px.bar(
            promo_df,
            x="Scenario",
            y="Predicted Orders",
            text="Predicted Orders",
            color="Scenario",
            color_discrete_sequence=[ACCENT_COLOR, "#38bdf8", "#818cf8", "#22c55e"],
            title="Forecasted Demand Across Marketing Campaign Channels",
        )
        fig_promo.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        chart_layout(fig_promo, height=270, showlegend=False)
        st.plotly_chart(fig_promo, use_container_width=True, theme="streamlit")


# ==============================================================================
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    section(
        "01",
        "Model Evaluation Scoreboard",
        "Trained on weeks 1–131; evaluated on hold-out validation weeks 132–145 to simulate real out-of-sample forward deployment.",
    )

    # Format Benchmarks DataFrame with clean column config
    st.dataframe(
        benchmarks_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Model Architecture": st.column_config.TextColumn("Model Architecture", width="medium"),
            "Family": st.column_config.TextColumn("Model Family", width="medium"),
            "Validation WAPE (%)": st.column_config.ProgressColumn(
                "Validation WAPE (%)",
                help="Weighted Absolute Percentage Error (Lower is better)",
                format="%.2f%%",
                min_value=20.0,
                max_value=50.0,
            ),
            "Validation MAPE (%)": st.column_config.NumberColumn(
                "Validation MAPE (%)",
                format="%.2f%%",
            ),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Notes": st.column_config.TextColumn("Architecture Notes", width="large"),
        },
    )

    section("02", "Error Metric Comparison", "Comparison across WAPE (primary supply chain KPI) and MAPE.")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        fig_wape = px.bar(
            benchmarks_df,
            x="Validation WAPE (%)",
            y="Model Architecture",
            orientation="h",
            text="Validation WAPE (%)",
            color="Validation WAPE (%)",
            color_continuous_scale="Reds_r",
            title="Validation WAPE (%) — Volume-Weighted Error (Lower is Better)",
        )
        fig_wape.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        chart_layout(
            fig_wape,
            height=340,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_wape, use_container_width=True, theme="streamlit")

    with b_col2:
        fig_mape = px.bar(
            benchmarks_df,
            x="Validation MAPE (%)",
            y="Model Architecture",
            orientation="h",
            text="Validation MAPE (%)",
            color="Validation MAPE (%)",
            color_continuous_scale="Oranges_r",
            title="Validation MAPE (%) — Average Item-Level Percentage Error",
        )
        fig_mape.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        chart_layout(
            fig_mape,
            height=340,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_mape, use_container_width=True, theme="streamlit")

    section("03", "LightGBM Feature Importance (Gain & Splits)", "What drivers matter most for order volume prediction.")

    try:
        lgb_raw_model = bundle["model"]
        feat_names = bundle["features"]
        importances = lgb_raw_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df.tail(12),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Top 12 Predictive Features in Champion LightGBM Model",
        )
        chart_layout(fig_fi, height=390, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True, theme="streamlit")
    except Exception:
        st.info("Feature importance display unavailable for current model format.")

    with st.expander("📚 Why WAPE is Prioritized Over RMSE & MAPE in Kitchen Supply Chains"):
        st.markdown(
            r"""
**WAPE (Weighted Absolute Percentage Error)**

$$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100$$

- **Volume Weighting:** High-demand staples (e.g. Rice Bowls, Beverages) account for the vast majority of raw ingredients and kitchen labor. WAPE weights errors by true order volume, ensuring a 20-order error on a 500-order item is not penalized the same as a 20-order error on a 20-order dish.
- **Why Not Standard MAPE?** MAPE divides by each individual row's actual order count ($|y - \hat{y}| / y$). When a local hub only serves 1 or 2 portions of a niche starter, a slight deviation blows up the percentage to several hundred percent, distorting operational evaluation.
- **Why Not RMSE?** Raw RMSE squares errors and is dominated solely by the top 5 largest fulfillment hubs, hiding systematic under-stocking at regional facilities.
            """
        )


# ==============================================================================
# TAB 3: PRODUCT & KITCHEN ANALYTICS
# ==============================================================================
with tab_analytics:
    section(
        "01",
        "Operational Portfolio Overview",
        "Historical demand patterns across 119.5M meals, 77 fulfillment centers, and 51 catalog dishes.",
    )

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        with st.container(border=True):
            st.metric(
                label="ACTIVE HUBS",
                value=f"{len(centers_df):,}",
                help="Decentralized fulfillment center network across all operating cities.",
            )
    with t2:
        with st.container(border=True):
            st.metric(
                label="ACTIVE DISHES",
                value=f"{len(meals_df):,}",
                help="Total distinct catalog meals and beverage SKUs.",
            )
    with t3:
        with st.container(border=True):
            st.metric(
                label="CUISINES",
                value=f"{meals_df['cuisine'].nunique()}",
                help="Distinct culinary categories (Continental, Indian, Italian, Thai).",
            )
    with t4:
        with st.container(border=True):
            st.metric(
                label="CATEGORIES",
                value=f"{meals_df['category'].nunique()}",
                help="Meal categories across beverages, bowls, snacks, and mains.",
            )

    section("02", "Demand Composition by Category & Cuisine", "Volume distribution across food categories and styles.")

    a_col1, a_col2 = st.columns(2)

    by_category = summary_data.get("by_category", [])
    if by_category:
        cat_df = pd.DataFrame(by_category).sort_values("total_orders", ascending=False)
        with a_col1:
            fig_cat = px.bar(
                cat_df,
                x="total_orders",
                y="category",
                orientation="h",
                text="total_orders",
                color="total_orders",
                color_continuous_scale="Teal",
                title="Total Historical Orders by Meal Category",
            )
            fig_cat.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            chart_layout(
                fig_cat,
                height=420,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True, theme="streamlit")

    by_cuisine = summary_data.get("by_cuisine", [])
    if by_cuisine:
        cui_df = pd.DataFrame(by_cuisine)
        cuisine_palette = [ACCENT_COLOR, "#38bdf8", "#10b981", "#fbbf24", "#f87171"]
        with a_col2:
            fig_cui = px.pie(
                cui_df,
                values="total_orders",
                names="cuisine",
                hole=0.52,
                color_discrete_sequence=cuisine_palette,
                title="Historical Demand Share by Cuisine",
            )
            fig_cui.update_traces(textinfo="percent+label")
            chart_layout(fig_cui, height=420)
            st.plotly_chart(fig_cui, use_container_width=True, theme="streamlit")

    weekly_history = summary_data.get("weekly_platform_demand", [])
    if weekly_history:
        section("03", "Platform Total Weekly Order Trajectory", "Multi-year network order volume trends (Weeks 1 to 145).")
        w_hist_df = pd.DataFrame(weekly_history)
        fig_weekly = px.line(
            w_hist_df,
            x="week",
            y="num_orders",
            markers=True,
            title="Total Platform Weekly Meal Demand (Historical Weeks 1–145)",
        )
        fig_weekly.update_traces(line_color=ACCENT_COLOR, line_width=2.2, marker=dict(size=4))
        chart_layout(
            fig_weekly,
            xaxis_title="Operational Week Number",
            yaxis_title="Total Network Orders",
            height=310,
        )
        st.plotly_chart(fig_weekly, use_container_width=True, theme="streamlit")