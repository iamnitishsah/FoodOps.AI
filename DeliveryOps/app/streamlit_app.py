import sys
import os

# Guarantee absolute and relative import resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DELIVERYOPS_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(DELIVERYOPS_DIR)
for p in [PROJECT_ROOT, DELIVERYOPS_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from DeliveryOps.src.inference import DeliveryOpsEngine
from DeliveryOps.src.data_loader import load_operational_summary, load_benchmark_metrics

# Try importing the theme file (relies on you having copied theme.py to DeliveryOps/app/)
try:
    from DeliveryOps.app.theme import apply_theme, masthead, section, chart_layout, ACCENT_COLOR
except ImportError:
    from theme import apply_theme, masthead, section, chart_layout, ACCENT_COLOR

# ------------------------------------------------------------------------------
# APP CONFIGURATION & THEMING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodOps.AI — DeliveryOps",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()


@st.cache_resource(show_spinner="Loading trained ETA models...")
def load_engine():
    """Caches the model bundle in memory so it doesn't reload on every UI click."""
    return DeliveryOpsEngine()


@st.cache_data(show_spinner="Loading operational metadata...")
def get_data():
    summary = load_operational_summary()
    benchmarks = load_benchmark_metrics()
    return summary, benchmarks


# Initialize Core Resources
try:
    engine = load_engine()
    summary_data, benchmarks_df = get_data()
    model_loaded = True
except Exception as e:
    st.error(f"Initialization Error: Unable to load DeliveryOps artifacts. Error: {e}")
    model_loaded = False

if not model_loaded:
    st.stop()

# Primary Navigation Tabs
tab_pred, tab_benchmarks, tab_analytics = st.tabs([
    "ETA Dispatch Simulator",
    "Model Benchmarks & Evaluation",
    "Portfolio & Dispatch Analytics"
])

# ==============================================================================
# TAB 1: ETA DISPATCH SIMULATOR
# ==============================================================================
with tab_pred:
    st.markdown(
        "Adjust market congestion, routing distances, and order complexity to simulate customer-facing delivery promises.")

    col1, col2, col3 = st.columns(3)

    with col1:
        section("01", "Logistics & Timing")
        market_id = st.selectbox("Market ID", options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        order_hour = st.slider("Local Hour of Day", 0, 23, 19)
        order_day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", options=[0, 1, 2, 3, 4, 5, 6], index=4)
        driving_duration = st.number_input("Estimated Drive Time (seconds)", min_value=60, max_value=3600, value=600)
        place_duration = st.number_input("Estimated Place Duration (seconds)", min_value=60, max_value=1200, value=251)

    with col2:
        section("02", "Market Congestion")
        onshift_dashers = st.number_input("Total On-Shift Dashers", min_value=0, max_value=200, value=50)
        busy_dashers = st.number_input("Total Busy Dashers", min_value=0, max_value=200, value=45)
        outstanding_orders = st.number_input("Total Outstanding Orders", min_value=0, max_value=300, value=60)

        st.caption("Congestion Metrics:")
        busy_ratio = busy_dashers / max(1, onshift_dashers)
        out_ratio = outstanding_orders / max(1, onshift_dashers)
        st.write(f"- Dasher Utilization: **{busy_ratio:.2f}**")
        st.write(f"- Order/Dasher Ratio: **{out_ratio:.2f}**")

    with col3:
        section("03", "Order Complexity")
        store_category = st.text_input("Store Category (e.g., mexican, pizza)", value="mexican")
        order_protocol = st.selectbox("Order Protocol", options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        total_items = st.number_input("Total Items", min_value=1, max_value=50, value=3)
        num_distinct_items = st.number_input("Distinct Items", min_value=1, max_value=50, value=3)
        subtotal = st.number_input("Subtotal (cents)", min_value=100, max_value=20000, value=2500)
        min_price = st.number_input("Min Item Price (cents)", min_value=0, max_value=10000, value=500)
        max_price = st.number_input("Max Item Price (cents)", min_value=0, max_value=10000, value=1200)

    st.divider()

    if st.button("Simulate Customer ETA", type="primary", use_container_width=True):
        input_data = {
            "market_id": float(market_id),
            "store_primary_category": store_category.lower().strip(),
            "order_protocol": float(order_protocol),
            "total_items": int(total_items),
            "subtotal": int(subtotal),
            "num_distinct_items": int(num_distinct_items),
            "min_item_price": int(min_price),
            "max_item_price": int(max_price),
            "total_onshift_dashers": float(onshift_dashers),
            "total_busy_dashers": float(busy_dashers),
            "total_outstanding_orders": float(outstanding_orders),
            "estimated_order_place_duration": int(place_duration),
            "estimated_store_to_consumer_driving_duration": float(driving_duration),
            "order_hour": int(order_hour),
            "order_day_of_week": int(order_day_of_week)
        }

        with st.spinner("Calculating delivery permutations..."):
            results = engine.simulate_eta(input_data)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            with st.container(border=True):
                st.metric(label="OPTIMISTIC ETA (P10)", value=f"{results['p10_min']} min", delta="- Best Case Scenario",
                          delta_color="normal")
        with res_col2:
            with st.container(border=True):
                st.metric(label="EXPECTED ETA (P50)", value=f"{results['eta_min']} min", delta="Model Average",
                          delta_color="off")
        with res_col3:
            with st.container(border=True):
                st.metric(label="PROMISED ETA (P90)", value=f"{results['p90_min']} min", delta="+ Worst Case Scenario",
                          delta_color="inverse")

        st.info(
            f"**Customer Promise Window:** {results['p10_min']} to {results['p90_min']} minutes (Spread: {results['spread_min']} min)")

# ==============================================================================
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    section("01", "Delivery ETA Scoreboard",
            "Validating Mean Absolute Error (MAE) and Promised Delivery (P90) thresholds.")

    st.dataframe(
        benchmarks_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Model Architecture": st.column_config.TextColumn("Model Architecture", width="medium"),
            "Family": st.column_config.TextColumn("Model Family", width="medium"),
            "Test MAE (min)": st.column_config.ProgressColumn(
                "Test MAE (min)",
                help="Mean Absolute Error in minutes (Lower is better)",
                format="%.2f",
                min_value=8.0,
                max_value=14.0,
            ),
            "P90 Coverage (%)": st.column_config.NumberColumn(
                "P90 Coverage (%)",
                format="%.2f%%",
                help="Percent of deliveries arriving before the P90 estimated limit"
            ),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Notes": st.column_config.TextColumn("Architecture Notes", width="large"),
        },
    )

    b_col1, b_col2 = st.columns([1, 1.5], gap="large")
    with b_col1:
        section("02", "Error Metric Comparison")
        error_scale = [[0.0, "#10b981"], [0.5, "#f59e0b"], [1.0, "#ef4444"]]
        fig_mae = px.bar(
            benchmarks_df,
            x="Test MAE (min)",
            y="Model Architecture",
            orientation="h",
            text="Test MAE (min)",
            color="Test MAE (min)",
            color_continuous_scale=error_scale,
            title="Test MAE Across Architectures",
        )
        fig_mae.update_traces(texttemplate="%{text:.2f} m", textposition="outside", cliponaxis=False)
        chart_layout(fig_mae, height=360, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_mae, width="stretch", theme="streamlit")

    with b_col2:
        section("03", "LightGBM Global Feature Importance")
        try:
            # Extract importances natively from the LGBM model stored in our engine
            lgb_raw_model = engine.model_p50
            feat_names = engine.feature_names
            importances = lgb_raw_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance",
                                                                                                 ascending=True)

            fig_fi = px.bar(
                fi_df.tail(10),
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance",
                color="Importance",
                color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#6366f1"], [1.0, ACCENT_COLOR]],
                title="Top 10 Drivers of ETA (Gain/Splits)",
            )
            fig_fi.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
            chart_layout(fig_fi, height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_fi, width="stretch", theme="streamlit")
        except Exception:
            st.info("Feature importance is currently unavailable.")

# ==============================================================================
# TAB 3: PORTFOLIO & DISPATCH ANALYTICS
# ==============================================================================
with tab_analytics:
    section("01", "Fulfillment Network Overview", "Logistics footprint across ~196k historical deliveries.")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        with st.container(border=True):
            st.metric("ACTIVE MARKETS", value="6", help="Distinct regional operating zones.")
    with t2:
        with st.container(border=True):
            st.metric("UNIQUE STORES", value="5,600+", help="Restaurants fulfilling orders.")
    with t3:
        with st.container(border=True):
            st.metric("CUISINE TYPES", value="75", help="Target Encoded Categories.")
    with t4:
        with st.container(border=True):
            st.metric("AVERAGE ETA", value="44.2 min", help="Global Network P50 Time.")

    a_col1, a_col2 = st.columns(2)

    by_cat = summary_data.get("by_category", [])
    if by_cat:
        with a_col1:
            section("02", "Average ETA by Category")
            cat_df = pd.DataFrame(by_cat).sort_values("avg_duration_min", ascending=True)
            fig_cat = px.bar(
                cat_df,
                x="avg_duration_min",
                y="category",
                orientation="h",
                text="avg_duration_min",
                color="avg_duration_min",
                color_continuous_scale=[[0.0, "#10b981"], [0.5, "#f59e0b"], [1.0, "#ef4444"]],
                title="Historical ETA Duration by Cuisine (Minutes)",
            )
            fig_cat.update_traces(texttemplate="%{text:.1f} m", textposition="outside", cliponaxis=False)
            chart_layout(fig_cat, height=380, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_cat, width="stretch", theme="streamlit")

    daily_vol = summary_data.get("daily_platform_demand", [])
    if daily_vol:
        with a_col2:
            section("03", "Daily Dispatch Volume")
            vol_df = pd.DataFrame(daily_vol)
            fig_vol = px.line(
                vol_df,
                x="day",
                y="num_orders",
                markers=True,
                title="Total Network Orders by Day of Week",
            )
            fig_vol.update_traces(
                line_color=ACCENT_COLOR,
                line_width=3,
                marker=dict(size=6),
                hovertemplate="%{x}<br>Orders: %{y:,.0f}<extra></extra>",
            )
            chart_layout(fig_vol, height=380, xaxis_title="Day of Week", yaxis_title="Dispatched Orders")
            st.plotly_chart(fig_vol, width="stretch", theme="streamlit")