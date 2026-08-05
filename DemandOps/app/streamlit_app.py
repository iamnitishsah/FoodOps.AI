"""FoodOps.AI — DemandOps Streamlit Application.

Interactive dashboard around the LightGBM demand model:

  1. Forecast & What-If   10-week planner (recursive), plus next-week price and promo scenarios
  2. Model Benchmarks     scoreboard from models/results.json and LightGBM feature importance
  3. Portfolio Analytics  demand composition, weekly trajectory and data coverage

Everything shown is derived from the notebook artifacts; nothing is hard-coded.
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
    get_pair_state,
    build_feature_row,
    forecast_pair,
    safety_stock,
    simulate_price_sensitivity,
    simulate_promo_scenarios,
    check_feature_parity,
)
from DemandOps.src.data_loader import (
    load_panel,
    load_planned_inputs,
    load_results,
    load_benchmark_metrics,
    load_error_band,
    build_pair_catalog,
    build_analytics,
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

HORIZON = 10          # weeks in the planner
HISTORY_WEEKS = 26    # weeks of history drawn on the forecast chart

st.set_page_config(
    page_title="FoodOps.AI — DemandOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()


# ------------------------------------------------------------------------------
# RESOURCES
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and history, verifying training/serving parity...")
def get_context():
    """Load everything once. The parity check fails loudly if serving drifts from training."""
    panel = load_panel()
    bundle = load_model_bundle()
    check_feature_parity(panel)
    results = load_results()
    return {
        "panel": panel,
        "bundle": bundle,
        "results": results,
        "catalog": build_pair_catalog(panel),
        "analytics": build_analytics(panel),
        "planned": load_planned_inputs(),
        "benchmarks": load_benchmark_metrics(),
        "band": load_error_band(panel),
    }


try:
    ctx = get_context()
except Exception as e:  # noqa: BLE001
    st.error(f"Initialization error: {type(e).__name__}: {e}")
    st.stop()

panel, bundle = ctx["panel"], ctx["bundle"]
catalog, analytics = ctx["catalog"], ctx["analytics"]
planned, benchmarks_df, band = ctx["planned"], ctx["benchmarks"], ctx["band"]
kpis = analytics["kpis"]
lgb_wape = ctx["results"].get("lgb", {}).get("wape")

masthead(
    week_label=(f"Panel: weeks {kpis['first_week']}–{kpis['last_week']} · {kpis['hubs']} hubs · "
                f"{kpis['dishes']} dishes · {kpis['pairs']:,} hub–dish pairs"),
    model_label=(f"LightGBM · {lgb_wape:.2f}% WAPE (1-week-ahead)" if lgb_wape is not None else "LightGBM"),
)


def make_default_plan(state: dict, planned_df: pd.DataFrame, horizon: int):
    """Planning table pre-filled from the official plan (test.csv) where available."""
    weeks = state["last_week"] + 1 + np.arange(horizon)
    plan = pd.DataFrame({
        "week": weeks,
        "base_price": state["last_base_price"],
        "checkout_price": state["last_checkout_price"],
        "emailer_for_promotion": False,
        "homepage_featured": False,
    })
    prefilled = 0
    if len(planned_df):
        sel = planned_df[(planned_df["center_id"] == state["center_id"])
                         & (planned_df["meal_id"] == state["meal_id"])]
        sel = sel.drop_duplicates("week").set_index("week")
        for i, w in enumerate(weeks):
            if w in sel.index:
                r = sel.loc[w]
                plan.loc[i, "base_price"] = float(r["base_price"])
                plan.loc[i, "checkout_price"] = float(r["checkout_price"])
                plan.loc[i, "emailer_for_promotion"] = bool(r["emailer_for_promotion"])
                plan.loc[i, "homepage_featured"] = bool(r["homepage_featured"])
                prefilled += 1
    return plan, prefilled


tab_pred, tab_benchmarks, tab_analytics = st.tabs([
    "Forecast & What-If Simulator",
    "Model Benchmarks & Evaluation",
    "Portfolio & Kitchen Analytics",
])

# ==============================================================================
# TAB 1: FORECAST PLANNER & WHAT-IF SIMULATOR
# ==============================================================================
with tab_pred:
    col_input, col_output = st.columns([1.05, 1.35], gap="large")

    # --------------------------------------------------------------------------
    # LEFT: selection, plan, buffer
    # --------------------------------------------------------------------------
    with col_input:
        section("01", "Facility & Dish Selection", "Only dishes this hub has actually sold are listed.")

        hubs = catalog.drop_duplicates("center_id").sort_values("center_id")
        hub_labels = {
            int(r.center_id): f"Hub {int(r.center_id)} — {r.center_type} (City {int(r.city_code)}, {r.op_area} km²)"
            for r in hubs.itertuples()
        }
        center_id = st.selectbox("Fulfillment Hub", list(hub_labels), format_func=hub_labels.get)

        hub_meals = catalog[catalog["center_id"] == center_id]
        meal_labels = {
            int(r.meal_id): f"Meal {int(r.meal_id)} — {r.category} ({r.cuisine})"
            for r in hub_meals.itertuples()
        }
        meal_id = st.selectbox("Catalog Dish", list(meal_labels), format_func=meal_labels.get)

        state = get_pair_state(panel, center_id, meal_id)
        last_actual = float(state["orders_hist"][-1])
        recent_empty = int((state["orders_hist"][-4:] == 0).sum())

        if last_actual > 0:
            st.caption(f"✓ History loaded — week {state['last_week']} actual: **{last_actual:,.0f} orders**")
        else:
            st.caption(f"ℹ️ No order recorded in week {state['last_week']}; "
                       f"last recorded week: **{state['last_observed_week']}**.")
        if recent_empty:
            st.warning(
                f"{recent_empty} of the last 4 weeks have no recorded orders. The model reads those weeks "
                "as zero demand, so this forecast is less reliable than usual."
            )

        section("02", "Planned Prices & Promotions",
                f"Weeks {state['last_week'] + 1}–{state['last_week'] + HORIZON}. Edit any cell; the forecast updates.")

        default_plan, n_prefilled = make_default_plan(state, planned, HORIZON)
        if n_prefilled:
            st.caption(f"Pre-filled from the official plan for {n_prefilled} of {HORIZON} weeks; "
                       "other weeks start from the latest known price with no promotion.")
        else:
            st.caption("Starts from the latest known price with no promotion.")

        plan_edit = st.data_editor(
            default_plan,
            key=f"plan_{center_id}_{meal_id}",
            hide_index=True,
            disabled=["week"],
            num_rows="fixed",
            width="stretch",
            column_config={
                "week": st.column_config.NumberColumn("Week", format="%d"),
                "base_price": st.column_config.NumberColumn("Base price", min_value=1.0, step=1.0, format="%.2f"),
                "checkout_price": st.column_config.NumberColumn("Checkout price", min_value=1.0, step=1.0, format="%.2f"),
                "emailer_for_promotion": st.column_config.CheckboxColumn("📧 Email"),
                "homepage_featured": st.column_config.CheckboxColumn("🏠 Homepage"),
            },
        )

        price_cols = plan_edit[["base_price", "checkout_price"]]
        if price_cols.isna().any().any() or (price_cols <= 0).any().any():
            st.error("Every week needs a positive base price and checkout price.")
            st.stop()

        plan = plan_edit.copy()
        plan["emailer_for_promotion"] = plan["emailer_for_promotion"].astype(int)
        plan["homepage_featured"] = plan["homepage_featured"].astype(int)

        wk1 = plan.iloc[0]
        wk1_pct = (wk1["checkout_price"] - wk1["base_price"]) / wk1["base_price"]
        st.markdown(price_badge(wk1_pct), unsafe_allow_html=True)

        section("03", "Kitchen Prep Buffer", "A planning choice layered on the forecast, not a statistical interval.")
        buffer_pct = st.slider("Safety buffer on forecast (%)", 0, 40, 15, step=5) / 100.0

    # --------------------------------------------------------------------------
    # RIGHT: forecast outputs and what-if charts
    # --------------------------------------------------------------------------
    with col_output:
        section("04", f"{HORIZON}-Week Forecast", "Week 1 uses true history; later weeks feed earlier predictions back in.")

        fc = forecast_pair(bundle, state, plan)
        fc["prep_quantity"] = fc["predicted_orders"].apply(lambda x: safety_stock(x, buffer_pct))
        fc["revenue"] = fc["predicted_orders"] * fc["checkout_price"]

        first = fc.iloc[0]
        delta_str = None
        if last_actual > 0:
            delta_str = f"{(first['predicted_orders'] / last_actual - 1) * 100:+.1f}% vs. week {state['last_week']}"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.metric(f"WEEK {int(first['week'])} FORECAST", f"{first['predicted_orders']:,.0f}",
                          delta=delta_str, help="One-week-ahead forecast built from true history.")
        with m2:
            with st.container(border=True):
                st.metric(f"{HORIZON}-WEEK TOTAL", f"{fc['predicted_orders'].sum():,.0f}",
                          help="Sum of the recursive weekly forecasts. Accuracy decreases after week 1.")
        with m3:
            with st.container(border=True):
                st.metric("WEEK 1 PREP QTY", f"{first['prep_quantity']:,.0f}",
                          delta=f"+{buffer_pct * 100:.0f}% buffer", delta_color="off",
                          help="Forecast plus the safety buffer chosen on the left.")
        with m4:
            with st.container(border=True):
                st.metric("EST. REVENUE", f"${fc['revenue'].sum():,.0f}",
                          delta=f"over {HORIZON} weeks", delta_color="off",
                          help="Sum of checkout price × forecast orders.")

        if band is not None:
            lo, hi = first["predicted_orders"] * band["lo"], first["predicted_orders"] * band["hi"]
            st.caption(f"Week {int(first['week'])} typical range: **{lo:,.0f} to {hi:,.0f} orders** · {band['label']}. "
                       "Later weeks carry compounding error, so treat them as directional.")

        # Forecast chart: recorded history (gaps where nothing was recorded) + forecast
        hist_weeks = state["weeks"][-HISTORY_WEEKS:]
        hist_orders = state["orders_hist"][-HISTORY_WEEKS:]
        hist_y = np.where(hist_orders > 0, hist_orders, np.nan)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist_weeks, y=hist_y, mode="lines+markers", name="Recorded orders",
            line=dict(color="#3b82f6", width=2.4), marker=dict(size=5), connectgaps=False,
            hovertemplate="Week %{x}<br>Actual: %{y:,.0f}<extra></extra>",
        ))
        fx, fy = fc["week"].tolist(), fc["predicted_orders"].tolist()
        if hist_orders[-1] > 0:
            fx, fy = [int(hist_weeks[-1])] + fx, [float(hist_orders[-1])] + fy
        fig_fc.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines+markers", name="Forecast",
            line=dict(color=ACCENT_COLOR, width=2.8, dash="dash"), marker=dict(size=6),
            hovertemplate="Week %{x}<br>Forecast: %{y:,.0f}<extra></extra>",
        ))
        if band is not None:
            p1 = float(first["predicted_orders"])
            fig_fc.add_trace(go.Scatter(
                x=[int(first["week"])], y=[p1], mode="markers", name="Week 1 range",
                marker=dict(size=1, color=ACCENT_COLOR),
                error_y=dict(type="data", symmetric=False, array=[p1 * (band["hi"] - 1)],
                             arrayminus=[p1 * (1 - band["lo"])], color=ACCENT_COLOR, thickness=2.2, width=7),
                hovertemplate=f"Typical range: {p1 * band['lo']:,.0f}–{p1 * band['hi']:,.0f}<extra></extra>",
            ))
        chart_layout(
            fig_fc,
            title=f"Meal {meal_id} at Hub {center_id}: recorded demand and {HORIZON}-week forecast",
            xaxis_title="Operational week", yaxis_title="Weekly orders", height=340, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc, width="stretch", theme="streamlit")

        st.dataframe(
            fc[["week", "checkout_price", "emailer_for_promotion", "homepage_featured",
                "predicted_orders", "prep_quantity", "revenue"]],
            hide_index=True, width="stretch",
            column_config={
                "week": st.column_config.NumberColumn("Week", format="%d"),
                "checkout_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "emailer_for_promotion": st.column_config.CheckboxColumn("Email"),
                "homepage_featured": st.column_config.CheckboxColumn("Homepage"),
                "predicted_orders": st.column_config.NumberColumn("Forecast orders", format="%.0f"),
                "prep_quantity": st.column_config.NumberColumn("Prep qty", format="%.0f"),
                "revenue": st.column_config.NumberColumn("Est. revenue", format="$%.0f"),
            },
        )

        # ----- What-if: next week, from true history -----
        wk1_week = int(first["week"])
        base_p, chk_p = float(wk1["base_price"]), float(wk1["checkout_price"])
        em1, hp1 = int(wk1["emailer_for_promotion"]), int(wk1["homepage_featured"])

        section("05", "Price Elasticity & Revenue Sensitivity",
                f"Week {wk1_week} demand across ±30% price changes, other inputs as planned.")
        price_df = simulate_price_sensitivity(bundle, state, base_p, em1, hp1, week=wk1_week)
        opt = price_df.loc[price_df["expected_revenue"].idxmax()]

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=price_df["checkout_price"], y=price_df["predicted_orders"], mode="lines+markers",
            name="Demand curve", line=dict(color=ACCENT_COLOR, width=2.8), marker=dict(size=5),
            hovertemplate="Price: $%{x:.2f}<br>Forecast: %{y:,.0f} orders<extra></extra>",
        ))
        fig_price.add_trace(go.Scatter(
            x=[chk_p], y=[first["predicted_orders"]], mode="markers", name="Planned price",
            marker=dict(size=12, symbol="star", color="#3b82f6", line=dict(color="#1d4ed8", width=1.8)),
            hovertemplate=f"Planned: ${chk_p:.2f}<br>Forecast: {first['predicted_orders']:,.0f}<extra></extra>",
        ))
        fig_price.add_trace(go.Scatter(
            x=[opt["checkout_price"]], y=[opt["predicted_orders"]], mode="markers", name="Revenue-maximizing price",
            marker=dict(size=11, symbol="diamond", color="#10b981", line=dict(color="#047857", width=1.8)),
            hovertemplate=(f"Max revenue price: ${opt['checkout_price']:.2f}<br>"
                           f"Forecast: {opt['predicted_orders']:,.0f}<br>Revenue: ${opt['expected_revenue']:,.0f}<extra></extra>"),
        ))
        chart_layout(
            fig_price, title=f"Price sensitivity for Meal {meal_id} (week {wk1_week})",
            xaxis_title="Checkout price", yaxis_title="Forecast weekly orders", height=320, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_price, width="stretch", theme="streamlit")
        st.caption("Model-implied relationship learned from observational history, not a causal elasticity. "
                   "Tree models give stepwise curves; read the direction, not the decimals.")

        section("06", "Promotional Channel Lift", f"Week {wk1_week} demand under each promotion combination.")
        promo_df = simulate_promo_scenarios(bundle, state, base_p, chk_p, week=wk1_week)
        fig_promo = px.bar(
            promo_df, x="Scenario", y="Predicted Orders", text="Predicted Orders", color="Scenario",
            color_discrete_sequence=[ACCENT_COLOR, "#0284c7", "#6366f1", "#10b981"],
            title="Forecast demand by marketing channel",
        )
        fig_promo.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        chart_layout(fig_promo, height=280, showlegend=False)
        st.plotly_chart(fig_promo, width="stretch", theme="streamlit")

        with st.expander("How to read this forecast"):
            st.markdown(
                f"- **Week 1** is built from true history; **weeks 2–{HORIZON}** feed earlier predictions back in as lags, "
                "so errors compound.\n"
                "- The scoreboard on the next tab measures **one-week-ahead** accuracy only.\n"
                "- Weeks with no recorded order enter the lag features as zero (a feature-engineering assumption).\n"
                f"- Deployed model: `{bundle['source']}`, trained on weeks 1–{bundle['train_week']}.\n"
                "- Prep quantity = forecast × (1 + buffer). The buffer is a policy knob."
            )

        with st.expander("Model inputs for the first forecast week"):
            row = build_feature_row(state, wk1_week, chk_p, base_p, em1, hp1)
            shown = pd.DataFrame({"feature": list(row.keys()), "value": [str(v) for v in row.values()]})
            st.dataframe(shown, hide_index=True, width="stretch")

# ==============================================================================
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    section(
        "01", "Model Evaluation Scoreboard",
        "Trained on weeks 1–131 and scored on validation weeks 132–145, one week ahead with true history available.",
    )
    st.dataframe(
        benchmarks_df, width="stretch", hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Model Architecture": st.column_config.TextColumn("Model", width="medium"),
            "Family": st.column_config.TextColumn("Family", width="medium"),
            "Validation WAPE (%)": st.column_config.ProgressColumn(
                "Validation WAPE (%)", help="Weighted absolute percentage error (lower is better)",
                format="%.2f%%", min_value=20.0, max_value=50.0),
            "Validation MAPE (%)": st.column_config.NumberColumn("Validation MAPE (%)", format="%.2f%%"),
            "Bias (%)": st.column_config.NumberColumn(
                "Bias (%)", format="%+.2f%%",
                help="Total forecast vs total actual. Negative = under-forecast (stockout risk)."),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Notes": st.column_config.TextColumn("Notes", width="large"),
        },
    )
    callout(
        "<b>Reading the scores.</b> Differences of a few tenths of a WAPE point are within noise for one split of "
        "14 weeks. All figures are one-week-ahead; multi-week forecasts in the planner use recursive predictions "
        "and are expected to be less accurate."
    )

    section("02", "Error Metric Comparison", "WAPE is the primary supply-chain KPI; bias shows over- vs under-forecasting.")
    error_scale = [[0.0, "#10b981"], [0.33, "#3b82f6"], [0.66, "#f59e0b"], [1.0, "#ef4444"]]
    b1, b2, b3 = st.columns(3)
    with b1:
        fig = px.bar(benchmarks_df, x="Validation WAPE (%)", y="Model Architecture", orientation="h",
                     text="Validation WAPE (%)", color="Validation WAPE (%)",
                     color_continuous_scale=error_scale, title="WAPE (%) — lower is better")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        chart_layout(fig, height=320, yaxis=dict(autorange="reversed", title=""), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", theme="streamlit")
    with b2:
        fig = px.bar(benchmarks_df, x="Validation MAPE (%)", y="Model Architecture", orientation="h",
                     text="Validation MAPE (%)", color="Validation MAPE (%)",
                     color_continuous_scale=error_scale, title="MAPE (%) — item-level error")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        chart_layout(fig, height=320, yaxis=dict(autorange="reversed", title=""), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", theme="streamlit")
    with b3:
        bias_df = benchmarks_df.assign(
            Direction=np.where(benchmarks_df["Bias (%)"] > 0, "Over-forecast", "Under-forecast"))
        fig = px.bar(bias_df, x="Bias (%)", y="Model Architecture", orientation="h", text="Bias (%)",
                     color="Direction", title="Bias (%) — total forecast vs actual",
                     color_discrete_map={"Over-forecast": "#f59e0b", "Under-forecast": "#3b82f6"})
        fig.update_traces(texttemplate="%{text:+.2f}%", textposition="outside", cliponaxis=False)
        chart_layout(fig, height=320, yaxis=dict(autorange="reversed", title=""), showlegend=False)
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    section("03", "LightGBM Feature Importance (Gain)", "Share of total loss reduction attributed to each feature.")
    booster = bundle["model"].booster_
    fi = pd.DataFrame({"Feature": booster.feature_name(),
                       "Gain": booster.feature_importance(importance_type="gain")})
    fi["Gain share (%)"] = fi["Gain"] / fi["Gain"].sum() * 100
    fi = fi.sort_values("Gain share (%)").tail(12)
    fig_fi = px.bar(fi, x="Gain share (%)", y="Feature", orientation="h", text="Gain share (%)",
                    color="Gain share (%)",
                    color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#6366f1"], [1.0, ACCENT_COLOR]],
                    title="Top 12 features in the deployed LightGBM model")
    fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    chart_layout(fig_fi, height=390, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, width="stretch", theme="streamlit")

# ==============================================================================
# TAB 3: PORTFOLIO & KITCHEN ANALYTICS
# ==============================================================================
with tab_analytics:
    section("01", "Operational Portfolio Overview",
            f"Recorded demand across {kpis['hubs']} hubs and {kpis['dishes']} dishes, weeks "
            f"{kpis['first_week']}–{kpis['last_week']}.")
    t1, t2, t3, t4 = st.columns(4)
    for col, label, value, hint in [
        (t1, "HUBS", f"{kpis['hubs']:,}", "Fulfillment centers in the panel."),
        (t2, "DISHES", f"{kpis['dishes']:,}", "Distinct catalog meals."),
        (t3, "HUB–DISH PAIRS", f"{kpis['pairs']:,}", "Combinations that appear at least once in the data."),
        (t4, "RECORDED ORDERS", f"{kpis['total_orders'] / 1e6:,.1f}M", "Total orders across all weeks."),
    ]:
        with col:
            with st.container(border=True):
                st.metric(label, value, help=hint)

    section("02", "Demand Composition by Category & Cuisine", "Volume distribution across food categories and styles.")
    a1, a2 = st.columns(2)
    with a1:
        cat_df = analytics["by_category"]
        fig_cat = px.bar(cat_df, x="total_orders", y="category", orientation="h", text="total_orders",
                         color="total_orders",
                         color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#0d9488"], [1.0, "#10b981"]],
                         title="Total recorded orders by meal category")
        fig_cat.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        chart_layout(fig_cat, height=420, yaxis=dict(autorange="reversed", title=""), coloraxis_showscale=False)
        st.plotly_chart(fig_cat, width="stretch", theme="streamlit")
    with a2:
        cui_df = analytics["by_cuisine"]
        fig_cui = px.pie(cui_df, values="total_orders", names="cuisine", hole=0.52,
                         color_discrete_sequence=[ACCENT_COLOR, "#0284c7", "#10b981", "#d97706", "#8b5cf6"],
                         title="Demand share by cuisine")
        fig_cui.update_traces(textinfo="percent+label",
                              marker=dict(line=dict(color="rgba(128, 128, 128, 0.25)", width=1.5)))
        chart_layout(fig_cui, height=420)
        st.plotly_chart(fig_cui, width="stretch", theme="streamlit")

    weekly = analytics["weekly"]
    section("03", "Platform Weekly Order Trajectory", "Network-wide order volume by week.")
    fig_weekly = px.line(weekly, x="week", y="num_orders", markers=True,
                         title=f"Total platform weekly orders (weeks {kpis['first_week']}–{kpis['last_week']})")
    fig_weekly.update_traces(line_color=ACCENT_COLOR, line_width=2.5, marker=dict(size=4.5),
                             hovertemplate="Week %{x}<br>Orders: %{y:,.0f}<extra></extra>")
    chart_layout(fig_weekly, xaxis_title="Operational week", yaxis_title="Total orders",
                 yaxis=dict(tickformat=","), height=320)
    st.plotly_chart(fig_weekly, width="stretch", theme="streamlit")

    section("04", "Data Coverage", "Share of hub–dish pairs with at least one recorded order in each week.")
    early = 100 - weekly.loc[weekly["week"] <= 131, "coverage_pct"].mean()
    late = 100 - weekly.loc[weekly["week"] > 131, "coverage_pct"].mean()
    fig_cov = px.area(weekly, x="week", y="coverage_pct", title="Recorded hub–dish pairs per week (%)")
    fig_cov.update_traces(line_color="#3b82f6", hovertemplate="Week %{x}<br>Coverage: %{y:.1f}%<extra></extra>")
    chart_layout(fig_cov, xaxis_title="Operational week", yaxis_title="% of pairs with an order",
                 yaxis=dict(range=[0, 100]), height=280)
    st.plotly_chart(fig_cov, width="stretch", theme="streamlit")
    if early > 1.5 * late:
        headline = "Gaps are concentrated early in the history."
    elif late > 1.5 * early:
        headline = "Gaps are concentrated in recent weeks."
    else:
        headline = "Gaps are spread fairly evenly across the history."
    callout(
        f"<b>{headline}</b> {early:.1f}% of hub–dish cells are empty in weeks 1–131 versus {late:.1f}% in the most "
        "recent 14 weeks. Empty cells are zero-filled in the model's lag features, an assumption rather than a measured zero."
    )