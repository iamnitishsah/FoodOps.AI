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
import plotly.graph_objects as go

from DeliveryOps.src.inference import DeliveryOpsEngine, FEATURE_LABELS, format_feature_value, store_history_tier
from DeliveryOps.src import data_loader as dl

try:
    from DeliveryOps.app.theme import apply_theme, masthead, section, callout, chart_layout, ACCENT_COLOR
except ImportError:
    from theme import apply_theme, masthead, section, callout, chart_layout, ACCENT_COLOR

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

GREEN, AMBER, RED, INDIGO, SKY = "#10b981", "#f59e0b", "#ef4444", "#6366f1", "#0284c7"
ERROR_SCALE = [[0.0, GREEN], [0.5, AMBER], [1.0, RED]]

# The platform's order-place estimate is almost determined by the order protocol (251 s or 446 s)
PLACE_ESTIMATE = {1: 446, 2: 251, 3: 251, 4: 251, 5: 251, 6: 446}
SEGMENT_TITLES = {"market": "Market", "hour": "Hour of day", "day": "Day", "telemetry": "Dispatch telemetry",
                  "store history": "Store history"}


@st.cache_resource(show_spinner="Loading trained ETA models...")
def load_engine():
    """Caches the model bundle in memory so it doesn't reload on every UI click."""
    return DeliveryOpsEngine()


@st.cache_data(show_spinner="Loading model card...")
def get_card():
    return dl.load_model_card()


@st.cache_data(show_spinner="Loading delivery data...")
def get_data_products(_store_table):
    """Aggregates built once from the processed feature table (None when the table is not shipped)."""
    table = dl.load_feature_table()
    if table is None:
        return None
    stores = dl.build_store_catalog(table, _store_table)
    return {"analytics": dl.build_analytics(table), "ranges": dl.input_ranges(table), "stores": stores,
            "store_labels": stores["label"].to_dict()}


@st.cache_resource(show_spinner="Verifying training/serving parity...")
def run_parity(_engine):
    """Rebuild features for test-week orders and compare them with the saved feature table."""
    table = dl.load_feature_table()
    return None if table is None else _engine.parity_check(table)


# Initialize Core Resources
try:
    engine = load_engine()
    card = get_card()
    products = get_data_products(engine.store_table)
    parity = run_parity(engine)
    model_loaded = True
except Exception as e:
    st.error(f"Initialization Error: Unable to load DeliveryOps artifacts. Error: {e}")
    model_loaded = False

if not model_loaded:
    st.stop()

if parity is not None and not parity["ok"]:
    st.error(
        "Training/serving parity check failed, so the app will not serve estimates. "
        f"Mismatched features: {parity['mismatched_features'] or 'none'}; "
        f"max prediction difference: {parity['max_prediction_diff_s']:.3g} s."
    )
    st.stop()

ranges = products["ranges"] if products else dl.FALLBACK_RANGES
analytics = products["analytics"] if products else None
stores = products["stores"] if products else pd.DataFrame()
store_labels = products["store_labels"] if products else {}

scoreboard = dl.card_scoreboard(card)
intervals = dl.card_intervals(card)
segments = dl.card_segments(card)

test_mae = dl.dig(card, "test", "point_models", "lightgbm P50", "MAE (min)")
test_bias = dl.dig(card, "test", "point_models", "lightgbm P50", "bias (min)")
test_cov80 = dl.dig(card, "test", "intervals", "80% interval coverage % (nominal 80)")

# Masthead
data_info = card.get("data", {})
try:
    start, end = pd.Timestamp(data_info["train_window"][0]), pd.Timestamp(data_info["test_window"][1])
    week_label = (f"{start:%b %d} – {end:%b %d, %Y} · {len(engine.levels['market_id'])} Markets · "
                  f"{data_info['rows_after_cleaning']:,} Orders")
except Exception:
    week_label = "Historical panel"
model_label = (f"LightGBM P10/P50/P90 · {test_mae:.2f} min test MAE" if test_mae is not None
               else "LightGBM P10/P50/P90")
masthead(week_label=week_label, model_label=model_label)

mismatches = dl.version_mismatches(card)
if mismatches:
    st.warning("Library versions differ from the ones the models were trained with: "
               + ", ".join(f"{pkg} {trained} (trained) vs {installed} (installed)" for pkg, trained, installed in mismatches)
               + ". Pin the trained versions in requirements.txt if predictions look off.")


# ------------------------------------------------------------------------------
# CHART HELPERS
# ------------------------------------------------------------------------------
def band_chart(sweep: pd.DataFrame, x: str, xtitle: str, title: str, selected=None) -> go.Figure:
    """Expected time (P50) with the P10-P90 window as a band."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep[x], y=sweep["p90_min"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=sweep[x], y=sweep["p10_min"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(255, 107, 53, 0.18)", name="P10–P90 window", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sweep[x], y=sweep["p50_min"], mode="lines+markers",
                             line=dict(color=ACCENT_COLOR, width=3), marker=dict(size=6), name="Expected (P50)",
                             customdata=sweep[["p10_min", "p90_min"]].to_numpy(),
                             hovertemplate="%{x}<br>Expected: %{y:.1f} min<br>Window: %{customdata[0]:.1f} to "
                                           "%{customdata[1]:.1f} min<extra></extra>"))
    if selected is not None:
        fig.add_vline(x=selected, line_dash="dash", line_color="rgba(128, 128, 128, 0.65)")
    chart_layout(fig, height=340, title=title, xaxis_title=xtitle, yaxis_title="Delivery time (min)",
                 legend=dict(orientation="h", y=-0.22))
    return fig


def waterfall_chart(explanation: dict) -> go.Figure:
    """Contribution of each input to the expected time, in minutes."""
    c = explanation["contributions"]
    base, pred = explanation["base_min"], explanation["prediction_min"]
    labels = (["Average order"] + [f"{r.label} = {r.value}" if r.value else r.label for r in c.itertuples()]
              + ["Expected time (P50)"])
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(c) + ["total"],
        y=labels,
        x=[base] + c["shap_min"].tolist() + [pred],
        text=[f"{base:.1f}"] + [f"{v:+.1f}" for v in c["shap_min"]] + [f"{pred:.1f}"],
        textposition="outside",
        increasing=dict(marker=dict(color=RED)),
        decreasing=dict(marker=dict(color=GREEN)),
        totals=dict(marker=dict(color=ACCENT_COLOR)),
        connector=dict(line=dict(color="rgba(128, 128, 128, 0.45)")),
        hoverinfo="skip",
    ))
    running = base + np.cumsum(c["shap_min"].to_numpy())
    low, high = min(base, running.min()), max(base, running.max())
    pad = max(2.0, (high - low) * 0.3)
    chart_layout(fig, height=430, title="Red adds time, green removes time (minutes)",
                 xaxis=dict(range=[low - pad, high + pad], title="Delivery time (min)"),
                 yaxis=dict(autorange="reversed", automargin=True), showlegend=False)
    return fig


def segment_row(name: str, label: str):
    table = segments.get(name)
    if table is None:
        return None
    match = table[table["Segment"] == label]
    return None if match.empty else match.iloc[0]


# Primary Navigation Tabs
tab_pred, tab_benchmarks, tab_analytics = st.tabs([
    "ETA Simulator",
    "Model Benchmarks & Evaluation",
    "Network & Delivery Analytics"
])

# ==============================================================================
# TAB 1: ETA SIMULATOR
# ==============================================================================
with tab_pred:
    st.markdown(
        "Enter an order and the market's dispatch load to see the expected delivery time, the P10 to P90 promise "
        "window and the factors behind it.")

    col1, col2, col3 = st.columns(3)

    with col1:
        section("01", "Market & Timing")
        market_id = st.selectbox("Market", options=engine.levels["market_id"])
        order_hour = st.slider("Local hour of day", 7, 22, 18, help="The model was trained on orders from 07:00 to 22:59 local time.")
        weekday_name = st.selectbox("Day of week", options=dl.WEEKDAYS, index=4)
        drive = ranges["estimated_store_to_consumer_driving_duration"]
        drive_s = st.number_input("Platform drive-time estimate (seconds)", min_value=drive["lo"],
                                  max_value=drive["hi"], value=drive["median"], step=30)
        protocol = st.selectbox("Order protocol", options=engine.levels["order_protocol"])
        default_place = PLACE_ESTIMATE.get(int(protocol), 251)
        place_s = st.radio("Order-place estimate (seconds)", options=[251, 446], index=[251, 446].index(default_place),
                           horizontal=True, key=f"place_{protocol}",
                           help="Set by the platform and nearly determined by the protocol: 251 s for protocols 2, 3 "
                                "and 5, 446 s for protocols 1 and 6. Protocol 4 mixes both.")

    with col2:
        section("02", "Dispatch Load")
        telemetry_on = st.toggle("Dispatch telemetry available", value=True,
                                 help="Switch off to simulate an order without live dasher data, as in the "
                                      "orders where telemetry was missing.")
        onshift_r, busy_r, out_r = (ranges["total_onshift_dashers"], ranges["total_busy_dashers"],
                                    ranges["total_outstanding_orders"])
        onshift = st.number_input("Dashers on shift", min_value=0, max_value=onshift_r["hi"],
                                  value=onshift_r["median"], disabled=not telemetry_on)
        busy = st.number_input("Busy dashers", min_value=0, max_value=busy_r["hi"], value=busy_r["median"],
                               disabled=not telemetry_on)
        outstanding = st.number_input("Outstanding orders", min_value=0, max_value=out_r["hi"],
                                      value=out_r["median"], disabled=not telemetry_on)

        st.caption("Load metrics:")
        if not telemetry_on:
            st.write("- Telemetry unavailable: the model sees the dasher inputs and both ratios as missing.")
        elif onshift == 0:
            st.write("- No dasher on shift: the load ratios are undefined and the model sees them as missing.")
        else:
            st.write(f"- Busy dashers per on-shift dasher: **{busy / onshift:.2f}**")
            st.write(f"- Outstanding orders per on-shift dasher: **{outstanding / onshift:.2f}**")

    with col3:
        section("03", "Store & Order")
        store_modes = ["Known store", "New store (no history)"] if len(stores) else ["New store (no history)"]
        store_mode = st.radio("Store", options=store_modes, horizontal=True)
        store_id = None
        if store_mode == "Known store":
            store_id = st.selectbox("Select store", options=stores.index.tolist(), format_func=store_labels.get)
            category = stores.loc[store_id, "category"]
            st.caption(f"Category: {category} · most common market: {int(stores.loc[store_id, 'market'])}")
        else:
            categories = engine.levels["store_primary_category"]
            category = st.selectbox("Store category", options=categories,
                                    index=categories.index("american") if "american" in categories else 0)

        summary = engine.store_summary(store_id)
        if summary:
            st.caption(f"Training history: {summary['orders']:,} orders ({summary['tier']}), average "
                       f"{summary['avg_min']:.1f} min, smoothed store effect {summary['encoding_min']:.1f} min.")
        else:
            st.caption(f"No training history: the model uses the overall training average of "
                       f"{engine.store_prior / 60:.1f} min and is less accurate for new stores.")

        items_r, price_r = ranges["total_items"], ranges["subtotal"]
        total_items = st.number_input("Items in cart", min_value=max(items_r["lo"], 1), max_value=items_r["hi"],
                                      value=min(max(items_r["median"], 1), items_r["hi"]))
        num_distinct = st.number_input("Distinct items", min_value=1, max_value=int(total_items),
                                       value=min(int(total_items), 3))
        subtotal = st.number_input("Subtotal (cents)", min_value=max(price_r["lo"], 1), max_value=price_r["hi"],
                                   value=price_r["median"], step=100)
        min_r, max_r = ranges["min_item_price"], ranges["max_item_price"]
        min_price = st.number_input("Cheapest item (cents)", min_value=min_r["lo"], max_value=min_r["hi"],
                                    value=min_r["median"], step=50)
        max_price = st.number_input("Priciest item (cents)", min_value=int(min_price),
                                    max_value=max(max_r["hi"], int(min_price)),
                                    value=max(max_r["median"], int(min_price)), step=50)

    order = {
        "market_id": int(market_id),
        "store_id": store_id,
        "store_primary_category": category,
        "order_protocol": int(protocol),
        "total_items": int(total_items),
        "subtotal": int(subtotal),
        "num_distinct_items": int(num_distinct),
        "min_item_price": int(min_price),
        "max_item_price": int(max_price),
        "total_onshift_dashers": float(onshift) if telemetry_on else np.nan,
        "total_busy_dashers": float(busy) if telemetry_on else np.nan,
        "total_outstanding_orders": float(outstanding) if telemetry_on else np.nan,
        "estimated_order_place_duration": int(place_s),
        "estimated_store_to_consumer_driving_duration": float(drive_s),
        "order_hour": int(order_hour),
        "order_day_of_week": dl.WEEKDAYS.index(weekday_name),
    }

    st.divider()
    section("04", "Customer Delivery Window")
    results = engine.simulate_eta(order)

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        with st.container(border=True):
            st.metric(label="OPTIMISTIC ETA (P10)", value=f"{results['p10_min']} min",
                      delta=f"{results['p10_min'] - results['p50_min']:+.1f} min vs expected", delta_color="off")
    with res_col2:
        with st.container(border=True):
            st.metric(label="EXPECTED ETA (P50)", value=f"{results['p50_min']} min", delta="Model median",
                      delta_color="off")
    with res_col3:
        with st.container(border=True):
            st.metric(label="PROMISED ETA (P90)", value=f"{results['p90_min']} min",
                      delta=f"{results['p90_min'] - results['p50_min']:+.1f} min vs expected", delta_color="off")

    st.info(
        f"**Customer Promise Window:** {results['p10_min']} to {results['p90_min']} minutes "
        f"(width: {results['spread_min']} min)")
    if test_mae is not None and test_bias is not None:
        coverage_note = f"; the P10 to P90 window held {test_cov80:.1f}% of orders (nominal 80%)" if test_cov80 else ""
        st.caption(
            f"Context: on the unseen test week the expected time missed by {test_mae:.2f} min on average and ran "
            f"{abs(test_bias):.1f} min {'short' if test_bias < 0 else 'long'}{coverage_note}.")

    # --- Reliability for this kind of order (test-week segments from the model card) ---
    tier = summary["tier"] if summary else store_history_tier(0)
    telemetry_state = ("telemetry missing" if not telemetry_on
                       else "zero on-shift" if onshift == 0 else "telemetry present")
    lines = []
    for title, seg_name, label in [(f"Store history: {tier}", "store history", tier),
                                   (f"Dispatch: {telemetry_state}", "telemetry", telemetry_state),
                                   (f"Market {int(market_id)}", "market", str(int(market_id)))]:
        row = segment_row(seg_name, label)
        if row is not None:
            lines.append(f"<b>{title}</b>: typical test-week error {row['LightGBM MAE (min)']:.2f} min, "
                         f"bias {row['Bias (min)']:+.1f} min, P90 coverage {row['P90 coverage (%)']:.1f}%")
    if lines:
        section("05", "How Reliable Is This Estimate?", "Test-week results for orders like this one.")
        callout("<br>".join(lines))

    # --- What-if charts ---
    section("06", "What-If Analysis", "The same order with one input changed. The shaded band is the P10 to P90 window.")
    w_col1, w_col2 = st.columns(2)
    with w_col1:
        hours = engine.sweep(order, "order_hour", range(7, 23))
        st.plotly_chart(band_chart(hours, "order_hour", "Local hour of day", "Delivery time across the day",
                                   selected=order["order_hour"]), width="stretch", theme="streamlit")
    with w_col2:
        if not telemetry_on:
            st.info("The dispatch-load what-if needs telemetry. Switch it on to see how outstanding orders move the estimate.")
        elif onshift == 0:
            st.info("Set at least one dasher on shift to see how outstanding orders move the estimate.")
        else:
            top = max(int(outstanding) * 2, out_r["median"] * 2, 20)
            loads = engine.sweep(order, "total_outstanding_orders", np.unique(np.linspace(0, top, 25).round().astype(int)))
            st.plotly_chart(band_chart(loads, "total_outstanding_orders", "Outstanding orders",
                                       "Delivery time as dispatch load rises", selected=order["total_outstanding_orders"]),
                            width="stretch", theme="streamlit")

    # --- Why this ETA ---
    section("07", "Why This ETA",
            "How each input moves the expected time away from the average order (SHAP values of the P50 model).")
    try:
        st.plotly_chart(waterfall_chart(engine.explain(order)), width="stretch", theme="streamlit")
    except ImportError:
        st.info("Install the `shap` package to see the breakdown of this estimate.")

    with st.expander("How this ETA was produced"):
        st.markdown(
            "The 20 model features are rebuilt from the inputs above with the same definitions used in training: "
            "the two load ratios (missing when nobody is on shift or telemetry is missing), the store encoding from "
            "the table saved in the model bundle (a new store receives the training average) and the saved category "
            "levels. The P10, P50 and P90 models then predict, and the three values are sorted so that P10 ≤ P50 ≤ P90.")
        features = engine.build_features(pd.DataFrame([order]))
        st.dataframe(pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in engine.features],
            "Value": [format_feature_value(f, features[f].iloc[0]) for f in engine.features],
        }), hide_index=True, width="stretch")
        if parity is not None:
            st.success(f"Training/serving parity verified at start-up on {parity['n_rows']} test-week orders "
                       f"(largest prediction difference {parity['max_prediction_diff_s']:.1e} s).")
        else:
            st.info("The feature table is not available, so the start-up parity check was skipped.")

# ==============================================================================
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    section("01", "Delivery ETA Scoreboard",
            "Mean absolute error (MAE) of the expected time on the unseen test week, with the validation folds beside it.")

    st.dataframe(
        scoreboard,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Model Architecture": st.column_config.TextColumn("Model Architecture", width="medium"),
            "Family": st.column_config.TextColumn("Model Family", width="medium"),
            "Test MAE (min)": st.column_config.ProgressColumn(
                "Test MAE (min)",
                help="Mean Absolute Error in minutes on the test week (Lower is better)",
                format="%.2f",
                min_value=8.0,
                max_value=14.0,
            ),
            "Test bias (min)": st.column_config.NumberColumn(
                "Test bias (min)", format="%.2f",
                help="Mean prediction minus actual. Negative means the model under-estimates on average."),
            "Validation MAE A (min)": st.column_config.NumberColumn("Validation MAE A", format="%.2f"),
            "Validation MAE B (min)": st.column_config.NumberColumn("Validation MAE B", format="%.2f"),
            "Role": st.column_config.TextColumn("Role", width="small"),
        },
    )

    b_col1, b_col2 = st.columns([1, 1.5], gap="large")
    with b_col1:
        section("02", "Error Metric Comparison")
        fig_mae = px.bar(
            scoreboard,
            x="Test MAE (min)",
            y="Model Architecture",
            orientation="h",
            text="Test MAE (min)",
            color="Test MAE (min)",
            color_continuous_scale=ERROR_SCALE,
            title="Test MAE Across Architectures",
        )
        fig_mae.update_traces(texttemplate="%{text:.2f} m", textposition="outside", cliponaxis=False)
        chart_layout(fig_mae, height=360, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_mae, width="stretch", theme="streamlit")

    with b_col2:
        section("03", "Interval Coverage vs Nominal",
                "Share of orders at or below P10 and P90, and inside the P10 to P90 window. Intervals are raw, not calibrated.")
        cov = intervals[intervals["Nominal"].notna()]
        if not cov.empty:
            long = cov.melt(id_vars=["Metric", "Nominal"], value_vars=["Fold A", "Fold B", "Test week"],
                            var_name="Evaluation", value_name="Coverage (%)")
            fig_cov = px.bar(long, x="Metric", y="Coverage (%)", color="Evaluation", barmode="group",
                             color_discrete_map={"Fold A": INDIGO, "Fold B": SKY, "Test week": ACCENT_COLOR},
                             text="Coverage (%)", title="Coverage on validation folds and the test week")
            fig_cov.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig_cov.add_trace(go.Scatter(x=cov["Metric"], y=cov["Nominal"], mode="markers", name="Nominal",
                                         marker=dict(symbol="line-ew", size=64, line=dict(width=3, color="#94a3b8"))))
            chart_layout(fig_cov, height=360, yaxis=dict(range=[0, 105]), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_cov, width="stretch", theme="streamlit")
        else:
            st.info("Interval results are not available in the model card.")

    st.dataframe(
        intervals, width="stretch", hide_index=True,
        column_config={
            "Nominal": st.column_config.NumberColumn("Nominal", format="%.0f"),
            "Fold A": st.column_config.NumberColumn("Fold A", format="%.2f"),
            "Fold B": st.column_config.NumberColumn("Fold B", format="%.2f"),
            "Test week": st.column_config.NumberColumn("Test week", format="%.2f"),
        },
    )

    c_col1, c_col2 = st.columns([1, 1.5], gap="large")
    with c_col1:
        section("04", "LightGBM Feature Importance")
        try:
            booster = engine.models["p50"].booster_
            gain = pd.DataFrame({"Feature": booster.feature_name(),
                                 "Gain share (%)": booster.feature_importance(importance_type="gain")})
            gain["Gain share (%)"] = gain["Gain share (%)"] / gain["Gain share (%)"].sum() * 100
            gain = gain.sort_values("Gain share (%)", ascending=True).tail(10)
            fig_fi = px.bar(
                gain, x="Gain share (%)", y="Feature", orientation="h", text="Gain share (%)",
                color="Gain share (%)",
                color_continuous_scale=[[0.0, SKY], [0.5, INDIGO], [1.0, ACCENT_COLOR]],
                title="Top 10 Features of the P50 Model (Share of Gain)",
            )
            fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            chart_layout(fig_fi, height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_fi, width="stretch", theme="streamlit")
        except Exception:
            st.info("Feature importance is currently unavailable.")

    with c_col2:
        section("05", "Feature Ablation",
                f"Each change is measured against the 21-feature reference on two validation folds; positive means worse. "
                f"Noise floor (seed spread): {dl.ABLATION_NOISE_FLOOR['A']:.3f} min (A), {dl.ABLATION_NOISE_FLOOR['B']:.3f} min (B).")
        st.dataframe(
            dl.ABLATION, width="stretch", hide_index=True,
            column_config={
                "Change from the reference": st.column_config.TextColumn("Change from the reference", width="medium"),
                "Delta MAE, Fold A (min)": st.column_config.NumberColumn("ΔMAE A", format="%+.3f"),
                "Delta MAE, Fold B (min)": st.column_config.NumberColumn("ΔMAE B", format="%+.3f"),
                "Days better (of 14)": st.column_config.NumberColumn("Days better", format="%d"),
                "Outcome": st.column_config.TextColumn("Outcome", width="large"),
            },
        )

    section("06", "Test-Week Segments", "Where the model is strongest and weakest on the held-out week.")
    available = [name for name in SEGMENT_TITLES if name in segments]
    if available:
        seg_name = st.selectbox("Segment by", options=available, format_func=SEGMENT_TITLES.get)
        seg = segments[seg_name]
        st.dataframe(
            seg, width="stretch", hide_index=True,
            column_config={
                "Orders": st.column_config.NumberColumn("Orders", format="%d"),
                "Share (%)": st.column_config.NumberColumn("Share (%)", format="%.1f"),
                "LightGBM MAE (min)": st.column_config.NumberColumn("LightGBM MAE", format="%.2f"),
                "Ridge MAE (min)": st.column_config.NumberColumn("Ridge MAE", format="%.2f"),
                "Bias (min)": st.column_config.NumberColumn("Bias", format="%.2f"),
                "P10 coverage (%)": st.column_config.NumberColumn("P10 coverage", format="%.1f%%"),
                "P90 coverage (%)": st.column_config.NumberColumn("P90 coverage", format="%.1f%%"),
                "Mean width (min)": st.column_config.NumberColumn("Mean width", format="%.1f"),
            },
        )
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            mae_long = seg.melt(id_vars="Segment", value_vars=["LightGBM MAE (min)", "Ridge MAE (min)"],
                                var_name="Model", value_name="MAE (min)")
            mae_long["Model"] = mae_long["Model"].str.replace(" MAE (min)", "", regex=False)
            fig_seg = px.bar(mae_long, x="Segment", y="MAE (min)", color="Model", barmode="group",
                             color_discrete_map={"LightGBM": ACCENT_COLOR, "Ridge": INDIGO},
                             title=f"MAE by {SEGMENT_TITLES[seg_name].lower()}")
            chart_layout(fig_seg, height=340, legend=dict(orientation="h", y=-0.25),
                         xaxis=dict(type="category"))
            st.plotly_chart(fig_seg, width="stretch", theme="streamlit")
        with s_col2:
            fig_p90 = px.bar(seg, x="Segment", y="P90 coverage (%)", text="P90 coverage (%)",
                             title=f"P90 coverage by {SEGMENT_TITLES[seg_name].lower()}")
            fig_p90.update_traces(marker_color=SKY, texttemplate="%{text:.1f}", textposition="outside",
                                  cliponaxis=False)
            fig_p90.add_hline(y=90, line_dash="dash", line_color="rgba(128, 128, 128, 0.7)",
                              annotation_text="nominal 90%", annotation_position="top left")
            chart_layout(fig_p90, height=340, yaxis=dict(range=[70, 100]), xaxis=dict(type="category"))
            st.plotly_chart(fig_p90, width="stretch", theme="streamlit")
    else:
        st.info("Segment results are not available in the model card.")

    section("07", "Limitations & How the Model Was Built")
    limitations = card.get("limitations", [])
    if limitations:
        with st.expander("Documented limitations", expanded=True):
            st.markdown("\n".join(f"- {item}" for item in limitations))
    decisions = card.get("decisions", [])
    if decisions:
        with st.expander("Modeling decisions"):
            st.markdown("\n".join(f"- {item}" for item in decisions))

# ==============================================================================
# TAB 3: NETWORK & DELIVERY ANALYTICS
# ==============================================================================
with tab_analytics:
    if analytics is None:
        st.info("The processed feature table is not available, so network analytics are hidden.")
    else:
        kpis = analytics["kpis"]
        section("01", "Delivery Network Overview",
                f"Logistics footprint across {kpis['orders']:,} cleaned deliveries of 15 minutes to 2 hours.")

        t1, t2, t3, t4, t5 = st.columns(5)
        with t1:
            with st.container(border=True):
                st.metric("ORDERS", value=f"{kpis['orders']:,}", help="Cleaned deliveries, Jan 21 to Feb 17, 2015.")
        with t2:
            with st.container(border=True):
                st.metric("ACTIVE MARKETS", value=f"{kpis['markets']}", help="Distinct regional operating zones.")
        with t3:
            with st.container(border=True):
                st.metric("UNIQUE STORES", value=f"{kpis['stores']:,}", help="Stores that fulfilled orders.")
        with t4:
            with st.container(border=True):
                st.metric("MEDIAN DELIVERY", value=f"{kpis['median_min']:.1f} min", help="Order placed to doorstep.")
        with t5:
            with st.container(border=True):
                st.metric("90TH PERCENTILE", value=f"{kpis['p90_min']:.1f} min", help="9 in 10 deliveries finish by then.")

        section("02", "Delivery Time by Hour", "Median and 90th percentile delivery time for each local hour.")
        hour_long = analytics["by_hour"].melt(id_vars=["order_hour", "orders"], value_vars=["median", "p90"],
                                              var_name="Statistic", value_name="Minutes")
        hour_long["Statistic"] = hour_long["Statistic"].map({"median": "Median", "p90": "90th percentile"})
        fig_hour = px.line(hour_long, x="order_hour", y="Minutes", color="Statistic", markers=True,
                           color_discrete_map={"Median": ACCENT_COLOR, "90th percentile": INDIGO},
                           title="Delivery Time by Local Hour (Minutes)")
        fig_hour.update_traces(line_width=3, marker=dict(size=6))
        chart_layout(fig_hour, height=360, xaxis_title="Local hour of day", yaxis_title="Delivery time (min)",
                     legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_hour, width="stretch", theme="streamlit")

        a_col1, a_col2 = st.columns(2)
        with a_col1:
            section("03", "Delivery Time by Weekday")
            fig_day = px.bar(analytics["by_weekday"], x="weekday", y="median", text="median", color="median",
                             color_continuous_scale=ERROR_SCALE, title="Median Delivery Time by Weekday (Minutes)")
            fig_day.update_traces(texttemplate="%{text:.1f} m", textposition="outside", cliponaxis=False)
            chart_layout(fig_day, height=360, xaxis_title="Day of week", yaxis_title="Median delivery time (min)",
                         coloraxis_showscale=False)
            st.plotly_chart(fig_day, width="stretch", theme="streamlit")
        with a_col2:
            section("04", "Delivery Time by Market")
            by_market = analytics["by_market"].copy()
            by_market["market"] = by_market["market"].astype(str)
            fig_mkt = px.bar(by_market, x="market", y="median", text="median", color="median",
                             color_continuous_scale=ERROR_SCALE, title="Median Delivery Time by Market (Minutes)")
            fig_mkt.update_traces(texttemplate="%{text:.1f} m", textposition="outside", cliponaxis=False)
            fig_mkt.add_trace(go.Scatter(x=by_market["market"], y=by_market["p90"], mode="markers", name="90th percentile",
                                         marker=dict(symbol="diamond", size=9, color=INDIGO)))
            chart_layout(fig_mkt, height=360, xaxis_title="Market", yaxis_title="Delivery time (min)",
                         xaxis=dict(type="category"), coloraxis_showscale=False,
                         legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_mkt, width="stretch", theme="streamlit")

        b2_col1, b2_col2 = st.columns([1.4, 1])
        with b2_col1:
            section("05", "Dispatch Load vs Delivery Time",
                    "Deciles of outstanding orders per on-shift dasher: busier markets deliver slower.")
            load = analytics["load"]
            fig_load = go.Figure()
            fig_load.add_trace(go.Scatter(x=load["load"], y=load["p90"], mode="lines", line=dict(width=0),
                                          hoverinfo="skip", showlegend=False))
            fig_load.add_trace(go.Scatter(x=load["load"], y=load["median"], mode="lines+markers",
                                          line=dict(color=ACCENT_COLOR, width=3), marker=dict(size=7), name="Median",
                                          fill="tonexty", fillcolor="rgba(255, 107, 53, 0.14)",
                                          hovertemplate="Load %{x}<br>Median: %{y:.1f} min<extra></extra>"))
            fig_load.add_trace(go.Scatter(x=load["load"], y=load["p90"], mode="lines+markers",
                                          line=dict(color=INDIGO, width=2, dash="dot"), marker=dict(size=6),
                                          name="90th percentile",
                                          hovertemplate="Load %{x}<br>90th percentile: %{y:.1f} min<extra></extra>"))
            chart_layout(fig_load, height=360, title="Delivery Time by Dispatch Load (Minutes)",
                         xaxis_title="Outstanding orders per on-shift dasher (decile median)",
                         yaxis_title="Delivery time (min)", xaxis=dict(type="category"),
                         legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_load, width="stretch", theme="streamlit")
        with b2_col2:
            section("06", "Telemetry Coverage by Market")
            tel = analytics["telemetry"].copy()
            tel["market"] = tel["market"].astype(str)
            tel_long = tel.melt(id_vars="market", var_name="State", value_name="Share of orders (%)")
            fig_tel = px.bar(tel_long, x="market", y="Share of orders (%)", color="State", barmode="group",
                             color_discrete_map={"Telemetry missing (%)": RED, "Zero on-shift (%)": AMBER},
                             title="Orders Without Usable Dispatch Data")
            chart_layout(fig_tel, height=360, xaxis_title="Market", xaxis=dict(type="category"),
                         legend=dict(orientation="h", y=-0.25, title_text=""))
            st.plotly_chart(fig_tel, width="stretch", theme="streamlit")

        st.dataframe(
            analytics["special"].assign(**{"Median (min)": analytics["special"]["Median (min)"].round(1)}),
            width="stretch", hide_index=True,
            column_config={"Orders": st.column_config.NumberColumn("Orders", format="%d"),
                           "Median (min)": st.column_config.NumberColumn("Median delivery (min)", format="%.1f")},
        )