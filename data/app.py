import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from decision_engine import (
    calculate_financial_impact,
    classify_risk,
    get_action,
    calculate_baseline_eta,
    calculate_optimized_eta,
    generate_notification
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Smart Logistics Intelligence",
    layout="wide"
)

# ======================================================
# PREMIUM DARK THEME STYLING
# ======================================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: #f1f5f9;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #334155;
    text-align: center;
}

.section-card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

.badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.title("🚚 Smart Logistics Risk Intelligence Platform")
st.caption("AI-Driven Delay Prediction • Risk Optimization • Financial Exposure Modeling")

st.divider()
st.sidebar.markdown("## ⚙ Analysis Mode")

mode = st.sidebar.radio(
    "Select Mode",
    ["Single Shipment", "Batch Shipment (CSV Upload)"]
)

# ======================================================
# LOAD MODEL
# ======================================================

model = joblib.load("models/delay_model.pkl")
scaler = joblib.load("models/scaler.pkl")

df_system = pd.read_csv("data/processed/dataset_with_risk_levels.csv")

operational_base_time = df_system["Waiting_Time"].mean()

df_system["traffic_level"] = df_system.apply(
    lambda row: "Heavy" if row["Traffic_Status_Heavy"] == 1
    else ("Detour" if row["Traffic_Status_Detour"] == 1 else "Clear"),
    axis=1
)

traffic_impact = (
    df_system.groupby("traffic_level")["delay_probability"]
    .mean()
)

clear_factor = traffic_impact.min()

# ======================================================
# SIDEBAR INPUT PANEL
# ======================================================

with st.sidebar:
    st.header("📦 Shipment Inputs")

    latitude = st.number_input("Latitude", value=19.0760)
    longitude = st.number_input("Longitude", value=72.8777)

    traffic = st.selectbox("Traffic Level", ["Clear", "Detour", "Heavy"])

    asset_utilization = st.slider("Asset Utilization (%)", 50, 100, 75)
    precipitation = st.slider("Precipitation (mm)", 0, 50, 10)
    waiting_time = st.slider("Waiting Time (min)", 10, 60, 30)

    hour = st.slider("Hour of Day", 0, 23, 14)
    peak_hour = 1 if hour in [8,9,10,17,18,19] else 0

    st.markdown("---")
    st.header("💰 Financial Inputs")

    order_value = st.number_input("Order Value ($)", 10, 10000, 500)
    shipping_cost = st.number_input("Shipping Cost ($)", 5, 500, 50)
    is_express = st.checkbox("Express Shipping")

    analyze = st.button("🚀 Analyze Shipment")

# ======================================================
# ANALYSIS
# ======================================================

if mode == "Single Shipment" and analyze:


    training_features = scaler.feature_names_in_
    feature_means = df_system[training_features].mean()

    input_data = pd.DataFrame([feature_means], columns=training_features)

    input_data["Latitude"] = latitude
    input_data["Longitude"] = longitude
    input_data["Precipitation(mm)"] = precipitation
    input_data["Waiting_Time"] = waiting_time
    input_data["Asset_Utilization"] = asset_utilization
    input_data["hour"] = hour
    input_data["peak_hour"] = peak_hour

    input_data["Traffic_Status_Heavy"] = 0
    input_data["Traffic_Status_Detour"] = 0

    if traffic == "Heavy":
        input_data["Traffic_Status_Heavy"] = 1
    elif traffic == "Detour":
        input_data["Traffic_Status_Detour"] = 1

    input_data = input_data[training_features]

    scaled_data = scaler.transform(input_data)
    delay_probability = model.predict_proba(scaled_data)[0][1]

    # ------------------------
    # Core Logic
    # ------------------------

    risk = classify_risk(delay_probability)

    baseline_eta = calculate_baseline_eta(
        delay_probability,
        operational_base_time
    )

    optimized_eta = calculate_optimized_eta(
        delay_probability,
        risk,
        operational_base_time,
        clear_factor
    )

    action = get_action(risk, asset_utilization)
    formatted_action = action.replace("_", " ")

    message = generate_notification(
        risk,
        baseline_eta,
        optimized_eta,
        traffic
    )

    delay_hours, expected_loss = calculate_financial_impact(
        delay_probability,
        operational_base_time,
        order_value,
        shipping_cost,
        is_express
    )

    eta_improvement = baseline_eta - optimized_eta
    improvement_percent = (
        (eta_improvement / baseline_eta) * 100
        if baseline_eta != 0 else 0
    )
    
    # ======================================================
    # EXECUTIVE SUMMARY CARD
    # ======================================================

    st.markdown(f"""
    <div class="section-card">
    <h3>🧠 Executive Summary</h3>
    Delay Probability: <b>{round(delay_probability,3)}</b><br>
    Risk Level: <b>{risk}</b><br>
    ETA Improvement: <b>{round(improvement_percent,1)}%</b><br>
    Financial Exposure: <b>${expected_loss:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # KPI ROW
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Delay Probability", round(delay_probability,3))
    col2.metric("Risk Level", risk)
    col3.metric("Baseline ETA", round(baseline_eta,2))
    col4.metric("Optimized ETA", round(optimized_eta,2))

    st.divider()

    # ======================================================
    # ACTION + FINANCIAL PANELS
    # ======================================================

    colA, colB = st.columns(2)

    # Risk Color
    risk_colors = {
        "Low": "#22c55e",
        "Medium": "#facc15",
        "High": "#f97316",
        "Critical": "#ef4444"
    }

    risk_color = risk_colors.get(risk, "#ffffff")

    colA.markdown(f"""
    <div class="section-card">
    <h3>🚦 Recommended Action</h3>
    <span class="badge" style="background-color:{risk_color}; color:black;">
    {formatted_action}
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Financial color
    financial_color = "#ca2828"
    if expected_loss > 100:
        financial_color = "#ef4444"
    elif expected_loss > 40:
        financial_color = "#facc15"

    colB.markdown(f"""
    <div class="section-card">
    <h3>💰 Financial Exposure</h3>
    Expected Delay: <b>{round(delay_hours,2)} minutes</b><br><br>
    Expected Loss: <span style="color:{financial_color}; font-weight:700;">
    ${expected_loss:.2f}
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ======================================================
    # VISUAL ANALYTICS
    # ======================================================

    st.subheader("📈 Risk vs Financial Impact")

    fig, ax = plt.subplots()
    ax.bar(
        ["Delay Probability", "Financial Loss"],
        [delay_probability, expected_loss]
    )

    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    st.pyplot(fig)

    st.divider()

    # ======================================================
    # CUSTOMER MESSAGE
    # ======================================================

    st.subheader("📩 Customer Communication")
    st.info(message)
# ======================================================
# BATCH SHIPMENT ANALYSIS
# ======================================================

if mode == "Batch Shipment (CSV Upload)":

    st.header("📂 Batch Shipment Intelligence")

    uploaded_file = st.file_uploader(
        "Upload Shipment CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_batch = pd.read_csv(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.dataframe(df_batch.head())

        required_cols = [
            "Latitude", "Longitude", "Precipitation(mm)",
            "Waiting_Time", "Asset_Utilization",
            "hour", "Traffic_Status_Heavy",
            "Traffic_Status_Detour",
            "order_value", "shipping_cost"
        ]

        missing = [col for col in required_cols if col not in df_batch.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:

            training_features = scaler.feature_names_in_

            results = []

            for _, row in df_batch.iterrows():

                input_data = pd.DataFrame(
                    [df_system[training_features].mean()],
                    columns=training_features
                )

                for col in training_features:
                    if col in row:
                        input_data[col] = row[col]

                scaled_data = scaler.transform(input_data)
                delay_probability = model.predict_proba(scaled_data)[0][1]

                risk = classify_risk(delay_probability)

                baseline_eta = calculate_baseline_eta(
                    delay_probability,
                    operational_base_time
                )

                optimized_eta = calculate_optimized_eta(
                    delay_probability,
                    risk,
                    operational_base_time,
                    clear_factor
                )

                delay_hours, expected_loss = calculate_financial_impact(
                    delay_probability,
                    operational_base_time,
                    row["order_value"],
                    row["shipping_cost"],
                    False
                )

                priority_score = delay_probability * expected_loss

                results.append({
                    "delay_probability": delay_probability,
                    "risk": risk,
                    "baseline_eta": baseline_eta,
                    "optimized_eta": optimized_eta,
                    "expected_loss": expected_loss,
                    "priority_score": priority_score
                })

            df_results = pd.concat(
                [df_batch.reset_index(drop=True),
                 pd.DataFrame(results)],
                axis=1
            )

            st.divider()
            st.subheader("📊 Batch Executive Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Shipments", len(df_results))
            col2.metric(
                "Total Expected Loss ($)",
                f"${df_results['expected_loss'].sum():,.2f}"
            )
            col3.metric(
                "High/Critical Shipments",
                len(df_results[df_results["risk"].isin(["High", "Critical"])])
            )

            st.divider()
            st.subheader("🏆 Top Priority Shipments")

            df_sorted = df_results.sort_values(
                "priority_score",
                ascending=False
            )

            st.dataframe(df_sorted.head(10))

            csv_output = df_sorted.to_csv(index=False).encode('utf-8')

            st.download_button(
                "📥 Download Full Analysis",
                csv_output,
                "batch_analysis_results.csv",
                "text/csv"
            )
