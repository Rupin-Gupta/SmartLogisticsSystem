import streamlit as st
import pandas as pd
import numpy as np
import joblib

from decision_engine import (
    classify_risk,
    get_action,
    calculate_baseline_eta,
    calculate_optimized_eta,
    generate_notification
)
from decision_engine import calculate_financial_impact

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(page_title="Smart Logistics System", layout="centered")
# ======================================
# PREMIUM UI STYLING
# ======================================

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
    border-radius: 15px;
    border: 1px solid #334155;
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

st.title("🚚 Smart Logistics Risk Intelligence System")
# ======================================
# MODE SELECTION
# ======================================

mode = st.radio(
    "Select Analysis Mode",
    ["Single Shipment", "Batch Shipment"],
    horizontal=True
)

# ======================================
# LOAD MODEL & SCALER  
# ======================================

model = joblib.load("models/delay_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ======================================
# LOAD SYSTEM DATA (FOR MEANS & CONSTANTS)
# ======================================

df_system = pd.read_csv("data/processed/dataset_with_risk_levels.csv")

# Training feature order (EXACTLY as used in model training)
training_features = [
    'Latitude', 'Longitude', 'Inventory_Level', 'Temperature', 'Humidity',
    'Precipitation(mm)', 'Waiting_Time', 'User_Transaction_Amount',
    'User_Purchase_Frequency', 'Asset_Utilization', 'Demand_Forecast',
    'hour', 'day_of_week', 'peak_hour',
    'Traffic_Status_Detour', 'Traffic_Status_Heavy',
    'Asset_ID_Truck_10', 'Asset_ID_Truck_2', 'Asset_ID_Truck_3',
    'Asset_ID_Truck_4', 'Asset_ID_Truck_5', 'Asset_ID_Truck_6',
    'Asset_ID_Truck_7', 'Asset_ID_Truck_8', 'Asset_ID_Truck_9',
    'Asset_ID_Truck_10'
]

# Compute feature means
feature_means = df_system[training_features].mean()

# Operational baseline time
operational_base_time = df_system["Waiting_Time"].mean()

# Reconstruct traffic level
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

# ======================================
# USER INPUTS (Only Important Ones)
# ======================================
if mode == "Single Shipment":
    st.divider()
    st.write("Enter shipment details to assess delivery risk and ETA optimization.")
    latitude = st.number_input("Latitude", value=19.0760)
    longitude = st.number_input("Longitude", value=72.8777)

    traffic = st.selectbox("Traffic Level", ["Clear", "Detour", "Heavy"])

    asset_utilization = st.slider(
        "Asset Utilization (%)",
        min_value=50,
        max_value=100,
        value=75
    )

    precipitation = st.slider(
        "Precipitation (mm)",
        min_value=0,
        max_value=50,
        value=10
    )

    waiting_time = st.slider(
        "Expected Waiting Time (minutes)",
        min_value=10,
        max_value=60,
        value=30
    )

    hour = st.slider("Hour of Day", 0, 23, 14)
    peak_hour = 1 if hour in [8,9,10,17,18,19] else 0
    user_transaction_amount = st.number_input(
        "User Transaction Amount ($)",
        min_value=10,
        max_value=10000,
        value=100
    )   
    shipping_cost = st.number_input(
        "Shipping Cost ($)",
        min_value=1,
        max_value=1000,
        value=10
    )

# ======================================
# ANALYSIS
# ======================================

if mode == "Single Shipment" and st.button("Analyze Shipment"):


    # Get exact training feature order
    training_features = scaler.feature_names_in_

    # Start with mean values for all features
    feature_means = df_system[training_features].mean()

    # Create base input row
    input_data = pd.DataFrame([feature_means], columns=training_features)

    # Overwrite with user inputs
    input_data["Latitude"] = latitude
    input_data["Longitude"] = longitude
    input_data["Precipitation(mm)"] = precipitation
    input_data["Waiting_Time"] = waiting_time
    input_data["User_Transaction_Amount"] = user_transaction_amount
    input_data["Asset_Utilization"] = asset_utilization
    input_data["hour"] = hour
    input_data["peak_hour"] = peak_hour

    
    # Encode traffic
    
    input_data["Traffic_Status_Heavy"] = 0
    input_data["Traffic_Status_Detour"] = 0

    if traffic == "Heavy":
        input_data["Traffic_Status_Heavy"] = 1
    elif traffic == "Detour":
        input_data["Traffic_Status_Detour"] = 1


    # Ensure correct column order
    input_data = input_data[training_features]

    # Scale
    scaled_data = scaler.transform(input_data)

    # Predict
    delay_probability = model.predict_proba(scaled_data)[0][1]


    # ==================================
    # DECISION ENGINE
    # ==================================

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

    message = generate_notification(
        risk,
        baseline_eta,
        optimized_eta,
        traffic
    )
 # ==================================
# BASIC FINANCIAL IMPACT
# ==================================

# Estimate delay hours
    delay_hours = delay_probability * operational_base_time

    # SLA penalty
    if delay_hours > 60:
        sla_penalty = 0.15 * user_transaction_amount
    elif delay_hours > 30:
        sla_penalty = 0.08 * user_transaction_amount
    else:
        sla_penalty = 0

    # Refund risk
    refund_cost = 0.05 * user_transaction_amount * delay_probability

    # Extra shipping cost
    extra_shipping_cost = 0.2 * shipping_cost if delay_probability > 0.5 else 0

    # Total delay cost
    total_delay_cost = (
        sla_penalty +
        refund_cost +
        extra_shipping_cost
    )

    # Expected loss
    expected_loss = delay_probability * total_delay_cost


    # ==================================
    # DISPLAY
    # ==================================

    # st.subheader("📊 Analysis Results")

    # st.write("**Delay Probability:**", round(delay_probability, 3))
    # st.write("**Risk Level:**", risk)
    # st.write("**Action Taken:**", action)
    # st.write("**Baseline ETA:**", round(baseline_eta, 2), "minutes")
    # st.write("**Optimized ETA:**", round(optimized_eta, 2), "minutes")

    # if risk in ["High", "Critical"]:
    #     st.error("⚠ Shipment delay risk detected.")
    # elif risk == "Medium":
    #     st.warning("🟡 Moderate risk detected. Monitoring active.")
    # else:
    #     st.success("🟢 Shipment is on schedule.")

    # st.subheader("📩 Customer Notification")
    # st.info(message)
    # st.subheader("💰 Financial Impact")

    # col1, col2 = st.columns(2)

    # col1.metric("Expected Delay (min)", round(delay_hours, 2))
    # col2.metric("Expected Financial Loss ($)", f"${expected_loss:.2f}")

    # if expected_loss > 150:
    #     st.error("⚠ High financial exposure")
    # elif expected_loss > 60:
    #     st.warning("🟡 Moderate financial exposure")
    # else:
    #     st.success("🟢 Low financial exposure")
    st.divider()

    # ======================================
    # EXECUTIVE SUMMARY
    # ======================================

    st.markdown(f"""
    <div class="section-card">
    <h3>🧠 Executive Summary</h3>
    Delay Probability: <b>{round(delay_probability,3)}</b><br>
    Risk Level: <b>{risk}</b><br>
    Expected Financial Exposure: <b>${expected_loss:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # KPI ROW
    # ======================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Delay Probability", round(delay_probability, 3))
    col2.metric("Risk Level", risk)
    col3.metric("Baseline ETA", round(baseline_eta, 2))
    col4.metric("Optimized ETA", round(optimized_eta, 2))

    st.divider()

    # ======================================
    # ACTION + FINANCIAL
    # ======================================

    colA, colB = st.columns(2)

    colA.markdown(f"""
    <div class="section-card">
    <h3>🚦 Recommended Action</h3>
    <p style="font-size:18px;">{action}</p>
    </div>
    """, unsafe_allow_html=True)

    colB.markdown(f"""
    <div class="section-card">
    <h3>💰 Financial Impact</h3>
    Expected Delay: <b>{round(delay_hours,2)} min</b><br><br>
    Expected Loss: <b>${expected_loss:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ======================================
    # CUSTOMER MESSAGE
    # ======================================

    st.markdown(f"""
    <div class="section-card">
    <h3>📩 Customer Communication</h3>
    {message}
    </div>
    """, unsafe_allow_html=True)


# ======================================
# BATCH SHIPMENT ANALYSIS
# ======================================

st.divider()
if mode == "Batch Shipment":

    st.header("📂 Batch Shipment Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV for Batch Analysis",
        type=["csv"]
    )
    if uploaded_file is not None:

        df_batch = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df_batch.head())

        results = []

        training_features = scaler.feature_names_in_

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

            delay_hours = delay_probability * operational_base_time

    # BASIC COST MODEL (Same as single shipment)

    # SLA penalty
            if delay_hours > 60:
                sla_penalty = 0.15 * row["User_Transaction_Amount"]
            elif delay_hours > 30:
                sla_penalty = 0.08 * row["User_Transaction_Amount"]
            else:
                sla_penalty = 0

    # Refund risk
            refund_cost = 0.05 * row["User_Transaction_Amount"] * delay_probability

    # Extra shipping cost
            extra_shipping_cost = 0.2 * row["shipping_cost"] if delay_probability > 0.5 else 0

    # Total delay cost
            total_delay_cost = (
                sla_penalty +
                refund_cost +
                extra_shipping_cost
            )

    # Expected loss
            expected_loss = delay_probability * total_delay_cost

    # Priority score
            priority_score = delay_probability * expected_loss


            results.append({
                "delay_probability": delay_probability,
                "risk": risk,
                "expected_loss": expected_loss,
                "priority_score": priority_score
            })

        df_results = pd.concat(
            [df_batch.reset_index(drop=True),
            pd.DataFrame(results)],
            axis=1
        )

        st.divider()

        total_shipments = len(df_results)
        total_loss = df_results["expected_loss"].sum()
        high_risk_count = len(
            df_results[df_results["risk"].isin(["High", "Critical"])]
        )

        # =============================
        # EXECUTIVE SUMMARY CARD
        # =============================

        st.markdown(f"""
        <div class="section-card">
        <h3>📊 Portfolio Executive Intelligence</h3>
        Total Shipments: <b>{total_shipments}</b><br>
        High / Critical Risk: <b>{high_risk_count}</b><br>
        Total Financial Exposure: <b>${total_loss:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # =============================
        # KPI ROW
        # =============================

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Shipments", total_shipments)
        col2.metric("High Risk Shipments", high_risk_count)
        col3.metric("Portfolio Exposure ($)", f"${total_loss:,.2f}")

        st.divider()

        # =============================
        # RISK DISTRIBUTION
        # =============================

        st.subheader("📈 Risk Distribution")

        risk_counts = df_results["risk"].value_counts()

        st.bar_chart(risk_counts)

        st.divider()

        # =============================
        # TOP PRIORITY TABLE
        # =============================

        st.subheader("🏆 Highest Financial Exposure Shipments")

        df_sorted = df_results.sort_values(
            "expected_loss",
            ascending=False
        )

        st.dataframe(df_sorted.head(10), use_container_width=True)

        # =============================
        # DOWNLOAD BUTTON
        # =============================

        csv_output = df_sorted.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Full Portfolio Report",
            csv_output,
            "batch_analysis_results.csv",
            "text/csv"
        )
