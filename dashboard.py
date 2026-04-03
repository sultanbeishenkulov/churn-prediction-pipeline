import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from load import load_raw
from engineer import engineer_features
from drift import check_drift

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Monitor", layout="wide")
st.title("Customer Churn Prediction — Monitoring Dashboard")

API_URL = "https://churn-prediction-pipeline-1.onrender.com"
MODEL_DIR = Path("models")


# ── Load training data (cached) ────────────────────────────────────────────
@st.cache_data
def load_training_data():
    df = load_raw()
    df = engineer_features(df)
    return df


training_df = load_training_data()

# ── Sidebar: single-customer prediction ────────────────────────────────────
st.sidebar.header("Predict Churn for a Customer")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
monthly = st.sidebar.number_input("Monthly Charges", 18.0, 120.0, 70.0)
total = st.sidebar.number_input("Total Charges", 0.0, 9000.0, monthly * tenure)

if st.sidebar.button("Predict"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        prob = result["churn_probability"]
        risk = result["risk_level"]

        color = {"high": "red", "medium": "orange", "low": "green"}[risk]
        st.sidebar.markdown(
            f"### Churn probability: :{color}[{prob:.1%}]  \nRisk: **{risk.upper()}**"
        )

        st.sidebar.markdown("**Top churn drivers:**")
        for r in result["top_reasons"]:
            direction = "↑" if r["impact"] > 0 else "↓"
            st.sidebar.write(f"- {r['feature']}: {r['impact']:+.3f} {direction}")

    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API error: {e}")

# ── Main area: tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Distributions", "Drift Monitoring"])

# ── Tab 1: Model Performance ───────────────────────────────────────────────
with tab1:
    st.subheader("Training Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", "~0.85")
    col2.metric("Precision (Churn)", "~0.67")
    col3.metric("Recall (Churn)", "~0.78")

    st.markdown("---")

    # Churn distribution
    st.subheader("Target Distribution")
    churn_counts = training_df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    churn_counts["Churn"] = churn_counts["Churn"].map({1: "Churn", 0: "No Churn"})
    fig = px.bar(churn_counts, x="Churn", y="Count", color="Churn",
                 color_discrete_map={"Churn": "#EF553B", "No Churn": "#636EFA"})
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP summary image (if available)
    shap_path = MODEL_DIR / "plots" / "shap_summary.png"
    if shap_path.exists():
        st.subheader("SHAP Feature Importance")
        st.image(str(shap_path), use_column_width=True)

# ── Tab 2: Feature Distributions ──────────────────────────────────────────
with tab2:
    st.subheader("Key Feature Distributions (Training Data)")

    feat_cols = ["tenure", "MonthlyCharges", "TotalCharges", "contract_risk",
                 "service_count", "charge_ratio"]
    available = [c for c in feat_cols if c in training_df.columns]

    cols = st.columns(2)
    for i, col_name in enumerate(available):
        with cols[i % 2]:
            fig = px.histogram(training_df, x=col_name, nbins=30,
                               title=col_name, color_discrete_sequence=["#636EFA"])
            fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Drift Monitoring ───────────────────────────────────────────────
with tab3:
    st.subheader("PSI Drift Detection")
    st.markdown(
        "Upload a CSV of **new / live customer data** to compare against the "
        "training distribution. PSI > 0.2 = significant drift."
    )

    uploaded = st.file_uploader("Upload live data CSV", type=["csv"])

    if uploaded:
        live_raw = pd.read_csv(uploaded)
        live_raw["TotalCharges"] = pd.to_numeric(live_raw["TotalCharges"], errors="coerce")
        live_df = engineer_features(live_raw)

        # Only compare numeric features that exist in both
        numeric_train = training_df.select_dtypes(include="number")
        numeric_live = live_df.select_dtypes(include="number")
        common = list(set(numeric_train.columns) & set(numeric_live.columns) - {"Churn"})

        drift_results = check_drift(
            numeric_train[common],
            numeric_live[common],
        )

        drift_df = pd.DataFrame([
            {"Feature": k, "PSI": v["psi"], "Status": v["status"]}
            for k, v in drift_results.items()
        ]).sort_values("PSI", ascending=False)

        # Color-coded bar chart
        color_map = {"stable": "#2CA02C", "monitor": "#FF7F0E", "drift": "#D62728"}
        fig = px.bar(drift_df, x="Feature", y="PSI", color="Status",
                     color_discrete_map=color_map, title="PSI by Feature")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                      annotation_text="Drift threshold (0.2)")
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                      annotation_text="Monitor threshold (0.1)")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

        drifted = drift_df[drift_df["Status"] == "drift"]
        if not drifted.empty:
            st.error(f"⚠️ {len(drifted)} feature(s) show significant drift — consider retraining.")
        else:
            st.success("✅ No significant drift detected.")
    else:
        st.info("Upload a CSV to run drift detection.")