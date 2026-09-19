"""
Streamlit demo: Supply Chain Shipment Delay Risk Predictor

Run locally with:
    pip install streamlit scikit-learn pandas numpy joblib matplotlib
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Shipment Delay Risk Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.joblib")
    schema = joblib.load("models/feature_schema.joblib")
    return model, schema

model, schema = load_model()

st.title("🚢 Shipment Delay Risk Predictor")
st.caption("Predicts the probability a shipment will arrive late, based on route, carrier, "
           "customs, and congestion factors.")

with st.form("shipment_form"):
    col1, col2 = st.columns(2)

    with col1:
        carrier = st.selectbox("Carrier", ["MaerskLine", "MSC", "CMA_CGM", "COSCO",
                                            "Hapag_Lloyd", "ONE_Line", "Evergreen"])
        mode = st.selectbox("Shipping Mode", ["Sea", "Air", "Rail", "Road"])
        origin_port = st.selectbox("Origin Port", ["Shanghai", "Singapore", "Rotterdam",
                                                    "Los_Angeles", "Busan", "Hamburg", "Dubai",
                                                    "Antwerp", "Ningbo", "Hong_Kong"])
        dest_port = st.selectbox("Destination Port", ["New_York", "Long_Beach", "Felixstowe",
                                                        "Santos", "Mumbai", "Sydney", "Tokyo",
                                                        "Vancouver", "Le_Havre", "Durban"])
        season = st.selectbox("Season", ["Q1", "Q2", "Q3", "Q4"])

    with col2:
        distance_km = st.slider("Distance (km)", 100, 20000, 8000)
        planned_transit_days = st.slider("Planned Transit (days)", 1, 45, 18)
        customs_complexity = st.slider("Customs Complexity (0-1)", 0.0, 1.0, 0.4)
        num_customs_stops = st.slider("Number of Customs Stops", 0, 5, 1)
        weather_risk = st.slider("Weather Risk (0-1)", 0.0, 1.0, 0.3)
        shipment_value_usd = st.number_input("Shipment Value (USD)", 1000, 500000, 50000)
        is_hazmat = st.checkbox("Hazardous Materials")
        num_prior_shipments_route = st.slider("Prior Shipments on This Route", 0, 100, 10)

    submitted = st.form_submit_button("Predict Delay Risk")

if submitted:
    carrier_reliability = {
        "MaerskLine": 0.85, "MSC": 0.78, "CMA_CGM": 0.75, "COSCO": 0.68,
        "Hapag_Lloyd": 0.82, "ONE_Line": 0.72, "Evergreen": 0.70,
    }[carrier]
    origin_congestion_idx = np.random.uniform(0.2, 0.7)  # demo proxy; real system would look this up live
    dest_congestion_idx = np.random.uniform(0.2, 0.7)

    row = pd.DataFrame([{
        "distance_km": distance_km,
        "planned_transit_days": planned_transit_days,
        "origin_congestion_idx": origin_congestion_idx,
        "dest_congestion_idx": dest_congestion_idx,
        "customs_complexity": customs_complexity,
        "num_customs_stops": num_customs_stops,
        "weather_risk": weather_risk,
        "carrier_reliability_score": carrier_reliability,
        "log_shipment_value": np.log1p(shipment_value_usd),
        "is_hazmat": int(is_hazmat),
        "route_experience": np.log1p(num_prior_shipments_route),
        "congestion_gap": abs(origin_congestion_idx - dest_congestion_idx),
        "total_congestion": origin_congestion_idx + dest_congestion_idx,
        "transit_per_1000km": planned_transit_days / (distance_km / 1000 + 1e-6),
        "carrier": carrier,
        "mode": mode,
        "origin_port": origin_port,
        "dest_port": dest_port,
        "season": season,
    }])

    proba = model.predict_proba(row)[0, 1]
    st.metric("Predicted Delay Probability", f"{proba:.1%}")

    if proba > 0.5:
        st.error("⚠️ High risk of delay — consider buffer time or an alternate carrier/route.")
    else:
        st.success("✅ Low-to-moderate delay risk.")

    st.subheader("What's driving delay risk in general")
    st.image("plots/feature_importance.png",
              caption="Permutation feature importance from the trained model "
                      "(shows which factors most affect predictions across the whole test set).")

st.divider()
st.subheader("Model Comparison")
st.image("plots/roc_comparison.png", caption="ROC curves: Logistic Regression vs Gradient Boosting vs Neural Network")
