import streamlit as st
import numpy as np
import joblib

from src.risk_logic import risk_bucket, decision_from_risk

# ---------------- App Config ----------------
st.set_page_config(
    page_title="RiskShield",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ RiskShield – Fraud Risk Assessment System")
st.write(
    """
    This application demonstrates a **fraud risk scoring engine**.
    It evaluates transaction patterns and assigns a **risk level**
    instead of making hard yes/no decisions.
    """
)

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_model():
    return joblib.load("results/final_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("results/scaler.pkl")

model = load_model()
scaler = load_scaler()

# ---------------- Input Mode ----------------
st.subheader("Choose Input Mode")

mode = st.radio(
    "Select input mode:",
    ["Demo (Pre-filled sample)", "Manual (Advanced)"]
)

st.info(
    "⚠️ The dataset uses anonymized PCA-based features (V1–V28). "
    "These represent internal transaction signals, not raw card details."
)

inputs = []

# ---------------- DEMO MODE ----------------

time_value = st.number_input(
    "Transaction Time (seconds since first transaction)",
    value=0.0,
    step=1.0
)
inputs.append(time_value)

if mode == "Demo (Pre-filled sample)":
    st.subheader("Demo Transaction (Internal Features)")


    demo_values = [0.0] * 28  # V1–V28
    demo_amount = 18.0

    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.number_input(
                f"Feature V{i+1}",
                value=demo_values[i],
                disabled=True
            )
            inputs.append(demo_values[i])

    amount = st.number_input(
        "Transaction Amount",
        value=demo_amount,
        disabled=True
    )
    inputs.append(amount)

# ---------------- MANUAL MODE ----------------
else:
    st.subheader("Manual Feature Input (Advanced)")

    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            val = st.number_input(
                f"Feature V{i+1}",
                value=0.0,
                step=0.1
            )
            inputs.append(val)

    amount = st.number_input(
        "Transaction Amount",
        value=0.0,
        step=1.0
    )
    inputs.append(amount)

# ---------------- Prediction ----------------


if st.button("Assess Fraud Risk"):
    X = np.array(inputs).reshape(1, -1)

    # DEBUG (temporary)
    st.write("Input shape:", X.shape)
    st.write("Scaler expects features:", scaler.n_features_in_)

    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    risk = risk_bucket(prob)
    decision = decision_from_risk(risk)

    st.subheader("Risk Assessment Result")

    st.metric("Fraud Risk Score", f"{prob:.3f}")

    if risk == "LOW":
        st.success("Risk Level: LOW")
    elif risk == "MEDIUM":
        st.warning("Risk Level: MEDIUM")
    else:
        st.error("Risk Level: HIGH")

    st.write(f"**System Decision:** {decision}")

    st.caption(
        "Decisions are based on configurable risk thresholds "
        "and are meant to assist, not replace, human judgment."
    )

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "RiskShield | ML-based Fraud Risk Scoring System\n\n"
    "Built for academic demonstration and decision-support analysis."
)
