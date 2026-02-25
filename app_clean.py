import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

EXPECTED_FEATURES = model.n_features_in_

st.set_page_config(page_title="Vehicle Claim Fraud Detection")

st.title("🚗 Vehicle Claim Fraud Detection System")
st.write("Enter claim details to predict fraud")

# -------- USER FRIENDLY INPUTS --------
age = st.number_input("Age of Policy Holder", 18, 100, 25)
driver_rating = st.slider("Driver Rating (1 = Worst, 5 = Best)", 1, 5, 2)
past_claims = st.slider("Past Number of Claims", 0, 10, 3)
deductible = st.number_input("Deductible Amount", 0, 10000, 0)
vehicle_age = st.slider("Age of Vehicle (Years)", 0, 20, 8)

police_report = st.selectbox("Police Report Filed", ["No", "Yes"])
witness = st.selectbox("Witness Present", ["No", "Yes"])
agent_type = st.selectbox("Agent Type", ["Internal", "External"])

# Encode categorical
police_report = 1 if police_report == "Yes" else 0
witness = 1 if witness == "Yes" else 0
agent_type = 1 if agent_type == "External" else 0

# -------- BUILD FULL FEATURE VECTOR --------
# Start with safe default values
input_values = [1] * EXPECTED_FEATURES


# Assign important features to fixed positions
input_values[10] = age
input_values[16] = driver_rating
input_values[19] = past_claims
input_values[15] = deductible
input_values[20] = vehicle_age
input_values[22] = police_report
input_values[23] = witness
input_values[24] = agent_type

# -------- PREDICTION --------
if st.button("Predict Fraud"):
    input_array = np.array(input_values).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.metric("Fraud Probability", f"{probability:.4f}")

    if probability > 0.15:
        st.error("⚠️ Fraudulent Claim Detected")
    else:
        st.success("✅ Genuine Claim")
