import streamlit as st
import numpy as np
import joblib

# ===============================
# Load model & scaler
# ===============================
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Default Prediction", layout="centered")

st.title("💳 Credit Card Default Prediction App")
st.write("Predict whether a customer is likely to default on credit card payment.")

# ===============================
# Mapping dictionaries
# ===============================
gender_map = {"Male": 1, "Female": 2}

education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}

marriage_map = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}

pay_status_map = {
    "No consumption": -2,
    "Paid duly": -1,
    "Revolving credit": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3 months delay": 3,
    "4 months delay": 4,
    "5 months delay": 5,
    "6 months delay": 6,
    "7 months delay": 7,
    "8 months delay": 8
}

# ===============================
# User Inputs
# ===============================
st.header("📋 Customer Information")

LIMIT_BAL = st.number_input("Credit Limit", min_value=10000, max_value=1000000, step=10000)
AGE = st.number_input("Age", min_value=18, max_value=100)

SEX = gender_map[st.selectbox("Gender", gender_map.keys())]
EDUCATION = education_map[st.selectbox("Education Level", education_map.keys())]
MARRIAGE = marriage_map[st.selectbox("Marital Status", marriage_map.keys())]

st.header("💰 Repayment Status (Past 6 Months)")

PAY_0 = pay_status_map[st.selectbox("Last Month (PAY_0)", pay_status_map.keys())]
PAY_2 = pay_status_map[st.selectbox("2 Months Ago (PAY_2)", pay_status_map.keys())]
PAY_3 = pay_status_map[st.selectbox("3 Months Ago (PAY_3)", pay_status_map.keys())]
PAY_4 = pay_status_map[st.selectbox("4 Months Ago (PAY_4)", pay_status_map.keys())]
PAY_5 = pay_status_map[st.selectbox("5 Months Ago (PAY_5)", pay_status_map.keys())]
PAY_6 = pay_status_map[st.selectbox("6 Months Ago (PAY_6)", pay_status_map.keys())]

st.header("🧾 Bill Amounts")

BILL_AMT1 = st.number_input("Bill Amount 1", 0, 1000000)
BILL_AMT2 = st.number_input("Bill Amount 2", 0, 1000000)
BILL_AMT3 = st.number_input("Bill Amount 3", 0, 1000000)
BILL_AMT4 = st.number_input("Bill Amount 4", 0, 1000000)
BILL_AMT5 = st.number_input("Bill Amount 5", 0, 1000000)
BILL_AMT6 = st.number_input("Bill Amount 6", 0, 1000000)

st.header("💸 Payment Amounts")

PAY_AMT1 = st.number_input("Payment Amount 1", 0, 1000000)
PAY_AMT2 = st.number_input("Payment Amount 2", 0, 1000000)
PAY_AMT3 = st.number_input("Payment Amount 3", 0, 1000000)
PAY_AMT4 = st.number_input("Payment Amount 4", 0, 1000000)
PAY_AMT5 = st.number_input("Payment Amount 5", 0, 1000000)
PAY_AMT6 = st.number_input("Payment Amount 6", 0, 1000000)

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict Default Risk"):
    features = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                          PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                          BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
                          PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default\n\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Default\n\nProbability: {probability*100:.2f}%")
