import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Load model bundle
# -------------------------------
bundle = joblib.load("price_model.pkl")
model = bundle["best_model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Selling Price Prediction")
st.write("Predict the **selling price** of a car using Machine Learning")

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Car Details")

car_name = st.number_input("Car Name (Encoded)", min_value=0, step=1)
year = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, step=1)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
fuel_type = st.selectbox("Fuel Type (Encoded)", [0, 1, 2])
seller_type = st.selectbox("Seller Type (Encoded)", [0, 1])
transmission = st.selectbox("Transmission (Encoded)", [0, 1])
owner = st.selectbox("Owner", [0, 1, 2, 3])

# -------------------------------
# Create input dataframe
# -------------------------------
input_data = pd.DataFrame([[
    car_name,
    year,
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    owner
]], columns=feature_columns)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price 🚀"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    st.success(f"💰 Estimated Selling Price: **₹ {prediction[0]:.2f} Lakhs**")
