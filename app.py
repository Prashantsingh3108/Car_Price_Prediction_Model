import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Car Price Prediction App")

# User input
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

# Simple encoding example
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
fuel_encoded = fuel_map[fuel]

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[year, km_driven, fuel_encoded]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")

