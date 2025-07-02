# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Fetal Health Predictor", layout="centered")
st.title("👶 Fetal Health Classification App")
st.markdown("Enter the fetal CTG values below to classify fetal health condition.")

# Define feature names (must match dataset column order)
feature_names = [
    "baseline value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
    "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance",
    "histogram_tendency"
]

# Create input fields
input_values = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_values.append(val)

# Predict button
if st.button("Predict Fetal Health"):
    input_df = pd.DataFrame([input_values], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    label_map = {0: "1 - Normal", 1: "2 - Suspect", 2: "3 - Pathological"}
    st.success(f"🧠 Predicted Fetal Health Class: **{label_map[prediction]}**")
