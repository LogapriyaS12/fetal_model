{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "0e09c98c-81a0-465d-88f7-4f3a854b55a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load model and scaler\n",
    "model = joblib.load('rf_model.pkl')\n",
    "scaler = joblib.load('scaler.pkl')\n",
    "\n",
    "st.set_page_config(page_title=\"Fetal Health Predictor\", layout=\"centered\")\n",
    "st.title(\"👶 Fetal Health Classification App\")\n",
    "st.markdown(\"Enter the fetal CTG values below to classify fetal health condition.\")\n",
    "\n",
    "# Define feature names (must match dataset column order)\n",
    "feature_names = [\n",
    "    \"baseline value\", \"accelerations\", \"fetal_movement\", \"uterine_contractions\",\n",
    "    \"light_decelerations\", \"severe_decelerations\", \"prolongued_decelerations\",\n",
    "    \"abnormal_short_term_variability\", \"mean_value_of_short_term_variability\",\n",
    "    \"percentage_of_time_with_abnormal_long_term_variability\",\n",
    "    \"mean_value_of_long_term_variability\", \"histogram_width\", \"histogram_min\",\n",
    "    \"histogram_max\", \"histogram_number_of_peaks\", \"histogram_number_of_zeroes\",\n",
    "    \"histogram_mode\", \"histogram_mean\", \"histogram_median\", \"histogram_variance\",\n",
    "    \"histogram_tendency\"\n",
    "]\n",
    "\n",
    "# Create input fields\n",
    "input_values = []\n",
    "for feature in feature_names:\n",
    "    val = st.number_input(f\"{feature}\", value=0.0)\n",
    "    input_values.append(val)\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Predict Fetal Health\"):\n",
    "    input_df = pd.DataFrame([input_values], columns=feature_names)\n",
    "    input_scaled = scaler.transform(input_df)\n",
    "    prediction = model.predict(input_scaled)[0]\n",
    "\n",
    "    label_map = {0: \"1 - Normal\", 1: \"2 - Suspect\", 2: \"3 - Pathological\"}\n",
    "    st.success(f\"🧠 Predicted Fetal Health Class: **{label_map[prediction]}**\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
