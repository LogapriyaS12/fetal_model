# Fetal Health Classification using Machine Learning

## Overview  
This project builds a machine-learning model to classify fetal health (Normal / Suspect / Pathological) from Cardiotocography (CTG) dataset. The model is trained using [dataset source/link], and deployed as a web application using Streamlit.

## Dataset  
- fetal_health.csv — contains features (baseline heart rate, accelerations, uterine contractions, etc.) + label (health class)
- Data source: [if from publicly available dataset, mention URL]

## Model Training  
- Preprocessing: standard scaling of features  
- Algorithm: Random Forest Classifier  
- Model and scaler saved as `rf_model.pkl` and `scaler.pkl`

## Running the App  
1. Clone this repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run the Streamlit app: `streamlit run app.py` (or appropriate filename)  
4. Input the CTG features in UI → get predicted fetal health condition  

## Dependencies  
Listed in `requirements.txt`

## Usage & Disclaimer  
This project is for educational/demo purposes only. Predictions should not be used for actual medical diagnosis.  

## Author  
Logapriya — BE AIML Student  
