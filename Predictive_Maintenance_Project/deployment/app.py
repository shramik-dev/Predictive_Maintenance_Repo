import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download model
model_path = hf_hub_download(
    repo_id="Shramik121/predictive-maintenance-model",
    filename="best_predictive_maintenance_xgb.pkl",
    repo_type="model",
    token=os.getenv("HF_TOKEN")
)
model = joblib.load(model_path)

st.title(" Predictive Maintenance System")
st.write("Predict Engine Condition (Healthy / Faulty)")

engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=1500.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0, value=3.5)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=5.0)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temp", min_value=0.0, value=85.0)
coolant_temp = st.number_input("Coolant Temp", min_value=0.0, value=90.0)

input_df = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

if st.button(" Predict"):
    pred = model.predict(input_df)[0]
    label_map = {0: "Healthy", 1: "Faulty"}
    if pred == 1:
        st.error(f" Engine Status: **{label_map[pred]}**")
    else:
        st.success(f" Engine Status: **{label_map[pred]}**")
