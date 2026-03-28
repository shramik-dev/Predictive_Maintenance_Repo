import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ----------------------------

# Page Config

# ----------------------------

st.set_page_config(
page_title="Predictive Maintenance",
page_icon="🔧",
layout="centered"
)

# ----------------------------

# Load Model (Cached)

# ----------------------------

@st.cache_resource
def load_model():
model_path = hf_hub_download(
repo_id="Shramik121/predictive-maintenance-model",
filename="best_predictive_maintenance_xgb.pkl",
repo_type="model"
)
return joblib.load(model_path)

model = load_model()

# ----------------------------

# Title

# ----------------------------

st.title("🔧 Predictive Maintenance System")
st.write("Predict Engine Condition (Healthy / Faulty)")

# ============================

# OPTION 1: Manual Input

# ============================

st.subheader("🔹 Single Prediction")

engine_rpm = st.number_input("Engine RPM", value=1500)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.5)
fuel_pressure = st.number_input("Fuel Pressure", value=5.0)
coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temp", value=85.0)
coolant_temp = st.number_input("Coolant Temp", value=90.0)

input_df = pd.DataFrame([{
'Engine rpm': engine_rpm,
'Lub oil pressure': lub_oil_pressure,
'Fuel pressure': fuel_pressure,
'Coolant pressure': coolant_pressure,
'lub oil temp': lub_oil_temp,
'Coolant temp': coolant_temp
}])

if st.button("🔍 Predict Single"):
try:
pred = model.predict(input_df)[0]
label_map = {0: "Healthy", 1: "Faulty"}

```
    if pred == 1:
        st.error(f"⚠️ {label_map[pred]} Engine")
    else:
        st.success(f"✅ {label_map[pred]} Engine")

except Exception as e:
    st.error(f"Prediction Error: {e}")
```

# ============================

# OPTION 2: CSV Upload

# ============================

st.subheader("📂 Bulk Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

required_cols = [
'Engine rpm',
'Lub oil pressure',
'Fuel pressure',
'Coolant pressure',
'lub oil temp',
'Coolant temp'
]

if uploaded_file is not None:
try:
data = pd.read_csv(uploaded_file)

```
    st.write("### 📊 Uploaded Data Preview")
    st.dataframe(data.head())

    # Validate columns
    if not all(col in data.columns for col in required_cols):
        st.error("❌ Invalid CSV format")
        st.info(f"Required columns: {required_cols}")

    else:
        predictions = model.predict(data)

        # Add predictions
        label_map = {0: "Healthy", 1: "Faulty"}
        data["Prediction"] = [label_map[p] for p in predictions]

        st.write("### ✅ Predictions")
        st.dataframe(data)

        # Download results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error processing file: {e}")
    st.warning("⚠️ Please upload a valid CSV file.")
```
