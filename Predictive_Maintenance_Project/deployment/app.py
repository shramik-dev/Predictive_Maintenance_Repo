import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os
import traceback

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="centered"
)

# Load Model (Cached)
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Shramik121/predictive-maintenance-model",
            filename="best_predictive_maintenance_xgb.pkl",
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.code(traceback.format_exc())
        return None

model = load_model()

if model is None:
    st.warning("⚠️ App cannot run without the model. Check the error above.")
    st.stop()
