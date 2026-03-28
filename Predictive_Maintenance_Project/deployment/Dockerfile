FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Pre-download and cache the model during build
RUN python -c "
from huggingface_hub import hf_hub_download
import joblib
import os
model_path = hf_hub_download(
    repo_id='Shramik121/predictive-maintenance-model',
    filename='best_predictive_maintenance_xgb.pkl',
    repo_type='model'
)
print('Model downloaded to:', model_path)
"

# Expose the correct port
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]
