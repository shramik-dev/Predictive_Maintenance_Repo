# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Pre-download the model during build (so it's cached in the image)
RUN python -c '
from huggingface_hub import hf_hub_download
import joblib
import os

print("Downloading model from Hugging Face Hub...")
model_path = hf_hub_download(
    repo_id="Shramik121/predictive-maintenance-model",
    filename="best_predictive_maintenance_xgb.pkl",
    repo_type="model"
)
print(f"Model successfully downloaded and cached at: {model_path}")

# Optional: Test load the model
model = joblib.load(model_path)
print("Model loaded successfully during build!")
'

# Expose the port that HF Docker Spaces expect
EXPOSE 7860

# Run the Streamlit app correctly bound to 0.0.0.0:7860
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.headless=true"]
