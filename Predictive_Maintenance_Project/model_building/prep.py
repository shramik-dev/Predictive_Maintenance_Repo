
# ----------------------------
# Libraries
# ----------------------------
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ----------------------------
# Setup Hugging Face API
# ----------------------------
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Missing Hugging Face token. Please set HF_TOKEN.")

api = HfApi(token=token)

# ----------------------------
# Load Dataset
# ----------------------------
data_path = "/content/Predictive_Maintenance_Project/data/engine_data.csv"

df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully.")
print("Shape:", df.shape)

# ----------------------------
# Define Target Variable
# ----------------------------
target = "Engine Condition"

# Remove rows where target is missing
df = df.dropna(subset=[target])

# ----------------------------
# Define Features
# ----------------------------
numeric_features = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

# No categorical features in this dataset
X = df[numeric_features]
y = df[target]

# ----------------------------
# Data Cleaning
# ----------------------------
# Fill missing numeric values with mean
X = X.fillna(X.mean())

# ----------------------------
# Train-Test Split
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Data split done:")
print(f"Train Shape: {Xtrain.shape}")
print(f"Test Shape: {Xtest.shape}")

# ----------------------------
# Save Locally
# ----------------------------
os.makedirs("/content/Predictive_Maintenance_Project/data/processed", exist_ok=True)

Xtrain_path = "/content/Predictive_Maintenance_Project/data/processed/Xtrain.csv"
Xtest_path = "/content/Predictive_Maintenance_Project/data/processed/Xtest.csv"
ytrain_path = "/content/Predictive_Maintenance_Project/data/processed/ytrain.csv"
ytest_path = "/content/Predictive_Maintenance_Project/data/processed/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("✅ Train/Test datasets saved locally.")

# ----------------------------
# Upload to Hugging Face
# ----------------------------
repo_id = "Shramik121/predictive-maintenance-dataset"

files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="dataset",
    )

print("🚀 All split files uploaded successfully to Hugging Face!")
