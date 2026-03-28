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
    raise ValueError("Missing Hugging Face token.")

api = HfApi(token=token)

# ----------------------------
# Load Dataset FROM HF
# ----------------------------
df = pd.read_csv("Predictive_Maintenance_Project/data/engine_data.csv")
print("Dataset loaded successfully.")


print("✅ Dataset loaded from Hugging Face successfully.")
print("Shape:", df.shape)

# ----------------------------
# Define Target Variable
# ----------------------------
target = "Engine Condition"

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

X = df[numeric_features]
y = df[target]

# ----------------------------
# Data Cleaning
# ----------------------------
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
# Save Locally (temporary)
# ----------------------------
os.makedirs("processed", exist_ok=True)

Xtrain_path = "processed/Xtrain.csv"
Xtest_path = "processed/Xtest.csv"
ytrain_path = "processed/ytrain.csv"
ytest_path = "processed/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("✅ Train/Test datasets saved locally.")

# ----------------------------
# OPTIONAL: Upload splits to SAME dataset repo
# ----------------------------
repo_id = "Shramik121/engine-dataset"

files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed/{os.path.basename(file_path)}",
        repo_id=repo_id,
        repo_type="dataset",
    )

print("🚀 Processed files uploaded to Hugging Face (optional step).")
