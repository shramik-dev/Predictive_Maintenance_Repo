import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import os

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment("Predictive-Maintenance-XGBoost")

# ----------------------------
# Hugging Face setup
# ----------------------------
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Missing Hugging Face token.")

api = HfApi(token=token)

# ----------------------------
# Dataset paths (from HF)
# ----------------------------
Xtrain_path = "hf://datasets/1samjack1/predictive-maintenance-dataset/processed/Xtrain.csv"
Xtest_path  = "hf://datasets/1samjack1/predictive-maintenance-dataset/processed/Xtest.csv"
ytrain_path = "hf://datasets/1samjack1/predictive-maintenance-dataset/processed/ytrain.csv"
ytest_path  = "hf://datasets/1samjack1/predictive-maintenance-dataset/processed/ytest.csv"

# ----------------------------
# Load data
# ----------------------------
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).iloc[:, 0]
ytest = pd.read_csv(ytest_path).iloc[:, 0]

print("✅ Data loaded successfully")

# ----------------------------
# Model Training + Tuning (XGBoost)
# ----------------------------
best_model = None
best_score = 0

for n_estimators in [100, 200]:
    for max_depth in [3, 5, 7]:
        for lr in [0.01, 0.1]:

            with mlflow.start_run():

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )

                # Train
                model.fit(Xtrain, ytrain)

                # Predict
                y_pred = model.predict(Xtest)

                # Metrics
                acc = accuracy_score(ytest, y_pred)
                f1 = f1_score(ytest, y_pred, average="weighted")

                # ----------------------------
                # Log to MLflow
                # ----------------------------
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("learning_rate", lr)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)

                print(f"n={n_estimators}, depth={max_depth}, lr={lr}, acc={acc:.4f}, f1={f1:.4f}")

                # Save best model
                if acc > best_score:
                    best_score = acc
                    best_model = model

# ----------------------------
# Final Evaluation
# ----------------------------
print("\n✅ Best Accuracy:", best_score)

y_pred_best = best_model.predict(Xtest)

print("\nClassification Report:")
print(classification_report(ytest, y_pred_best))

# ----------------------------
# Save Model
# ----------------------------
model_path = "best_predictive_maintenance_xgb.pkl"
joblib.dump(best_model, model_path)

print(f"✅ Model saved locally: {model_path}")

# ----------------------------
# Log best model in MLflow
# ----------------------------
mlflow.log_artifact(model_path, artifact_path="model")

# ----------------------------
# Upload to Hugging Face Model Hub
# ----------------------------
repo_id = "1samjack1/predictive-maintenance-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo {repo_id} already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Created new repo: {repo_id}")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("🚀 XGBoost model uploaded to Hugging Face successfully!")
