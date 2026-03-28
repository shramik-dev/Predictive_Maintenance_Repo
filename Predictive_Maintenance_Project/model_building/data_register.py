
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# -------------------------------
# Hugging Face Token (from GitHub Secrets)
# -------------------------------
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Missing Hugging Face token. Please set HF_TOKEN in GitHub Actions secrets.")

# -------------------------------
# Repository Configuration
# -------------------------------
repo_id = "Shramik121/predictive-maintenance-dataset"
repo_type = "dataset"

# -------------------------------
# Initialize API
# -------------------------------
api = HfApi(token=token)

# -------------------------------
# Step 1: Check if dataset repo exists
# -------------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    
    print(f"Dataset repository '{repo_id}' created successfully.")

# -------------------------------
# Step 2: Upload Dataset Folder
# -------------------------------
data_folder_path = "/content/Predictive_Maintenance_Project/data"

if not os.path.exists(data_folder_path):
    raise FileNotFoundError(f"Data folder not found at {data_folder_path}")

api.upload_folder(
    folder_path=data_folder_path,
    repo_id=repo_id,
    repo_type=repo_type
)

print("✅ Data successfully uploaded to Hugging Face Dataset Hub!")
