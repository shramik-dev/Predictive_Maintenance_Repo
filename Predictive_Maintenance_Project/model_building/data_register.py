from huggingface_hub import HfApi
import os

# -------------------------------
# Hugging Face Token
# -------------------------------
token = os.getenv("HF_TOKEN1")

if not token:
    raise ValueError("Missing Hugging Face token.")

# -------------------------------
# Existing Dataset Repo
# -------------------------------
repo_id = "1samjack1/engine-dataset"
repo_type = "dataset"

# -------------------------------
# Initialize API
# -------------------------------
api = HfApi(token=token)

# -------------------------------
# Check if dataset exists
# -------------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="Predictive_Maintenance_Project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
