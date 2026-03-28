from huggingface_hub import HfApi
import os

# -------------------------------
# Hugging Face Token
# -------------------------------
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Missing Hugging Face token.")

# -------------------------------
# Existing Dataset Repo
# -------------------------------
repo_id = "Shramik121/engine-dataset"
repo_type = "dataset"

# -------------------------------
# Initialize API
# -------------------------------
api = HfApi(token=token)

# -------------------------------
# Check if dataset exists
# -------------------------------
try:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset '{repo_id}' found and ready to use.")
    print(f"Files in dataset: {[file.rfilename for file in info.siblings]}")

except Exception as e:
    print(f"❌ Error accessing dataset: {e}")
    raise

print("🚀 No upload needed. Using existing Hugging Face dataset.")
