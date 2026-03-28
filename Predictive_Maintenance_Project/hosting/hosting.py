
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ----------------------------
# Initialize API
# ----------------------------
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError(" Hugging Face token not found. Please set HF_TOKEN.")

api = HfApi(token=token)

# ----------------------------
# Space Details
# ----------------------------
repo_id = "Shramik121/predictive-maintenance-app"
repo_type = "space"

# ----------------------------
# Step 1: Create Space (if not exists)
# ----------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Space '{repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f"🚀 Creating Space '{repo_id}'...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="streamlit",   # Required for Streamlit apps
        private=False,           # Change to True if needed
        exist_ok=False
    )

    print(f" Space created successfully!")
    print(f" Wait 1-2 minutes for initialization:")
    print(f"https://huggingface.co/spaces/{repo_id}")

except Exception as e:
    print(f" Error while creating Space: {e}")
    raise

# ----------------------------
# Step 2: Upload Deployment Files
# ----------------------------
api.upload_folder(
    folder_path="Predictive_Maintenance_Project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",   # Upload to root
    commit_message="🚀 Deploy Predictive Maintenance Streamlit App"
)

print("\n Upload complete!")
print("⏳ App will build in 2–5 minutes.")

print(f"\n🌐 Visit your app here:")
print(f"https://huggingface.co/spaces/{repo_id}")
