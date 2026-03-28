from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize API
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("❌ HF_TOKEN not found.")

api = HfApi(token=token)
repo_id = "Shramik121/predictive-maintenance-app"

# Step 1: Delete broken Space
try:
    api.delete_repo(repo_id=repo_id, repo_type="space", token=token)
    print("✅ Deleted broken Space")
except Exception as e:
    print(f"Delete skipped: {e}")

# Step 2: Create Space fresh
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="streamlit",   # ← must be streamlit, not docker
    private=False,
    exist_ok=True,
    token=token
)
print("✅ Space created")

# Step 3: Upload README.md FIRST (mandatory)
readme_content = b"""---
title: Predictive Maintenance App
emoji: \xf0\x9f\x94\xa7
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---
"""
api.upload_file(
    path_or_fileobj=readme_content,
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Add README metadata",
    token=token
)
print("✅ README uploaded")

# Step 4: Upload deployment files
api.upload_folder(
    folder_path="Predictive_Maintenance_Project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
    commit_message="🚀 Deploy Predictive Maintenance Streamlit App",
    token=token
)

print("\n✅ Upload complete!")
print("⏳ App will build in 2–5 minutes.")
print(f"\n🌐 https://huggingface.co/spaces/{repo_id}")
