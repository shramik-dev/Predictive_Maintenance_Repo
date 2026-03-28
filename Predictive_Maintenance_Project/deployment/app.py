from huggingface_hub import HfApi, create_repo
import os

token = os.getenv("HF_TOKEN")
api = HfApi(token=token)
repo_id = "Shramik121/predictive-maintenance-app"

# Step 1: Delete
try:
    api.delete_repo(repo_id=repo_id, repo_type="space", token=token)
    print("✅ Deleted")
except Exception as e:
    print(f"Delete skipped: {e}")

# Step 2: Create with docker sdk
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
    token=token
)
print("✅ Space created")

# Step 3: Upload README first
readme = """---
title: Predictive Maintenance App
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
""".encode("utf-8")

api.upload_file(
    path_or_fileobj=readme,
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Add README",
    token=token
)
print("✅ README uploaded")

# Step 4: Upload files
api.upload_folder(
    folder_path="Predictive_Maintenance_Project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
    commit_message="Deploy app",
    token=token
)
print("✅ Done!")
print(f"🌐 https://huggingface.co/spaces/{repo_id}")
