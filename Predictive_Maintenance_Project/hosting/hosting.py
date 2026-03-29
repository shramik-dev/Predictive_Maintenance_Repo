from huggingface_hub import HfApi, create_repo
import os

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(" HF_TOKEN not found.")

api = HfApi(token=token)
repo_id = "Shramik121/predictive-maintenance-app"

# Step 1: Delete old Space
try:
    api.delete_repo(repo_id=repo_id, repo_type="space", token=token)
    print(" Deleted old Space")
except Exception:
    print(" No existing Space to delete")

# Step 2: Create fresh Space
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="Stremlit",
    private=False,
    exist_ok=True,
    token=token
)
print(" Space created")

# Step 3: Upload README first
api.upload_file(
    path_or_fileobj="""---
title: Predictive Maintenance App
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
""".encode("utf-8"),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Add README",
    token=token
)
print(" README uploaded")

# Step 4: Upload deployment files
api.upload_folder(
    folder_path="Predictive_Maintenance_Project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
    commit_message=" Deploy Predictive Maintenance App",
    token=token
)

print(" Upload complete! App will build in 2-5 min.")
print(f" https://huggingface.co/spaces/{repo_id}")
