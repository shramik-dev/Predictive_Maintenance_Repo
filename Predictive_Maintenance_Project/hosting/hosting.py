from huggingface_hub import HfApi, create_repo
import os

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("❌ HF_TOKEN not found.")

api = HfApi(token=token)
repo_id = "1samjack1/predictive-maintenance-app"

# Never delete — just create if not exists
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
    token=token
)
print("✅ Space ready")

# Upload README
api.upload_file(
    path_or_fileobj="""---
title: Predictive Maintenance App
emoji: 🔧
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
print("✅ README uploaded")

# Upload deployment files — overwrites existing files automatically
api.upload_folder(
    folder_path="Predictive_Maintenance_Project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
    commit_message="🚀 Deploy app",
    token=token
)

print("✅ Done!")
print(f"🌐 https://huggingface.co/spaces/{repo_id}")
