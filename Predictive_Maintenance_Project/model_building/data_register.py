from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError
import os

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Missing Hugging Face token.")

repo_id = "1samjack1/engine-dataset"
repo_type = "dataset"

api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"🚀 Creating dataset '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        exist_ok=True,
        token=token
    )
    print(f"✅ Dataset created!")
except Exception as e:
    print(f"❌ Error: {e}")
    raise

api.upload_folder(
    folder_path="Predictive_Maintenance_Project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
    commit_message="Upload engine dataset",
    token=token
)

print("✅ Dataset uploaded!")
print(f"🌐 https://huggingface.co/datasets/{repo_id}")
