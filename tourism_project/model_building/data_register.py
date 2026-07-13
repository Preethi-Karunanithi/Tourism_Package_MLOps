from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing. Please configure it.")

api = HfApi(token=token)

repo_id = "PreethiKarunanithi/Preethi-first-project-space"
repo_type = "dataset"

try:
    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Dataset repository '{repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating...")
    
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=token
    )

    print(f"Dataset repository '{repo_id}' created.")


api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully!")
