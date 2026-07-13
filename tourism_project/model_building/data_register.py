

from google.colab import userdata
from huggingface_hub import HfApi

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Get Hugging Face token from GitHub Actions secret
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing. Please add it in GitHub Secrets.")

# Initialize API client
api = HfApi(token=token)

repo_id = "PreethiKarunanithi/Preethi-first-project-space"
repo_type = "dataset"

# Check if dataset repository exists
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


# Upload dataset folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully!")

token = os.getenv("HF_TOKEN")


api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "PreethiKarunanithi/Preethi-first-project-space"    # please create your space and repository

repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
