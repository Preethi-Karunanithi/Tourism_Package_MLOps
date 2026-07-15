import os
from huggingface_hub import HfApi

# --------------------------------------------------
# Hugging Face Authentication
# --------------------------------------------------

token = os.getenv("HF_TOKEN")

if token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=token)

# --------------------------------------------------
# Upload Streamlit App to Hugging Face Space
# --------------------------------------------------

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="PreethiKarunanithi/Preethi-first-project-space",
    repo_type="space",
    path_in_repo=""
)

print("Deployment uploaded successfully to Hugging Face Space.")
