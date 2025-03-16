from huggingface_hub import HfApi, login
import os

# Login to Hugging Face (Ensure you have a token set up)
login()  # If you are already logged in, this step can be skipped

# Define repository details
repo_id = "447AnushkaD/nllb_bn_finetuned"  # Your Hugging Face username and model name
local_path = r"C:\Users\chatn\synapse\nllb_bn_finetuned"  # Ensure the path is correct

# Validate the folder path
if not os.path.exists(local_path):
    raise FileNotFoundError(f"Model folder not found at: {local_path}")

api = HfApi()

print("Creating the repository if it doesn't exist...")
api.create_repo(repo_id, repo_type="model", exist_ok=True)

print("Uploading model files... This may take a while.")
api.upload_folder(
    folder_path=local_path,
    repo_id=repo_id,
    repo_type="model",
)

print(f"Upload successful! Check your model at: https://huggingface.co/{repo_id}")
