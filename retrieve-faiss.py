from huggingface_hub import HfApi

api = HfApi()
# Change this to your repo name
repo_id = "Credioni/your_faiss_repo"

# Create a repo (if it doesn’t exist)
api.create_repo(repo_id, exist_ok=True)

# Upload the FAISS index file
api.upload_file(
    path_or_fileobj="faiss_index.bin",
    path_in_repo="faiss_index.bin",
    repo_id=repo_id,
    repo_type="model"
)

print(f"FAISS index uploaded to: https://huggingface.co/{repo_id}")

