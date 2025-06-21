from huggingface_hub import create_repo, upload_folder

# Replace with your actual Hugging Face username
username = "sunnypatel782"
repo_name = "self-healing-sentiment-model"

# Create a new repo on Hugging Face
create_repo(repo_id=f"{username}/{repo_name}", private=False, exist_ok=True)

# Upload model folder
upload_folder(
    repo_id=f"{username}/{repo_name}",
    folder_path="./model",  # Replace with your model directory if different
    path_in_repo=".",       # Upload to the root of the repo
)

print("✅ Upload complete! View at:")
print(f"https://huggingface.co/{username}/{repo_name}")
