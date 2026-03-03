# ingest.py
# Handles cloning of GitHub repositories for DevMind.

import os
from git import Repo

def clone_repo(github_url: str, dest: str = "./repo") -> str:
    """Clones a GitHub repo to dest if it doesn't already exist."""
    if os.path.exists(dest) and os.path.exists(os.path.join(dest, ".git")):
        print(f"Repo already exists at {dest}. Skipping clone.")
    else:
        print(f"Cloning {github_url} into {dest}...")
        Repo.clone_from(github_url, dest)
    return dest
