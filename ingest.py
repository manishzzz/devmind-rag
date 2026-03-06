# ingest.py - Windows-safe repo cloning with force-delete
import os
import stat
import shutil
from pathlib import Path

def _force_remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree on Windows.
    Git sets pack files as read-only. This removes that flag before retry.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"[WARN] Could not remove {path}: {e}")


def _safe_delete_folder(folder_path: str):
    """Delete folder safely on both Windows and Linux."""
    path = Path(folder_path)
    if not path.exists():
        return

    print(f"[INGEST] Removing old folder: {folder_path}")
    try:
        # Windows-safe: pass error handler that removes read-only flag
        shutil.rmtree(
            str(path),
            onerror=_force_remove_readonly
        )
        print(f"[INGEST] Removed: {folder_path}")
    except Exception as e:
        print(f"[WARN] rmtree failed ({e}), trying os.system fallback...")
        # Last resort: use system command
        if os.name == "nt":  # Windows
            os.system(f'rmdir /s /q "{folder_path}"')
        else:  # Linux/Mac
            os.system(f'rm -rf "{folder_path}"')


def clone_repo(github_url: str, dest: str = "./repo") -> str:
    """Clone repo fresh every time. Windows-safe deletion."""
    import git

    # Always delete old repo first (Windows-safe)
    _safe_delete_folder(dest)

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"[INGEST] Cloning {github_url} -> {dest}")

    token = os.getenv("GITHUB_TOKEN", "")
    if token and "github.com" in github_url:
        url_parts = github_url.replace("https://", "")
        auth_url = f"https://{token}@{url_parts}"
        git.Repo.clone_from(auth_url, dest)
    else:
        git.Repo.clone_from(github_url, dest)

    print(f"[INGEST] Clone complete: {dest}")
    return str(dest_path)
