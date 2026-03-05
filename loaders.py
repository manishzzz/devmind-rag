# loaders.py
# Robust file loading for the DevMind LangChain backend.
# Handles recursive loading and strips problematic characters.

import os
import re
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

def sanitize_text(text: str) -> str:
    """Strips non-ASCII characters that cause Streamlit encoding errors."""
    # Specifically targeting the corrupted sequences like 'ðŸ'
    # but more generally keeping only printable UTF-8 characters
    # and avoiding common emoji sequences that break specific environments.
    return re.sub(r'[^\x00-\x7f]', '', text)

class RobustTextLoader(TextLoader):
    """TextLoader that handles encoding errors gracefully."""
    def load(self) -> List[Document]:
        try:
            with open(self.file_path, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(self.file_path, encoding="latin-1") as f:
                text = f.read()
        
        # Sanitize to prevent UI crashes downstream
        clean_text = sanitize_text(text)
        metadata = {"source": self.file_path}
        return [Document(page_content=clean_text, metadata=metadata)]

def generate_tree(startpath: str) -> str:
    """Generates a string representation of the file tree for LLM structural awareness."""
    tree = ["Repository Structure:"]
    for root, dirs, files in os.walk(startpath):
        # Skip hidden/binary folders for the tree map
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        # Only show a subset of files to keep the tree document efficient
        for f in files[:20]: # Cap at 20 files per dir for the map
            if not f.endswith(('.pyc', '.png', '.jpg', '.exe')):
                tree.append(f"{subindent}{f}")
        if len(files) > 20:
            tree.append(f"{subindent}... ({len(files)-20} more files)")
    return "\n".join(tree)

def get_documents(repo_path: str) -> List[Document]:
    """Recursively loads all files and adds a structural map document."""
    if not os.path.exists(repo_path):
        return []
    
    docs = []
    
    # Industrial Standard: Add a 'Structural Map' as the first document
    # This gives the LLM a bird's eye view for questions like "Explain the folder structure"
    tree_content = generate_tree(repo_path)
    docs.append(Document(
        page_content=tree_content, 
        metadata={"source": "VIRTUAL_REPOSITORY_MAP.txt", "type": "structure_map"}
    ))

    for root, dirs, files in os.walk(repo_path):
        if ".git" in root.split(os.sep): continue

        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.zip', '.exe', '.bin', '.pdf', '.pyc')):
                continue
            
            try:
                loader = RobustTextLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"[!] Skipping {file_path}: {e}")
                continue
    
    print(f"[*] Loaded {len(docs)} documents (including structural map) from {repo_path}")
    return docs
