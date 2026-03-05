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

def get_documents(repo_path: str) -> List[Document]:
    """Recursively loads all files from the repository using a robust crawler."""
    if not os.path.exists(repo_path):
        return []
    
    docs = []
    # Industry standard: don't rely on generic globs that skip dot-folders.
    # We walk the directory manually to catch .github, .gitpod, etc.
    for root, dirs, files in os.walk(repo_path):
        # We don't want to index the .git directory itself
        if ".git" in root.split(os.sep): continue

        for file in files:
            file_path = os.path.join(root, file)
            # Skip common binary/large files to keep the index efficient
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.zip', '.exe', '.bin', '.pdf', '.pyc')):
                continue
            
            try:
                # Use our robust loader for character sanitization
                loader = RobustTextLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"[!] Skipping {file_path}: {e}")
                continue
    
    print(f"[*] Loaded {len(docs)} documents from {repo_path}")
    return docs
