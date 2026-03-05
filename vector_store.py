# vector_store.py
# LangChain FAISS vector store management for DevMind.
# Handles building, saving, and loading the vector index.

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_embeddings():
    """Returns the embedding model, using local HuggingFace to avoid API limits."""
    print("[*] Using local HuggingFace embeddings for reliable indexing.")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(documents: List[Document], storage_dir: str = "./storage_faiss") -> FAISS:
    """Builds and persists a FAISS vector store with industrial splitting."""
    embeddings = get_embeddings()
    
    # Industry standard: use specialized separators for different file types
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # We use a broad splitter that handles YAML, JSON, and Code logic better
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""], # Priority to double newlines (blocks)
        is_separator_regex=False
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    # Cleanup old storage to prevent stale/corrupted data poison
    import shutil
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    
    print(f"[*] Building FAISS index with {len(split_docs)} chunks...")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    os.makedirs(storage_dir, exist_ok=True)
    vector_store.save_local(storage_dir)
    print(f"[*] Vector store saved to {storage_dir}")
    return vector_store

def load_vector_store(storage_dir: str = "./storage_faiss") -> FAISS:
    """Loads a persisted FAISS vector store."""
    embeddings = get_embeddings()
    if not os.path.exists(storage_dir):
        raise FileNotFoundError(f"Storage directory not found: {storage_dir}")
    
    return FAISS.load_local(storage_dir, embeddings, allow_dangerous_deserialization=True)
