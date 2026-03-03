# main.py
# Command-line interface and entry point for the DevMind RAG Assistant.

import os
import sys
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
from ingest import clone_repo
from graph_index import build_index, load_index
from query_engine import get_answer

def main():
    """Main setup and REPL loop for querying the repository codebase."""
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <github_url>")
        sys.exit(1)
        
    github_url = sys.argv[1]
    repo_name = github_url.rstrip("/").split("/")[-1]
    repo_path = os.path.join(os.getenv("GITHUB_REPO_PATH", "./repo"), repo_name)
    storage_dir = os.path.join("./storage", repo_name)
    
    # 1. Clone repository
    clone_repo(github_url, dest=repo_path)
    
    # 2. Build or load index
    if os.path.exists(storage_dir) and os.path.isdir(storage_dir):
        print("Found existing index in ./storage, loading it...")
        index = load_index(storage_dir=storage_dir)
    else:
        print("No index found, building index...")
        index = build_index(repo_path=repo_path, storage_dir=storage_dir)
        
    print("\nDevMind is ready! Type 'exit' or 'quit' to stop.")
    
    # 3. REPL loop
    while True:
        try:
            query = input("\nQuery: ")
            if query.lower() in ("exit", "quit"):
                break
            
            response_text, category = get_answer(index, query)
            print(f"\n[{category}] RESPONSE:\n{response_text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
