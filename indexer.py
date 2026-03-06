# indexer.py
# Ingests EVERY readable file in a GitHub repo. Zero file left behind.

import os
import hashlib
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)

STORAGE_DIR = "./storage"

# All file extensions we can read as text
READABLE_EXTENSIONS = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift",
    ".kt", ".rs", ".scala", ".r", ".m", ".sh", ".bash",
    ".ps1", ".lua", ".dart", ".ex", ".exs", ".clj", ".hs",
    # Config & DevOps
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
    ".env", ".properties", ".xml", ".json", ".jsonc",
    # Docs
    ".md", ".mdx", ".rst", ".txt", ".adoc",
    # Web
    ".html", ".css", ".scss", ".less", ".vue", ".svelte",
    # Build
    ".gradle", ".cmake", ".makefile", ".dockerfile",
    # Data
    ".csv", ".tsv",
    # GitHub special files (no extension)
}

# Folders to always skip
SKIP_FOLDERS = {
    "node_modules", "__pycache__", ".git", "dist", "build",
    "target", ".idea", ".vscode", "vendor", "venv", ".env",
    "coverage", ".pytest_cache", "eggs", ".eggs",
    "*.egg-info", ".tox", "htmlcov"
}

# Files to skip even if readable
SKIP_FILES = {
    ".DS_Store", "Thumbs.db", "package-lock.json",
    "yarn.lock", "poetry.lock", "Pipfile.lock"
}


def _get_local_embeddings():
    """Local HuggingFace embeddings - zero API calls, no rate limits."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from llama_index.embeddings.langchain import LangchainEmbedding
    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return LangchainEmbedding(hf)


def _should_skip_folder(folder_name: str) -> bool:
    return folder_name in SKIP_FOLDERS or folder_name.startswith(".") and folder_name not in {".github", ".devcontainer"}


def _is_readable_file(filepath: Path) -> bool:
    """Check if file is readable text."""
    # Check by extension
    if filepath.suffix.lower() in READABLE_EXTENSIONS:
        return True
    # Check files with no extension (Makefile, Dockerfile, CODEOWNERS etc.)
    if filepath.suffix == "" and filepath.stat().st_size < 100_000:
        return True
    return False


def _read_file_safe(filepath: Path) -> str | None:
    """Read file content safely, return None if binary or unreadable."""
    try:
        # Skip huge files (>500KB)
        if filepath.stat().st_size > 500_000:
            return None
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        # Skip empty files
        if len(content) < 10:
            return None
        return content
    except Exception:
        return None


def _walk_repo(repo_path: Path) -> list[Document]:
    """Walk entire repo and return all readable Documents."""
    documents = []
    skipped = 0
    loaded = 0

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)

        # Skip unwanted folders IN PLACE (modifies dirs to prevent descending)
        dirs[:] = [
            d for d in dirs
            if not _should_skip_folder(d)
        ]

        for filename in files:
            if filename in SKIP_FILES:
                continue

            filepath = root_path / filename

            if not _is_readable_file(filepath):
                skipped += 1
                continue

            content = _read_file_safe(filepath)
            if content is None:
                skipped += 1
                continue

            # Relative path for metadata
            try:
                rel_path = str(filepath.relative_to(repo_path))
            except ValueError:
                rel_path = str(filepath)

            # Rich metadata for better retrieval
            doc = Document(
                text=content,
                metadata={
                    "file_path": rel_path,
                    "file_name": filename,
                    "file_ext": filepath.suffix.lower(),
                    "file_size": filepath.stat().st_size,
                    "folder": str(filepath.parent.relative_to(repo_path)),
                    "is_config": filepath.suffix.lower() in {
                        ".yml", ".yaml", ".xml", ".json", ".toml",
                        ".properties", ".gradle"
                    },
                    "is_test": (
                        "test" in rel_path.lower() or
                        filename.lower().startswith("test") or
                        filename.lower().endswith("test.java") or
                        filename.lower().endswith("_test.py") or
                        filename.lower().endswith("spec.js")
                    ),
                    "is_hidden": filename.startswith("."),
                }
            )
            documents.append(doc)
            loaded += 1

    print(f"[INDEX] Loaded: {loaded} files | Skipped: {skipped} files")
    return documents


def _chunk_documents(documents: list[Document]) -> list:
    """Smart chunking based on file type."""
    from llama_index.core.node_parser import (
        CodeSplitter, MarkdownNodeParser, SentenceSplitter
    )

    CODE_LANGS = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".cpp": "cpp", ".c": "c",
        ".cs": "c_sharp", ".rb": "ruby", ".rs": "rust",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".php": "php",
    }

    code_nodes = []
    md_nodes = []
    other_nodes = []

    md_parser = MarkdownNodeParser()
    default_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Group by type
    code_docs_by_lang: dict[str, list] = {}
    md_docs = []
    other_docs = []

    for doc in documents:
        ext = doc.metadata.get("file_ext", "")
        if ext in CODE_LANGS:
            lang = CODE_LANGS[ext]
            code_docs_by_lang.setdefault(lang, []).append(doc)
        elif ext in {".md", ".mdx", ".rst"}:
            md_docs.append(doc)
        else:
            other_docs.append(doc)

    # Chunk code by language
    for lang, docs in code_docs_by_lang.items():
        try:
            splitter = CodeSplitter(
                language=lang,
                chunk_lines=40,
                chunk_overlap_lines=5
            )
            code_nodes.extend(splitter.get_nodes_from_documents(docs))
        except Exception:
            # Fallback to sentence splitter if language not supported
            other_docs.extend(docs)

    # Chunk markdown
    if md_docs:
        try:
            md_nodes.extend(md_parser.get_nodes_from_documents(md_docs))
        except Exception:
            other_docs.extend(md_docs)

    # Chunk everything else
    if other_docs:
        other_nodes.extend(default_splitter.get_nodes_from_documents(other_docs))

    all_nodes = code_nodes + md_nodes + other_nodes
    print(f"[INDEX] Chunks created: {len(all_nodes)}")
    return all_nodes


def build_index(repo_path: str, lc_llm=None, lc_embeddings=None):
    """
    Full repo ingestion. Zero LLM calls during build.
    Uses only local embeddings - no API rate limits.
    """
    # CRITICAL: No LLM during indexing
    Settings.llm = None
    Settings.embed_model = _get_local_embeddings()
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    repo = Path(repo_path)
    if not repo.exists():
        raise ValueError(f"Repo path does not exist: {repo_path}")

    print(f"[INDEX] Scanning repo: {repo_path}")
    documents = _walk_repo(repo)

    if not documents:
        raise ValueError("No readable files found in repo.")

    print(f"[INDEX] Chunking {len(documents)} documents...")
    nodes = _chunk_documents(documents)

    print(f"[INDEX] Building vector index from {len(nodes)} chunks...")
    index = VectorStoreIndex(
        nodes=nodes,
        show_progress=True,
    )

    index.storage_context.persist(STORAGE_DIR)
    print(f"[INDEX] Complete. Saved to {STORAGE_DIR}")

    # Save index stats for display in UI
    stats = {
        "total_files": len(documents),
        "total_chunks": len(nodes),
        "repo_path": str(repo_path),
    }
    import json
    with open(f"{STORAGE_DIR}/stats.json", "w") as f:
        json.dump(stats, f)

    return index


def load_index(lc_llm, lc_embeddings):
    """Load saved index. Embeddings must match what was used during build."""
    Settings.llm = None
    Settings.embed_model = _get_local_embeddings()
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    # Set LLM only for query time
    from llama_index.llms.langchain import LangChainLLM
    Settings.llm = LangChainLLM(llm=lc_llm)
    return index


def get_index_stats() -> dict:
    """Return stats about the current index."""
    import json
    stats_file = Path(f"{STORAGE_DIR}/stats.json")
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    return {}
