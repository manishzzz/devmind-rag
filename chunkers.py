# chunkers.py
# Provides CodeSplitter and MarkdownNodeParser for DevMind ingestion.

from llama_index.core.node_parser import TokenTextSplitter, MarkdownNodeParser

def get_code_splitter(language: str) -> TokenTextSplitter:
    """Returns a TokenTextSplitter since CodeSplitter lacks tree-sitter on Python 3.13."""
    return TokenTextSplitter(chunk_size=160, chunk_overlap=16)

def get_markdown_parser() -> MarkdownNodeParser:
    """Returns a MarkdownNodeParser for .md/.mdx files."""
    return MarkdownNodeParser()
