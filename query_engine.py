# query_engine.py - Hybrid retrieval: direct file read + vector search
import re
import os
from pathlib import Path
from llama_index.core import Settings, get_response_synthesizer
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.schema import TextNode, NodeWithScore
from prompts import PROMPT_MAP, QUERY_CLASSIFIER_PROMPT

# Matches class names like LRUCache, BinaryTree, QuickSort etc.
SPECIFIC_CLASS_PATTERN = re.compile(
    r'\b([A-Z][a-zA-Z0-9]+)\b'
)

REPO_PATH = "./repo"  # Where the cloned repo lives


def classify_query(query: str, lc_llm) -> str:
    try:
        from langchain_core.messages import HumanMessage
        prompt = QUERY_CLASSIFIER_PROMPT.format(query=query)
        response = lc_llm.invoke([HumanMessage(content=prompt)])
        category = response.content.strip().upper()
        valid = {"FUNCTION", "ARCHITECTURE", "DEBUG", "ISSUE_PR"}
        return category if category in valid else "ARCHITECTURE"
    except Exception:
        return "ARCHITECTURE"


def extract_target_class(query: str) -> str | None:
    """Extract the most likely specific class name from query."""
    # Look for CamelCase words that look like class names
    matches = SPECIFIC_CLASS_PATTERN.findall(query)
    # Filter out common English words
    skip = {
        "What", "How", "Why", "When", "Where", "Which", "The",
        "This", "That", "Does", "Can", "Is", "Are", "Was",
        "Explain", "Show", "List", "Find", "Tell", "Give",
        "Describe", "Implement", "Use", "Get", "Set"
    }
    for match in matches:
        if match not in skip and len(match) > 3:
            return match
    return None


def find_file_in_repo(class_name: str, repo_path: str = REPO_PATH) -> str | None:
    """
    Search entire cloned repo for the exact file matching class_name.
    Returns full file content as string, or None if not found.
    """
    repo = Path(repo_path)
    if not repo.exists():
        print(f"[QUERY] Repo path not found: {repo_path}")
        return None

    # Try exact filename match first: LRUCache.java
    target_filename = f"{class_name}.java"
    matches = list(repo.rglob(target_filename))

    # Also try case-insensitive
    if not matches:
        target_lower = target_filename.lower()
        matches = [
            f for f in repo.rglob("*.java")
            if f.name.lower() == target_lower
        ]

    # Also try other extensions
    if not matches:
        for ext in [".py", ".ts", ".js", ".cs", ".go", ".cpp"]:
            found = list(repo.rglob(f"{class_name}{ext}"))
            if found:
                matches = found
                break

    if not matches:
        print(f"[QUERY] File not found in repo: {target_filename}")
        return None

    # Read the file
    target_file = matches[0]
    try:
        content = target_file.read_text(encoding="utf-8", errors="ignore")
        print(f"[QUERY] Direct file read: {target_file}")
        return content, str(target_file)
    except Exception as e:
        print(f"[QUERY] Could not read {target_file}: {e}")
        return None


def get_answer(index, query: str, lc_llm) -> tuple:
    Settings.llm = LangChainLLM(llm=lc_llm)
    category = classify_query(query, lc_llm)
    prompt = PROMPT_MAP[category]
    target_class = extract_target_class(query)

    all_nodes = []
    sources = []
    used_direct_read = False

    # ── STRATEGY 1: Direct file read (for specific class queries) ──
    if target_class:
        result = find_file_in_repo(target_class, REPO_PATH)
        if result:
            file_content, file_path_str = result
            # Create a node directly from file content
            # This GUARANTEES the exact file is in context
            direct_node = NodeWithScore(
                node=TextNode(
                    text=f"FILE: {file_path_str}\n\n{file_content}",
                    metadata={
                        "file_path": file_path_str,
                        "file_name": f"{target_class}.java",
                        "source": "direct_read",
                    }
                ),
                score=1.0  # Highest possible score
            )
            all_nodes.append(direct_node)
            sources.append(file_path_str)
            used_direct_read = True
            print(f"[QUERY] Using direct file read for {target_class}")

    # ── STRATEGY 2: Vector search for supporting context ──
    try:
        vector_nodes = index.as_retriever(
            similarity_top_k=5
        ).retrieve(query)

        # If we have direct read, only add vector nodes from SAME file
        # to avoid polluting context with wrong classes
        for node in vector_nodes:
            fname = node.metadata.get("file_name", "")
            fpath = node.metadata.get("file_path", "")

            if used_direct_read and target_class:
                # Only add if from the same target file
                if target_class.lower() in fname.lower():
                    all_nodes.append(node)
                    if fpath not in sources:
                        sources.append(fpath)
            else:
                # No direct read - use all vector results
                all_nodes.append(node)
                if fpath not in sources:
                    sources.append(fpath)

    except Exception as e:
        print(f"[QUERY] Vector search failed: {e}")
        if not all_nodes:
            return f"Query error: {str(e)}", category, []

    if not all_nodes:
        return (
            f"NOT FOUND IN INDEX: Could not find '{target_class}' "
            f"in repo or index.",
            category,
            []
        )

    # ── Synthesize answer ──
    try:
        synthesizer = get_response_synthesizer(
            text_qa_template=prompt,
            response_mode="compact",
        )
        response = synthesizer.synthesize(query, nodes=all_nodes)
        response_text = str(response)

        # Red flag detection
        red_flags = [
            "typically", "commonly", "standard practice",
            "usually", "in most projects", "likely uses"
        ]
        if any(f in response_text.lower() for f in red_flags):
            response_text = (
                "⚠️ **WARNING: Answer may contain inferred content not from codebase.**\n\n"
                + response_text
            )

        if used_direct_read:
            response_text = (
                f"✅ **[Direct file read: {target_class}]**\n\n"
                + response_text
            )

        return response_text, category, sources[:5]

    except Exception as e:
        return f"Synthesis error: {str(e)}", category, sources
