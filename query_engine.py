# query_engine.py
# Handles query classification and LLM querying logic for DevMind.

import asyncio
from llama_index.llms.ollama import Ollama
from prompts import (
    QUERY_CLASSIFIER_PROMPT, FUNCTION_EXPLAIN_PROMPT,
    ARCHITECTURE_PROMPT, DEBUG_PROMPT, ISSUE_PR_PROMPT
)

PROMPT_MAP = {
    "FUNCTION":     FUNCTION_EXPLAIN_PROMPT,
    "ARCHITECTURE": ARCHITECTURE_PROMPT,
    "DEBUG":        DEBUG_PROMPT,
    "ISSUE_PR":     ISSUE_PR_PROMPT,
}


def _ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def classify_query(query: str) -> str:
    """Classifies the given query into one of the four categories."""
    _ensure_event_loop()
    
    # Use global LLM from Settings (handling both Gemini and Ollama)
    from llama_index.core import Settings
    llm = Settings.llm
    
    system_prompt = "You are a direct query classifier. Respond ONLY with one word from: FUNCTION, ARCHITECTURE, DEBUG, ISSUE_PR."
    user_prompt = QUERY_CLASSIFIER_PROMPT.format(query=query)
    prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

    response_obj = llm.complete(prompt)
    category = response_obj.text.strip().upper()

    # Extract only the first word in case LLM adds explanation
    for word in category.split():
        if word in PROMPT_MAP:
            return word
    return "FUNCTION"  # Fallback


def get_answer(index, query: str):
    """Classifies query, sets appropriate prompt, and returns (response, category)"""
    _ensure_event_loop()
    category = classify_query(query)
    prompt = PROMPT_MAP[category]

    query_engine = index.as_query_engine(
        text_qa_template=prompt,
        similarity_top_k=6,
    )

    _ensure_event_loop()
    response = query_engine.query(query)
    return str(response), category
