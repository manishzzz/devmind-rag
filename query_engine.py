# query_engine.py
# Handles query classification and LLM querying logic for DevMind.

import asyncio
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


def _run_with_failover(func, *args, **kwargs):
    """Executes a function and fails over to the next LLM in the pool if rate limited."""
    import streamlit as st
    from llama_index.core import Settings
    
    if not hasattr(st, "session_state") or "llm_pool" not in st.session_state or not st.session_state.llm_pool:
        return func(*args, **kwargs)

    max_retries = len(st.session_state.llm_pool)
    for _ in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                # Rotate to next LLM
                st.session_state.current_llm_idx = (st.session_state.current_llm_idx + 1) % len(st.session_state.llm_pool)
                next_llm_info = st.session_state.llm_pool[st.session_state.current_llm_idx]
                Settings.llm = next_llm_info["llm"]
                st.toast(f"Rate limited. Switching to {next_llm_info['name']}...", icon="ðŸ”„")
                continue
            else:
                raise e
    return func(*args, **kwargs) # Last attempt


def classify_query(query: str) -> str:
    """Classifies the given query into one of the four categories."""
    _ensure_event_loop()
    from llama_index.core import Settings
    
    system_prompt = "You are a direct query classifier. Respond ONLY with one word from: FUNCTION, ARCHITECTURE, DEBUG, ISSUE_PR."
    user_prompt = QUERY_CLASSIFIER_PROMPT.format(query=query)
    prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

    def do_complete():
        return Settings.llm.complete(prompt)

    response_obj = _run_with_failover(do_complete)
    category = response_obj.text.strip().upper()

    for word in category.split():
        if word in PROMPT_MAP:
            return word
    return "FUNCTION"


def get_answer(index, query: str):
    """Classifies query, sets appropriate prompt, and returns (response, category)"""
    _ensure_event_loop()
    category = classify_query(query)
    prompt = PROMPT_MAP[category]

    def do_query():
        query_engine = index.as_query_engine(
            text_qa_template=prompt,
            similarity_top_k=6,
        )
        return query_engine.query(query)

    response = _run_with_failover(do_query)
    return str(response), category
