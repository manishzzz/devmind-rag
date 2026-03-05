# prompts.py
# Updated for LangChain-based DevMind RAG Assistant.

from langchain_core.prompts import PromptTemplate

# 1. Industrial Standard Grounding System Prompt
SYSTEM_GROUNDING_TEMPLATE = (
    "You are an expert DevMind Codebase RAG Assistant. "
    "Answer ONLY based on the provided context fragments. "
    "If the information is not present in the context, explicitly say: 'I could not find that information in the codebase.' "
    "Do NOT use external knowledge. Be precise, technical, and professional.\n\n"
    "Codebase Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
)

# 2. Conversational QA Prompt
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=SYSTEM_GROUNDING_TEMPLATE + "Chat History:\n{chat_history}\n\nUser Question: {question}\n\nAnswer:"
)

# 3. Query Categorization (kept for UI routing if needed, though LangChain handles general QA well)
QUERY_CLASSIFIER_PROMPT = (
    "Classify this codebase question: {query}\n"
    "Categories: FUNCTION | ARCHITECTURE | DEBUG | ISSUE_PR\n"
    "Return only the category name."
)
