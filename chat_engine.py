# chat_engine.py
# LangChain Conversational RAG logic for DevMind.
# Updated for LangChain 0.4.x modular structure.

import os
from typing import Tuple, List, Dict
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def get_llm():
    """Returns a Chat model, prioritizing Google Gemini Flash with multi-alias fallbacks."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and "models/" in api_key:
        api_key = api_key.replace("models/", "")

    # Industrial standard: Try standard alias, then latest alias, then 8b, then Pro
    model_aliases = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-001", "gemini-1.5-pro-latest"]
    
    last_err = None
    for model_name in model_aliases:
        try:
            print(f"[*] Attempting to use Google Gemini: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            # Test the model with a tiny probe to verify it's NOT a 404
            llm.invoke("Hi") 
            return llm
        except Exception as e:
            print(f"[!] {model_name} failed: {e}")
            last_err = e
            continue

    # Fallback to Groq if all Google attempts fail
    print(f"[!] All Gemini attempts failed. Falling back to Groq.")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        return ChatGroq(model_name="llama-3.1-70b-versatile", groq_api_key=groq_api_key)
    
    # If no Groq key, raise the last Gemini error
    raise last_err

def create_chat_engine(vector_store, prompt_template: str = ""):
    """Creates a modern LangChain retrieval chain."""
    llm = get_llm()
    
    # Industrial standard: using a professional system prompt with strict grounding
    system_prompt = (
        "You are an industrial-standard Codebase RAG Assistant. "
        "Strict Rules:\n"
        "1. ONLY answer based on the provided context fragments. If the answer is not in the context, say 'I don't have enough information in the indexed codebase to answer this.'\n"
        "2. Do NOT use outside knowledge about libraries or frameworks unless it's explicitly shown in the context (files like package.json, requirements.txt, etc.).\n"
        "3. When referring to files, ALWAYS use the 'source' path provided in the context.\n"
        "4. If asked about a file that is not in the context, explicitly state: 'That file was not found in the current index.'\n"
        "5. Keep responses technical, concise, and professional.\n\n"
        "Provided Context:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 7}), question_answer_chain)

def query_chat_engine(chain, query: str) -> Tuple[str, List[Document]]:
    """Executes a query and returns the answer and source documents."""
    response = chain.invoke({"input": query})
    return response["answer"], response["context"]
