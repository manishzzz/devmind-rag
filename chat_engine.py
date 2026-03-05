# chat_engine.py
# LangChain Conversational RAG logic for DevMind.
# Updated for LangChain 0.4.x modular structure.

import os
from typing import Tuple, List, Dict
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def get_llm():
    """Returns a Chat model, prioritizing Google Gemini Flash, then SambaNova, then Groq."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    # 1. Try Google Gemini Flash (Latest)
    if google_api_key:
        if "models/" in google_api_key:
            google_api_key = google_api_key.replace("models/", "")
        
        model_aliases = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"]
        for model_name in model_aliases:
            try:
                print(f"[*] Attempting Google Gemini: {model_name}")
                llm = ChatGoogleGenerativeAI(
                    model=model_name, 
                    google_api_key=google_api_key,
                    temperature=0.1,
                    convert_system_message_to_human=True
                )
                llm.invoke("Hi") # Probe
                return llm
            except Exception as e:
                print(f"[!] {model_name} failed: {e}")
                continue

    # 2. Try SambaNova (Llama 3.3 70B - High Speed, Professional)
    if sambanova_api_key:
        try:
            print("[*] Attempting SambaNova (Llama 3.3 70B)")
            return ChatOpenAI(
                model="Meta-Llama-3.3-70B-Instruct",
                openai_api_key=sambanova_api_key,
                openai_api_base="https://api.sambanova.ai/v1",
                temperature=0.1
            )
        except Exception as e:
            print(f"[!] SambaNova failed: {e}")

    # 3. Try Groq (Llama 3.1 70B - Ultra Fast)
    if groq_api_key:
        try:
            print("[*] Attempting Groq (Llama 3.1 70B)")
            return ChatGroq(
                model_name="llama-3.1-70b-versatile", 
                groq_api_key=groq_api_key,
                temperature=0.1
            )
        except Exception as e:
            print(f"[!] Groq failed: {e}")

    raise ValueError("All LLM providers failed. Check your API keys in .env.")

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
