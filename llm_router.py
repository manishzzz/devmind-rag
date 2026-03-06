# llm_router.py - 6-level LLM fallback chain, zero crashes
import os
from dotenv import load_dotenv
load_dotenv()

def get_working_llm():
    errors = []

    # 1. Cerebras - fastest, most generous free tier
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="llama-3.3-70b",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            base_url="https://api.cerebras.ai/v1",
            temperature=0.1,
            max_tokens=2048
        )
        llm.invoke("hi")
        print("[LLM] Active: Cerebras - llama-3.3-70b")
        return llm, "Cerebras - Llama 3.3 70B"
    except Exception as e:
        errors.append(f"Cerebras: {e}")

    # 2. Mistral - free, no daily token limits
    try:
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(
            model="open-mistral-7b",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.1,
            max_tokens=2048
        )
        llm.invoke("hi")
        print("[LLM] Active: Mistral - open-mistral-7b")
        return llm, "Mistral - open-mistral-7b"
    except Exception as e:
        errors.append(f"Mistral: {e}")

    # 3. Groq - use 8b model (10x fewer tokens than 70b)
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=2048
        )
        llm.invoke("hi")
        print("[LLM] Active: Groq - llama-3.1-8b-instant")
        return llm, "Groq - Llama 3.1 8B"
    except Exception as e:
        errors.append(f"Groq: {e}")

    # 4. Gemini 2.0 Flash
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        llm.invoke("hi")
        print("[LLM] Active: Google Gemini 2.0 Flash")
        return llm, "Google Gemini 2.0 Flash"
    except Exception as e:
        errors.append(f"Gemini: {e}")

    # 5. SambaNova - 8B model
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="Meta-Llama-3.1-8B-Instruct",
            api_key=os.getenv("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
            temperature=0.1,
            max_tokens=2048
        )
        llm.invoke("hi")
        print("[LLM] Active: SambaNova - Llama 3.1 8B")
        return llm, "SambaNova - Llama 3.1 8B"
    except Exception as e:
        errors.append(f"SambaNova: {e}")

    # 6. Ollama local - last resort
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2", temperature=0.1)
        llm.invoke("hi")
        print("[LLM] Active: Ollama local llama3.2")
        return llm, "Ollama (local)"
    except Exception as e:
        errors.append(f"Ollama: {e}")

    raise RuntimeError("All 6 LLMs failed:\n" + "\n".join(errors))


def get_working_embeddings():
    # 1. Google embedding-001 - free, reliable
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        emb = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        emb.embed_query("test")
        print("[EMB] Active: Google embedding-001")
        return emb, "Google embedding-001"
    except Exception as e:
        print(f"[EMB] Google failed: {e}")

    # 2. HuggingFace local - no API key needed
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("[EMB] Active: HuggingFace MiniLM (local)")
        return emb, "HuggingFace MiniLM (local)"
    except Exception as e:
        print(f"[EMB] HuggingFace failed: {e}")
        
    raise RuntimeError("All embedding models failed.")
