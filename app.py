
# app.py
# Premium Streamlit UI for DevMind Ã¢â‚¬â€œ Codebase-Aware RAG Assistant.

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Fix asyncio event loop for Streamlit + LlamaIndex Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import nest_asyncio
nest_asyncio.apply()

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Configure Models (Gemini if API Key exists, else local Ollama) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from llama_index.core import Settings

if os.getenv("GOOGLE_API_KEY"):
    from llama_index.llms.gemini import Gemini
    from llama_index.embeddings.gemini import GeminiEmbedding
    
    Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"))
    MODELS_MODE = "Gemini (Cloud)"
else:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    Settings.llm = Ollama(model="llama2", request_timeout=600.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    MODELS_MODE = "Ollama (Local)"

from ingest import clone_repo
from graph_index import build_index, load_index
from query_engine import get_answer

load_dotenv()

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Page Config Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
st.set_page_config(
    page_title="DevMind Ã¢â‚¬â€œ Codebase RAG Assistant",
    page_icon="Ã°Å¸Â§Â ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Custom CSS Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Ã¢â€â‚¬Ã¢â€â‚¬ Root & Background Ã¢â€â‚¬Ã¢â€â‚¬ */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"], .stApp header { background: transparent !important; }
[data-testid="stSidebar"] { background: #111320 !important; border-right: 1px solid #1e2235; }

/* Ã¢â€â‚¬Ã¢â€â‚¬ Hero Banner Ã¢â€â‚¬Ã¢â€â‚¬ */
.hero-banner {
    background: linear-gradient(135deg, #1a1f3a 0%, #0f1629 50%, #151b35 100%);
    border: 1px solid #2a3158;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Chat Messages Ã¢â€â‚¬Ã¢â€â‚¬ */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
}
.msg-user {
    background: linear-gradient(135deg, #1e2a4a, #1a2240);
    border: 1px solid #2e3d6b;
    border-radius: 14px 14px 4px 14px;
    padding: 0.9rem 1.2rem;
    align-self: flex-end;
    max-width: 85%;
    font-size: 0.97rem;
    color: #c7d2fe;
}
.msg-assistant {
    background: linear-gradient(135deg, #131824, #0f1520);
    border: 1px solid #1e2940;
    border-radius: 14px 14px 14px 4px;
    padding: 1rem 1.3rem;
    max-width: 92%;
    font-size: 0.97rem;
    color: #e2e8f0;
    position: relative;
}
.msg-assistant::before {
    content: 'Ã°Å¸Â§Â ';
    position: absolute;
    top: -12px; left: 10px;
    font-size: 1.1rem;
}
.category-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.badge-FUNCTION   { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb55; }
.badge-ARCHITECTURE { background: #2d1f5e; color: #a78bfa; border: 1px solid #7c3aed55; }
.badge-DEBUG      { background: #3b1f1f; color: #f87171; border: 1px solid #dc262655; }
.badge-ISSUE_PR   { background: #1f3b2a; color: #4ade80; border: 1px solid #16a34a55; }

/* Ã¢â€â‚¬Ã¢â€â‚¬ Sidebar UI Ã¢â€â‚¬Ã¢â€â‚¬ */
.sidebar-section {
    background: #161927;
    border: 1px solid #1e2438;
    border-radius: 12px;
    padding: 1rem 1rem 0.75rem;
    margin-bottom: 1rem;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.status-dot.off { background: #ef4444; box-shadow: 0 0 6px #ef4444; animation: none; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

/* Ã¢â€â‚¬Ã¢â€â‚¬ Input area Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stTextInput"] > div > div > input {
    background: #131824 !important;
    border: 1px solid #2a3158 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.97rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Buttons Ã¢â€â‚¬Ã¢â€â‚¬ */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* Ã¢â€â‚¬Ã¢â€â‚¬ Spinner / Success / Warning Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stSpinner"] { color: #818cf8 !important; }
.stSuccess { background: #0f2a1e !important; border-color: #16a34a !important; }
.stWarning { background: #2a1d0a !important; }
.stError   { background: #2a0f0f !important; }

/* Code blocks inside answers */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0b0f1c !important;
    border: 1px solid #1e2438 !important;
    border-radius: 6px !important;
    color: #a5b4fc !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Session State Init Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
if "index" not in st.session_state:
    st.session_state.index = None
if "storage_dir" not in st.session_state:
    st.session_state.storage_dir = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sidebar Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2.8rem;'>Ã°Å¸Â§Â </div>
        <div style='font-size:1.2rem; font-weight:700; 
             background:linear-gradient(90deg,#818cf8,#c084fc);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text;'>DevMind</div>
        <div style='font-size:0.78rem; color:#475569; margin-top:2px;'>Codebase RAG Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Ã¢â€â‚¬Ã¢â€â‚¬ Status Ã¢â€â‚¬Ã¢â€â‚¬
    is_ready = st.session_state.index_ready
    dot_class = "status-dot" if is_ready else "status-dot off"
    status_text = "Index Ready" if is_ready else "No Index Loaded"
    st.markdown(f"""
    <div class='sidebar-section' style='margin-bottom:1.2rem;'>
        <div class='sidebar-label'>Status</div>
        <span class='{dot_class}'></span>
        <span style='font-size:0.88rem; color:{"#4ade80" if is_ready else "#f87171"};'>{status_text}</span>
        <div style='font-size:0.7rem; color:#64748b; margin-top:4px;'>Mode: {MODELS_MODE}</div>
    </div>
    """, unsafe_allow_html=True)

    # Ã¢â€â‚¬Ã¢â€â‚¬ Ingest Ã¢â€â‚¬Ã¢â€â‚¬
    st.markdown("<div class='sidebar-label'>Ã°Å¸â€œÂ¦ Ingest a Repository</div>", unsafe_allow_html=True)
    repo_url = st.text_input(
        "GitHub Repo URL",
        placeholder="https://github.com/user/repo",
        label_visibility="collapsed",
    )

    if st.button("Ã¢Å¡Â¡ Ingest & Build Index"):
        if repo_url.strip():
            with st.spinner("Cloning repo and building knowledge graphÃ¢â‚¬Â¦"):
                try:
                    repo_name = repo_url.rstrip("/").split("/")[-1]
                    repo_path = os.path.join(os.getenv("GITHUB_REPO_PATH", "./repo"), repo_name)
                    storage_dir = os.path.join("./storage", repo_name)

                    clone_repo(repo_url, dest=repo_path)
                    index = build_index(repo_path=repo_path, storage_dir=storage_dir)
                    st.session_state.index = index
                    st.session_state.storage_dir = storage_dir
                    st.session_state.index_ready = True
                    st.session_state.chat_history = []
                    st.success("Ã¢Å“â€¦ Index built successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ã¢ÂÅ’ Error: {e}")
        else:
            st.warning("Please provide a valid GitHub URL.")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Ã¢â€â‚¬Ã¢â€â‚¬ Load existing Ã¢â€â‚¬Ã¢â€â‚¬
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>Ã°Å¸â€™Â¾ Load Existing Index</div>", unsafe_allow_html=True)
    manual_storage = st.text_input(
        "Storage path",
        value="./storage",
        label_visibility="collapsed",
    )
    if st.button("Ã°Å¸â€œâ€š Load Index"):
        with st.spinner("Loading from storageÃ¢â‚¬Â¦"):
            try:
                index = load_index(storage_dir=manual_storage)
                st.session_state.index = index
                st.session_state.storage_dir = manual_storage
                st.session_state.index_ready = True
                st.session_state.chat_history = []
                st.success("Ã¢Å“â€¦ Loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Ã¢ÂÅ’ Could not load index: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Ã¢â€â‚¬Ã¢â€â‚¬ Clear chat Ã¢â€â‚¬Ã¢â€â‚¬
    if st.session_state.chat_history:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Ã°Å¸â€”â€˜Ã¯Â¸Â Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # Ã¢â€â‚¬Ã¢â€â‚¬ Info Ã¢â€â‚¬Ã¢â€â‚¬
    st.markdown("""
    <div style='position:absolute; bottom:1.5rem; left:0; right:0; text-align:center;'>
        <div style='font-size:0.72rem; color:#334155;'>Powered by LlamaIndex Ã‚Â· Ollama</div>
    </div>
    """, unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Main Area Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
st.markdown("""
<div class='hero-banner'>
    <p class='hero-title'>Ã°Å¸Â§Â  DevMind</p>
    <p class='hero-sub'>
        Ask anything about your codebase Ã¢â‚¬â€ architecture, functions, bugs, PRs, and more.
        Powered by a graph-based RAG engine.
    </p>
</div>
""", unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬ Category legend Ã¢â€â‚¬Ã¢â€â‚¬
st.markdown("""
<div style='display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1.5rem;'>
    <span class='category-badge badge-FUNCTION'>FUNCTION</span>
    <span class='category-badge badge-ARCHITECTURE'>ARCHITECTURE</span>
    <span class='category-badge badge-DEBUG'>DEBUG</span>
    <span class='category-badge badge-ISSUE_PR'>ISSUE / PR</span>
</div>
""", unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬ Chat history Ã¢â€â‚¬Ã¢â€â‚¬
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        st.markdown(f"<div class='msg-user'>Ã°Å¸â€™Â¬ {entry['query']}</div>", unsafe_allow_html=True)
        cat = entry['category']
        st.markdown(f"""
        <div class='msg-assistant'>
            <span class='category-badge badge-{cat}'>{cat}</span><br>
            {entry['answer'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    if not st.session_state.index_ready:
        st.markdown("""
        <div style='text-align:center; padding:4rem 0;'>
            <div style='font-size:3.5rem; margin-bottom:1rem;'>Ã°Å¸â€œâ€š</div>
            <div style='color:#475569; font-size:1.05rem;'>
                Paste a GitHub URL in the sidebar and click <strong style='color:#818cf8;'>Ingest &amp; Build Index</strong> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:4rem 0;'>
            <div style='font-size:3.5rem; margin-bottom:1rem;'>Ã°Å¸â€™Â¬</div>
            <div style='color:#475569; font-size:1.05rem;'>
                Index is ready! Ask your first question below.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬ Input bar Ã¢â€â‚¬Ã¢â€â‚¬
st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

# Using st.chat_input for modern UX and to fix the infinite rerun loop
prompt = st.chat_input("Ask a question about your code...")

if prompt:
    if not st.session_state.index_ready:
        st.warning("Ã¢Å¡Â Ã¯Â¸Â Please ingest a repo or load an existing index first.")
    else:
        # Display the user message immediately for responsiveness
        st.markdown(f"<div class='msg-user'>Ã°Å¸â€™Â¬ {prompt}</div>", unsafe_allow_html=True)
        
        with st.spinner("ThinkingÃ¢â‚¬Â¦"):
            try:
                response_text, category = get_answer(st.session_state.index, prompt.strip())
                st.session_state.chat_history.append({
                    "query": prompt.strip(),
                    "answer": response_text,
                    "category": category,
                })
                # Single rerun to update the history display correctly
                st.rerun()
            except Exception as e:
                st.error(f"Ã¢ÂÅ’ Error: {e}")
