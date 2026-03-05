# app.py
# Main Streamlit UI for the DevMind "Industrial Standard" Codebase RAG Assistant.
# Migrated to LangChain + FAISS for robust performance and professional features.

import os
import streamlit as st
from dotenv import load_dotenv
from ingest import clone_repo
from loaders import get_documents
from vector_store import build_vector_store, load_vector_store
from chat_engine import get_llm, create_chat_engine
from prompts import QUERY_CLASSIFIER_PROMPT

# Load environment variables
load_dotenv()

#  Page Config 
st.set_page_config(
    page_title="DevMind Codebase RAG Assistant",
    page_icon="[IQ]",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS (Industrial, Sleek, Professional)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/*  Root & Background  */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"], .stApp header { background: transparent !important; }
[data-testid="stSidebar"] { background: #1e293b !important; border-right: 1px solid #334155; }

/*  Hero Banner  */
.hero-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
}

/*  Chat UI  */
.msg-user {
    background: #334155;
    border-radius: 12px 12px 2px 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    align-self: flex-end;
}
.msg-assistant {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px 12px 12px 2px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/*  Sidebar UI  */
.sidebar-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 1.5rem;
}

/* Buttons */
.stButton > button {
    background: #0ea5e9 !important;
    color: white !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

#  Session State Init 
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "chat_history_log" not in st.session_state:
    st.session_state.chat_history_log = []

#  Sidebar 
with st.sidebar:
    st.markdown("<div style='text-align:center;'><h2>DevMind [IQ]</h2></div>", unsafe_allow_html=True)
    
    # Status
    is_ready = st.session_state.index_ready
    st.markdown(f"**Status:** {'[OK] Index Active' if is_ready else '[!] No Index Loaded'}")
    
    #  Ingest 
    st.markdown("<div class='sidebar-label'>Repository Ingestion</div>", unsafe_allow_html=True)
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")

    if st.button("Build Industrial Index"):
        if repo_url.strip():
            with st.spinner("Analyzing codebase and generating vector store..."):
                try:
                    repo_name = repo_url.rstrip("/").split("/")[-1]
                    repo_path = os.path.join("./repo", repo_name)
                    storage_dir = os.path.join("./storage_faiss", repo_name)

                    clone_repo(repo_url, dest=repo_path)
                    docs = get_documents(repo_path)
                    vector_store = build_vector_store(docs, storage_dir=storage_dir)
                    
                    st.session_state.chat_chain = create_chat_engine(vector_store, "General Codebase Assistant")
                    st.session_state.index_ready = True
                    st.session_state.chat_history_log = []
                    st.success("Industrial Index Built!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion Fail: {e}")
        else:
            st.warning("Please provide a URL.")

    #  Load existing 
    st.markdown("<div class='sidebar-label'>Load Existing FAISS Index</div>", unsafe_allow_html=True)
    manual_storage = st.text_input("Index Directory", value="./storage_faiss")
    if st.button("Load RAG Engine"):
        with st.spinner("Loading LangChain context..."):
            try:
                # Find the first subdirectory if the user gave the root
                if manual_storage == "./storage_faiss":
                    if not os.path.exists("./storage_faiss"):
                        st.error("Storage directory not found.")
                        st.stop()
                    subs = [d for d in os.listdir("./storage_faiss") if os.path.isdir(os.path.join("./storage_faiss", d))]
                    if subs: manual_storage = os.path.join("./storage_faiss", subs[0])
                
                vector_store = load_vector_store(storage_dir=manual_storage)
                st.session_state.chat_chain = create_chat_engine(vector_store, "General Codebase Assistant")
                st.session_state.index_ready = True
                st.session_state.chat_history_log = []
                st.success("Loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Load Fail: {e}")

#  Main Area 
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>[+] Industrial DevMind</div>
    <div class='hero-sub'>LangChain-Powered Codebase Awareness with Professional Memory.</div>
</div>
""", unsafe_allow_html=True)

#  Chat History Display 
for chat in st.session_state.chat_history_log:
    st.markdown(f"<div class='msg-user'>{chat['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='msg-assistant'>{chat['a']}</div>", unsafe_allow_html=True)

#  Query Input 
query = st.chat_input("Query your codebase...")

if query:
    if not st.session_state.index_ready:
        st.warning("Please load or build an index first.")
    else:
        st.markdown(f"<div class='msg-user'>{query}</div>", unsafe_allow_html=True)
        with st.spinner("Retrieving industrial context..."):
            try:
                # Modern LangChain 0.4.x invocation
                result = st.session_state.chat_chain.invoke({"input": query})
                answer = result["answer"]
                
                # Metadata sources from context
                context_docs = result.get("context", [])
                if not context_docs:
                    full_answer = "I could not find any relevant files in the index to answer this question. Please ensure the repository is fully ingested."
                else:
                    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in context_docs]))
                    # Get relative path for cleaner display
                    clean_sources = []
                    for s in sources:
                        # Try to make it relative to the repo root
                        parts = s.split(os.sep)
                        if "repo" in parts:
                            clean_sources.append(os.sep.join(parts[parts.index("repo")+2:]))
                        else:
                            clean_sources.append(os.path.basename(s))

                    source_text = "\n\n**Sources Analyzed:**\n" + "\n".join([f"- `{s}`" for s in clean_sources[:5]])
                    full_answer = f"{answer}{source_text}"
                
                st.session_state.chat_history_log.append({"q": query, "a": full_answer})
                st.rerun()
            except Exception as e:
                st.error(f"Query Error: {e}")

    #  Clear chat 
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All History"):
        st.session_state.chat_history_log = []
        st.rerun()
