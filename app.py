# app.py - Streamlit UI for DevMind
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from ingest import clone_repo, _safe_delete_folder
from indexer import build_index, load_index, get_index_stats
from query_engine import get_answer, extract_target_class

st.set_page_config(
    page_title="DevMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; }
    .main-title { font-size: 3rem; font-weight: 800; color: #58a6ff; }
    .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-right: 5px; }
    .badge-function { background-color: rgba(88, 166, 255, 0.2); color: #58a6ff; border: 1px solid #58a6ff; }
    .badge-architecture { background-color: rgba(187, 128, 255, 0.2); color: #bc8cff; border: 1px solid #bc8cff; }
    .badge-debug { background-color: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }
    .badge-issue { background-color: rgba(63, 185, 80, 0.2); color: #3fb950; border: 1px solid #3fb950; }
</style>
""", unsafe_allow_html=True)

# --- Init LLM once ---
if "llm" not in st.session_state:
    with st.spinner("Initializing LLM chain..."):
        try:
            from llm_router import get_working_llm, get_working_embeddings
            llm, llm_name = get_working_llm()
            embeddings, emb_name = get_working_embeddings()
            st.session_state.llm = llm
            st.session_state.embeddings = embeddings
            st.session_state.llm_name = llm_name
            st.session_state.emb_name = emb_name
            st.session_state.index = None
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"LLM initialization failed: {str(e)}")
            st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='color:#58a6ff;'>🧠 DevMind</h1>", unsafe_allow_html=True)
    st.caption("Codebase RAG Assistant · PropertyGraph Edition")
    st.divider()
    
    # Status
    st.markdown("**SYSTEM STATUS**")
    stats = get_index_stats()
    if st.session_state.get("index") or stats:
        st.success("✅ Index Active")
        if stats:
            st.metric("Files Indexed", stats.get("total_files", 0))
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        
        if st.session_state.get("current_repo"):
            repo_name = st.session_state.current_repo.split("/")[-1]
            st.caption(f"📍 Active Repo: {repo_name}")
        else:
            st.caption("📍 Active Repo: Unnamed")
    else:
        st.warning("⚠️ No Index Loaded")
    
    st.info(f"LLM: {st.session_state.llm_name}")
    st.info(f"Embeddings: {st.session_state.emb_name}")
    st.divider()

    # Ingest
    st.markdown("**REPOSITY INGESTION**")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
    
    if st.button("🚀 Ingest & Build Index", type="primary", use_container_width=True):
        if not repo_url.strip():
            st.error("Please enter a valid GitHub URL")
        else:
            # CRITICAL: Clear old index from memory first
            st.session_state.index = None
            st.session_state.chat_history = []
            st.session_state.current_repo = ""

            # CRITICAL: Delete old folders before building (Windows-safe)
            _safe_delete_folder("./storage")
            _safe_delete_folder("./repo")
            print("[APP] Cleared old context folders")

            with st.spinner("Phase 1: Cloning fresh repository..."):
                try:
                    # Clear old repo folder too
                    if os.path.exists("./repo"):
                        shutil.rmtree("./repo")
                    repo_path = clone_repo(repo_url.strip())
                    st.toast("Cloning complete!", icon="✅")
                except Exception as e:
                    st.error(f"Clone failed: {e}")
                    st.stop()
            
            with st.spinner("Phase 2: Building PropertyGraph Index (Processing nodes)..."):
                try:
                    idx = build_index(
                        repo_path,
                        st.session_state.llm,
                        st.session_state.embeddings
                    )
                    st.session_state.index = idx
                    st.session_state.current_repo = repo_url.strip()
                    st.session_state.repo_path = repo_path
                    st.session_state.chat_history = []
                    st.success("Fresh index built and ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Index build failed: {e}")
    
    st.divider()
    
    # Load existing
    st.markdown("**PERSISTENT STORAGE**")
    storage_dir = st.text_input("Index Path", value="./storage")
    if st.button("📂 Load Existing Index", use_container_width=True):
        with st.spinner("Loading indexed shards..."):
            try:
                from indexer import load_index
                idx = load_index(
                    st.session_state.llm,
                    st.session_state.embeddings
                )
                st.session_state.index = idx
                st.session_state.chat_history = []
                
                # Restore repo name from stats if possible
                stats = get_index_stats()
                if stats and "repo_path" in stats:
                    st.session_state.current_repo = stats["repo_path"]
                    st.session_state.repo_path = stats["repo_path"]
                
                st.success("Index loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load index from {storage_dir}: {e}")

# --- Main area ---
st.markdown("<h1 class='main-title'>DevMind</h1>", unsafe_allow_html=True)

current_repo = st.session_state.get("current_repo", "")
if current_repo:
    repo_name = current_repo.split("/")[-1]
    st.caption(f"Answering from: {repo_name} (2026 Industrial-Grade)")
else:
    st.caption("No repo indexed yet · 2026 Industrial-Grade")

# Category badges
c1, c2, c3, c4 = st.columns(4)
c1.markdown("<div class='badge badge-function'>FUNCTION</div>", unsafe_allow_html=True)
c2.markdown("<div class='badge badge-architecture'>ARCHITECTURE</div>", unsafe_allow_html=True)
c3.markdown("<div class='badge badge-debug'>DEBUG</div>", unsafe_allow_html=True)
c4.markdown("<div class='badge badge-issue'>ISSUE/PR</div>", unsafe_allow_html=True)

st.divider()

# Chat history
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else None):
        if msg["role"] == "assistant" and "category" in msg:
            color_map = {
                "FUNCTION": "#58a6ff", "ARCHITECTURE": "#bc8cff",
                "DEBUG": "#f85149", "ISSUE_PR": "#3fb950"
            }
            color = color_map.get(msg["category"], "#58a6ff")
            st.markdown(f"<span style='color:{color}; font-weight:700;'>[{msg['category']}]</span>", unsafe_allow_html=True)
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📝 Source files used"):
                for s in msg["sources"]:
                    st.code(s, language="text")

# Input
if prompt := st.chat_input("Ask about codebase logic, architecture, or bugs..."):
    if not st.session_state.get("index"):
        st.warning("Please build or load a repository index first.")
    else:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Analyzing graph nodes and reasoning..."):
                try:
                    import query_engine
                    query_engine.REPO_PATH = st.session_state.get("repo_path", "./repo")
                    
                    answer, category, sources = get_answer(
                        st.session_state.index, prompt, st.session_state.llm
                    )
                    
                    # Warn if answer mentions a different class than what was asked
                    asked_class = extract_target_class(prompt)
                    if asked_class:
                        answered_class = extract_target_class(answer[:200])
                        if answered_class and answered_class != asked_class:
                            st.warning(
                                f"Asked about: {asked_class} | "
                                f"Answer mentions: {answered_class} — "
                                f"verify the response carefully"
                            )
                    
                    color_map = {
                        "FUNCTION": "#58a6ff", "ARCHITECTURE": "#bc8cff",
                        "DEBUG": "#f85149", "ISSUE_PR": "#3fb950"
                    }
                    color = color_map.get(category, "#58a6ff")
                    
                    st.markdown(f"<span style='color:{color}; font-weight:700;'>[{category}]</span>", unsafe_allow_html=True)
                    st.markdown(answer)

                    if sources:
                        with st.expander("📝 Source files used"):
                            for s in sources:
                                st.code(s, language="text")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "category": category,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Retrieval Error: {str(e)}")
