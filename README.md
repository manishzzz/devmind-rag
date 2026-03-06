# 🧠 DevMind — Codebase Intelligence Assistant

> Ask anything about any GitHub repository. Zero hallucination. Fully grounded answers.

## ✨ Features

- **Free & Unlimited LLM** — Groq Llama 3.3 70B (primary) · Gemini 2.0 Flash (fallback)
- **Local Embeddings** — HuggingFace `all-MiniLM-L6-v2` · No API cost
- **FAISS Vector Search** — Fast, local, persistent index
- **Anti-Hallucination** — Strict grounding; cites exact source files
- **Structural Awareness** — Understands project folder hierarchy
- **Streamlit UI** — Industrial dark design, chat-native interface

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Setup
```bash
git clone <this-repo>
cd devmind

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)
```

### 3. Run
```bash
streamlit run app.py
```

## 🔑 API Keys (Free)

| Provider | Usage | Get Key |
|----------|-------|---------|
| **Groq** | Primary LLM (required) | [console.groq.com](https://console.groq.com) |
| Google Gemini | Fallback LLM (optional) | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

## 📁 Project Structure

```
devmind/
├── app.py            # Streamlit UI & session management
├── chat_engine.py    # LLM selection + retrieval chain
├── loaders.py        # File loading + repo tree generation
├── vector_store.py   # FAISS index build/load
├── ingest.py         # GitHub repo cloning
├── requirements.txt  # Python dependencies
├── Dockerfile        # Cloud Run deployment
├── .env.example      # API key template
└── .gitignore
```

## 🐳 Deploy to Cloud Run

```bash
gcloud run deploy devmind \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars "GROQ_API_KEY=your_key"
```

## 💡 Usage

1. Paste a GitHub repository URL in the sidebar
2. Click **Build Index** — DevMind clones and indexes the repo
3. Ask questions in natural language
4. Get grounded, cited answers referencing actual source files

## ⚙️ Tech Stack

- **LangChain** 0.3+ (retrieval chain)
- **FAISS** (local vector store)
- **Groq** (free LLM API — Llama 3.3 70B)
- **HuggingFace** (local sentence embeddings)
- **Streamlit** (web UI)
