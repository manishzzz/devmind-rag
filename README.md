  # DevMind - Codebase-Aware RAG Assistant

DevMind is an advanced AI assistant that understands your codebase using LlamaIndex's `PropertyGraphIndex`. It classifies your questions to provide accurate answers regarding function explanations, architecture overview, debugging, and issue/PR summaries.

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your `OPENAI_API_KEY`.

3. **Run Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Run via CLI**:
   ```bash
   python main.py https://github.com/username/repo
   ```
