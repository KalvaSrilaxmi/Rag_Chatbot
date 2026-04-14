# RAG Document Q&A Bot

A professional Retrieval-Augmented Generation (RAG) application that answers user questions from uploaded documents.  
The system ingests files from the `data/` folder, creates embeddings, stores them in FAISS, retrieves the most relevant chunks, and generates grounded answers with source citations (filename + page).  
It supports local/offline inference with Ollama and can also run with OpenAI.

**Repository:** [KalvaSrilaxmi/Rag_Chatbot](https://github.com/KalvaSrilaxmi/Rag_Chatbot)

## Tech Stack

| Category | Tool/Library | Version |
|---|---|---|
| Language | Python | 3.10+ recommended |
| Orchestration | `langchain` | `>=0.2.0` |
| Community Integrations | `langchain-community` | `>=0.2.0` |
| Core Interfaces | `langchain-core` | `>=0.2.0` |
| Text Splitting | `langchain-text-splitters` | `>=0.2.0` |
| OpenAI Integration | `langchain-openai` | `>=0.1.0` |
| Embeddings | `sentence-transformers` | `>=2.7.0` |
| Vector DB | `faiss-cpu` | `>=1.9.0` |
| PDF Parsing | `pypdf` | `>=4.2.0` |
| DOCX Parsing | `python-docx` | `>=1.1.0` |
| Environment Management | `python-dotenv` | `>=1.0.1` |
| UI | `streamlit` | `>=1.34.0` |

## Architecture Overview

The project follows a standard RAG pipeline:

1. **Ingestion**: Load documents from `data/` (`PDF` mandatory, `TXT/DOCX` optional).  
2. **Chunking**: Split extracted text into overlapping chunks and attach metadata (`filename`, `page`).  
3. **Embedding**: Convert chunks into dense vectors using a sentence-transformer model (batch processing).  
4. **Vector Storage**: Save vectors to a persistent local FAISS index in `vectorstore/`.  
5. **Retrieval**: For each query, run top-k similarity search and filter weak matches with a threshold.  
6. **Answer Generation**: Send only retrieved context to the LLM and return a grounded answer with citations, or fallback to `"Answer not found in documents."`

## Chunking Strategy

- **Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk size**: `500`
- **Chunk overlap**: `50`
- **Separators**: `["\n\n", "\n", ".", " ", ""]`

### Why this strategy?

- Preserves semantic continuity across chunk boundaries.
- Reduces context loss for multi-sentence ideas.
- Balances retrieval precision with context coverage.
- Works reliably across mixed document styles (reports, notes, structured text).

## Embedding Model & Vector Database

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: `FAISS` (local, persistent)

### Why these choices?

- `all-MiniLM-L6-v2` is lightweight, fast, and effective for semantic similarity on CPU.
- FAISS provides fast local vector search with no external infrastructure.
- Both are practical for internship demos and small-to-medium document collections.

## Setup Instructions

### 1) Clone repository

```bash
git clone https://github.com/KalvaSrilaxmi/Rag_Chatbot.git
cd Rag_Chatbot
```

On macOS/Linux, use `cp` instead of `copy` when copying `.env.example` to `.env` (see step 3).

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

**Windows (PowerShell / CMD):**

```bash
copy .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Then edit `.env` with your provider settings. **Do not commit `.env`**; it is listed in `.gitignore` and must stay local.

### 4) Add documents

- Place 4-5 meaningful documents in `data/`
- At least one must be a PDF (mandatory)

### 5) Run indexing

```bash
python index.py
```

### 6) Launch Streamlit app

```bash
python -m streamlit run app.py
```

### Optional: Run CLI mode

```bash
python main.py
```

## Environment Variables

Create a `.env` file using the template below:

```env
# LLM provider: "ollama" or "openai"
LLM_PROVIDER="ollama"

# Ollama settings
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="llama3"

# OpenAI settings
OPENAI_API_KEY="your-openai-api-key-here"
OPENAI_MODEL="gpt-3.5-turbo"

# Embedding model
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

> **Security Warning:** Never commit real API keys to GitHub. Keep `.env` private and share only `.env.example`.

## Example Queries

Use questions like:

1. **What is Artificial Intelligence in healthcare?**  
   Expected: Grounded answer from AI healthcare document with citations.

2. **What are the key impacts of climate change?**  
   Expected: Relevant climate findings and page-level citations.

3. **What role does the WTO play in global trade?**  
   Expected: Trade-focused answer with proper source metadata.

4. **What are common cybersecurity threats and defenses?**  
   Expected: Threat/defense summary from cybersecurity doc with citations.

5. **What is Blockchain?** *(not in corpus example)*  
   Expected: `"Answer not found in documents."`

## Known Limitations

- Retrieval quality depends on chunking and embedding model; very nuanced queries may need reranking.
- PDF extraction works best with machine-readable PDFs; scanned/image-only PDFs need OCR integration.
- CPU-only embedding + local LLM can be slower on low-resource machines.
- Current setup is optimized for small-to-medium datasets, not large-scale production indexing.

## Project Structure

```text
.
├── data/                 # Input documents (PDF/TXT/DOCX)
├── vectorstore/          # Persistent FAISS index files
├── src/
│   ├── ingest.py         # Document loading + cleaning
│   ├── chunk.py          # Chunking strategy + metadata
│   ├── embed.py          # Embedding model loader
│   ├── vector_store.py   # FAISS create/load logic
│   ├── retriever.py      # Top-k retrieval + threshold filtering
│   └── qa_chain.py       # Prompting + grounded answer generation
├── index.py              # Indexing entry point
├── main.py               # CLI chat entry point
├── app.py                # Streamlit app entry point
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```
