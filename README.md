# Attic AI Challenge: RAG Backend Engine

A robust Retrieval-Augmented Generation (RAG) backend API built with **FastAPI** and **FAISS**. This engine handles high-fidelity PDF extraction, recursive document chunking, and grounded asynchronous querying.

---

## 🛠️ Technical Design & Rationale

### 1. Robust Text Extraction
* **Engine:** `pdfplumber`
* **Rationale:** Chosen for its superior handling of visual layouts, multi-column text, and table structures compared to stream-based parsers.
* **Edge Case Handling:**
    * **Empty Pages:** Validation layer skips null-content pages to prevent "noise" in the vector store.
    * **Scanned PDFs:** [INSERT: e.g., "Handled via Tesseract OCR integration" OR "System returns a 422 Unprocessable Entity for non-selectable text"].
    * **Tables:** Preserves spatial alignment to ensure tabular data remains coherent during the embedding process.

### 2. Chunking & Indexing Strategy
The system supports two distinct chunking strategies, configurable via environment variables to allow for performance tuning and strategy ablation:

* **Primary Method:** `RecursiveCharacterTextSplitter`
* **Secondary Method:** Fixed-Size Chunking (Character-based)
* **Parameters:**
    * **Chunk Size:** `500` tokens (Optimized for granular context retrieval).
    * **Overlap:** `50` tokens (10% ratio).
* **Justification:** * The **Recursive approach** respects structural boundaries (paragraphs, then sentences), ensuring that semantic units stay intact.
    * An **Overlap of 10%** ensures "semantic continuity," preventing critical information loss or "context clipping" at chunk boundaries.
* **Vector Store:** **FAISS (In-Memory)**. Selected for sub-millisecond similarity search and efficient ephemeral indexing per request/session.
### 3. Embedding & Model Selection
* **Embeddings:** **[INSERT: e.g., text-embedding-3-small]** — Chosen for high MTEB scores and memory efficiency.
* **LLM Integration:** Supports dual routing via **OpenAI** (GPT-4o-mini) and **Hugging Face** (Qwen-2.5 via `InferenceClient`).
* **Grounding:** Every response includes metadata-derived citations (Source Filename & Page Number).

---

## 🏗️ System Architecture

The backend follows a modular, service-oriented architecture designed for **Separation of Concerns** and **Scalability**.

### Component Flow
1.  **Ingestion Service (`processor.py`):** Validates PDF binary streams and extracts raw text.
2.  **Vector Service (`vector_store.py`):** Manages real-time embedding generation and FAISS index management.
3.  **RAG Orchestrator (`main.py`):** Handles asynchronous FastAPI request/response cycles and prompt engineering.
4.  **Security & Isolation:** Every API request/session generates an isolated FAISS index instance. No data persists across sessions, ensuring 100% multi-tenant isolation.

### API Specifications
* **Framework:** FastAPI (Asynchronous)
* **Validation:** Pydantic models for strict request/response schema enforcement.
* **Documentation:** Interactive OpenAPI (Swagger) docs available at `/docs`.

---

## 🚀 Getting Started

### 1. Environment Configuration
Create a `.env` file in the root directory:

```env
# API Keys
OPENAI_API_KEY=<your_openai_key_here>
HF_TOKEN=<your_huggingface_token_here>

# App Settings
PORT=8000

# Model Selection & Routing
LLM_PROVIDER=openai # Options: 'openai', 'huggingface'
BASE_MODEL=gpt-4o-mini

# --- RAG Hyperparameters ---
# Controls the size of each text segment
CHUNK_SIZE=500
# Controls the 'bridge' context between segments
CHUNK_OVERLAP=50
# Number of document chunks retrieved per query
TOP_K=3

```

### 2. Local Run (Python)

#### Clone and enter repo
```
git clone https://github.com/sysharl/attic-ai-challenge
cd attic-ai-challenge
```

#### Install dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Start FastAPI server
```
uvicorn app.main:app --reload --port 8000
```

### 3. Docker Run (Containerized)

#### Build
```
docker build -t attic-backend .
```
#### Run (Automatically loads .env)
```
docker run -p 8000:8000 --env-file .env attic-backend
Test the API at http://localhost:8000/docs
```
