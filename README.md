# Attic AI Challenge: RAG Backend Engine

A robust Retrieval-Augmented Generation (RAG) backend API built with **FastAPI** and **FAISS**. This engine handles high-fidelity PDF extraction, recursive document chunking, and grounded asynchronous querying.

---

## 🛠️ Technical Design & Rationale

### 1. Robust Text Extraction
* **Engine:** `pdfplumber`
* **Rationale:** Chosen for its superior handling of visual layouts, multi-column text, and table structures compared to stream-based parsers.
* **Edge Case Handling:**
    * **Empty Pages:** Validation layer skips null-content pages to prevent "noise" in the vector store.
    * **Tables:** Preserves spatial alignment to ensure tabular data remains coherent during the embedding process.

### 2. Chunking & Indexing Strategy

The system supports two distinct chunking strategies, configurable via environment variables to allow for performance tuning and strategy ablation:

### Chunking Methods
- **Primary Method:** RecursiveCharacterTextSplitter  
- **Secondary Method:** Fixed-Size Chunking (character-based)

### Parameters
- **Chunk Size:** 500 units 
  - Optimized for the 512-token limit of the selected embedding models.
- **Overlap:** 50 tokens (10%)
- **Top K:** 5 to give more context for modern llm models, still within the context window with enough for reasoning

### Justification
- The **recursive approach** respects structural boundaries (paragraphs `\n\n` → sentences `\n` → spaces), ensuring semantic units remain intact.
- A **10% overlap** ensures semantic continuity, acting as a bridge to prevent information loss or "context clipping" at chunk boundaries.

### Vector Store
- **FAISS (In-Memory)**
  - Chosen for sub-millisecond similarity search
  - Enables efficient ephemeral indexing per request/session
  - Optimized for ~20-page PDF document processing

---

### 3. Embedding & Model Selection

### Embeddings
- **Model:** `intfloat/e5-small-v2`
- Selected after evaluating small-class open-source models for the best balance between:
  - Retrieval accuracy (**39.38 MTEB**)
  - CPU-only latency ~35ms (CPU-only)

### Selection Rationale
- Based on our internal ablation study using a Fixed-Size Chunking baseline, e5-small-v2 was selected over BGE and GTE due to its superior processing speed:
  - Query Efficiency: It demonstrated the lowest query latency (~0.035s), making it 7x faster than BGE-Small in our CPU environment.
  - Consistency: Maintained a 90% Hit Rate@3, ensuring the LLM always receives relevant context from the 20-page document.

### Grounding
- Each response includes metadata-derived citations:
  - Source filename
  - Page number  
- This reduces *hallucinations by omission*

---

## Internal Evaluation Results
- The following tests were conducted on a 20-page technical PDF using Fixed-Size Chunking (Size=500, Overlap=50) to establish a performance baseline for the retrieval pipeline.

| Model Name             | MTEB Retrieval| Context Window | Indexing (20pg) | Query Latency | Hit Rate@3 |
|------------------------|---------------|----------------|-----------------|---------------|------------|
| e5-small-v2            | 39.38         | 512 tokens     | 10.28s          | ~0.035s       | 90%        |
| bge-small-en-v1.5      | 36.26         | 512 tokens     | 11.24s          | ~0.256s       | 90%        |
| gte-small              | 35.99         | 512 tokens     | 9.10s           | ~0.774s       | 90%        |

*Note: Retrieval scores reflect local performance on specific document types during testing. Query latency includes embedding generation and FAISS similarity search.
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
HUGGINGFACE_API_KEY=<your_huggingface_token_here>

# App Settings
PORT=8000

# Model Selection & Routing
LLM_PROVIDER=huggingface # Options: 'openai', 'huggingface'
BASE_MODEL=<model> # optional

# --- RAG Hyperparameters ---
# Controls the size of each text segment
CHUNK_SIZE=500
# Controls the 'bridge' context between segments
CHUNK_OVERLAP=50
# Number of document chunks retrieved per query
TOP_K=5

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
