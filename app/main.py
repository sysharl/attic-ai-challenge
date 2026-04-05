import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import List

from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile, APIRouter)
from pydantic import BaseModel
from werkzeug.utils import secure_filename

from app.config import API_PREFIX

from app.core.answer_generation import generate_answer
from app.core.embeddings import encode_query, retrieve_top_k
from app.services.session_manager import create_session, get_session

app = FastAPI(title="PDF Q&A API")
# configure logging
logging.basicConfig(level=logging.INFO)
# -----------------------
# Request / Response Models
# -----------------------
class AskRequest(BaseModel):
    session_id: str
    question: str


# -----------------------
# Health Check
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


UPLOAD_FOLDER = "./data"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------
# Upload Endpoint
# -----------------------
@app.post("/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    strategy: str = Form("recursive"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Create batch folder
    batch_id = datetime.today().strftime("%Y-%m-%d") + "_" + str(uuid.uuid4())[:8]
    batch_dir = os.path.join(UPLOAD_FOLDER, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    results = []

    for file in files:
        if not allowed_file(file.filename):
            logging.warning(f"Skipping invalid file: {file.filename}")
            continue

        try:
            # 2. Secure and Save the file
            safe_name = secure_filename(file.filename)
            file_path = os.path.join(batch_dir, safe_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            session_id = create_session(file_path, strategy=strategy)
            session = get_session(session_id)

            results.append(
                {
                    "file_name": file.filename,
                    "session_id": session_id,
                    "chunks": len(session["chunks"]),
                }
            )

        except Exception as e:
            results.append({"file_name": file.filename, "error": str(e)})

    if not results:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    return {"batch_id": batch_id, "total_files": len(results), "files": results}


# -----------------------
# Ask Endpoint
# -----------------------
@app.post("/ask")
async def ask(request: AskRequest):
    session = get_session(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    index = session.get("index")
    chunks = session.get("chunks")

    if index is None or chunks is None:
        raise HTTPException(status_code=400, detail="Document not properly processed")

    try:
        # Encode query
        query_embedding = encode_query(request.question)

        # Retrieve top-k indices
        indices = retrieve_top_k(index, query_embedding)

        results = []
        for idx in indices:
            if 0 <= idx < len(chunks):
                results.append(chunks[idx])

    except Exception as e:
        print("Retrieval error:", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve answer")

    # -----------------------
    # No Results Case
    # -----------------------
    if not results:
        return {"answer": "No relevant information found.", "sources": []}

    # -----------------------
    # Prepare chunks for LLM
    # -----------------------
    formatted_chunks = []

    for i, r in enumerate(results):
        text = getattr(r, "page_content", None) or r.get("text", "")
        metadata = (
            getattr(r, "metadata", {})
            if hasattr(r, "metadata")
            else r.get("metadata", {})
        )
        page = metadata.get("page", "N/A")
        source = metadata.get("source", "Unknown Document")

        formatted_chunks.append(
            {"text": text, "page": page, "source": source, "chunk_index": i}
        )

    return {"answer": "answer_text", "sources": formatted_chunks}
    # -----------------------
    # Generate Answer
    # -----------------------
    response = generate_answer(
        query=request.question,
        chunks=formatted_chunks
    )

    return response
