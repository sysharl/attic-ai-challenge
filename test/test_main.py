import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# 1. Basic Health Check
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# 2. Test PDF Upload (Successful Case)
def test_upload_pdf():
    # Create a dummy PDF content for testing
    file_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\ntrailer<</Root 1 0 R>>%%EOF"
    files = {"file": ("test.pdf", file_content, "application/pdf")}

    response = client.post("/upload", files=files)

    # Assert successful ingestion
    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["message"] == "File processed successfully"


# 3. Test Query Logic (Requires session_id)
def test_query_endpoint():
    # Mock a query to an existing session
    # Note: In a real test, you'd upload first to get a valid session_id
    payload = {
        "session_id": "test-session-123",
        "query": "What is the capital of France?",
    }
    response = client.post("/query", json=payload)

    # Even if the session doesn't exist, we test the error handling
    if response.status_code == 404:
        assert response.json()["detail"] == "Session not found"
    else:
        assert response.status_code == 200
        assert "answer" in response.json()
        assert "sources" in response.json()


# 4. Edge Case: Invalid File Type
def test_upload_invalid_file():
    files = {"file": ("test.txt", b"Hello World", "text/plain")}
    response = client.post("/upload", files=files)

    # Verify the backend rejects non-PDF files (400 or 422 depending on your logic)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


# 5. Edge Case: Empty Query
def test_empty_query():
    payload = {"session_id": "test-session", "query": ""}
    response = client.post("/ask", json=payload)

    # Verify the API design catches empty inputs
    assert response.status_code == 422  # FastAPI's default for validation error
