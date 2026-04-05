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
    file_content = b"%PDF-1.4..."
    files = [("files", ("test.pdf", file_content, "application/pdf"))]

    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    assert "batch_id" in response.json()
    assert len(response.json()["files"]) > 0

# 3. Test Query Logic (Requires session_id)
def test_query_endpoint():
    payload = {
        "session_id": "non-existent-id",
        "question": "What is in the attic?", 
    }
    response = client.post("/ask", json=payload) 

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"

# 4. Edge Case: Invalid File Type
def test_upload_invalid_file():
    files = [("files", ("test.txt", b"Hello World", "text/plain"))]
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "No valid PDF files uploaded"


# 5. Edge Case: Empty Query
def test_empty_query():
    payload = {"session_id": "test-session", "query": ""}
    response = client.post("/ask", json=payload)
    
    assert response.status_code == 422  
