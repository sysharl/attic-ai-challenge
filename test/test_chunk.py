from app.core.embeddings import build_faiss_index, encode_chunks
from app.core.pdf_processing import chunk_fixed, chunk_recursive, extract_text

def test_chunk_fixed_overlap():
    # Mock the input structure expected by the function
    text_blocks = [{"page_number": 1, "text": "abcdefghij"}]
    filename = "test.pdf"
    
    # chunk_size=4, overlap=1
    # 1st chunk: index 0 to 4 -> "abcd"
    # 2nd chunk: start = (4 - 1) = 3. index 3 to 7 -> "defg"
    # 3rd chunk: start = (7 - 1) = 6. index 6 to 10 -> "ghij"
    
    documents = chunk_fixed(text_blocks, filename, chunk_size=4, overlap=1)
    
    # Extract just the text for the assertion
    chunks = [doc.page_content for doc in documents]
    
    assert chunks == ["abcd", "defg", "ghij"]
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["source"] == filename


def test_chunk_recursive_metadata():
    text_blocks = [{"page_number": 1, "text": "Para1\n\nPara2"}]
    filename = "test.pdf"
    
    # Using a small chunk size to force a split at the separator
    documents = chunk_recursive(text_blocks, filename, chunk_size=10, overlap=0)
    
    assert len(documents) == 2
    assert documents[0].page_content == "Para1"
    assert documents[1].page_content == "Para2"
