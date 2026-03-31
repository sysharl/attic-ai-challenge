from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pdfplumber
from config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text(file_path: str) -> list[dict]:
    """
    Extracts text from PDF preserving paragraphs and returns list of page dicts.
    """
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages.append({"page_number": i+1, "text": text})
    return pages

def chunk_fixed(text_blocks: list, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    """
    Chunk text into fixed-size character windows with overlap.
    Returns: list of LangChain Documents
    """

    documents = []

    for item in text_blocks:
        page_num = item["page_number"]
        text = item["text"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_num}
                )
            )

            start += chunk_size - overlap

    return documents

def chunk_recursive(text_blocks: list, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []

    for item in text_blocks:
        page_num = item["page_number"]
        text = item["text"]

        chunks = splitter.split_text(text)
        char_start = 0
        
        for i, chunk in enumerate(chunks):
            char_end = char_start + len(chunk)
            
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "chunk_index": i,        # chunk number on this page
                        "char_start": char_start,
                        "char_end": char_end
                    }
                )
            )
            char_start = char_end
    return documents