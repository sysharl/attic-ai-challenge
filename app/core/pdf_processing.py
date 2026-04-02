import logging

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text(file_path: str) -> list[dict]:
    """
    Extracts text from PDF preserving paragraphs and returns list of page dicts.
    """
    pages = []

    # 1. Immediate check: Does the file path even exist?
    if not file_path:
        return []

    try:
        with pdfplumber.open(file_path) as pdf:
            # 2. Check if the PDF object was created and has pages
            if pdf is None or not pdf.pages:
                return []

            for i, page in enumerate(pdf.pages):
                # 3. Extraction with error handling per page
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                except Exception as page_err:
                    logging.warning(
                        f"Warning: Could not extract page {i+1}: {page_err}"
                    )
                    continue  # Skip this page and move to the next

                # 4. Content Validation: Ensure text isn't just None, empty, or whitespace
                if text and isinstance(text, str) and text.strip():
                    pages.append(
                        {
                            "page_number": i + 1,
                            "text": text.strip(),  # Cleaning up leading/trailing whitespace
                        }
                    )

    except Exception as e:
        logging.error(f"Error opening PDF {file_path}: {e}")
        return []  # Return empty list so downstream code doesn't crash on 'None'

    # 5. Final safety check: Return empty list if no valid text was found across all pages
    return pages


def chunk_fixed(
    text_blocks: list, filename: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
) -> list:
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
        local_chunk_count = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,  # <--- Capture the filename
                        "page": page_num,  # Page number from PDF
                        "chunk_index": local_chunk_count,  # Index for this page
                    },
                )
            )
            local_chunk_count += 1
            start += chunk_size - overlap
    return documents


def chunk_recursive(
    text_blocks: list, filename: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
) -> list:
    if not text_blocks:
        logging.warning("Warning: No text blocks provided to chunker. Skipping.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )

    documents = []

    for item in text_blocks:
        if item is None or not isinstance(item, dict):
            continue

        page_num = item.get("page_number", 0)  # Use .get() to avoid KeyErrors
        text = item.get("text", "")

        if not text:  # Skip empty pages
            continue

        chunks = splitter.split_text(text)
        char_start = 0

        for i, chunk in enumerate(chunks):
            char_end = char_start + len(chunk)

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "source": filename,
                        "chunk_index": i,  # chunk number on this page
                        "char_start": char_start,
                        "char_end": char_end,
                    },
                )
            )
            char_start = char_end
    return documents
