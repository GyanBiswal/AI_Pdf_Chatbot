# from pypdf import PdfReader
# from typing import List, Dict

# def load_and_split_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
#     """Load PDF and split text into overlapping chunks for retrieval."""
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + " "

#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk_text = text[start:end]

#         # Just use generic metadata
#         chunks.append({
#             "text": chunk_text,
#             "metadata": {"source": "pdf"}
#         })

#         start += chunk_size - overlap

#     return chunks



from pypdf import PdfReader
from typing import List, Dict

def load_and_split_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Load PDF and split text into overlapping chunks for better retrieval.
    Returns a list of dicts: {"text": "...", "metadata": {"source": "pdf"}}
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Clean text: remove extra newlines and spaces
            page_text = " ".join(page_text.split())
            text += page_text + " "

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "metadata": {"source": "pdf"}})
        start += chunk_size - overlap

    return chunks




"""
    1. file_path: where your PDF is saved (like "uploaded.pdf").

    2. chunk_size: how big each text piece should be (default 1000 characters).

    3. overlap: how much the next chunk should repeat from the previous one (default 200 characters).
"""