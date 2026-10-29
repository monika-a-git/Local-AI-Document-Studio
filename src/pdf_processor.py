"""
PDF extraction and preprocessing module.
Features:
- File validation (size limit, extension, PDF header integrity)
- Corrupt/encrypted PDF detection
- SHA-256 duplicate detection hash
- Accurate 1-indexed page numbering
- Recursive character text splitting
"""

from pathlib import Path
import hashlib
from typing import List, Dict, Any, Union
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
)


def compute_file_hash(file_input: Union[str, Path, bytes]) -> str:
    """Calculate SHA-256 hash of a file or byte array for duplicate detection."""
    sha256 = hashlib.sha256()
    if isinstance(file_input, (str, Path)):
        with open(file_input, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
    elif isinstance(file_input, bytes):
        sha256.update(file_input)
    else:
        raise TypeError("file_input must be a file path or bytes.")
    return sha256.hexdigest()


def validate_pdf(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate that a PDF exists, is within size limits, is unencrypted, and is readable.
    Returns a dictionary of document metadata if valid.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a file with a .pdf extension.")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("The provided PDF file is empty (0 bytes).")

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"File size exceeds the {MAX_UPLOAD_SIZE_MB}MB safety limit for CPU execution."
        )

    # Validate PDF structure using pypdf
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Malformed or corrupt PDF file: {exc}") from exc

    if reader.is_encrypted:
        raise ValueError(
            "The PDF is encrypted or password-protected. Please decrypt it before uploading."
        )

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("The PDF contains no pages.")

    doc_info = reader.metadata or {}
    title = getattr(doc_info, "title", None) or path.stem
    author = getattr(doc_info, "author", None) or "Unknown Author"

    return {
        "filename": path.name,
        "file_size": file_size,
        "total_pages": total_pages,
        "title": str(title),
        "author": str(author),
        "hash": compute_file_hash(path),
    }


def extract_pdf_pages(pdf_path: Union[str, Path]) -> List[Document]:
    """
    Extract readable text from PDF pages with accurate 1-indexed page metadata.
    """
    path = Path(pdf_path)
    metadata_info = validate_pdf(path)
    reader = pypdf.PdfReader(str(path))

    documents: List[Document] = []
    file_hash = metadata_info["hash"]
    total_pages = metadata_info["total_pages"]

    for idx, page in enumerate(reader.pages):
        page_number = idx + 1  # 1-indexed for human readability and citations
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        clean_text = text.strip()
        if clean_text:
            doc = Document(
                page_content=clean_text,
                metadata={
                    "source": path.name,
                    "file_hash": file_hash,
                    "page": page_number,
                    "page_index": idx,
                    "total_pages": total_pages,
                    "title": metadata_info["title"],
                },
            )
            documents.append(doc)

    if not documents:
        raise ValueError(
            "No readable text found in this PDF. It may be scanned or image-only."
        )

    return documents


def process_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Load a PDF, extract non-empty pages, split into overlapping chunks,
    and attach deterministic source and page metadata.
    """
    pages = extract_pdf_pages(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Attach chunk index to each chunk's metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
