"""
Persistent vector store module using ChromaDB and local Ollama embeddings.
Supports multi-PDF incremental indexing, document-specific deletion,
metadata filtering (document selection), and collection statistics.
"""

from pathlib import Path
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Union

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
)

CHROMA_PATH = CHROMA_DIR
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL


def get_embeddings(model: str = EMBEDDING_MODEL):
    """Use a locally running Ollama embedding model."""
    return OllamaEmbeddings(model=model)


def get_chroma_client() -> chromadb.PersistentClient:
    """Return low-level PersistentClient for direct administrative operations."""
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_vector_store(embedding_function=None) -> Chroma:
    """Open or create the persistent ChromaDB collection."""
    if embedding_function is None:
        embedding_function = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_PATH),
    )


def index_chunks(chunks: List[Document], doc_name: Optional[str] = None) -> Chroma:
    """
    Save chunks using deterministic IDs to avoid duplicate insertions.
    Supports incremental indexing across multiple documents.
    """
    if not chunks:
        raise ValueError("No text chunks were provided.")

    vector_store = get_vector_store()
    ids = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", doc_name or "unknown.pdf")
        page = chunk.metadata.get("page", 1)
        start_index = chunk.metadata.get("start_index", i)
        chunk_index = chunk.metadata.get("chunk_index", i)

        # Create unique deterministic ID
        unique_text = f"{source}|p{page}|c{chunk_index}|s{start_index}|{chunk.page_content[:64]}"
        chunk_id = hashlib.sha256(unique_text.encode("utf-8")).hexdigest()
        ids.append(chunk_id)

    vector_store.add_documents(documents=chunks, ids=ids)
    return vector_store


def delete_document_vectors(source_name: str) -> int:
    """
    Delete all vector embeddings and chunks associated with a specific document.
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        # Check matching count before deletion
        matches = collection.get(where={"source": source_name})
        count = len(matches.get("ids", []))
        if count > 0:
            collection.delete(where={"source": source_name})
        return count
    except Exception as exc:
        print(f"Error deleting vectors for {source_name}: {exc}")
        return 0


def clear_vector_store() -> bool:
    """Wipe all embeddings and chunks from the Chroma collection."""
    try:
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        # Re-create empty collection
        client.create_collection(COLLECTION_NAME)
        return True
    except Exception as exc:
        print(f"Error clearing vector store: {exc}")
        return False


def get_indexed_sources() -> List[str]:
    """Return a list of all distinct document sources currently stored in ChromaDB."""
    try:
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        data = collection.get()
        metadatas = data.get("metadatas", [])
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))
    except Exception:
        return []


def get_collection_stats() -> Dict[str, Any]:
    """Return vector count and source summary of the Chroma database."""
    try:
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        total_count = collection.count()
        sources = get_indexed_sources()
        return {
            "total_chunks": total_count,
            "document_count": len(sources),
            "documents": sources,
        }
    except Exception:
        return {
            "total_chunks": 0,
            "document_count": 0,
            "documents": [],
        }


def retrieve_chunks(
    question: str,
    k: int = DEFAULT_TOP_K,
    selected_sources: Optional[List[str]] = None,
    return_scores: bool = False,
) -> Union[List[Document], List[Tuple[Document, float]]]:
    """
    Retrieve the most relevant chunks for a question.
    Optionally filter by selected documents (metadata-filtered search).
    If return_scores is True, returns list of (Document, score) tuples.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = get_vector_store()

    # Construct metadata filter if specific documents are selected
    filter_dict = None
    if selected_sources:
        if len(selected_sources) == 1:
            filter_dict = {"source": selected_sources[0]}
        else:
            filter_dict = {"source": {"$in": selected_sources}}

    if return_scores:
        try:
            return vector_store.similarity_search_with_relevance_scores(
                question, k=k, filter=filter_dict
            )
        except Exception:
            # Fallback to similarity_search_with_score
            results = vector_store.similarity_search_with_score(
                question, k=k, filter=filter_dict
            )
            # Normalize L2 distance to 0..1 relevance score approximately
            normalized = []
            for doc, distance in results:
                score = max(0.0, 1.0 - (distance / 2.0))
                normalized.append((doc, score))
            return normalized

    return vector_store.similarity_search(question, k=k, filter=filter_dict)
