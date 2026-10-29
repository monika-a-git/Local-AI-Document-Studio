"""
Local AI Document Studio - Core Package
"""

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    UPLOAD_DIR,
    CHROMA_DIR,
)
from src.pdf_processor import process_pdf, validate_pdf, compute_file_hash
from src.document_manager import document_manager
from src.vector_store import (
    get_vector_store,
    index_chunks,
    retrieve_chunks,
    delete_document_vectors,
)
from src.retrieval import hybrid_search_and_rerank
from src.rag_pipeline import answer_question, format_context
from src.content_generator import generate_content_asset
from src.visual_builder import generate_poster_image, generate_brochure_image, generate_chart_image
from src.export_manager import (
    export_to_pdf,
    export_to_docx,
    export_to_markdown,
    export_to_txt,
    export_to_csv,
    export_image_to_png,
)
from src.evaluation import evaluate_question, evaluate_indexing_performance

__all__ = [
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "UPLOAD_DIR",
    "CHROMA_DIR",
    "process_pdf",
    "validate_pdf",
    "compute_file_hash",
    "document_manager",
    "get_vector_store",
    "index_chunks",
    "retrieve_chunks",
    "delete_document_vectors",
    "hybrid_search_and_rerank",
    "answer_question",
    "format_context",
    "generate_content_asset",
    "generate_poster_image",
    "generate_brochure_image",
    "generate_chart_image",
    "export_to_pdf",
    "export_to_docx",
    "export_to_markdown",
    "export_to_txt",
    "export_to_csv",
    "export_image_to_png",
    "evaluate_question",
    "evaluate_indexing_performance",
]

