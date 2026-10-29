"""
Advanced retrieval engine with adaptive hybrid search, lexical scoring, and reranking.
Handles both broad summary/thematic queries and granular factual queries accurately.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

from src.config import SIMILARITY_THRESHOLD, DEFAULT_TOP_K
from src.vector_store import retrieve_chunks

META_SUMMARY_WORDS = {
    "summarize", "summary", "overview", "findings", "results",
    "takeaways", "conclusions", "digest", "briefing", "outline"
}

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "and", "in", "of", "to", "for",
    "how", "why", "who", "which", "on", "it", "this", "that", "these", "those",
    "by", "with", "from", "at", "as", "be", "was", "were", "or"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def is_thematic_query(query: str) -> bool:
    """Detect if query is asking for a general summary, overview, or explanation."""
    tokens = set(tokenize(query))
    return bool(tokens.intersection(META_SUMMARY_WORDS)) or len(tokens) <= 3


def compute_bm25_lexical_score(query_tokens: List[str], doc_tokens: List[str], avg_doc_len: float) -> float:
    """
    Lightweight, CPU-friendly BM25-inspired lexical matching score.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    k1 = 1.5
    b = 0.75
    doc_len = len(doc_tokens)
    score = 0.0

    doc_token_counts: Dict[str, int] = {}
    for t in doc_tokens:
        doc_token_counts[t] = doc_token_counts.get(t, 0) + 1

    for q in query_tokens:
        if q in doc_token_counts:
            tf = doc_token_counts[q]
            score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(1.0, avg_doc_len))))

    return min(1.0, score / max(1.0, len(query_tokens) * 2.0))


def hybrid_search_and_rerank(
    question: str,
    k: int = DEFAULT_TOP_K,
    selected_sources: Optional[List[str]] = None,
    threshold: float = 0.10,
    semantic_weight: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    """
    Perform adaptive hybrid retrieval:
    - If thematic/summary query: relies primarily on dense semantic similarity (weight 0.90+)
      and filters out meta-words like 'summarize' from lexical scoring so introductory chunks aren't dropped.
    - If specific factual query: combines dense similarity with BM25 lexical precision.
    - Preserves candidates so users always get comprehensive answers from their selected documents.
    """
    thematic = is_thematic_query(question)

    # For thematic queries or small selected documents, retrieve more chunks for a complete picture
    fetch_k = max(k * 2, 8)
    candidates = retrieve_chunks(
        question=question,
        k=fetch_k,
        selected_sources=selected_sources,
        return_scores=True,
    )

    if not candidates:
        return []

    # If all candidates come from a single small document (e.g. <= 8 chunks total),
    # keep them all sorted by relevance so the LLM gets full context!
    if len(candidates) <= k:
        return [(doc, round(float(score), 4)) for doc, score in candidates]

    query_tokens = tokenize(question)
    if thematic:
        # Filter out meta words like "summarize", "findings" from lexical scoring
        filtered_query_tokens = [t for t in query_tokens if t not in STOPWORDS and t not in META_SUMMARY_WORDS]
        if semantic_weight is None:
            semantic_weight = 0.92
    else:
        filtered_query_tokens = [t for t in query_tokens if t not in STOPWORDS]
        if semantic_weight is None:
            semantic_weight = 0.70

    if not filtered_query_tokens:
        filtered_query_tokens = query_tokens

    doc_lengths = [len(tokenize(doc.page_content)) for doc, _ in candidates]
    avg_len = sum(doc_lengths) / max(1, len(doc_lengths))

    scored_items: List[Tuple[Document, float]] = []

    for doc, semantic_score in candidates:
        doc_tokens = tokenize(doc.page_content)
        lexical_score = compute_bm25_lexical_score(filtered_query_tokens, doc_tokens, avg_len)

        # Fused hybrid score
        combined_score = (semantic_weight * float(semantic_score)) + ((1.0 - semantic_weight) * lexical_score)

        if combined_score >= threshold or len(scored_items) < k:
            scored_items.append((doc, round(combined_score, 4)))

    # Sort descending by fused score
    scored_items.sort(key=lambda item: item[1], reverse=True)

    # Always ensure at least min(len(candidates), k) items are returned
    if not scored_items and candidates:
        scored_items = [(doc, round(float(score), 4)) for doc, score in candidates]

    return scored_items[:k]


def verify_evidence_presence(query: str, documents: List[Document]) -> float:
    """
    Compute soft evidence presence score.
    Returns 1.0 for thematic/summary queries so broad requests are never rejected.
    """
    if not query.strip() or not documents:
        return 0.0

    if is_thematic_query(query):
        return 1.0

    q_tokens = set(tokenize(query))
    informative_tokens = q_tokens - STOPWORDS
    if not informative_tokens:
        return 1.0

    context_tokens = set()
    for doc in documents:
        context_tokens.update(tokenize(doc.page_content))

    overlap = informative_tokens.intersection(context_tokens)
    return len(overlap) / len(informative_tokens)
