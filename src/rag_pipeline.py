"""
Conversational RAG pipeline with follow-up query contextualization,
adaptive hybrid retrieval, source grounding, and evidence synthesis.
"""

import time
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.config import DEFAULT_LLM_MODEL, DEFAULT_TOP_K
from src.retrieval import hybrid_search_and_rerank, verify_evidence_presence

LLM_MODEL = DEFAULT_LLM_MODEL
ABSTENTION_MESSAGE = "I couldn't find enough evidence in the provided documents to answer your question."


def get_display_page(metadata: Dict[str, Any]) -> int:
    """
    Format human-readable 1-indexed page number.
    Handles legacy 0-indexed documents and new 1-indexed metadata consistently.
    """
    raw_page = metadata.get("page", 0)
    try:
        page_num = int(raw_page)
    except (ValueError, TypeError):
        page_num = 0

    return 1 if page_num == 0 else page_num


def format_context(documents: List[Document]) -> str:
    """Format retrieved passages with their source and page numbers."""
    parts = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        page = get_display_page(doc.metadata)

        parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)


def contextualize_query(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm: Optional[ChatOllama] = None,
) -> str:
    """
    Reformulate follow-up questions into standalone search queries using conversation context.
    If no history exists or question is self-contained, returns question unmodified.
    """
    if not chat_history or len(chat_history) < 2:
        return question

    referential_triggers = [
        "it", "its", "they", "them", "their", "this", "that", "these", "those",
        "he", "him", "his", "she", "her", "the former", "the latter",
        "what else", "tell me more", "how about", "and what", "why is that"
    ]
    tokens = re.findall(r"\b\w+\b", question.lower())
    has_trigger = any(t in tokens for t in referential_triggers) or len(tokens) <= 4

    if not has_trigger:
        return question

    recent_history = chat_history[-4:]
    history_lines = []
    for turn in recent_history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")[:200]
        history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines)

    rewrite_prompt = f"""Given the chat history and the follow-up question below, rephrase the follow-up question into a single, standalone search query that includes all necessary entities. Do NOT answer the question. Only output the rewritten question.

Chat History:
{history_str}

Follow-up Question: {question}

Standalone Search Query:"""

    try:
        if llm is None:
            llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        rephrased = llm.invoke(rewrite_prompt).content.strip()
        clean_rephrased = re.sub(r'^(Standalone Search Query:\s*|"|\')', '', rephrased).rstrip('"\'')
        return clean_rephrased if clean_rephrased else question
    except Exception:
        return question


def answer_question(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    selected_sources: Optional[List[str]] = None,
    k: int = 6,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Answer a question using retrieved document evidence with multi-PDF support,
    metadata filtering, adaptive hybrid reranking, follow-up query awareness, and helpful synthesis.
    """
    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    total_start = time.perf_counter()

    # Step 1: Query Contextualization for follow-ups
    rephrased_query = contextualize_query(question, chat_history)

    # Step 2: Adaptive Hybrid Retrieval
    retrieval_start = time.perf_counter()
    scored_documents = hybrid_search_and_rerank(
        question=rephrased_query,
        k=k,
        selected_sources=selected_sources,
    )
    retrieval_elapsed = time.perf_counter() - retrieval_start

    documents = [doc for doc, _ in scored_documents]

    # If completely empty, abstain cleanly
    if not documents:
        total_elapsed = time.perf_counter() - total_start
        return {
            "answer": ABSTENTION_MESSAGE,
            "sources": [],
            "retrieved_chunks": 0,
            "rephrased_query": rephrased_query,
            "has_sufficient_evidence": False,
            "retrieval_time_seconds": round(retrieval_elapsed, 3),
            "generation_time_seconds": 0.0,
            "total_time_seconds": round(total_elapsed, 3),
            "evidence_coverage": 0.0,
        }

    evidence_coverage = verify_evidence_presence(rephrased_query, documents)

    # Step 3: Build helpful, grounded prompt
    context = format_context(documents)

    prompt = f"""You are an expert AI Document Intelligence Analyst.
Your goal is to provide a helpful, highly accurate, and thorough answer based on the supplied DOCUMENT CONTEXT.

GUIDELINES:
1. Grounding: Answer strictly using facts, features, and concepts from the provided CONTEXT. Do not speculate or use outside knowledge.
2. Summaries & Overviews: If the user asks to summarize, give an overview, or explain results/findings, synthesize the context into clear sections, highlighting key points, features, and conclusions.
3. Citations: Cite supporting evidence inline using [Source: filename, Page: number].
4. Clarity: Format with clean Markdown, bold headers, and concise bullet points.
5. If the context does not contain any relevant information to the question at all, state:
   "{ABSTENTION_MESSAGE}"

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

DETAILED ANSWER:
"""

    gen_start = time.perf_counter()
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=temperature,
        )
        response = llm.invoke(prompt)
        raw_answer = response.content.strip()
    except Exception as exc:
        raise ConnectionError(
            f"Ollama execution failed. Ensure Ollama is running (`ollama serve`) and model '{LLM_MODEL}' is installed: {exc}"
        ) from exc

    gen_elapsed = time.perf_counter() - gen_start
    total_elapsed = time.perf_counter() - total_start

    has_evidence = ABSTENTION_MESSAGE.lower() not in raw_answer.lower()

    # Build unique source citations
    sources = []
    seen = set()

    for i, (doc, score) in enumerate(scored_documents):
        source = doc.metadata.get("source", "Unknown source")
        page = get_display_page(doc.metadata)
        key = (source, page)

        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source,
                "page": page,
                "score": round(score, 3),
                "excerpt": doc.page_content[:300].strip(),
            })

    return {
        "answer": raw_answer,
        "sources": sources,
        "retrieved_chunks": len(documents),
        "rephrased_query": rephrased_query,
        "has_sufficient_evidence": has_evidence,
        "retrieval_time_seconds": round(retrieval_elapsed, 3),
        "generation_time_seconds": round(gen_elapsed, 3),
        "total_time_seconds": round(total_elapsed, 3),
        "evidence_coverage": round(evidence_coverage, 2),
    }
