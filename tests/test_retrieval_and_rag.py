"""
Tests for hybrid search scoring, evidence verification, query contextualization, and RAG formatting.
"""

import unittest
from langchain_core.documents import Document

from src.retrieval import (
    tokenize,
    compute_bm25_lexical_score,
    verify_evidence_presence,
)
from src.rag_pipeline import (
    format_context,
    get_display_page,
    contextualize_query,
    ABSTENTION_MESSAGE,
)


class TestRetrievalAndRAG(unittest.TestCase):

    def test_tokenize(self):
        text = "Arduino Uno microcontroller with ATmega328P!"
        tokens = tokenize(text)
        self.assertIn("arduino", tokens)
        self.assertIn("atmega328p", tokens)
        self.assertIn("microcontroller", tokens)

    def test_bm25_lexical_score(self):
        q_tokens = ["arduino", "voltage"]
        doc_matching = ["arduino", "uno", "operates", "at", "5v", "operating", "voltage"]
        doc_unrelated = ["threat", "prediction", "in", "cloud", "digital", "twin"]

        score_match = compute_bm25_lexical_score(q_tokens, doc_matching, avg_doc_len=7.0)
        score_unrelated = compute_bm25_lexical_score(q_tokens, doc_unrelated, avg_doc_len=7.0)

        self.assertGreater(score_match, score_unrelated)
        self.assertEqual(score_unrelated, 0.0)

    def test_verify_evidence_presence(self):
        query = "What is the operating voltage?"
        docs_with_evidence = [
            Document(page_content="The operating voltage is 5V.", metadata={"source": "test.pdf", "page": 1})
        ]
        docs_without_evidence = [
            Document(page_content="Completely unrelated topic about cooking recipes.", metadata={"source": "food.pdf", "page": 1})
        ]

        cov_high = verify_evidence_presence(query, docs_with_evidence)
        cov_low = verify_evidence_presence(query, docs_without_evidence)

        self.assertGreater(cov_high, cov_low)

    def test_get_display_page(self):
        # Legacy 0-indexed page 0 should map to page 1
        self.assertEqual(get_display_page({"page": 0}), 1)
        # Already 1-indexed page 1 should remain 1
        self.assertEqual(get_display_page({"page": 1}), 1)
        # Page 5 should remain 5
        self.assertEqual(get_display_page({"page": 5}), 5)

    def test_format_context_citations(self):
        docs = [
            Document(
                page_content="ATmega328P is the main microcontroller chip.",
                metadata={"source": "arduino.pdf", "page": 3},
            )
        ]
        context = format_context(docs)
        self.assertIn("[Source: arduino.pdf, Page: 3]", context)
        self.assertIn("ATmega328P", context)

    def test_contextualize_query_triggers(self):
        # A standalone query without pronouns or history remains unchanged
        standalone = "What is the microcontroller of Arduino Uno?"
        res = contextualize_query(standalone, chat_history=[])
        self.assertEqual(res, standalone)


if __name__ == "__main__":
    unittest.main()

