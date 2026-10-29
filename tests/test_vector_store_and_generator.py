"""
Tests for Vector Store querying and Content Generator prompts/fallbacks.
"""

import unittest
from pathlib import Path
from langchain_core.documents import Document

from src.vector_store import (
    get_collection_stats,
    get_indexed_sources,
)
from src.content_generator import (
    build_system_prompt,
    generate_offline_fallback,
    GENERATION_TEMPLATES,
)


class TestVectorStoreAndGenerator(unittest.TestCase):

    def test_vector_store_stats(self):
        stats = get_collection_stats()
        self.assertIn("total_chunks", stats)
        self.assertIn("document_count", stats)
        self.assertIn("documents", stats)

    def test_sources_present(self):
        sources = get_indexed_sources()
        self.assertIsInstance(sources, list)
        if stats := get_collection_stats():
            if stats["document_count"] > 0:
                self.assertGreater(len(sources), 0)

    def test_build_system_prompt_all_templates(self):
        for key in GENERATION_TEMPLATES.keys():
            prompt = build_system_prompt(
                asset_type=key,
                title="Test Title",
                tone="Professional",
                audience="General Audience",
                specific_focus="Focus area",
            )
            self.assertIn("Test Title", prompt)
            self.assertIn("STRICT GROUNDING RULES", prompt)

    def test_generate_offline_fallback_templates(self):
        docs = [
            Document(page_content="ATmega328P 8-bit AVR RISC microcontroller.", metadata={"source": "arduino.pdf", "page": 1}),
            Document(page_content="Operating voltage 5V, recommended input voltage 7-12V.", metadata={"source": "arduino.pdf", "page": 2}),
        ]

        for asset_key in ["tri_fold_brochure", "extracted_table", "flowchart", "chart_data", "executive_summary"]:
            fallback = generate_offline_fallback(
                asset_type=asset_key,
                title="Asset Test",
                documents=docs,
                selected_sources=["arduino.pdf"],
            )
            self.assertIn("Asset Test", fallback)
            self.assertGreater(len(fallback), 50)


if __name__ == "__main__":
    unittest.main()

