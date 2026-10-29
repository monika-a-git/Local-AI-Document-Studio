
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from src.pdf_processor import process_pdf
from src.rag_pipeline import format_context


class TestLocalRAG(unittest.TestCase):

    def test_format_context_includes_source_and_page(self):
        docs = [
            Document(
                page_content="Machine learning learns patterns from data.",
                metadata={"source": "notes.pdf", "page": 0},
            )
        ]

        context = format_context(docs)

        self.assertIn("notes.pdf", context)
        self.assertIn("Page: 1", context)
        self.assertIn("Machine learning", context)

    def test_missing_pdf_raises_error(self):
        with TemporaryDirectory() as folder:
            missing_file = Path(folder) / "missing.pdf"

            with self.assertRaises(FileNotFoundError):
                process_pdf(str(missing_file))


if __name__ == "__main__":
    unittest.main()
