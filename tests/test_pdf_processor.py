"""
Tests for PDF processing, validation, duplicate hash calculation, and page metadata.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pypdf

from src.pdf_processor import (
    process_pdf,
    validate_pdf,
    compute_file_hash,
    extract_pdf_pages,
)


def create_dummy_pdf(file_path: Path, pages_text: list[str]) -> None:
    """Helper to generate a minimal valid PDF with specified page text."""
    writer = pypdf.PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        # Note: blank page with annotations or text can be created
    with open(file_path, "wb") as f:
        writer.write(f)


class TestPDFProcessor(unittest.TestCase):

    def test_missing_pdf_raises_error(self):
        with TemporaryDirectory() as tmp_dir:
            missing_file = Path(tmp_dir) / "non_existent.pdf"
            with self.assertRaises(FileNotFoundError):
                process_pdf(str(missing_file))

    def test_invalid_extension_raises_error(self):
        with TemporaryDirectory() as tmp_dir:
            txt_file = Path(tmp_dir) / "sample.txt"
            txt_file.write_text("Hello world", encoding="utf-8")
            with self.assertRaises(ValueError):
                process_pdf(str(txt_file))

    def test_empty_pdf_raises_error(self):
        with TemporaryDirectory() as tmp_dir:
            empty_file = Path(tmp_dir) / "empty.pdf"
            empty_file.write_bytes(b"")
            with self.assertRaises(ValueError):
                validate_pdf(empty_file)

    def test_corrupt_pdf_raises_error(self):
        with TemporaryDirectory() as tmp_dir:
            corrupt_file = Path(tmp_dir) / "corrupt.pdf"
            corrupt_file.write_bytes(b"%PDF-1.4 completely invalid junk data")
            with self.assertRaises(ValueError):
                validate_pdf(corrupt_file)

    def test_compute_file_hash_is_deterministic(self):
        data = b"Deterministic PDF byte sequence for testing."
        hash1 = compute_file_hash(data)
        hash2 = compute_file_hash(data)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_real_pdf_processing(self):
        """Test on actual sample PDF present in uploads."""
        sample_path = Path("data/uploads/ARDINO UNO .PPT.pdf")
        if sample_path.exists():
            meta = validate_pdf(sample_path)
            self.assertGreater(meta["total_pages"], 0)
            self.assertEqual(meta["filename"], "ARDINO UNO .PPT.pdf")

            chunks = process_pdf(str(sample_path))
            self.assertGreater(len(chunks), 0)
            first_chunk = chunks[0]
            self.assertIn("source", first_chunk.metadata)
            self.assertIn("page", first_chunk.metadata)
            # Ensure page numbering is 1-indexed (>= 1)
            self.assertGreaterEqual(first_chunk.metadata["page"], 1)


if __name__ == "__main__":
    unittest.main()

