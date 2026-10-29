"""
Tests for DocumentManager: registration, duplicate detection, metadata, and deletion.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.document_manager import DocumentManager


class TestDocumentManager(unittest.TestCase):

    def test_registration_and_retrieval(self):
        with TemporaryDirectory() as tmp_dir:
            reg_path = Path(tmp_dir) / "registry.json"
            mgr = DocumentManager(registry_path=reg_path)

            mgr.register_document(
                filename="test_doc.pdf",
                file_hash="abcdef1234567890",
                file_size_bytes=1024,
                page_count=5,
                chunk_count=12,
                title="Test Document",
            )

            docs = mgr.get_all_documents()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["filename"], "test_doc.pdf")
            self.assertEqual(docs[0]["page_count"], 5)
            self.assertEqual(docs[0]["chunk_count"], 12)

    def test_duplicate_detection(self):
        with TemporaryDirectory() as tmp_dir:
            reg_path = Path(tmp_dir) / "registry.json"
            mgr = DocumentManager(registry_path=reg_path)

            test_hash = "unique_hash_987654321"
            mgr.register_document(
                filename="first.pdf",
                file_hash=test_hash,
                file_size_bytes=2048,
                page_count=2,
                chunk_count=4,
            )

            # Check duplicate on same hash
            is_dup, dup_name = mgr.is_duplicate(test_hash)
            self.assertTrue(is_dup)
            self.assertEqual(dup_name, "first.pdf")

            # Check non-duplicate
            is_dup_new, _ = mgr.is_duplicate("different_hash_000")
            self.assertFalse(is_dup_new)

    def test_document_deletion(self):
        with TemporaryDirectory() as tmp_dir:
            reg_path = Path(tmp_dir) / "registry.json"
            mgr = DocumentManager(registry_path=reg_path)

            mgr.register_document(
                filename="to_delete.pdf",
                file_hash="hash_delete",
                file_size_bytes=500,
                page_count=1,
                chunk_count=2,
            )

            vector_deleted = False

            def dummy_delete_vectors(filename):
                nonlocal vector_deleted
                vector_deleted = True

            deleted = mgr.delete_document("to_delete.pdf", delete_vectors_fn=dummy_delete_vectors)
            self.assertTrue(deleted)
            self.assertTrue(vector_deleted)
            self.assertEqual(len(mgr.get_all_documents()), 0)


if __name__ == "__main__":
    unittest.main()

