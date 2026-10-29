"""
Document management and metadata registry module.
Manages document lifecycle, duplicate detection, metadata tracking, and safe deletion.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.config import UPLOAD_DIR, REGISTRY_FILE
from src.pdf_processor import validate_pdf, compute_file_hash


class DocumentManager:
    """Manages document registry and metadata persistence."""

    def __init__(self, registry_path: Path = REGISTRY_FILE):
        self.registry_path = registry_path
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Create empty registry file if it does not exist."""
        if not self.registry_path.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from JSON file."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_registry(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Save registry to JSON file safely."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return list of all registered documents."""
        data = self._load_registry()
        return list(data.values())

    def get_document_names(self) -> List[str]:
        """Return list of indexed document filenames."""
        data = self._load_registry()
        return list(data.keys())

    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific document."""
        data = self._load_registry()
        return data.get(filename)

    def is_duplicate(self, file_hash: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file with the given SHA-256 hash is already indexed.
        Returns (True, existing_filename) or (False, None).
        """
        data = self._load_registry()
        for name, meta in data.items():
            if meta.get("file_hash") == file_hash:
                return True, name
        return False, None

    def register_document(
        self,
        filename: str,
        file_hash: str,
        file_size_bytes: int,
        page_count: int,
        chunk_count: int,
        title: str = "",
        author: str = "",
    ) -> Dict[str, Any]:
        """Register or update document metadata in registry."""
        data = self._load_registry()
        entry = {
            "filename": filename,
            "file_hash": file_hash,
            "file_size_bytes": file_size_bytes,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "title": title or filename,
            "author": author or "Unknown",
            "indexed_at": datetime.now().isoformat(),
        }
        data[filename] = entry
        self._save_registry(data)
        return entry

    def delete_document(self, filename: str, delete_vectors_fn=None) -> bool:
        """
        Delete a specific document:
        1. Remove document chunks from vector store
        2. Remove physical PDF from uploads directory
        3. Remove metadata from registry
        """
        # Delete vectors if callback is provided
        if delete_vectors_fn:
            try:
                delete_vectors_fn(filename)
            except Exception as exc:
                print(f"Warning deleting vectors for {filename}: {exc}")

        # Delete physical file
        pdf_path = UPLOAD_DIR / filename
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception as exc:
                print(f"Warning deleting file {pdf_path}: {exc}")

        # Remove from registry
        data = self._load_registry()
        if filename in data:
            del data[filename]
            self._save_registry(data)
            return True
        return False

    def clear_all(self, clear_vectors_fn=None) -> int:
        """
        Clear all documents from the library, vector store, and registry.
        Returns count of removed documents.
        """
        data = self._load_registry()
        count = len(data)

        if clear_vectors_fn:
            try:
                clear_vectors_fn()
            except Exception as exc:
                print(f"Warning clearing vector collection: {exc}")

        # Delete all files in uploads dir
        for pdf_file in UPLOAD_DIR.glob("*.pdf"):
            try:
                pdf_file.unlink()
            except Exception:
                pass

        self._save_registry({})
        return count

    def sync_with_storage(self, known_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Scan UPLOAD_DIR and vector store sources to register any existing files.
        Ensures existing documents in data/uploads are registered automatically.
        """
        data = self._load_registry()
        modified = False

        # Scan UPLOAD_DIR for PDFs
        for pdf_file in UPLOAD_DIR.glob("*.pdf"):
            filename = pdf_file.name
            if filename not in data:
                try:
                    meta = validate_pdf(pdf_file)
                    data[filename] = {
                        "filename": filename,
                        "file_hash": meta["hash"],
                        "file_size_bytes": meta["file_size"],
                        "page_count": meta["total_pages"],
                        "chunk_count": 0,  # Will be updated when queried or re-indexed
                        "title": meta["title"],
                        "author": meta["author"],
                        "indexed_at": datetime.now().isoformat(),
                    }
                    modified = True
                except Exception:
                    continue

        if modified:
            self._save_registry(data)

        return list(data.values())


# Singleton instance
document_manager = DocumentManager()

