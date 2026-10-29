"""
Central configuration for Local AI Document Studio.
Optimized for Windows, CPU-only execution, and 16 GB RAM.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = DATA_DIR / "exports"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
REGISTRY_FILE = DATA_DIR / "document_registry.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Settings
COLLECTION_NAME = "local_rag_documents"

# Ollama Models (CPU-friendly 3B model and efficient embedding model)
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# RAG & Chunking Settings
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
SIMILARITY_THRESHOLD = 0.20  # Minimum relevance score to avoid irrelevant noise

# Upload limits (protects 16 GB RAM from memory exhaustion)
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf"}

# Color palettes for visual generators (Posters, Brochures, Charts)
COLOR_PALETTES = {
    "modern_blue": {
        "primary": "#1E3A8A",      # Deep Navy
        "secondary": "#3B82F6",    # Blue Accent
        "accent": "#60A5FA",       # Light Blue
        "background": "#F8FAFC",   # Off-white / light slate
        "card_bg": "#FFFFFF",      # Pure White
        "text_primary": "#0F172A", # Dark Slate
        "text_secondary": "#475569" # Slate Muted
    },
    "emerald": {
        "primary": "#064E3B",
        "secondary": "#059669",
        "accent": "#34D399",
        "background": "#F0FDF4",
        "card_bg": "#FFFFFF",
        "text_primary": "#064E3B",
        "text_secondary": "#374151"
    },
    "sunset": {
        "primary": "#7C2D12",
        "secondary": "#EA580C",
        "accent": "#FB923C",
        "background": "#FFF7ED",
        "card_bg": "#FFFFFF",
        "text_primary": "#431407",
        "text_secondary": "#78350F"
    },
    "slate_dark": {
        "primary": "#0F172A",
        "secondary": "#334155",
        "accent": "#64748B",
        "background": "#F1F5F9",
        "card_bg": "#FFFFFF",
        "text_primary": "#0F172A",
        "text_secondary": "#334155"
    }
}

