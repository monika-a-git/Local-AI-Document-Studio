"""
Ollama client utilities and diagnostics for local execution.
Provides health checks, model verification, and troubleshooting instructions.
"""

import logging
from typing import Dict, Any, List
import requests

from src.config import DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> Dict[str, Any]:
    """
    Check if the local Ollama daemon is reachable and whether required models exist.
    Timeout is set to 2.0s to ensure fast, non-blocking UI startup.
    """
    status: Dict[str, Any] = {
        "connected": False,
        "available_models": [],
        "has_llm": False,
        "has_embedding": False,
        "missing_models": [],
        "error_message": "",
        "instructions": "",
    }

    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if response.status_code == 200:
            status["connected"] = True
            data = response.json()
            models: List[str] = [m.get("name", "") for m in data.get("models", [])]
            status["available_models"] = models

            # Check if required models or base names are present
            llm_base = DEFAULT_LLM_MODEL.split(":")[0]
            embed_base = DEFAULT_EMBEDDING_MODEL.split(":")[0]

            status["has_llm"] = any(
                DEFAULT_LLM_MODEL in m or m.startswith(llm_base) for m in models
            )
            status["has_embedding"] = any(
                DEFAULT_EMBEDDING_MODEL in m or m.startswith(embed_base) for m in models
            )

            if not status["has_llm"]:
                status["missing_models"].append(DEFAULT_LLM_MODEL)
            if not status["has_embedding"]:
                status["missing_models"].append(DEFAULT_EMBEDDING_MODEL)

            if status["missing_models"]:
                pull_cmds = "\n".join(
                    [f"  ollama pull {m}" for m in status["missing_models"]]
                )
                status["instructions"] = (
                    f"Required models are missing. Run in your terminal:\n{pull_cmds}"
                )
        else:
            status["error_message"] = f"Ollama returned HTTP status {response.status_code}"
    except requests.exceptions.RequestException as exc:
        status["error_message"] = (
            "Cannot connect to Ollama. Make sure Ollama is installed and running."
        )
        status["instructions"] = (
            "To start Ollama, run in your terminal:\n"
            "  ollama serve\n\n"
            "Then pull the required models:\n"
            f"  ollama pull {DEFAULT_LLM_MODEL}\n"
            f"  ollama pull {DEFAULT_EMBEDDING_MODEL}"
        )

    return status

