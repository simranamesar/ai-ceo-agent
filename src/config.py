"""Load YAML config and resolve env-var-backed LLM settings.

Single source of truth for paths, model names, and source settings so the
rest of the pipeline never hard-codes anything.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read config.yaml and inject resolved LLM credentials from the env."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    llm = cfg.get("llm", {})
    llm["provider"] = os.getenv("LLM_PROVIDER", llm.get("provider", "ollama"))
    llm["base_url"] = os.getenv(llm.get("base_url_env", "LLM_BASE_URL"), "http://localhost:11434/v1")
    llm["model"] = os.getenv(llm.get("model_env", "LLM_MODEL"), "qwen3:8b")
    llm["api_key"] = os.getenv(llm.get("api_key_env", "LLM_API_KEY"), "ollama")
    cfg["llm"] = llm
    return cfg
