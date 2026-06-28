"""Quick LLM connectivity check before running the full pipeline.

Run:  python -m scripts.check_llm
Verifies that the configured endpoint (Ollama by default) is reachable and the
model name actually exists, with a clear message if not.
"""
from __future__ import annotations

from src.config import load_config
from src.llm.client import LLMClient


def main() -> None:
    cfg = load_config()
    print(f"base_url = {cfg['llm']['base_url']}")
    print(f"model    = {cfg['llm']['model']}")
    print("Sending a 1-token test prompt...\n")
    try:
        reply = LLMClient(cfg).chat("You are a connectivity test.",
                                    "Reply with the single word: OK")
        print("LLM reply:", reply.strip()[:200])
        print("\nConnectivity OK - run:  python -m pipeline.run_pipeline")
    except Exception as e:
        print("LLM call FAILED:\n ", e)
        print("\nChecklist:")
        print("  1. Is Ollama running?         ollama serve")
        print("  2. Is the model pulled?       ollama list   (then: ollama pull qwen3:8b)")
        print("  3. Does the model name match? set LLM_MODEL or edit src/config.py default")


if __name__ == "__main__":
    main()
