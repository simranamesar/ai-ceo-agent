"""CEO briefing: what happened / why it matters / what to do next."""
from __future__ import annotations

import json
from typing import Any

from src.agent import prompts
from src.llm.client import LLMClient

_EMPTY = {"what_happened": "", "why_it_matters": "", "what_to_do_next": ""}


def generate_briefing(
    cfg: dict[str, Any],
    llm: LLMClient,
    intelligence: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    company = cfg["company"]["name"]

    def titles(key: str, n: int = 5) -> list[str]:
        return [f.get("title", "") for f in (intelligence.get(key) or [])[:n] if isinstance(f, dict)]

    summary = {
        "opportunities": titles("opportunities"),
        "risks": titles("risks"),
        "trends": titles("trends"),
        "recommendations": [r.get("recommendation", "") for r in recommendations[:5]],
    }
    system = prompts.BRIEFING_SYSTEM.format(company=company)
    user = "FINDINGS SUMMARY:\n" + json.dumps(summary, indent=2) + "\n\nReturn ONLY the JSON object."
    try:
        brief = llm.json_chat(system, user)
    except Exception as e:
        print(f"[briefing] generation/parse failed: {e}")
        return dict(_EMPTY)
    if not isinstance(brief, dict):
        return dict(_EMPTY)
    for k in _EMPTY:
        brief.setdefault(k, "")
    return brief
