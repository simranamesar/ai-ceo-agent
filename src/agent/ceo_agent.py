"""CEO agent: consumes the intelligence object and emits prioritised
recommendations. Each recommendation cites finding labels (O1, R2, T3, ...);
the agent resolves those back to the evidence attached to those findings, so
every recommendation carries its sources, expected impact and risk level.
"""
from __future__ import annotations

from typing import Any

from src.agent import prompts
from src.llm.client import LLMClient


def _index_findings(intelligence: dict[str, Any]) -> list[tuple[str, str, dict]]:
    """Return [(label, kind, finding)] with labels O1.., R1.., T1.."""
    indexed: list[tuple[str, str, dict]] = []
    for kind, prefix in (("opportunities", "O"), ("risks", "R"), ("trends", "T")):
        for i, f in enumerate(intelligence.get(kind, []) or [], 1):
            if isinstance(f, dict):
                indexed.append((f"{prefix}{i}", kind, f))
    return indexed


def _findings_context(indexed: list[tuple[str, str, dict]]) -> str:
    lines = []
    for label, kind, f in indexed:
        detail = f.get("description") or f.get("category") or ""
        lines.append(f"[{label}] ({kind[:-1]}) {f.get('title','')} - {detail}")
    return "\n".join(lines)


class CEOAgent:
    def __init__(self, cfg: dict[str, Any], llm: LLMClient) -> None:
        self.cfg = cfg
        self.llm = llm
        self.company = cfg["company"]["name"]

    def recommend(self, intelligence: dict[str, Any]) -> list[dict[str, Any]]:
        indexed = _index_findings(intelligence)
        if not indexed:
            return []
        label_map = {label: f for label, _, f in indexed}

        system = prompts.CEO_SYSTEM.format(company=self.company)
        user = (
            "FINDINGS (each has a label):\n"
            + _findings_context(indexed)
            + "\n\nIn \"evidence_refs\" cite the finding labels (e.g. \"O1\", \"R2\") "
            "that each recommendation is based on. Return ONLY the JSON array."
        )
        try:
            recs = self.llm.json_chat(system, user)
        except Exception as e:
            print(f"[ceo] recommendation generation/parse failed: {e}")
            return []
        if not isinstance(recs, list):
            return []

        out: list[dict[str, Any]] = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            r["evidence"] = self._resolve_evidence(r.get("evidence_refs", []), label_map, indexed)
            out.append(r)
        return out

    @staticmethod
    def _resolve_evidence(refs, label_map, indexed) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ref in refs or []:
            finding = label_map.get(str(ref).strip().upper())
            if not finding:
                continue
            for ev in finding.get("evidence", []) or []:
                key = ev.get("url") or ev.get("title", "")
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(ev)
        # Fallback so a recommendation is never evidence-free.
        if not evidence and indexed:
            evidence = list(indexed[0][2].get("evidence", []) or [])
        return evidence
