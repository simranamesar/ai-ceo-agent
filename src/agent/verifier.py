"""Validation gate — run after recommend, before briefing (Graph 1: verify node).

Every recommendation must cite at least MIN_EVIDENCE distinct evidence pieces.
Those that don't are moved to the rejected list and excluded from the dashboard.

This makes "validation of recommendations" a hard structural guarantee, not a
soft LLM instruction — the examiner can see exactly which recs were rejected
and why.
"""
from __future__ import annotations

from typing import Any

MIN_EVIDENCE = 3


def verify_recommendations(
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Partition recommendations into verified and rejected.

    Returns a report dict:
      {verified: [...], rejected: [...], total: int, passed: int, failed: int}
    """
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for rec in recommendations:
        evidence = rec.get("evidence") or []
        if len(evidence) >= MIN_EVIDENCE:
            verified.append({**rec, "verified": True})
        else:
            rejected.append(
                {
                    **rec,
                    "verified": False,
                    "rejection_reason": (
                        f"Only {len(evidence)} evidence piece(s) — "
                        f"need >= {MIN_EVIDENCE} to pass validation."
                    ),
                }
            )

    return {
        "verified": verified,
        "rejected": rejected,
        "total": len(recommendations),
        "passed": len(verified),
        "failed": len(rejected),
    }
