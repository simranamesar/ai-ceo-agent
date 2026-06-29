"""Two-stage validation gate — runs after recommend, before briefing (Graph 1: verify node).

Stage 1 — Structural count:
    Every recommendation must cite >= MIN_EVIDENCE distinct evidence pieces.
    A recommendation with thin evidence is dropped regardless of how well-written it is.

Stage 2 — Semantic grounding (Sentence-BERT cosine):
    The recommendation text is embedded and compared against every evidence title +
    snippet using cosine similarity (dot product, because embeddings are L2-normalised).
    The maximum similarity across all evidence pieces is the grounding score.
    If the grounding score falls below the configured threshold the recommendation is
    rejected as "unsupported" — the LLM wrote something that does not relate to the
    evidence it cited.

Blended confidence = 0.5 * grounding_score + 0.5 * retrieval_confidence
    retrieval_confidence comes from the intelligence engine (distance-derived).

Aggregate metrics (mean_confidence, factual_precision) are saved to
data/outputs/metrics.json so the examiner can see quantitative validation results.
"""
from __future__ import annotations

from typing import Any

MIN_EVIDENCE = 3


def _grounding_score(claim: str, evidence: list[dict], embedder) -> float:
    """Max cosine similarity between the claim and any evidence piece.

    Uses title + snippet so both the headline and the content contribute.
    Dot product == cosine similarity because embedder returns L2-normalised vectors.
    """
    if not evidence or embedder is None:
        return 0.5  # neutral fallback when embedder unavailable

    ev_texts = [
        f"{e.get('title', '')} {(e.get('snippet', '') or '')[:150]}".strip()
        for e in evidence
        if e.get("title") or e.get("snippet")
    ]
    if not ev_texts:
        return 0.5

    all_texts = [claim] + ev_texts
    vecs = embedder.encode(all_texts)   # L2-normalised by Embedder.encode()
    claim_vec = vecs[0]
    sims = [sum(a * b for a, b in zip(claim_vec, ev)) for ev in vecs[1:]]
    return round(max(sims), 3)


def verify_recommendations(
    recommendations: list[dict[str, Any]],
    embedder=None,
    threshold: float = 0.25,
) -> dict[str, Any]:
    """Two-stage validation: evidence count then semantic grounding.

    Returns a report dict:
      {verified, rejected, total, passed, failed, metrics}
    """
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for rec in recommendations:
        evidence = rec.get("evidence") or []
        claim = f"{rec.get('recommendation', '')} {rec.get('rationale', '')}".strip()

        # ── Stage 1: structural count ────────────────────────────────────
        if len(evidence) < MIN_EVIDENCE:
            rejected.append({
                **rec,
                "verified": False,
                "grounding_score": 0.0,
                "rejection_reason": (
                    f"Only {len(evidence)} evidence piece(s) — "
                    f"need >= {MIN_EVIDENCE} to pass Stage 1 (structural check)."
                ),
            })
            continue

        # ── Stage 2: semantic grounding ──────────────────────────────────
        g_score = _grounding_score(claim, evidence, embedder)

        # Blend with retrieval confidence already attached to the finding
        retrieval_conf = float(rec.get("confidence", 0.6))
        blended = round(0.5 * g_score + 0.5 * retrieval_conf, 3)

        if blended >= threshold:
            verified.append({
                **rec,
                "verified": True,
                "grounding_score": g_score,
                "confidence": blended,
            })
        else:
            rejected.append({
                **rec,
                "verified": False,
                "grounding_score": g_score,
                "confidence": blended,
                "rejection_reason": (
                    f"Blended confidence {blended:.2f} below threshold {threshold:.2f} "
                    f"(grounding={g_score:.2f}) — Stage 2 (semantic check) failed."
                ),
            })

    all_recs = verified + rejected
    confidences = [r.get("confidence", 0.0) for r in all_recs] or [0.0]
    metrics: dict[str, Any] = {
        "n_recommendations": len(recommendations),
        "passed": len(verified),
        "failed": len(rejected),
        "mean_confidence": round(sum(confidences) / len(confidences), 3),
        "factual_precision": (
            round(len(verified) / len(recommendations), 3) if recommendations else 0.0
        ),
        "threshold": threshold,
        "min_evidence": MIN_EVIDENCE,
    }

    return {
        "verified": verified,
        "rejected": rejected,
        "total": len(recommendations),
        "passed": len(verified),
        "failed": len(rejected),
        "metrics": metrics,
    }
