"""Offline tests for stage 6: CEO recommendations resolve evidence from cited
findings, and briefing fills its three sections. Stub LLM, no network.
Run:  python -m tests.strategy_test
"""
from __future__ import annotations

from src.agent.briefing import generate_briefing
from src.agent.ceo_agent import CEOAgent

_INTEL = {
    "company": "NVIDIA",
    "opportunities": [
        {"title": "Expand DC GPUs", "description": "demand",
         "evidence": [{"source": "news", "title": "AI demand", "url": "u1", "snippet": "x"}]}
    ],
    "risks": [
        {"title": "China export controls", "category": "regulatory",
         "evidence": [{"source": "news", "title": "Export rule", "url": "u2", "snippet": "y"}]}
    ],
    "trends": [],
}


class _RecLLM:
    def json_chat(self, system, user):
        assert "O1" in user and "R1" in user  # findings were labelled in the prompt
        return [
            {"recommendation": "Scale DC capacity", "priority": "High",
             "rationale": "ride demand", "expected_impact": "revenue",
             "risk_level": "Medium", "evidence_refs": ["O1", "R1"]},
            {"recommendation": "No-ref rec", "priority": "Low", "evidence_refs": []},
        ]


class _BriefLLM:
    def json_chat(self, system, user):
        return {"what_happened": "demand up", "why_it_matters": "revenue",
                "what_to_do_next": "scale capacity"}


def test_recommendations_resolve_evidence() -> None:
    cfg = {"company": {"name": "NVIDIA"}}
    recs = CEOAgent(cfg, _RecLLM()).recommend(_INTEL)
    assert len(recs) == 2
    urls = {e["url"] for e in recs[0]["evidence"]}
    assert urls == {"u1", "u2"}          # both cited findings' evidence merged
    # fallback: the no-ref rec still gets non-empty evidence
    assert len(recs[1]["evidence"]) >= 1


def test_briefing_fills_sections() -> None:
    cfg = {"company": {"name": "NVIDIA"}}
    brief = generate_briefing(cfg, _BriefLLM(), _INTEL, [])
    assert set(brief.keys()) >= {"what_happened", "why_it_matters", "what_to_do_next"}
    assert brief["what_to_do_next"] == "scale capacity"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} stage-6 offline tests passed.")
