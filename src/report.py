"""Readable end-of-run terminal report (titles, recommendations, briefing,
sentiment) so you can sanity-check results without opening the JSON files."""
from __future__ import annotations

from typing import Any


def _rule() -> None:
    print("-" * 64)


def print_report(cfg, intel, recs, brief, sentiment=None) -> None:
    print("\n" + "=" * 64)
    print(f"  STRATEGIC INTELLIGENCE REPORT - {cfg['company']['name']}")
    print("=" * 64)

    for kind, label in (("opportunities", "OPPORTUNITIES"),
                        ("risks", "RISKS"), ("trends", "TRENDS")):
        items = intel.get(kind, []) or []
        print(f"\n{label} ({len(items)})")
        _rule()
        for i, it in enumerate(items, 1):
            tag = it.get("impact") or it.get("severity") or it.get("direction") or ""
            conf = it.get("confidence", "")
            print(f" {i}. {it.get('title','')}   [{tag} | conf {conf}]")

    print(f"\nRECOMMENDATIONS ({len(recs)})")
    _rule()
    for i, r in enumerate(recs, 1):
        print(f" {i}. [{r.get('priority','-')}] {r.get('recommendation','')}")
        if r.get("expected_impact"):
            print(f"      impact: {r['expected_impact']}")

    if brief and any(brief.values()):
        print("\nCEO BRIEFING")
        _rule()
        for k in ("what_happened", "why_it_matters", "what_to_do_next"):
            if brief.get(k):
                print(f" {k.replace('_',' ').title()}: {brief[k]}")

    if sentiment:
        ov = sentiment.get("overall", {})
        print("\nSENTIMENT")
        _rule()
        print(f" overall  +{ov.get('positive',0)} ~{ov.get('neutral',0)} "
              f"-{ov.get('negative',0)}  (net {ov.get('net_polarity',0)})")
        for cat, d in sentiment.get("by_category", {}).items():
            print(f" {cat:8} net {d.get('net_polarity',0)}  (n={d.get('total',0)})")

    print("=" * 64 + "\n")
