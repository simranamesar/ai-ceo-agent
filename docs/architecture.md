# Architecture

## System architecture

```mermaid
flowchart LR
  subgraph Collect [Task 1: Live Collection]
    N[News RSS]:::src --> R1
    C[Company / IR]:::src --> R1
    R1[(data/raw/*.json)]
  end

  subgraph Process [Task 3: Processing]
    R1 --> CL[Clean] --> DD[Dedupe] --> CH[Chunk]
  end

  subgraph Store [Task 2: Knowledge Repository]
    CH --> EM[Embed all-MiniLM-L6-v2] --> VDB[(ChromaDB + index)]
  end

  subgraph Intel [Task 4-6: Intelligence]
    VDB --> RAG[RAG retrieval]
    RAG --> ENG[Opportunities / Risks / Trends + Evidence]
    VDB --> SENT[Sentiment]
    ENG --> REC[CEO Agent -> Recommendations]
    REC --> BRF[CEO Briefing]
  end

  ENG --> OUT[(data/outputs/*.json)]
  SENT --> OUT
  REC --> OUT
  BRF --> OUT
  OUT --> DASH[Streamlit Dashboard]

  classDef src fill:#eef,stroke:#446;
```

## Data flow
1. Collectors emit canonical `Document`s -> `data/raw/documents.json`
2. Clean + dedupe -> `data/processed/clean.json`
3. Chunk + embed + index -> ChromaDB (`data/chroma/`)
4. Intelligence engine + sentiment -> `data/outputs/intelligence.json`
5. CEO agent -> `data/outputs/recommendations.json`
6. Briefing -> `data/outputs/briefing.json`
7. Dashboard reads the outputs.

## Technology stack
- Python 3.11
- Collection: requests, feedparser, beautifulsoup4
- Store/retrieval: ChromaDB + sentence-transformers (all-MiniLM-L6-v2), RAG
- LLM: open model (Qwen3-8B) in-process via transformers, or an OpenAI-compatible endpoint (Ollama / vLLM)
- Dashboard: Streamlit + Plotly — 9 views: Strategic Advisor (live Q&A), Overview, Market
  Intelligence (incl. emerging tech & trends), Opportunities, Risks, Trends, Sentiment
  (distribution donut / news-vs-public / by-source / trend), Recommendations, CEO Briefing

## Design decisions
- Single OpenAI-compatible `LLMClient` -> local Ollama and the Frankfurt
  server share one code path; the "no paid API" rule is enforced in one file.
- JSON artifact at every stage -> inspectable, resumable, demo-friendly.
- One collector class per source behind a `BaseCollector` -> add/remove
  sources without touching the pipeline.
- Stable doc ids (hash of url|title) -> dedup is trivial and idempotent.

## AI pipeline
RAG (semantic retrieval over chunked docs) feeds structured prompts
(`src/agent/prompts.py`); the LLM returns JSON findings; the CEO agent reasons
over those findings to produce prioritised, evidence-cited recommendations and
the executive briefing.
