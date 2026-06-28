# AI CEO — Strategic Intelligence Agent

An **AI Strategy Consultant** for a publicly visible company (default: **NVIDIA, NVDA**).
It continuously **collects live public information**, builds a **vector knowledge
repository**, runs a **strategic intelligence engine** (opportunities / risks / trends),
and produces **evidence-based executive recommendations** plus a **CEO briefing** —
answering the question:

> *"If you were the CEO today, what would you do next and why?"*

The system uses **two explicit LangGraph graphs** as its backbone, plus an interactive
**Strategic Advisor** powered by a ReAct agent loop. All models are **open-source /
freely accessible — no paid API.**

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  DATA SOURCES (>= 3, keyless)                        │
│                                                                      │
│  Google News RSS        NVIDIA Newsroom + Blog RSS     HN API        │
│  (NVIDIA, earnings,     (company's own voice)          (NVIDIA, GPU, │
│   AI chip, data centre)                                 CUDA)        │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│        GRAPH 1 — Pipeline  ·  pipeline/graph.py                      │
│        LangGraph StateGraph · 8 nodes · linear order                 │
│                                                                      │
│  collect → process → index → intelligence → sentiment →              │
│            recommend → verify → brief                                │
│                                                                      │
│  PipelineState flows through every node.                             │
│  Artifacts written to data/outputs/*.json after each LLM stage.     │
│  verify node is the validation gate: drops recs with < 3 evidence.  │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  KNOWLEDGE REPOSITORY                                │
│  ChromaDB (persistent, hnsw:space=cosine)                            │
│  each chunk: text · embedding · doc_id · source · title · url        │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│        GRAPH 2 — ReAct Agent  ·  src/agent/react_agent.py            │
│        LangGraph MessagesState · reason ↔ act loop                   │
│                                                                      │
│  reason: LLM picks a tool (search_kb / fetch_live_news /             │
│           score_sentiment / compute_metrics)                         │
│  act:    ToolNode executes the chosen tool                           │
│  loop:   repeats until LLM produces a plain-text final answer        │
│                                                                      │
│  Triggered on every "Ask" click in the dashboard.                   │
│  MAX_TOOL_CALLS cap prevents infinite loops on small models.         │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD  ·  dashboard/app.py                │
│  Strategic Advisor · Overview · Market Intelligence · Opportunities  │
│  Risk Monitor · Trends · Sentiment · Recommendations · CEO Briefing  │
└──────────────────────────────────────────────────────────────────────┘
```

A Mermaid version of this diagram lives in `docs/architecture.md`.

---

## Data Flow

```
News RSS ───┐
Company RSS ┼──► collect_node() ──► data/raw/documents.json
Hacker News ┘           │
                         │   [GRAPH 1 — pipeline/graph.py]
                         ▼
                  process_node()     clean · dedupe (exact + near-semantic)
                         │  ──► data/processed/clean.json
                         ▼
                  index_node()       chunk 800/120 · embed MiniLM · upsert ChromaDB
                         │  ──► data/processed/chunks.json
                         ▼
           ┌─────────────┴──────────────────────┐
           ▼                                      ▼
  intelligence_node()                      sentiment_node()
  RAG retrieve → LLM extract              RoBERTa 3-class classifier
  opportunities / risks / trends           news vs public · trend
           │                                      │
           ▼                                      ▼
  recommend_node()                     data/outputs/sentiment.json
  CEOAgent → draft recs (LLM)
           │
           ▼
  verify_node()   ← validation gate: evidence >= 3 required
  rejected recs ──► data/outputs/verification.json
  verified recs ──► data/outputs/recommendations.json
           │
           ▼
  brief_node()    ← what happened / why it matters / what to do
  ──► data/outputs/briefing.json
           │
           ▼
  Streamlit dashboard reads *.json artifacts
  "Ask" button fires GRAPH 2 (ReAct loop) live
```

---

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Environment / deps | `python -m venv` + `requirements.txt` | reproducible; no extra tooling on the server |
| Collection | `requests`, `feedparser` | free public RSS / JSON APIs, **no auth keys** |
| HTML cleaning | `BeautifulSoup4` | robust against malformed news HTML |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | contextual, distilled-BERT, 384-dim, fast on CPU |
| Vector store | **ChromaDB** (persistent, `hnsw:space=cosine`) | persistence + metadata + ANN index |
| Retrieval | dense cosine top-k (custom RAG) | Module 3 vector-space model, upgraded to dense vectors |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | fine-tuned encoder, 3-class, Lab Week 4 family |
| Generative LLM | **Qwen3-8B** via vLLM (OpenAI-compatible endpoint) or in-process `transformers` | PDF rule: open / freely accessible only — **no paid API** |
| **Agent framework** | **LangGraph** (Graph 1 + Graph 2) | explicit stateful graphs; plan visible; easy to extend with conditional edges |
| **Tool calling** | `langchain-openai` + `ToolNode` | LLM autonomously selects tools; vLLM parses structured tool calls |
| Dashboard | **Streamlit** + **Plotly** | rapid, interactive, native Python |

---

## Agent Architecture

The system demonstrates all six required agent behaviours:

| Behaviour | Where it lives |
|---|---|
| **Planning before execution** | Graph 1 nodes are the explicit plan; PipelineState makes the current step visible |
| **Autonomous decision-making** | Graph 2 `reason` node — LLM picks which tool to call each turn; conditionally calls `fetch_live_news` when `search_kb` returns thin results |
| **Tool usage beyond the LLM** | 4 tools: `search_kb` (ChromaDB), `fetch_live_news` (live RSS), `score_sentiment` (RoBERTa), `compute_metrics` (Python math) |
| **Retrieval and use of evidence** | RAG chain feeds every LLM call; all findings carry cited chunk IDs + URLs |
| **Analysis of risks, opportunities, trends** | Graph 1 `intelligence_node` — IntelligenceEngine extracts three categories |
| **Validation of recommendations** | Graph 1 `verify_node` — hard gate: recommendations with < 3 evidence pieces are rejected and logged |

---

## AI Pipeline

### Graph 1 — Offline pipeline (`pipeline/graph.py`)

A `StateGraph(PipelineState)` with 8 nodes in fixed linear order:

1. **collect** — `collect_all()` pulls live documents from all enabled sources.
2. **process** — clean text, exact-dedup, semantic near-dedup (cosine ≥ 0.92).
3. **index** — chunk → embed (all-MiniLM-L6-v2) → upsert ChromaDB cosine index.
4. **intelligence** — three RAG + LLM passes: opportunities / risks / trends.
5. **sentiment** — RoBERTa labels each document; aggregated by source and time.
6. **recommend** — CEO agent turns findings into prioritised strategic actions.
7. **verify** — **validation gate**: each recommendation must cite ≥ 3 evidence pieces; failing ones are moved to `verification.json` and excluded from the dashboard.
8. **brief** — LLM produces the executive briefing (what happened / why / what next).

### Graph 2 — ReAct agent (`src/agent/react_agent.py`)

A `StateGraph(MessagesState)` with two nodes and a conditional edge:

- `reason` — LLM (Qwen3-8B via vLLM) is called with bound tool schemas; it emits either a tool call or a final answer.
- `act` — `ToolNode` executes the chosen tool and appends the result as a `tool` message.
- **Conditional edge**: `tool_calls` present → back to `reason`; no `tool_calls` → `END`.
- **Safety cap**: after `MAX_TOOL_CALLS` turns, the agent is forced to a final answer.

The LLM's system prompt instructs it to call `search_kb` first, and to call `fetch_live_news` if results are thin — making the branching decision autonomous, not hard-coded.

---

## Design Decisions

- **Two-graph split.** Graph 1 is the offline, deterministic pipeline; Graph 2 is the live, autonomous ReAct loop. Separating them keeps the expensive collection/indexing stages offline and makes each graph's purpose obvious.
- **LangGraph over LangChain AgentExecutor.** LangGraph's explicit node/edge model makes the control flow visible (the graph IS the architecture diagram) and lets us insert custom logic (the verify gate, the tool-cap) without fighting a hidden loop.
- **Validation as a structural node, not a prompt.** `verify_node` is a Python function that enforces the ≥ 3 evidence rule unconditionally. An LLM instruction to "only recommend things with evidence" is optional; a gate that drops the recommendation is a guarantee.
- **Model-driven tool selection (ReAct) over orchestrator-driven.** Qwen3 on vLLM with `--enable-auto-tool-choice --tool-call-parser hermes` emits structured tool calls; the agent demonstrates autonomous decision-making rather than a pre-scripted sequence.
- **Contextual embeddings over TF-IDF / Word2Vec.** Dense BERT-family vectors match meaning, not surface words.
- **Cosine, not Euclidean.** Vectors are L2-normalised; ChromaDB uses a cosine index. (Module 3.)
- **Semantic near-dedup.** Catches paraphrased syndication that exact-hash dedup misses.
- **Encoder for understanding, decoder for generation.** RoBERTa (encoder) classifies sentiment; Qwen3 (decoder) writes the briefing and recommendations.
- **RAG grounding, not fine-tuning.** No labelled dataset exists; RAG injects fresh evidence and cuts hallucination.
- **Keyless sources.** Three independent sources (Google News RSS, NVIDIA newsroom, Hacker News API) that need no API keys.
- **Open model only.** Selected via env vars at runtime; never a paid API.
- **Evidence-first.** Every finding and recommendation carries cited, traceable chunks (chunk ID, URL, title, snippet).

---

## Dashboard (Streamlit, 9 sections)

| Section | What it shows |
|---|---|
| Strategic Advisor | ReAct agent (Graph 2) — live Q&A with tool-call trace visible |
| Overview | Documents, sources, last updated, executive snapshot |
| Market Intelligence | Recent news, company announcements, trends, competitor mentions |
| Opportunity Monitor | Title, impact, evidence, confidence |
| Risk Monitor | Title, category, severity, evidence, confidence |
| Trends | Direction, evidence, confidence |
| Sentiment | Distribution donut, news vs public, by-source, time trend |
| Recommendations | Verification report + verified recs (priority, evidence, impact, risk) |
| CEO Briefing | What happened / Why it matters / What to do next |

---

## Project Structure

```
ai-ceo-agent/
├── config/config.yaml           # company, sources, processing, store, llm, sentiment
├── pipeline/
│   ├── graph.py                 # ★ GRAPH 1 — LangGraph pipeline (primary entry point)
│   └── run_pipeline.py          # legacy entry point (delegates to graph.py)
├── src/
│   ├── schema.py                # canonical Document + stable id
│   ├── config.py                # load config + resolve env (provider/model)
│   ├── collectors/              # news / company / hackernews + registry
│   ├── processing/              # cleaner, deduplicator (exact + near), chunker
│   ├── store/                   # embeddings (MiniLM), vector_store (ChromaDB)
│   ├── tools/                   # ★ Tool registry — search_kb, fetch_live_news,
│   │   └── registry.py          #     score_sentiment, compute_metrics
│   ├── agent/
│   │   ├── react_agent.py       # ★ GRAPH 2 — LangGraph ReAct loop (reason ↔ act)
│   │   ├── verifier.py          # ★ validation gate (≥ 3 evidence required)
│   │   ├── ceo_agent.py         # recommendation generation
│   │   ├── briefing.py          # executive briefing
│   │   ├── rag_chain.py         # shared RAG retrieval
│   │   ├── prompts.py           # all LLM prompt templates
│   │   └── qa.py                # simple RAG Q&A (used as fallback)
│   ├── intelligence/            # engine (opp/risk/trend), evidence, sentiment
│   ├── llm/client.py            # open-model client: local transformers OR OpenAI-compatible
│   └── report.py                # terminal summary
├── dashboard/app.py             # Streamlit executive dashboard (9 sections)
└── docs/                        # architecture.md, THEMES.md
```

★ = new files added for the LangGraph agent layer.

---

## How to Run

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the LLM endpoint (Graph 2 requires an OpenAI-compatible server)
#    Option A — vLLM on GPU server (recommended):
vllm serve Qwen/Qwen3-8B \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000
#    Option B — in-process transformers (Graph 1 only, no tool calling):
# export LLM_PROVIDER=local LLM_MODEL=Qwen/Qwen3-8B

# 3. Set endpoint env vars — pick ONE block below:

# Option A: Ollama running locally (model name must match what `ollama list` shows)
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen3:8b           # Ollama format — NOT Qwen/Qwen3-8B
export LLM_API_KEY=ollama

# Option B: Groq free hosted API (open model, no GPU needed, sign up at console.groq.com)
# export LLM_PROVIDER=ollama
# export LLM_BASE_URL=https://api.groq.com/openai/v1
# export LLM_API_KEY=gsk_...        # your Groq API key
# export LLM_MODEL=llama-3.3-70b-versatile

# Option C: vLLM on a remote server (replace IP with your server's actual address)
# export LLM_PROVIDER=ollama
# export LLM_BASE_URL=http://192.168.1.100:8000/v1
# export LLM_MODEL=Qwen/Qwen3-8B
# export LLM_API_KEY=EMPTY

# 4. Build the knowledge base + run the full agent pipeline (Graph 1)
python -m pipeline.graph

# 5. Launch the dashboard (Graph 2 fires on each "Ask" click)
streamlit run dashboard/app.py
```

---

## Mapping to Exam / Assignment Requirements

| Requirement | Where it lives |
|---|---|
| Live data collection (≥ 3 sources) | `src/collectors/*`, `config.yaml` |
| Knowledge repository | `src/store/vector_store.py` (ChromaDB) |
| Processing (clean / dedupe / embed / index) | `src/processing/*`, `src/store/embeddings.py` |
| Strategic intelligence (opp / risk / trend) | `src/intelligence/engine.py` |
| AI CEO agent (reason, prioritise, recommend) | `src/agent/ceo_agent.py` |
| Evidence-based recommendations | `src/intelligence/evidence.py`, verify gate |
| **Planning before execution** | Graph 1 `PipelineState` — plan is the node list |
| **Autonomous decision-making** | Graph 2 `reason` node — LLM picks tools |
| **Tool usage beyond the LLM** | `src/tools/registry.py` — 4 tools |
| **Validation of recommendations** | Graph 1 `verify_node`, `src/agent/verifier.py` |
| Dashboard (9 sections) | `dashboard/app.py` |
| Architecture documentation | this README + `docs/architecture.md` |
