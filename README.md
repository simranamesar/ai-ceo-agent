# AI CEO — Strategic Intelligence Agent

An **AI Strategy Consultant** for a publicly visible company (default: **NVIDIA, NVDA**).
It continuously **collects live public information**, builds a **vector knowledge
repository**, runs a **strategic intelligence engine** (opportunities / risks / trends),
and produces **evidence-based executive recommendations** plus a **CEO briefing** —
answering the question:

> *"If you were the CEO today, what would you do next and why?"*

It also exposes an interactive **Strategic Advisor** box: type any strategic question and get an
answer grounded in retrieved evidence with cited sources. The system uses **open-source /
freely accessible models only — no paid API.**

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES (>= 3, keyless)                   │
│                                                                        │
│  Google News RSS        NVIDIA Newsroom + Blog RSS     Hacker News API │
│  (NVIDIA, earnings,     (company's own voice)          (NVIDIA, GPU,   │
│   AI chip, data centre)                                 CUDA)          │
└───────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PROCESSING PIPELINE                             │
│  1. Clean      strip HTML, collapse whitespace, drop empties           │
│  2. Dedupe     exact (id hash) + semantic near-dup (cosine >= 0.92)    │
│  3. Chunk      800-char windows, 120 overlap, parent-doc back-ref      │
│  4. Embed      all-MiniLM-L6-v2 -> L2-normalised 384-dim vectors       │
└───────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       KNOWLEDGE REPOSITORY                             │
│  ChromaDB (persistent, hnsw:space = cosine)                            │
│  each chunk: text, embedding, doc_id, source, title, url, published    │
└───────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE + AI CEO AGENT (RAG)                   │
│  query -> embed -> retrieve top-8 chunks -> grounded LLM prompt        │
│  Intelligence engine : opportunities / risks / trends (+ evidence)     │
│  CEO agent           : prioritised recommendations (+ evidence)        │
│  Briefing            : what happened / why it matters / what to do     │
│  Sentiment (RoBERTa) : distribution donut · news/public · by-source    │
│  LLM: open model (Qwen3-8B) via in-process transformers OR an          │
│       OpenAI-compatible endpoint (Ollama / vLLM)                       │
└───────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       STREAMLIT DASHBOARD (9 sections)                 │
│  Strategic Advisor · Overview · Market Intelligence · Opportunities ·  │
│  Risk Monitor · Trends · Sentiment · Recommendations · CEO Briefing    │
└──────────────────────────────────────────────────────────────────────┘
```

A Mermaid version of this diagram lives in `docs/architecture.md`.

---

## Data Flow

```
News RSS ───┐
Company RSS ┼──► collect_all() ──► data/raw/*.json
Hacker News ┘            │
                          ▼
                   clean_documents()        strip HTML, normalise whitespace
                          │
                          ▼
                   dedupe_exact()           drop repeated id (sha1 of url|title)
                   dedupe_near()            drop cosine >= 0.92 (embedding) duplicates
                          │  ──► data/processed/clean.json
                          ▼
                   chunk_documents()        800/120 char windows  ──► chunks.json
                          │
                          ▼
                   Embedder.encode()        all-MiniLM-L6-v2, 384-dim, L2-normalised
                          │
                          ▼
                   VectorStore.add()        ChromaDB persistent, cosine index
                          │
              ┌───────────┼───────────────────────────────┐
              ▼           ▼                                 ▼
   IntelligenceEngine   SentimentAnalyzer            RagChain.retrieve()
   (opp / risk / trend) (RoBERTa, news vs public)    (top-k for Strategic Advisor)
              │           │                                 │
              ▼           ▼                                 ▼
   CEOAgent.recommend()  sentiment.json            live grounded answer + sources
   generate_briefing()
              │
              ▼
   data/outputs/{intelligence, recommendations, briefing}.json
              │
              ▼
   Streamlit dashboard (reads the JSON; Strategic Advisor queries the store live)
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
| Generative LLM | **Qwen3-8B** (open) via in-process `transformers` **or** an OpenAI-compatible endpoint (Ollama / vLLM) | PDF rule: open / freely accessible only — **no paid API** |
| RAG framework | custom (no LangChain / LangGraph) | full control + transparency; matches the taught "Basic RAG Pipeline" |
| Dashboard | **Streamlit** + **Plotly** | rapid, interactive, native Python |

---

## AI Pipeline

The system is a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Collect** — `collect_all()` pulls live documents from the enabled sources into one
   canonical `Document` schema (`id, source, title, text, url, published`).
2. **Process** — clean text, remove exact and **semantic near-duplicate** documents,
   then chunk into overlapping 800-char windows.
3. **Index** — embed each chunk with all-MiniLM-L6-v2 (384-dim, L2-normalised) and store
   it in a persistent ChromaDB collection with a cosine index.
4. **Retrieve** — for each analysis question, embed the query and pull the **top-8** nearest
   chunks; format them as a numbered, citable context block.
5. **Analyse** — the **intelligence engine** asks the LLM for opportunities / risks / trends
   grounded only in that context; each item is mapped back to the real chunks it cited and
   given a confidence score derived from retrieval similarity.
6. **Recommend & brief** — the **CEO agent** turns the findings into prioritised
   recommendations (recommendation · priority · evidence · expected impact · risk level),
   and a **briefing** answers what happened / why it matters / what to do next.
7. **Sentiment** — a RoBERTa 3-class encoder labels each document; results are aggregated
   into **news-vs-public** sentiment plus a time trend.
8. **Strategic Advisor** — the dashboard runs the same retrieve-then-generate loop live on any
   free-form question, returning an evidence-grounded answer.

For structured outputs the prompts demand strict JSON; a tolerant parser plus an
object-salvage fallback keep the pipeline robust to imperfect model output.

---

## Design Decisions

- **Contextual embeddings over TF-IDF / Word2Vec.** Dense BERT-family vectors match meaning,
  not surface words, fixing Word2Vec's one-static-vector-per-word limitation. (Embeddings
  evolution; Module 9.)
- **Cosine, not Euclidean.** Module 3 marks Euclidean as the wrong choice for text; vectors
  are L2-normalised so cosine equals the dot product, and Chroma uses a cosine index.
- **Minimal cleaning — keep stop-words.** The downstream model is contextual, so the Module 2
  "when NOT to remove stop-words" rule applies (negations carry meaning: "not bad").
- **Semantic near-dedup.** Embedding cosine catches paraphrased syndication that
  Jaccard / Levenshtein miss.
- **Encoder for understanding, decoder for generation.** BERT-family encoders embed and
  classify; a GPT-family decoder writes the briefing and recommendations (Module 9 split).
- **RAG grounding, not fine-tuning.** No labelled "CEO recommendation" dataset exists; RAG
  injects fresh evidence and cuts hallucination.
- **Custom RAG over LangChain / LangGraph.** Transparent, fully controllable, and a direct
  implementation of the taught "Basic RAG Pipeline" — easy to explain and extend live.
- **Keyless sources + the company's own voice.** Three independent sources that need no API
  keys, including NVIDIA's newsroom/blog as an "official voice" signal.
- **Open model only.** Selected via env vars at runtime (`LLM_PROVIDER` / `LLM_MODEL`); never
  a paid API, per the brief.
- **Evidence-first.** Every finding and recommendation carries cited, traceable evidence that
  points only at documents that were actually retrieved.

---

## Dashboard (Streamlit, 9 sections)

| Section | Exam mapping |
|---|---|
| Strategic Advisor | interactive Q&A — live RAG answer + cited evidence |
| Overview | Section 1 — company, industry, #documents, #sources, last update |
| Market Intelligence | Section 2 — recent news, company announcements, **emerging technologies & trends**, competitor activity |
| Opportunity Monitor | Section 3 — title, impact, evidence, confidence |
| Risk Monitor | Section 4 — title, category, severity, evidence, confidence |
| Trends | Deliverable 2 — emerging tech / industry developments, direction, evidence, confidence |
| Sentiment | Section 5 — distribution donut, news vs public, by-source, trend (Plotly) |
| Recommendations | Section 6 — recommendation, priority, evidence, impact, risk level |
| CEO Briefing | Section 7 — what happened / why it matters / what to do next |

---

## Project Structure

```
ai-ceo-agent/
├── config/config.yaml          # company, sources, processing, store, llm, sentiment
├── pipeline/run_pipeline.py     # end-to-end orchestrator (stages 1-7)
├── src/
│   ├── schema.py                # canonical Document + stable id
│   ├── config.py                # load config + resolve env (provider/model)
│   ├── collectors/              # news / company / hackernews + registry
│   ├── processing/              # cleaner, deduplicator (exact + near), chunker
│   ├── store/                   # embeddings (MiniLM), vector_store (ChromaDB)
│   ├── agent/                   # rag_chain, prompts, ceo_agent, briefing, qa (Strategic Advisor)
│   ├── intelligence/            # engine (opp/risk/trend), evidence, sentiment
│   ├── llm/client.py            # open-model client: local transformers OR OpenAI-compatible
│   └── report.py                # terminal summary
├── dashboard/app.py             # Streamlit executive dashboard (9 sections)
└── docs/                        # architecture.md, JUSTIFICATION.md, THEMES.md
```

---

## How to Run

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Choose the open model backend (examples)
#    a) in-process on a GPU server:
export LLM_PROVIDER=local
export LLM_MODEL=Qwen/Qwen3-8B
#    b) or an OpenAI-compatible endpoint (Ollama):
# export LLM_PROVIDER=ollama LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:8b

# 3. Build the knowledge base + analysis
python -m pipeline.run_pipeline

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

On a shared TensorFlow image, prepend `export USE_TF=0` so `transformers` stays on the
PyTorch path.

---

## Mapping to the Exam Tasks

| Exam task | Where it lives |
|---|---|
| Task 1 — Live data collection (>= 3 sources) | `src/collectors/*`, `config.yaml` |
| Task 2 — Knowledge repository | `src/store/vector_store.py` (ChromaDB) |
| Task 3 — Processing (clean / dedupe / embed / index) | `src/processing/*`, `src/store/embeddings.py` |
| Task 4 — Strategic intelligence (opp / risk / trend) | `src/intelligence/engine.py` |
| Task 5 — AI CEO agent (reason, prioritise, recommend) | `src/agent/ceo_agent.py`, `src/agent/qa.py` |
| Task 6 — Evidence-based recommendations | `src/intelligence/evidence.py`, label-linked evidence |
| Deliverable 2 — Dashboard (7 sections) | `dashboard/app.py` |
| Deliverable 3 — Architecture documentation | this README + `docs/` |
