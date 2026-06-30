# AI CEO Agent — Complete Study Guide

Everything you need to explain any part of this project in an exam or viva:
concepts, math, file breakdown, connections, and design decisions.

---

## 1. What the System Does (One Paragraph)

The system acts as an AI Strategy Consultant for NVIDIA. It pulls live news
from three independent public sources (no API keys), cleans and deduplicates
the articles, embeds them as vectors and stores them in a persistent database,
then runs a strategic intelligence engine (using an LLM + retrieval) to
surface opportunities, risks, and trends. Those findings pass through a
validation gate before becoming executive recommendations and a CEO briefing.
All of that runs as a LangGraph pipeline (Graph 1). A second LangGraph graph
(Graph 2) powers an interactive "Strategic Advisor" in the dashboard, where
the LLM autonomously decides which tools to call to answer a live question.

---

## 2. Core Concepts

### 2.1 RAG — Retrieval-Augmented Generation

**Problem it solves:** LLMs hallucinate. They also do not know about events
that happened after their training cutoff.

**How it works:**
1. At query time, embed the question into a vector.
2. Find the most similar document chunks in the vector database (retrieval).
3. Paste those chunks into the LLM's prompt as grounding context.
4. Ask the LLM to answer *only* using that context.

**Why it matters here:** Every finding (opportunity / risk / trend) and every
recommendation cites the real chunk IDs and URLs it came from. The LLM
cannot invent facts — the retrieval step forces it to reference real articles.

**Where it lives:** `src/agent/rag_chain.py` → `RagChain.retrieve()` then
`RagChain.build_context()` formats the numbered context block that is pasted
into every LLM prompt.

---

### 2.2 Vector Embeddings

**What they are:** A dense vector is a list of numbers (here 384 numbers) that
encodes the *meaning* of a sentence. Similar sentences end up close together
in the 384-dimensional space. Crucially, this is meaning-based, not
keyword-based — "GPU shortage" and "chip supply constraint" land nearby.

**Model used:** `all-MiniLM-L6-v2` from the `sentence-transformers` library.
It is a distilled BERT model, fast on CPU, and outputs 384-dimensional vectors.

**Where it lives:** `src/store/embeddings.py` → `Embedder.encode(texts)`

Key line:
```python
model.encode(texts, normalize_embeddings=True, ...)
```
`normalize_embeddings=True` is critical — it L2-normalises every vector
(see Section 3).

**Used in three places:**
- Near-dedup (`process_node`) — to detect paraphrased duplicate articles
- Indexing (`index_node`) — to build the ChromaDB vector index
- Retrieval (`intelligence_node`, `react_agent`) — to embed each query

---

### 2.3 Cosine Similarity

**The formula:**

```
           A · B
sim(A,B) = ————————
           |A| × |B|
```

Where `A · B` is the dot product (sum of element-wise products) and `|A|`
is the L2 norm (length) of vector A.

**Range:** −1 (opposite) to +1 (identical). In practice, text embeddings
sit between 0 and 1.

**What it measures:** The angle between two vectors. Two articles that talk
about the same thing will have a small angle → cosine close to 1.

**Why cosine and not Euclidean distance?** Euclidean distance is affected by
vector length. A short article and a long article about the same topic would
have very different Euclidean distances just because of length. Cosine
similarity only cares about the direction (the meaning), not the magnitude.

**ChromaDB config:**
```python
metadata={"hnsw:space": "cosine"}
```
This tells ChromaDB to build an HNSW index in cosine space. The
`query()` method returns `distance` values — in ChromaDB's cosine space,
`distance = 1 − cosine_similarity`. So `distance = 0` means identical,
`distance = 1` means completely unrelated.

---

### 2.4 L2 Normalisation

**What it is:** Dividing a vector by its own length so the resulting vector
has length exactly 1.

```
           v
v_hat  =  ———
          |v|
```

**Why it matters:** After L2 normalisation, `|A| = |B| = 1`, so the cosine
formula collapses to just the dot product:

```
sim(A, B)  =  A · B     (when both are unit vectors)
```

This is a huge win for speed: dot products are faster to compute than
the full cosine formula, and the math is identical.

**This is why the near-dedup code just uses a dot product:**
```python
def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# This equals cosine similarity because vectors are L2-normalised
if any(_dot(v, kv) >= threshold for kv in kept_vecs):
    continue
```
Located in `src/processing/deduplicator.py`.

---

### 2.5 HNSW — Hierarchical Navigable Small World

**What it is:** The approximate nearest-neighbour algorithm that ChromaDB
uses internally for fast retrieval.

**Problem without it:** With 300 chunks, brute-force comparison is fast.
With 10 million chunks, comparing the query vector to every single stored
vector would take seconds. HNSW builds a graph index so you only need to
check a small fraction of stored vectors.

**How it works (simplified):** Think of it like a highway system. HNSW builds
multiple layers — the top layer is a sparse "express highway" with far-apart
nodes, and lower layers are denser "local roads". A query starts at the
top layer, greedily navigates to the nearest node, then drops down to
finer layers. This finds approximate nearest neighbours in O(log n) time
instead of O(n).

**Where it lives:** Entirely inside ChromaDB — we just set `hnsw:space=cosine`
in `src/store/vector_store.py` and ChromaDB handles the rest.

---

### 2.6 Chunking

**Why chunk at all?** Embedding a 2000-word article as a single vector loses
local detail — the vector averages over everything. Shorter chunks give
sharper, more specific embeddings. Also, retrieval returns the *chunk* as
context, so smaller chunks mean more focused context for the LLM.

**How it works here (sliding window):**
```
Window size = 800 chars, Overlap = 120 chars

Article: [========================================]  (1500 chars)
Chunk 1: [============]                              (chars 0–799)
Chunk 2:         [============]                     (chars 680–1479)
```
The overlap of 120 chars ensures a sentence that straddles a boundary
appears in full in at least one chunk.

**Chunk ID format:** `{doc_id}_{index}` — e.g. `a3f21b_0`, `a3f21b_1`.
This ties every chunk back to its parent document.

**Where it lives:** `src/processing/chunker.py` → `chunk_documents()`

---

### 2.7 Deduplication — Exact and Near

**Exact dedup (`dedupe_exact`):**
Each document gets a stable ID computed as:
```python
sha1(url.lower() + "|" + title.lower())[:16]
```
If two documents share the same URL and title, they get the same ID and the
second is dropped. Located in `src/schema.py` (`make_doc_id`) and used in
`src/processing/deduplicator.py`.

**Near dedup (`dedupe_near`):**
Catches the same story published by multiple sites with slightly different
titles (paraphrased syndication). Algorithm:
1. Embed the first 600 chars of each article.
2. Walk through articles one by one.
3. If the new article's vector has cosine similarity ≥ 0.92 with any
   *already-kept* article's vector, drop it.
4. Otherwise keep it and add it to the kept list.

This is a greedy O(n²) scan — acceptable for a few hundred articles.

**Threshold of 0.92:** High enough to catch paraphrase (same story, different
wording) but low enough not to confuse articles that are *about* the same
company but describe different events.

---

### 2.8 Sentiment Analysis — RoBERTa

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Type:** Encoder-only transformer (BERT-family). Fine-tuned on millions of
tweets for 3-class sentiment classification: Positive / Neutral / Negative.

**Encoder vs Decoder:** RoBERTa is an *encoder* — it reads the whole sentence
at once (bidirectional attention), which makes it excellent at classification
tasks. Qwen3 is a *decoder* — it generates one token at a time. You use
encoders for understanding/classification and decoders for generation.

**Output per document:**
```python
{"label": "positive", "score": 0.87}
```
The `score` is the softmax probability for the predicted class.

**Aggregation — net polarity formula:**
```
             positive_count − negative_count
net_polarity = ————————————————————————————————
                        total_count
```
Range: −1 (all negative) to +1 (all positive). Zero means balanced or all
neutral.

Located in `src/intelligence/sentiment.py` → `SentimentAnalyzer`.

---

### 2.9 LangGraph

**What it is:** A Python library for building stateful AI workflows as
explicit graphs (nodes + edges).

**Two key concepts:**

**StateGraph** — you define a `TypedDict` that acts as a shared memory
object. Every node receives the full state dict, modifies what it needs,
and returns only the changed keys. LangGraph merges those changes back.

**MessagesState** — a built-in state type specifically for chat agents.
It holds a list of messages with a built-in reducer (new messages are
appended, not replaced).

**Why LangGraph over a plain for-loop?**
- The graph IS the architecture diagram — you can visualise it.
- You can add conditional edges, retries, and parallel branches without
  refactoring.
- The `verify` node as a structural step is much more visible than burying
  it in a function call.
- MessagesState handles the message accumulation logic for ReAct correctly.

**Our two graphs:**
- Graph 1: `StateGraph(PipelineState)` — linear 8-node pipeline
- Graph 2: `StateGraph(MessagesState)` — ReAct loop

---

### 2.9.1 LangGraph internals — how state actually flows (deep dive)

This section answers the question precisely: **when** does a node have
access to information, **what** does it actually see, and **how** does
execution move from node to node under the hood?

#### The mental model: channels + super-steps (Pregel/BSP)

LangGraph is internally an implementation of the **Pregel** model (the same
"bulk synchronous parallel" graph-processing model Google built for
large-scale graph algorithms — message-passing nodes, executed in lockstep
rounds called *super-steps*). Forget "function calls a function" — think of
it as:

1. Every field of your state schema (`PipelineState`, `MessagesState`) becomes
   a **channel** — a named slot of memory with a type and an **update rule
   (reducer)**.
2. Execution proceeds in **super-steps**. In each super-step, LangGraph
   determines which nodes are "active" (have new input pending), runs *all*
   of them, collects what each one returns, and writes those returns into
   the channels using each channel's reducer. Only then does it compute the
   next active set and start the next super-step.
3. A node never talks to another node directly. It reads channels (the state
   dict), and writes channels (the dict it returns). The graph topology
   (edges) just decides *which* node becomes active next — it carries no data
   itself.

This is why a node signature is always `def node_fn(state: SomeState) -> dict`
— `state` is "read the channels," the returned dict is "write some channels."

#### Channels and reducers — the part people miss

By default, a key in a `TypedDict` state schema gets the **`LastValue`
reducer**: whatever a node returns for that key *overwrites* the previous
value. This is what `PipelineState` uses for every field — there is no
`Annotated[...]` on any of them in [pipeline/graph.py](pipeline/graph.py):

```python
class PipelineState(TypedDict):
    cfg: dict[str, Any]
    docs: list[dict]
    processed: list[dict]
    intelligence: dict[str, Any]
    ...
```

So when `process_node` returns `{"processed": deduped}`, LangGraph does not
merge or append — it replaces whatever `state["processed"]` held before
(nothing, on the first run) with `deduped`. Returning `{}` (as `index_node`
does) changes nothing; that node's only job is the ChromaDB side effect.

`MessagesState` ([src/agent/react_agent.py](../src/agent/react_agent.py))
is different: it is a **pre-built schema** equivalent to
`{"messages": Annotated[list[BaseMessage], add_messages]}`. The
`add_messages` reducer does *not* overwrite the list — it **appends** the
new messages to the existing list (and replaces a message in place if its
`id` already exists, which supports streaming token updates). That single
annotation is the entire reason the ReAct loop accumulates a growing
conversation instead of each turn wiping out the last one. Without it, every
`return {"messages": [response]}` would erase the whole history.

#### Building the graph (compile-time, not run-time)

```python
g = StateGraph(PipelineState)
for name, fn in nodes:
    g.add_node(name, fn)          # registers a channel "writer" function
g.set_entry_point("collect")      # which node activates at super-step 0
for (a, _), (b, _) in zip(nodes, nodes[1:]):
    g.add_edge(a, b)              # static: a finishing always activates b
g.add_edge("brief", END)
pipeline = g.compile()            # validates graph, returns a runnable
```

`g.compile()` checks every edge references a real node, that `END` is
reachable, and turns the builder into a `Pregel` instance (the
`CompiledStateGraph` object) with `.invoke()`, `.stream()`, `.ainvoke()`.
**No node code has run yet** — compiling only builds the topology.

Graph 2 additionally uses a **conditional edge**:

```python
g.add_conditional_edges("reason", should_continue, {"act": "act", END: END})
g.add_edge("act", "reason")
```

`should_continue` is a plain Python function that receives the state
*after* `reason` just ran, and returns a string key. LangGraph looks that
key up in the `{"act": "act", END: END}` mapping to decide what activates
next. This is the mechanism behind "the LLM decides" — the routing function
itself does nothing clever, it just reads `state["messages"][-1].tool_calls`;
the *intelligence* lives entirely in what the LLM chose to put in that last
message during the `reason` node.

#### Execution, step by step

Call `pipeline.invoke(initial_state_dict)`. This is what actually happens:

1. **Super-step 0 — seed.** The dict you pass to `invoke()` is written into
   every channel it touches. In `main()` ([pipeline/graph.py:201-213](../pipeline/graph.py)),
   this seeds `cfg`, `embedder`, `llm`, and empty placeholders for
   everything not yet computed. The entry-point node (`collect`) becomes
   the active set.
2. **Run the active node(s).** `collect_node(state)` is called with the
   *entire* current state dict — even keys it doesn't use (`intelligence`,
   `sentiment`, still empty) are present and readable. It returns
   `{"docs": docs}`.
3. **Apply reducers.** LangGraph takes `{"docs": docs}` and writes it into
   the `docs` channel via `LastValue` (overwrite). The shared state object
   now has real `docs`.
4. **Compute the next active set from edges.** `collect → process` is a
   static edge, so `process` becomes active for super-step 1.
5. **Repeat.** `process_node` receives the state *as it now stands* —
   including the `docs` that `collect_node` just wrote in the previous
   super-step. It reads `state["docs"]`, returns `{"processed": deduped}`.
   And so on through `index → intelligence → sentiment → recommend →
   verify → brief`.
6. **Termination.** When the active node's outgoing edge points at `END`
   (`brief → END`), there is no next active set, and `invoke()` returns the
   final accumulated state dict — every channel as it stood after the last
   write to it.

The critical point, answering "when does the StateGraph have info": **a
node always sees the full state as accumulated by every super-step that
ran before it — not just what the immediately preceding node returned.**
By the time `verify_node` runs, `state` already contains `cfg`, `embedder`,
`llm`, `docs`, `processed`, `intelligence`, `sentiment`, and
`recommendations` — five super-steps' worth of writes, all still present
because nothing overwrote them. `verify_node` only touches
`recommendations` and `verification`; everything else just rides along
unchanged in the dict.

#### Worked example — state growth across Graph 1's super-steps

| Super-step | Node that runs | Reads | Writes (returns) | State after this step |
|---|---|---|---|---|
| 0 (seed) | — | — | `cfg, embedder, llm` (+ empty placeholders) | `cfg, embedder, llm` only |
| 1 | `collect` | `cfg` | `{"docs": [...]}` | + `docs` |
| 2 | `process` | `cfg, embedder, docs` | `{"processed": [...]}` | + `processed` |
| 3 | `index` | `cfg, embedder, processed` | `{}` (side effect only) | unchanged |
| 4 | `intelligence` | `cfg, embedder, llm, processed` | `{"intelligence": {...}}` | + `intelligence` |
| 5 | `sentiment` | `cfg, processed` | `{"sentiment": {...}}` | + `sentiment` |
| 6 | `recommend` | `cfg, llm, intelligence` | `{"recommendations": [...]}` | + `recommendations` |
| 7 | `verify` | `cfg, embedder, recommendations` | `{"recommendations": [...], "verification": {...}}` | `recommendations` overwritten with verified-only list, + `verification` |
| 8 | `brief` | `cfg, llm, intelligence, recommendations` | `{"briefing": {...}}` | + `briefing` → returned by `invoke()` |

Because Graph 1 is a strict chain (one active node per super-step, no
branching, no parallel writers), there is never a conflict over who writes
a channel first — `LastValue` overwrite semantics are sufficient and safe.

#### Worked example — Graph 2's loop and the message list

```
invoke({"messages": [HumanMessage("What's our biggest AI risk?")]})

super-step 0 (seed):     messages = [Human]
super-step 1 (reason):   reads messages=[Human]
                          LLM emits AIMessage(tool_calls=[search_kb])
                          returns {"messages": [AIMessage]}
                          add_messages APPENDS → messages = [Human, AI(tool_call)]
                          should_continue(state) sees tool_calls → routes to "act"
super-step 2 (act):      ToolNode executes search_kb(...), wraps result as ToolMessage
                          returns {"messages": [ToolMessage]}
                          add_messages APPENDS → messages = [Human, AI(tool_call), Tool]
                          static edge act → reason
super-step 3 (reason):   reads the FULL messages list so far (all 3 messages)
                          LLM sees the tool result, decides: enough evidence → emits
                          a plain AIMessage with NO tool_calls
                          returns {"messages": [AIMessage(final)]}
                          add_messages APPENDS → messages = [Human, AI(tc), Tool, AI(final)]
                          should_continue(state) sees no tool_calls → routes to END
invoke() returns:        {"messages": [Human, AI(tc), Tool, AI(final)]}
```

Three things this makes concrete:
- **`reason` is called multiple times** (once per loop iteration) — each
  call is a fresh super-step, but it always reads the *cumulative* message
  list, which is exactly why the LLM "remembers" the tool result it just
  received: that `ToolMessage` is already sitting in `state["messages"]`
  before the next `reason` call starts.
- **The loop is just repeated graph traversal**, not a special LangGraph
  looping construct — `act → reason` is a normal static edge, and
  `reason`'s conditional edge re-enters `act` for as many iterations as the
  LLM keeps requesting tools (capped at `MAX_TOOL_CALLS = 8` inside `reason`
  itself, since LangGraph has no built-in step limit here).
- **`add_messages` is what prevents data loss.** If `MessagesState` used the
  default `LastValue` reducer instead, `super-step 2`'s
  `{"messages": [ToolMessage]}` would *replace* the whole list with a
  single `ToolMessage`, erasing the `HumanMessage` and the tool-call
  `AIMessage` — the agent would lose all prior context every turn.

#### Summary — direct answers

- **How does the StateGraph have info?** Every node's return value (a
  partial dict) is merged into a single shared state object using a
  per-key reducer (`LastValue`/overwrite by default, `add_messages`/append
  when annotated). The graph itself stores nothing else — it's purely the
  channel values plus whichever node is currently active.
- **When does a node have info?** At the moment it is invoked for a given
  super-step, it receives the state exactly as it stood after every prior
  super-step's writes were applied — never partial, never stale within a
  single linear path.
- **How does the flow run?** `compile()` builds a static graph of channels
  + edges once. `invoke()` then drives it through Pregel-style super-steps:
  seed → run active node(s) → reduce their outputs into channels →
  recompute the active set from edges (static or conditional) → repeat
  until the active set is empty (`END` reached).

#### 2.9.2 Diagram — StateGraph + memory, with the exact code file for each step

**Graph 1 — `pipeline/graph.py`: linear chain, overwrite memory**

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. SCHEMA — the shape of memory                                       │
│    class PipelineState(TypedDict): cfg, embedder, llm, docs, ...      │
│    pipeline/graph.py : lines 46-56                                    │
│    No Annotated[] on any field → every channel uses the default       │
│    LastValue (overwrite) reducer.                                     │
└────────────────────────────────┬───────────────────────────────────-─┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. BUILD — compile-time only, nothing runs yet                        │
│    g = StateGraph(PipelineState)            pipeline/graph.py : 168   │
│    g.add_node("collect", collect_node) ...  pipeline/graph.py : 171-181 (x8)│
│    g.set_entry_point("collect")             pipeline/graph.py : 183   │
│    g.add_edge(a, b)   (static, linear)      pipeline/graph.py : 184-186│
│    pipeline = g.compile()                   pipeline/graph.py : 188   │
└────────────────────────────────┬────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. SEED MEMORY — super-step 0                                         │
│    pipeline.invoke({cfg, embedder, llm, docs:[], processed:[], ...})  │
│    pipeline/graph.py : 201-213  (inside main())                       │
└────────────────────────────────┬────────────────────────────────────┘
                                  ▼
   ╔════════════════════════ MEMORY (in-process dict, lives only for ════════════════════════╗
   ║                          the duration of this one .invoke() call)                        ║
   ║  cfg, embedder, llm ─── present from super-step 0, read by every later node              ║
   ║  docs           ← written super-step 1                                                   ║
   ║  processed      ← written super-step 2                                                   ║
   ║  intelligence   ← written super-step 4                                                   ║
   ║  sentiment      ← written super-step 5                                                   ║
   ║  recommendations← written super-step 6, OVERWRITTEN super-step 7 (verified-only list)    ║
   ║  verification   ← written super-step 7                                                   ║
   ║  briefing       ← written super-step 8                                                   ║
   ╚════════════════════════════════════════════════════════════════════════════════════════╝
            │ each super-step: node reads memory → returns partial dict →
            │ LangGraph applies LastValue (overwrite) reducer back into memory
            ▼
  collect → process → index → intelligence → sentiment → recommend → verify → brief → END
  (node functions, in order, all in pipeline/graph.py : 63-159)
   collect_node:63   process_node:72   index_node:87   intelligence_node:99
   sentiment_node:113   recommend_node:123   verify_node:130   brief_node:153
            │
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. FINAL MEMORY returned to caller                                     │
│    final = pipeline.invoke(...)             pipeline/graph.py : 201   │
│    final["recommendations"], final["briefing"] etc. → print_report()  │
│    (src/report.py); each node also persists its own slice to          │
│    data/outputs/*.json as a side effect along the way.                │
└──────────────────────────────────────────────────────────────────────┘
```

**Graph 2 — `src/agent/react_agent.py`: looping chain, append-only memory**

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. SCHEMA — built-in MessagesState                                    │
│    {"messages": Annotated[list[BaseMessage], add_messages]}           │
│    imported from langgraph.graph         react_agent.py : 25          │
│    add_messages reducer = APPEND (and replace-by-id), not overwrite   │
└────────────────────────────────┬───────────────────────────────────-─┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. BUILD                                                                │
│    g = StateGraph(MessagesState)                  react_agent.py : 102 │
│    g.add_node("reason", reason)                   react_agent.py : 103 │
│    g.add_node("act", ToolNode(tools))              react_agent.py : 100,104│
│    g.set_entry_point("reason")                     react_agent.py : 105│
│    g.add_conditional_edges("reason", should_continue,                  │
│         {"act": "act", END: END})                  react_agent.py : 106│
│    g.add_edge("act", "reason")                      react_agent.py : 107│
│    agent = g.compile()                              react_agent.py : 108│
└────────────────────────────────┬────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. SEED MEMORY — super-step 0                                          │
│    agent.invoke({"messages": [HumanMessage(question)]})                │
│    react_agent.py : 127  (inside run_react_agent())                    │
└────────────────────────────────┬────────────────────────────────────┘
                                  ▼
  MEMORY = messages: [Human]
       │
       ▼  super-step 1 — reason()              react_agent.py : 69-91
       │     reads messages, calls llm_with_tools.invoke(messages)
       │     returns {"messages": [AIMessage(tool_calls=...)]}
       │     add_messages APPENDS → messages: [Human, AI(tc)]
       │     should_continue()  react_agent.py : 94-98  sees tool_calls → "act"
       ▼
  super-step 2 — act = ToolNode(tools)         react_agent.py : 100,104
       │     executes the chosen tool          src/tools/registry.py
       │     returns {"messages": [ToolMessage(result)]}
       │     add_messages APPENDS → messages: [Human, AI(tc), Tool]
       │     static edge "act" → "reason"      react_agent.py : 107
       ▼
  super-step 3 — reason() again
       │     reads ALL prior messages (full memory, not just the last one)
       │     LLM has enough evidence now → AIMessage with NO tool_calls
       │     returns {"messages": [AIMessage(final)]}
       │     add_messages APPENDS → messages: [Human, AI(tc), Tool, AI(final)]
       │     should_continue() sees no tool_calls → END
       ▼
  agent.invoke() returns final memory:
  messages = [Human, AI(tool_call), Tool(result), AI(final answer)]
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. run_react_agent()  react_agent.py : 115-149                         │
│    walks the final messages list, extracts the last plain-text         │
│    AIMessage as `answer`, and every tool_call as a `tool_trace` entry  │
└────────────────────────────────┬────────────────────────────────────┘
                                  ▼
            dashboard/app.py — "Ask" button renders answer + tool trace
```

The two diagrams make the contrast explicit: Graph 1's memory is a **flat
dict where each key is written once** (linear pipeline, overwrite reducer);
Graph 2's memory is a **single growing list** where every super-step adds
to it (loop, append reducer) — that accumulation is what gives the ReAct
agent something resembling short-term memory across reasoning turns.

---

### 2.10 ReAct Pattern — Reason + Act

**The pattern:** Instead of asking the LLM to answer directly, you give it
tools and let it decide which tool to call, observe the result, and decide
again — repeating until it has enough information to answer.

**Why it demonstrates autonomous decision-making:** The branching (call
`search_kb` or `fetch_live_news`?) is decided by the LLM each turn, not
hard-coded. The system prompt instructs it to prefer `search_kb` first and
fall back to `fetch_live_news` if results are thin — but the LLM makes that
judgement.

**The loop:**
```
User question
     ↓
[reason node] LLM reads messages + tool schemas
     ↓ if tool_calls present
[act node]    ToolNode executes the tool, appends result as a tool message
     ↓ always loops back
[reason node] LLM reads the tool result, decides next step
     ↓ if no tool_calls (plain text)
    END → return final answer
```

**How tool calling works technically:**
1. `llm.bind_tools(tools)` sends the tool schemas (name, description,
   parameters) to the LLM API as part of the request.
2. When the LLM wants to call a tool, it returns a special `AIMessage`
   with a `tool_calls` field (structured JSON: tool name + arguments).
3. `ToolNode` reads `tool_calls`, executes the matching Python function,
   and appends the result as a `ToolMessage`.
4. On the next `reason` turn, the LLM sees the tool result and continues.

**Safety cap:** `MAX_TOOL_CALLS = 8`. If the model loops 8 times, it is
forced to answer by stripping the tools and injecting a "give your final
answer now" message. This prevents infinite loops on small models that
might keep calling tools.

---

### 2.11 The Two-Stage Validation Gate

**What it is:** A Python function (`verify_recommendations`) that runs
*after* the LLM generates recommendations and *before* they reach the
dashboard. It applies two independent checks.

**Stage 1 — Structural count:**
Any recommendation with fewer than 3 distinct evidence pieces is rejected
immediately. This ensures every recommendation is at least weakly grounded.

**Stage 2 — Semantic grounding (Sentence-BERT cosine):**
Even if a recommendation has 3+ evidence pieces, it must actually relate
to them semantically. The recommendation text (+ rationale) is embedded
with the same `all-MiniLM-L6-v2` model and compared against every evidence
title + snippet via cosine similarity. The maximum similarity across all
evidence pieces is the **grounding score**.

```
grounding_score = max cosine_sim(embed(rec_text), embed(evidence_i))
                  over all evidence pieces i
```

This is then blended with the retrieval confidence already attached to the
finding:
```
blended_confidence = 0.5 × grounding_score + 0.5 × retrieval_confidence
```

If `blended_confidence < threshold (0.25)` the recommendation is rejected
as semantically unsupported — the LLM wrote something that does not relate
to the evidence it cited.

**Why two stages?**
- Stage 1 alone: the LLM could cite many irrelevant articles and pass.
- Stage 2 alone: the LLM could write a rec that sounds like the evidence
  but cites nothing and pass.
- Together: the recommendation must have real citations AND be semantically
  aligned with them.

**What gets saved:**
- `data/outputs/recommendations.json` — verified recs only (dashboard reads this)
- `data/outputs/verification.json` — full report including rejected + rejection reason
- `data/outputs/metrics.json` — `mean_confidence`, `factual_precision`, `threshold`

`factual_precision = passed / total` — the fraction of LLM-generated recommendations
that survived both validation stages.

Located in `src/agent/verifier.py` → `verify_recommendations()`.
Called by `verify_node` in `pipeline/graph.py`.

---

## 3. The Math, Summarised

### 3.1 Embedding and similarity

```
Document text → all-MiniLM-L6-v2 → v ∈ ℝ³⁸⁴ (L2-normalised)

Cosine similarity (general):
         A · B
sim =   ———————
        |A||B|

After L2 normalisation (|A| = |B| = 1):
sim = A · B    (just the dot product)
```

### 3.2 ChromaDB distance vs similarity

```
ChromaDB cosine distance = 1 − cosine_similarity

distance = 0   → vectors are identical
distance = 1   → vectors are orthogonal (unrelated)
distance = 2   → vectors are opposite (rare in practice for text)
```

So when the code computes confidence from distance:
```python
sim = 1 - sum(dists) / len(dists)   # distance → similarity
confidence = max(0.3, min(1.0, sim)) # clamp to [0.3, 1.0]
```
A chunk with distance 0.1 has similarity 0.9 → high confidence.

### 3.3 Near-dedup threshold

```
threshold = 0.92

Keep document D if: ∀ already-kept K,  sim(D, K) < 0.92
Drop document D if: ∃ already-kept K,  sim(D, K) ≥ 0.92
```

### 3.4 Semantic grounding score (verifier Stage 2)

```
grounding_score = max { A · B_i   for i = 1..N }

where:
  A   = embed(recommendation_text + rationale)          ∈ ℝ³⁸⁴
  B_i = embed(evidence_title_i + evidence_snippet_i)    ∈ ℝ³⁸⁴
  N   = number of evidence pieces

(dot product = cosine similarity because all vectors are L2-normalised)

blended_confidence = 0.5 × grounding_score + 0.5 × retrieval_confidence

factual_precision  = passed / total_drafted     ∈ [0, 1]
mean_confidence    = mean(blended_confidence)   across all recs
```

Example:
- Recommendation: "Expand into Asian chip markets"
- Evidence 1 title: "NVIDIA chip ban lifted in Asia"
- Evidence 2 title: "Asia GPU demand surges 40%"
- grounding_score = max(sim(rec, ev1), sim(rec, ev2)) ≈ 0.45
- retrieval_confidence (from RAG distance) ≈ 0.85
- blended = 0.5 × 0.45 + 0.5 × 0.85 = 0.65  →  passes threshold 0.25 ✅

### 3.5 Sentiment net polarity

```
              positive − negative
net_polarity = ———————————————————    ∈ [−1, +1]
                     total
```

Interpretation:
- `+1.0` → every article was positive
- `0.0`  → balanced or all neutral
- `−1.0` → every article was negative

### 3.5 Document ID hash

```python
key = url.strip().lower() + "|" + title.strip().lower()
id  = sha1(key.encode("utf-8")).hexdigest()[:16]   # 16 hex chars = 64 bits
```

64 bits gives a collision probability of ~1 in 10¹⁹ for 300 documents —
effectively zero. Deterministic: re-running collection gives the same IDs
for the same articles, so exact dedup works correctly.

---

## 4. File Breakdown and Connections

### Configuration layer

| File | What it does | Connects to |
|---|---|---|
| `config/config.yaml` | Single source of truth: company name, source URLs, chunk sizes, model names | Read by every module via `src/config.py` |
| `src/config.py` | Loads YAML + injects env vars (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) via `load_dotenv(override=True)` | Called once at startup by both `pipeline/graph.py` and `dashboard/app.py` |
| `.env` | Runtime secrets (gitignored). Groq API key lives here. | Loaded by `src/config.py` |
| `.env.example` | Template with placeholders (git-tracked). Never contains real keys. | Documentation only |

### Schema layer

| File | What it does | Connects to |
|---|---|---|
| `src/schema.py` | Defines `Document` TypedDict and `make_doc_id()` / `make_document()` factories | Used by every collector, cleaner, deduplicator, and the dashboard |

Every piece of data that flows through the pipeline is a `Document` dict with
keys: `id`, `source`, `title`, `text`, `url`, `published`.

### Collection layer

| File | What it does | Connects to |
|---|---|---|
| `src/collectors/base.py` | Abstract `BaseCollector` class with `collect()` method | Inherited by all three collectors |
| `src/collectors/news_collector.py` | Google News RSS → list of Documents | Reads config feeds/queries; uses `feedparser` |
| `src/collectors/company_collector.py` | NVIDIA newsroom + blog RSS | Same pattern as above |
| `src/collectors/hackernews_collector.py` | Hacker News Firebase API (top/new stories filtered for NVIDIA) | HTTP GET to `hacker-news.firebaseio.com` |
| `src/collectors/registry.py` | Orchestrates all enabled collectors. Calls each, exact-dedupes the union, saves `data/raw/documents.json` | Called by `collect_node` in `pipeline/graph.py` |

### Processing layer

| File | What it does | Connects to |
|---|---|---|
| `src/processing/cleaner.py` | Strips HTML, normalises whitespace, drops empty/too-short docs | Called by `process_node` |
| `src/processing/deduplicator.py` | `dedupe_exact` (hash) and `dedupe_near` (cosine ≥ 0.92) | Uses `Embedder` for near-dedup; called by `process_node` |
| `src/processing/chunker.py` | Sliding window (800 chars / 120 overlap) → chunk dicts with `chunk_id` | Called by `index_node`; outputs feed `VectorStore.add()` |

### Storage layer

| File | What it does | Connects to |
|---|---|---|
| `src/store/embeddings.py` | Wraps `sentence-transformers`; lazy-loads `all-MiniLM-L6-v2`; returns L2-normalised vectors | Called by `process_node`, `index_node`, `intelligence_node`, ReAct agent |
| `src/store/vector_store.py` | Wraps ChromaDB with `hnsw:space=cosine`. Methods: `add()`, `query()`, `reset()`, `count()` | Called by `index_node` to write; called by `RagChain` to read |

### Intelligence layer

| File | What it does | Connects to |
|---|---|---|
| `src/intelligence/evidence.py` | `build_evidence()` — de-duplicates retrieved chunks and trims each to a 300-char snippet for citation | Called by `IntelligenceEngine._finalize()` |
| `src/intelligence/engine.py` | `IntelligenceEngine` — runs 3 RAG+LLM passes (opportunities, risks, trends). Computes confidence from retrieval distance. | Uses `RagChain`, `LLMClient`, `prompts`, `evidence.py` |
| `src/intelligence/sentiment.py` | `SentimentAnalyzer` — scores each document with RoBERTa. Aggregates into overall / by-source / time trend. | Called by `sentiment_node`; also used as a tool in Graph 2 |

### Agent layer

| File | What it does | Connects to |
|---|---|---|
| `src/agent/prompts.py` | All LLM system prompt templates (opportunities, risks, trends, CEO, briefing, Q&A) | Imported by `IntelligenceEngine`, `CEOAgent`, `generate_briefing` |
| `src/agent/rag_chain.py` | `RagChain` — `retrieve(query)` → embed → ChromaDB query; `build_context()` → numbered text block | Shared by `IntelligenceEngine`, `CEOAgent`, and `search_kb` tool |
| `src/agent/ceo_agent.py` | `CEOAgent.recommend()` — takes intelligence findings, asks LLM for strategic actions, resolves evidence refs back to real chunks | Uses `LLMClient`, `prompts.CEO_SYSTEM` |
| `src/agent/verifier.py` | `verify_recommendations()` — partition recs into passed (≥3 evidence) and rejected. Hard Python gate. | Called by `verify_node` in `pipeline/graph.py` |
| `src/agent/briefing.py` | `generate_briefing()` — asks LLM for `what_happened / why_it_matters / what_to_do_next` JSON | Uses `LLMClient`, `prompts.BRIEFING_SYSTEM` |
| `src/agent/react_agent.py` | **Graph 2** — `build_react_agent()` assembles the ReAct `MessagesState` graph; `run_react_agent()` is the public API the dashboard calls | Uses `make_tools`, `ChatOpenAI`, `ToolNode` from LangGraph |

### LLM layer

| File | What it does | Connects to |
|---|---|---|
| `src/llm/client.py` | `LLMClient` — routes to OpenAI-compatible remote endpoint (Ollama/Groq/vLLM) or in-process `transformers`. `json_chat()` parses JSON from raw LLM text, handles `<think>` blocks. | Used by `IntelligenceEngine`, `CEOAgent`, `generate_briefing` |

Note: Graph 1 uses `LLMClient` (plain chat). Graph 2 uses `ChatOpenAI`
(LangChain's wrapper) because only LangChain knows how to attach tool
schemas and parse tool calls.

### Tools layer (Graph 2 only)

| File | What it does | Connects to |
|---|---|---|
| `src/tools/registry.py` | `make_tools(cfg, store, embedder)` — factory that returns 4 `@tool`-decorated functions with dependencies injected via closure | Called by `build_react_agent()` |

The 4 tools:

| Tool | What it does | When the LLM calls it |
|---|---|---|
| `search_kb(query)` | RAG retrieve from ChromaDB | Always first — to find stored evidence |
| `fetch_live_news(topic)` | Live Google News RSS request | When `search_kb` returns thin results |
| `score_sentiment(topic)` | Retrieve docs + run RoBERTa | When mood/sentiment context is needed |
| `compute_metrics(type)` | Count docs, source breakdown, date range | When statistical context is needed |

### Pipeline (Graph 1)

| File | What it does |
|---|---|
| `pipeline/graph.py` | Assembles `StateGraph(PipelineState)` with 8 nodes in linear order. `main()` is the entry point (`python -m pipeline.graph`). |
| `pipeline/run_pipeline.py` | Legacy imperative entry point — kept as fallback. Does the same stages as function calls instead of LangGraph nodes. |

### Dashboard

| File | What it does | Connects to |
|---|---|---|
| `dashboard/app.py` | Streamlit app with 9 sections. Reads all `data/outputs/*.json` at startup. "Ask" button calls `run_react_agent()` live. Shows verification report, tool trace expander, sentiment charts. | Reads JSON artifacts; imports `run_react_agent` from `src/agent/react_agent.py` |

---

## 5. How Data Flows (End to End)

```
[3 RSS/API sources]
        │  HTTP requests (no auth)
        ▼
src/collectors/registry.py → collect_all()
        │  list[Document]  (raw, may have duplicates)
        ▼  saved to: data/raw/documents.json
src/processing/cleaner.py → clean_documents()
        │  drops HTML, short docs
        ▼
src/processing/deduplicator.py → dedupe_exact()  (hash)
                               → dedupe_near()   (cosine ≥ 0.92)
        │  saved to: data/processed/clean.json
        ▼
src/processing/chunker.py → chunk_documents()    (800/120 sliding window)
        │
src/store/embeddings.py   → Embedder.encode()    (384-dim L2-normalised vectors)
        │
src/store/vector_store.py → VectorStore.add()    (upsert to ChromaDB cosine index)
        │  saved to: data/processed/chunks.json + data/chroma/
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
src/intelligence/engine.py            src/intelligence/sentiment.py
   3 RAG+LLM passes                      RoBERTa per-document scoring
   (opportunities, risks, trends)         aggregate by source + day
        │  saved: data/outputs/            │  saved: data/outputs/
        │         intelligence.json        │         sentiment.json
        ▼
src/agent/ceo_agent.py → CEOAgent.recommend()   (LLM, cites O1/R2/T3 labels)
        │  draft recommendations
        ▼
src/agent/verifier.py → verify_recommendations()
        │  Stage 1: len(evidence) >= 3
        │  Stage 2: Sentence-BERT grounding score >= threshold
        │  saved: data/outputs/recommendations.json (verified only)
        │  saved: data/outputs/verification.json   (full report incl. rejected)
        │  saved: data/outputs/metrics.json         (mean_confidence, factual_precision)
        ▼
src/agent/briefing.py → generate_briefing()       (LLM)
        │  saved: data/outputs/briefing.json
        ▼
dashboard/app.py reads all *.json at startup
"Ask" button → src/agent/react_agent.py → Graph 2 (live ReAct loop)
```

---

## 6. PipelineState — What Gets Passed Between Nodes

```python
class PipelineState(TypedDict):
    cfg            # config dict — available to every node
    embedder       # Embedder instance — reused across nodes (lazy-loaded model)
    llm            # LLMClient instance — reused across nodes
    docs           # list[Document]  — filled by collect_node
    processed      # list[Document]  — filled by process_node
    intelligence   # dict            — filled by intelligence_node
    sentiment      # dict            — filled by sentiment_node
    recommendations # list[dict]     — filled by recommend_node, updated by verify_node
    verification   # dict            — filled by verify_node
    briefing       # dict            — filled by brief_node
```

Each node returns only the keys it changed. LangGraph merges them back.
This is why `index_node` returns `{}` — it writes to disk as a side effect
(ChromaDB) but doesn't add anything new to the state dict.

---

## 7. Design Decisions

### Why two graphs instead of one?

Graph 1 is the offline, expensive, deterministic pipeline — it runs once and
writes artifacts to disk. Graph 2 is the live, interactive, non-deterministic
ReAct loop — it runs on every "Ask" click.

Keeping them separate means:
- You can re-run Graph 1 to refresh the knowledge base without touching the
  dashboard.
- You can run the dashboard (Graph 2) without re-running the whole pipeline.
- Each graph has a clear, single purpose.

### Why LangGraph instead of a plain for-loop?

A for-loop would work, but it makes the execution plan invisible. With
LangGraph, the graph definition IS the architecture — you can print it,
visualise it, and a reader can understand the pipeline just by looking at
the node/edge definitions. Adding a conditional edge (e.g., retry on failure)
is one line of code. The verify node as an explicit named node makes the
validation gate impossible to miss.

### Why the Python validation gate instead of prompting the LLM?

"Only recommend things with ≥ 3 evidence pieces" in a prompt is a suggestion.
`if len(evidence) < 3: rejected.append(rec)` is a guarantee. The rejected
recommendations are also saved to `verification.json` so the examiner can
see that the gate is real and not just empty code.

### Why cosine similarity instead of Euclidean distance?

Two documents about the same topic but with different lengths would have very
different Euclidean distances because longer documents accumulate more
vector magnitude. Cosine similarity only measures the direction (the meaning),
ignoring magnitude. L2-normalising the vectors makes this even more principled
— every vector is a unit vector on a hypersphere, and the geometry is purely
directional.

### Why an encoder (RoBERTa) for sentiment, not the LLM?

Two reasons:
1. RoBERTa is fine-tuned for this exact task (sentiment classification on
   social/news text) and runs extremely fast on CPU (16 docs per second).
   The LLM is slow and unpredictable for classification.
2. Demonstrating both encoder (understanding) and decoder (generation)
   architectures in one project shows that you understand when to use each.

### Why Groq (free API) instead of a local model for the LLM?

The assignment says open models only — no paid API. Groq provides free hosting
for open-source models (Qwen3, Llama 3) via an OpenAI-compatible endpoint.
Running a 27B model locally would require a GPU the exam environment doesn't
have. Groq makes the demo fast and reliable without paying for proprietary
APIs like GPT-4.

### Why not fine-tune the LLM?

Fine-tuning requires labelled training data ("here are 1000 examples of
correct NVIDIA strategic recommendations"). That data doesn't exist and
creating it would be months of work. RAG injects fresh, cited evidence at
inference time — achieving grounding without training. Fine-tuning also
locks the model to past knowledge; RAG uses today's articles.

### Why three independent data sources?

The assignment requires ≥ 3 independent sources, but there is a conceptual
reason too: each source has a different editorial perspective. Google News
aggregates mainstream financial and tech media; NVIDIA's own newsroom is
company-positive; Hacker News reflects the developer/technical community.
Having all three gives a more complete signal.

### Why semantic near-dedup on top of exact dedup?

Exact dedup catches reposts of the same URL. But the same story gets picked
up and rewritten by dozens of outlets. Without near-dedup, the same Reuters
story about NVIDIA earnings would appear 15 times with different titles,
flooding the knowledge base and skewing retrieval toward the most syndicated
story. A cosine threshold of 0.92 catches paraphrase while preserving genuinely
different takes on the same event.

---

## 8. Quick Answer Sheet

**Q: What is RAG?**
Retrieval-Augmented Generation. Embed a question, find similar document chunks in a
vector database, paste them into the LLM prompt as grounding context so the LLM
answers from real evidence rather than from memory.

**Q: Why L2-normalise embeddings?**
So cosine similarity equals the dot product. Faster to compute, and eliminates
the effect of document length on similarity scores.

**Q: What does ChromaDB distance = 0.1 mean?**
Cosine similarity = 1 − 0.1 = 0.9. Very similar. Distance 0 = identical,
distance 1 = completely unrelated.

**Q: Why is the near-dedup threshold 0.92?**
High enough to catch paraphrased articles (same story, rewritten) but low enough
to keep articles about the same company that describe different events.

**Q: How does LangGraph's StateGraph actually hold and pass state?**
Every field of the state schema is a "channel." A node receives the full
current state dict, returns only the keys it changed, and LangGraph writes
those into the channels using a reducer — `LastValue` (overwrite) by default,
or `add_messages` (append) when the field is `Annotated[list, add_messages]`
like `MessagesState.messages`. Execution proceeds in Pregel-style "super-steps":
run the active node(s) → reduce their outputs into channels → use the edges
(static or conditional) to pick the next active node(s) → repeat until `END`.
A node always sees everything every prior super-step wrote, not just what the
immediately preceding node returned. See 2.9.1 for the full worked walkthrough.

**Q: Why doesn't returning `{"messages": [response]}` in Graph 2 erase the
previous conversation?**
Because `MessagesState.messages` is `Annotated[list[BaseMessage], add_messages]`
— its reducer appends rather than overwrites. `PipelineState` fields have no
such annotation, so they use the default overwrite reducer instead (fine
there, since Graph 1 is a strict linear chain with one writer per channel).

**Q: What is the ReAct pattern?**
Reason + Act loop. The LLM reasons about which tool to call, executes it via
ToolNode, observes the result, and repeats until it produces a plain-text
final answer with no tool call.

**Q: How does tool calling work?**
`llm.bind_tools(tools)` sends tool schemas (JSON describing each tool's name,
description, and parameters) to the API. The LLM returns an `AIMessage` with a
`tool_calls` field. `ToolNode` reads it and calls the matching Python function.

**Q: Why is the validation gate a Python function, not a prompt?**
A prompt instruction is optional — the LLM can ignore it. A Python `if` statement
is a guarantee. Rejected recommendations are saved with their rejection reason,
making the gate auditable.

**Q: What are the two validation stages?**
Stage 1 (structural): `len(evidence) >= 3` — must cite at least 3 real articles.
Stage 2 (semantic): `grounding_score = max cosine_sim(rec_text, evidence_i)` must
reach the blended confidence threshold (0.25). A recommendation that cites irrelevant
articles fails Stage 2 even if it passed Stage 1.

**Q: What is factual_precision?**
`passed / total_drafted` — the fraction of LLM-generated recommendations that
survived both validation stages. Saved to `metrics.json` so the examiner can
see a quantitative validation result.

**Q: What is the grounding score?**
Maximum cosine similarity between the embedded recommendation text and any embedded
evidence piece (title + snippet). Uses the same L2-normalised MiniLM embeddings as
the rest of the pipeline, so dot product = cosine similarity.

**Q: What is net polarity?**
`(positive_count − negative_count) / total_count`. Ranges −1 to +1. Zero means
balanced sentiment or all neutral.

**Q: How is confidence calculated when the LLM doesn't give one?**
From the average retrieval distance of the evidence chunks:
`confidence = 1 − mean_distance`, clamped to [0.3, 1.0].

**Q: What does MAX_TOOL_CALLS = 8 do?**
Safety cap. After 8 tool calls the agent is forced to a final answer by stripping
the bound tools and injecting "give your final answer now." Prevents infinite loops
on smaller models.

**Q: Why use `feedparser` instead of the official NVIDIA API?**
There is no official NVIDIA news API. `feedparser` parses RSS feeds that NVIDIA
publishes publicly — no auth key required, which satisfies the assignment's
keyless data source requirement.
