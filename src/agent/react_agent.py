"""Graph 2 — ReAct agent loop (reason → act → reason → …).

The LLM autonomously decides which tool to call each turn and loops until it
produces a plain-text final answer (no tool call). This is where "autonomous
decision-making" and "tool usage beyond the LLM" live.

Key behaviours:
  - Starts with search_kb; if it returns thin results, the model calls
    fetch_live_news before concluding (conditional branching, not hard-coded).
  - MAX_TOOL_CALLS cap prevents infinite loops on a small model.
  - Works only with an OpenAI-compatible endpoint (vLLM / TGI) — the
    local-transformers path in LLMClient does not expose a tool-calling API.

Launch vLLM with:
    vllm serve Qwen/Qwen3-8B \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import load_config
from src.store.embeddings import Embedder
from src.store.vector_store import VectorStore
from src.tools.registry import make_tools

_SYSTEM_TEMPLATE = (
    "You are an AI strategic advisor for {company}. "
    "You have access to four tools: search_kb (searches the knowledge base), "
    "fetch_live_news (fetches fresh articles), score_sentiment (RoBERTa sentiment), "
    "and compute_metrics (counts, date ranges). "
    "Always call search_kb first to gather evidence before answering. "
    "If search_kb returns fewer than three relevant results, call fetch_live_news "
    "for supplementary information. "
    "Once you have enough evidence, provide a clear, decision-oriented answer "
    "with explicit references to what the tools returned."
)

MAX_TOOL_CALLS = 8  # safety cap — prevents an 8B model from looping forever


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_react_agent(cfg: dict[str, Any], store: VectorStore, embedder: Embedder):
    """Return a compiled LangGraph ReAct agent (Graph 2)."""
    tools = make_tools(cfg, store, embedder)

    llm = ChatOpenAI(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        api_key=cfg["llm"].get("api_key", "EMPTY"),
        temperature=cfg["llm"].get("temperature", 0.2),
        max_tokens=cfg["llm"].get("max_tokens", 2048),
    )
    llm_with_tools = llm.bind_tools(tools)
    system_msg = SystemMessage(
        content=_SYSTEM_TEMPLATE.format(company=cfg["company"]["name"])
    )

    # -- reason node: LLM decides next action ----------------------------
    def reason(state: MessagesState) -> dict:
        messages = list(state["messages"])

        # Prepend system message on the first turn
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_msg] + messages

        # Count tool calls already made — enforce the cap
        tool_call_count = sum(
            1
            for m in messages
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        if tool_call_count >= MAX_TOOL_CALLS:
            # Strip tools so the model is forced to produce a final answer
            forced = llm.invoke(
                messages
                + [HumanMessage(content="You have gathered enough information. Provide your final answer now.")]
            )
            return {"messages": [forced]}

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # -- routing: tool call → act, plain text → END ----------------------
    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "act"
        return END

    tool_node = ToolNode(tools)

    g = StateGraph(MessagesState)
    g.add_node("reason", reason)
    g.add_node("act", tool_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", should_continue, {"act": "act", END: END})
    g.add_edge("act", "reason")
    return g.compile()


# ---------------------------------------------------------------------------
# Public helper — used by the dashboard
# ---------------------------------------------------------------------------

def run_react_agent(
    question: str,
    cfg: dict[str, Any] | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Invoke Graph 2 and return the final answer plus the tool-call trace."""
    cfg = cfg or load_config()
    embedder = embedder or Embedder(cfg["store"]["embedding_model"])
    store = store or VectorStore(cfg)

    agent = build_react_agent(cfg, store, embedder)
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result["messages"]

    # Extract the last plain-text AI response as the final answer
    final_answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            final_answer = m.content or ""
            break

    # Build the tool trace for dashboard display
    tool_trace = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tool_trace.append({"tool": tc["name"], "input": tc["args"]})

    return {
        "answer": final_answer.strip(),
        "tool_trace": tool_trace,
        "steps": len(tool_trace),
    }
