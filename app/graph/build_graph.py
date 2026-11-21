from __future__ import annotations

import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.llm import LLM
from app.graph.nodes import Nodes, route_after_agent, route_after_reflect
from app.graph.state import AgentState
from app.indexing.fetch_repo import RepoContext
from app.tools.code_search import CodeIndex
from app.tools.github_tools import make_tools

_saver = None


def default_checkpointer() -> BaseCheckpointSaver:
    global _saver
    if _saver is None:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(get_settings().checkpoint_path, check_same_thread=False)
        _saver = SqliteSaver(conn)
    return _saver


def build_graph(ctx: RepoContext, llm: LLM, index: CodeIndex | None = None,
                checkpointer: BaseCheckpointSaver | None = None, enable_tests: bool = True):
    read_tools, write_tools = make_tools(ctx, index, enable_tests=enable_tests)
    n = Nodes(llm, ctx, read_tools, write_tools)

    g = StateGraph(AgentState)
    g.add_node("planner", n.planner)
    g.add_node("agent", n.agent)
    g.add_node("tools", n.tools)
    g.add_node("approval", n.approval)
    g.add_node("reflect", n.reflect)

    g.add_edge(START, "planner")
    g.add_edge("planner", "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "approval": "approval", "reflect": "reflect", "end": END})
    g.add_edge("tools", "agent")
    g.add_edge("approval", "agent")
    g.add_conditional_edges("reflect", route_after_reflect, {"agent": "agent", "end": END})

    return g.compile(checkpointer=checkpointer or default_checkpointer())
