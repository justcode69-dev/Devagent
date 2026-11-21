"""Runs agents and indexing jobs in background threads and records every step as an event."""
from __future__ import annotations

import threading
import time
from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.config import redact
from app.db.store import Store
from app.graph.build_graph import build_graph
from app.graph.llm import LLM
from app.graph.nodes import text_of
from app.indexing.fetch_repo import RepoContext
from app.tools.code_search import CodeIndex


def update_to_events(chunk: dict) -> list[tuple[str, dict]]:
    """Translate one LangGraph `updates` chunk into UI-friendly events."""
    evs: list[tuple[str, dict]] = []
    for node, upd in chunk.items():
        if node == "__interrupt__":
            for i in upd:
                evs.append(("approval_required", i.value))
            continue
        if not isinstance(upd, dict):
            continue
        if node == "planner":
            evs.append(("plan", {"steps": upd.get("plan", [])}))
        for m in upd.get("messages", []) if node != "planner" else []:
            if isinstance(m, AIMessage) and m.tool_calls:
                evs.append(("tool_call", {"calls": [{"name": c["name"], "args": c["args"]}
                                                    for c in m.tool_calls]}))
            elif isinstance(m, AIMessage):
                evs.append(("draft", {"text": text_of(m)}))
            elif isinstance(m, ToolMessage):
                c = text_of(m)
                evs.append(("tool_result", {"name": m.name, "chars": len(c), "preview": c[:600]}))
            elif node == "reflect":
                evs.append(("reflect_feedback", {"text": text_of(m)}))
        if upd.get("final_answer"):
            evs.append(("final", {"text": upd["final_answer"],
                                  "stopped_by_guard": bool(upd.get("stopped_by_guard"))}))
    return evs


class Runner:
    def __init__(self, store: Store, llm_factory: Callable[[], LLM],
                 index_factory: Callable[[], CodeIndex | None],
                 ctx_factory: Callable[[str], RepoContext] = RepoContext.from_url,
                 checkpointer=None, enable_tests: bool = True):
        self.store = store
        self._llm_factory, self._index_factory, self._ctx_factory = llm_factory, index_factory, ctx_factory
        self._llm: LLM | None = None
        self._index: CodeIndex | None = None
        self._ctxs: dict[str, RepoContext] = {}
        self._run_repo: dict[str, str] = {}
        self.checkpointer = checkpointer
        self.enable_tests = enable_tests
        self._lock = threading.Lock()

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = self._llm_factory()
        return self._llm

    @property
    def index(self) -> CodeIndex | None:
        """Raises if the embedder/Qdrant can't load (used by indexing, which reports it)."""
        if self._index is None:
            self._index = self._index_factory()
        return self._index

    def safe_index(self) -> CodeIndex | None:
        """For agent runs: a broken index must not kill the run; search_code just degrades."""
        try:
            return self.index
        except Exception:
            return None

    def ctx(self, repo_id: str) -> RepoContext:
        with self._lock:
            if repo_id not in self._ctxs:
                self._ctxs[repo_id] = self._ctx_factory(self.store.repo(repo_id)["url"])
            return self._ctxs[repo_id]

    # ---- indexing ----
    def index_repo(self, repo_id: str):
        try:
            self.store.update_repo(repo_id, status="fetching")
            ctx = self.ctx(repo_id)
            self.store.update_repo(repo_id, status="indexing", total=len(ctx.indexable_files()))
            n = self.index.index(ctx, progress=lambda i, t: self.store.update_repo(repo_id, progress=i))
            self.store.update_repo(repo_id, status="ready", chunks=n)
        except Exception as e:
            self.store.update_repo(repo_id, status="error", error=redact(str(e))[:500])

    # ---- agent runs ----
    def _graph(self, repo_id: str):
        kw = {"checkpointer": self.checkpointer} if self.checkpointer else {}
        return build_graph(self.ctx(repo_id), self.llm, self.safe_index(),
                           enable_tests=self.enable_tests, **kw)

    def _drive(self, run_id: str, repo_id: str, payload):
        cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 80}
        try:
            g = self._graph(repo_id)
            for chunk in g.stream(payload, cfg, stream_mode="updates"):
                for t, d in update_to_events(chunk):
                    self.store.add_event(run_id, t, d)
            st = g.get_state(cfg)
            v = st.values
            if st.next:
                self.store.update_run(run_id, status="awaiting_approval",
                                      tool_calls=v.get("tool_calls", 0), iterations=v.get("iterations", 0))
            else:
                self.store.update_run(run_id, status="done", final_answer=v.get("final_answer"),
                                      tool_calls=v.get("tool_calls", 0), iterations=v.get("iterations", 0),
                                      finished_at=time.time())
                self.store.add_event(run_id, "done", {"tool_calls": v.get("tool_calls", 0),
                                                      "iterations": v.get("iterations", 0)})
        except Exception as e:
            msg = redact(f"{type(e).__name__}: {e}")[:800]
            self.store.update_run(run_id, status="error", error=msg, finished_at=time.time())
            self.store.add_event(run_id, "error", {"message": msg})

    def start_run(self, repo_id: str, task: str, background: bool = True) -> str:
        run_id = self.store.add_run(repo_id, task)
        self._run_repo[run_id] = repo_id
        payload = {"repo_url": self.store.repo(repo_id)["url"], "task": task}
        self._spawn(self._drive, run_id, repo_id, payload, background=background)
        return run_id

    def pending_interrupt(self, run_id: str) -> dict | None:
        run = self.store.run(run_id)
        st = self._graph(run["repo_id"]).get_state({"configurable": {"thread_id": run_id}})
        for task in st.tasks:
            for i in task.interrupts:
                return i.value
        return None

    def approve(self, run_id: str, approved: bool, edited_body: str | None = None,
                background: bool = True):
        run = self.store.run(run_id)
        pending = self.pending_interrupt(run_id) or {}
        self.store.log_approval(run_id, pending.get("tool", "?"), pending.get("args", {}),
                                approved, edited_body)
        self.store.add_event(run_id, "approval_decision", {"approved": approved,
                                                           "tool": pending.get("tool"),
                                                           "edited": bool(edited_body)})
        self.store.update_run(run_id, status="running")
        cmd = Command(resume={"approved": approved, "edited_body": edited_body})
        self._spawn(self._drive, run_id, run["repo_id"], cmd, background=background)

    @staticmethod
    def _spawn(fn, *args, background: bool):
        if background:
            threading.Thread(target=fn, args=args, daemon=True).start()
        else:
            fn(*args)
