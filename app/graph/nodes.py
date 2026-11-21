from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from app.config import get_settings, redact
from app.graph.llm import LLM
from app.graph.state import AgentState, Plan, Reflection
from app.indexing.fetch_repo import RepoContext
from app.tools.github_tools import WRITE_TOOLS

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
MAX_REFLECTIONS = 2


def _prompt(name: str, **kw) -> str:
    return (PROMPTS / f"{name}.md").read_text().format(**kw)


def text_of(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, list):  # Gemini can return content blocks
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


def _last_ai(state: AgentState) -> AIMessage | None:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m
    return None


def _files_from_calls(calls) -> list[str]:
    return [c["args"].get("path", "") for c in calls if c["name"] == "read_file" and c["args"].get("path")]


def compact_messages(messages: list, budget: int) -> list:
    """Copy of `messages` whose tool results fit in `budget` chars, shrinking the OLDEST first.

    Free tiers limit tokens per minute/day, and every agent step re-sends the whole history.
    The newest results stay intact; older ones keep their head plus a note to re-read.
    State is not modified: the checkpoint keeps full results.
    """
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    total = sum(len(text_of(messages[i])) for i in tool_idx)
    if total <= budget:
        return messages
    out = list(messages)
    for i in tool_idx:  # oldest first
        if total <= budget:
            break
        body = text_of(out[i])
        keep = 600
        if len(body) <= keep + 200:
            continue
        short = (body[:keep] + f"\n[... {len(body) - keep} chars trimmed to save tokens; "
                 "call the tool again if you need this part ...]")
        total -= len(body) - len(short)
        out[i] = out[i].model_copy(update={"content": short})
    return out


class Nodes:
    def __init__(self, llm: LLM, ctx: RepoContext, read_tools: list[BaseTool],
                 write_tools: list[BaseTool]):
        self.ctx = ctx
        self.max_iter = get_settings().max_iterations
        self.context_budget = get_settings().context_char_budget
        self.read_map = {t.name: t for t in read_tools}
        self.write_map = {t.name: t for t in write_tools}
        self.agent_llm = llm.with_tools(read_tools + write_tools)
        self.planner_llm = llm.structured(Plan)
        self.reflect_llm = llm.structured(Reflection)
        # ToolNode only knows READ tools: even a routing bug cannot run a write tool here.
        self.tool_node = ToolNode(read_tools, handle_tool_errors=True)

    # 1. planner
    def planner(self, state: AgentState) -> dict:
        top = sorted({t["path"].split("/")[0] + ("/" if "/" in t["path"] else "")
                      for t in self.ctx.tree})[:60]
        plan: Plan = self.planner_llm.invoke([
            SystemMessage(_prompt("planner", repo=self.ctx.slug, tree="\n".join(top))),
            HumanMessage(state["task"]),
        ])
        steps = plan.steps[:6]
        system = _prompt("agent", repo=self.ctx.slug, ref=self.ctx.ref,
                         plan="\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
        return {"plan": steps, "iterations": 0, "reflections": 0, "tool_calls": 0,
                "files_read": [], "final_answer": None, "stopped_by_guard": False,
                "messages": [SystemMessage(system), HumanMessage(state["task"])]}

    # 2. agent (with loop guard)
    def agent(self, state: AgentState) -> dict:
        it = state.get("iterations", 0)
        if it >= self.max_iter:
            last = _last_ai(state)
            partial = text_of(last) if last and not last.tool_calls else ""
            return {"stopped_by_guard": True, "final_answer":
                    (partial or "I could not finish within the step limit.") +
                    f"\n\n_[Stopped by loop guard after {it} iterations; answer may be partial.]_"}
        msg = self.agent_llm.invoke(compact_messages(state["messages"], self.context_budget))
        update: dict = {"messages": [msg], "iterations": it + 1}
        writes = [c for c in (msg.tool_calls or []) if c["name"] in WRITE_TOOLS]
        if writes:
            update["pending_action"] = {"tool": writes[0]["name"], "args": writes[0]["args"]}
        return update

    # 3. tools (read-only)
    def tools(self, state: AgentState) -> dict:
        calls = _last_ai(state).tool_calls
        out = self.tool_node.invoke({"messages": state["messages"]})
        for m in out["messages"]:
            m.content = redact(text_of(m))
        return {"messages": out["messages"],
                "tool_calls": state.get("tool_calls", 0) + len(calls),
                "files_read": list(dict.fromkeys(state.get("files_read", []) + _files_from_calls(calls)))}

    # 4. approval (human-in-the-loop)
    def approval(self, state: AgentState) -> dict:
        calls = _last_ai(state).tool_calls
        # Ask for every decision FIRST: on resume LangGraph re-runs this node from the top,
        # so no side effect may happen before the interrupts.
        decisions = {c["id"]: interrupt({"type": "approval", "tool": c["name"],
                                         "args": c["args"], "tool_call_id": c["id"]})
                     for c in calls if c["name"] in WRITE_TOOLS}
        msgs = []
        for c in calls:
            if c["name"] in WRITE_TOOLS:
                d = decisions.get(c["id"]) or {}
                if d.get("approved"):
                    args = dict(c["args"])
                    if d.get("edited_body"):
                        args["body"] = d["edited_body"]
                    try:
                        result = self.write_map[c["name"]].invoke(args)
                    except Exception as e:
                        result = f"ERROR while executing {c['name']}: {redact(str(e))}"
                else:
                    result = "REJECTED by the human reviewer. Do not retry this action; report it."
            elif c["name"] in self.read_map:
                try:
                    result = redact(str(self.read_map[c["name"]].invoke(c["args"])))
                except Exception as e:
                    result = f"ERROR: {e}"
            else:
                result = f"ERROR: unknown tool {c['name']}"
            msgs.append(ToolMessage(content=result, tool_call_id=c["id"], name=c["name"]))
        return {"messages": msgs, "pending_action": None,
                "tool_calls": state.get("tool_calls", 0) + len(calls),
                "files_read": list(dict.fromkeys(state.get("files_read", []) + _files_from_calls(calls)))}

    # 5. reflect
    def reflect(self, state: AgentState) -> dict:
        answer = text_of(_last_ai(state))
        n = state.get("reflections", 0)
        if n >= MAX_REFLECTIONS or state.get("iterations", 0) >= self.max_iter:
            return {"final_answer": answer}
        r: Reflection = self.reflect_llm.invoke([HumanMessage(_prompt(
            "reflect", task=state["task"], plan="\n".join(state.get("plan", [])),
            files_read=", ".join(state.get("files_read", [])) or "(none)", answer=answer))])
        if r.complete and r.grounded:
            return {"final_answer": answer, "reflections": n + 1}
        return {"reflections": n + 1, "messages": [HumanMessage(
            f"Reviewer feedback: {r.feedback or 'Answer is incomplete or not grounded.'} "
            "Use tools to fix this, then give the complete final answer.")]}


# ---- routing ----
def route_after_agent(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
    last = _last_ai(state)
    if not last or not last.tool_calls:
        return "reflect"
    if any(c["name"] in WRITE_TOOLS for c in last.tool_calls):
        return "approval"
    return "tools"


def route_after_reflect(state: AgentState) -> str:
    return "end" if state.get("final_answer") is not None else "agent"
