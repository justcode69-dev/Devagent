"""Benchmark: runs every task in eval/tasks.yaml and writes eval/results.md.

Usage:  python -m eval.run_eval [--only arch,where_sign] [--no-index]
Metrics per task: pass/fail, tool calls, iterations, seconds. Free-tier friendly:
tasks run sequentially with a short pause between them.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.build_graph import build_graph
from app.graph.llm import LLM, default_models
from app.indexing.fetch_repo import RepoContext
from app.tools.code_search import CodeIndex

HERE = Path(__file__).parent


def score(t: dict, out: dict) -> bool:
    ans = (out.get("final_answer") or "").lower()
    ok = all(any(s.lower() in ans for s in group) for group in t.get("expect_any_of", []))
    if t.get("expect_files"):
        ok = ok and any(f in out.get("files_read", []) for f in t["expect_files"])
    return ok


def run_task(graph, t: dict) -> dict:
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 80}
    t0 = time.time()
    out = graph.invoke({"task": t["task"]}, cfg)
    while "__interrupt__" in out:  # auto-reviewer stands in for the human
        out = graph.invoke(Command(resume={"approved": t.get("approve", False)}), cfg)
    return {"id": t["id"], "passed": score(t, out), "tool_calls": out.get("tool_calls", 0),
            "iterations": out.get("iterations", 0), "seconds": round(time.time() - t0, 1),
            "guard": bool(out.get("stopped_by_guard")),
            "answer": (out.get("final_answer") or "")[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between tasks (rate limits)")
    ap.add_argument("--retry-wait", type=float, default=65.0,
                    help="seconds to wait before retrying a task that hit a rate limit")
    a = ap.parse_args()

    spec = yaml.safe_load((HERE / "tasks.yaml").read_text())
    tasks = [t for t in spec["tasks"] if not a.only or t["id"] in a.only.split(",")]
    ctx = RepoContext.from_url(spec["repo"])
    index = None
    if not a.no_index:
        index = CodeIndex()
        if not index.is_indexed(ctx.collection):
            print(f"Indexing {ctx.slug} ...")
            print(f"  {index.index(ctx)} chunks")
    graph = build_graph(ctx, LLM(default_models()), index, checkpointer=InMemorySaver())

    results = []
    for t in tasks:
        for attempt in range(2):
            try:
                r = run_task(graph, t)
                break
            except Exception as e:
                msg = f"ERROR {type(e).__name__}: {e}"
                r = {"id": t["id"], "passed": False, "tool_calls": 0, "iterations": 0,
                     "seconds": 0, "guard": False, "error": True, "answer": msg[:300]}
                if attempt == 0 and ("429" in msg or "rate limit" in msg.lower()):
                    print(f"  rate limited on {t['id']}, retrying in {a.retry_wait:.0f}s", flush=True)
                    time.sleep(a.retry_wait)
                    continue
                break
        results.append(r)
        status = "PASS" if r["passed"] else ("ERROR" if r.get("error") else "FAIL")
        print(f"{status:<5} {r['id']:<16} calls={r['tool_calls']:<3} {r['seconds']}s", flush=True)
        time.sleep(a.pause)

    n, ok = len(results), sum(r["passed"] for r in results)
    errs = sum(bool(r.get("error")) for r in results)
    ran = [r for r in results if not r.get("error")]
    avg = lambda k: round(sum(r[k] for r in ran) / max(len(ran), 1), 1)
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    err_note = f" ({errs} not run: LLM quota/rate-limit errors)" if errs else ""
    lines = [f"# DevAgent eval — {ctx.slug}", "",
             f"**{ok}/{n} tasks completed**{err_note} · avg {avg('tool_calls')} tool calls · "
             f"avg {avg('seconds')}s per task", "",
             "| Task | Result | Tool calls | Iterations | Time (s) |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {'✅' if r['passed'] else ('⚠️ error' if r.get('error') else '❌')}"
              f"{' (guard)' if r['guard'] else ''} | "
              f"{r['tool_calls']} | {r['iterations']} | {r['seconds']} |" for r in results]
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:3]))


if __name__ == "__main__":
    main()
