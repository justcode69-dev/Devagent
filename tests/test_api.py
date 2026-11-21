"""End-to-end API test: index -> run -> approval interrupt -> approve -> done, with SSE."""
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

import app.main as main
from app.db.store import Store
from app.graph.state import Plan, Reflection
from app.runner import Runner
from app.tools.code_search import CodeIndex, HashEmbedder
from tests.conftest import tc


@pytest.fixture
def api(tmp_path, ctx, make_llm, saver):
    llm, _ = make_llm(
        responses=[AIMessage("", tool_calls=[tc("search_code", {"query": "average"}, "c1")]),
                   AIMessage("", tool_calls=[tc("create_issue", {"title": "average([]) crashes",
                                                                  "body": "ZeroDivisionError"}, "w1")]),
                   AIMessage("Opened issue #1 for app/utils.py:2.")],
        structured=[Plan(steps=["search", "issue"]), Reflection(complete=True, grounded=True)])
    idx = CodeIndex(QdrantClient(":memory:"), HashEmbedder())
    r = Runner(Store(tmp_path / "t.db"), llm_factory=lambda: llm, index_factory=lambda: idx,
               ctx_factory=lambda url: ctx, checkpointer=saver, enable_tests=False)
    main.runner = r
    yield TestClient(main.app), r
    main.runner = None


def _wait(client, path, key, values, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(path).json()
        if body[key] in values:
            return body
        time.sleep(0.05)
    raise AssertionError(f"{path} stuck at {body[key]}")


def test_full_api_flow(api, fake_gh):
    client, runner = api
    assert client.post("/repos", json={"url": "not a url"}).status_code == 422
    rid = client.post("/repos", json={"url": "https://github.com/acme/demo"}).json()["repo_id"]
    repo = _wait(client, f"/repos/{rid}/status", "status", ("ready", "error"))
    assert repo["status"] == "ready" and repo["chunks"] > 0

    run_id = client.post("/runs", json={"repo_id": rid, "task": "find a bug and open an issue"}).json()["run_id"]
    run = _wait(client, f"/runs/{run_id}", "status", ("awaiting_approval", "error", "done"))
    assert run["status"] == "awaiting_approval"
    assert run["pending_approval"]["tool"] == "create_issue"
    assert fake_gh.issues == []

    assert client.post(f"/runs/{run_id}/approve", json={"approved": True}).status_code == 200
    run = _wait(client, f"/runs/{run_id}", "status", ("done", "error"))
    assert run["status"] == "done", run.get("error")
    assert run["final_answer"].startswith("Opened issue #1")
    assert run["tool_calls"] == 2
    assert fake_gh.issues[0]["title"] == "average([]) crashes"
    assert run["approvals"][0]["approved"] == 1 and run["approvals"][0]["tool"] == "create_issue"

    # Approving twice is refused.
    assert client.post(f"/runs/{run_id}/approve", json={"approved": True}).status_code == 409

    # SSE stream replays every step and ends.
    with client.stream("GET", f"/runs/{run_id}/stream") as s:
        body = "".join(s.iter_text())
    for ev in ["plan", "tool_call", "tool_result", "approval_required", "approval_decision", "final", "done", "end"]:
        assert f"event: {ev}" in body, ev
