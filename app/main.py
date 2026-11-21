"""FastAPI service for DevAgent.

POST /repos                 {url}                 -> index repo in background
GET  /repos/{id}/status                           -> indexing progress
POST /runs                  {repo_id, task}       -> start an agent run
GET  /runs/{id}                                   -> run summary + pending approval
GET  /runs/{id}/stream                            -> Server-Sent Events, every step live
POST /runs/{id}/approve     {approved, edited_body?} -> resume after interrupt()
"""
from __future__ import annotations

import asyncio
import json

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.db.store import Store
from app.graph.llm import LLM, default_models
from app.runner import Runner
from app.tools.code_search import CodeIndex
from app.tools.github_client import parse_repo_url

app = FastAPI(title="DevAgent", version="1.0")


def _default_runner() -> Runner:
    return Runner(Store(get_settings().db_path), llm_factory=lambda: LLM(default_models()),
                  index_factory=CodeIndex)


runner: Runner | None = None


def get_runner() -> Runner:
    global runner
    if runner is None:
        runner = _default_runner()
    return runner


class RepoIn(BaseModel):
    url: str


class RunIn(BaseModel):
    repo_id: str
    task: str = Field(min_length=3, max_length=2000)


class ApproveIn(BaseModel):
    approved: bool
    edited_body: str | None = Field(default=None, max_length=20000)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/repos", status_code=202)
def add_repo(body: RepoIn, bg: BackgroundTasks):
    try:
        owner, name = parse_repo_url(body.url)
    except ValueError as e:
        raise HTTPException(422, str(e))
    r = get_runner()
    rid = r.store.add_repo(body.url, f"{owner}/{name}")
    bg.add_task(r.index_repo, rid)
    return {"repo_id": rid, "status": "queued"}


@app.get("/repos/{repo_id}/status")
def repo_status(repo_id: str):
    repo = get_runner().store.repo(repo_id)
    if not repo:
        raise HTTPException(404, "repo not found")
    return repo


@app.post("/runs", status_code=202)
def start_run(body: RunIn):
    r = get_runner()
    repo = r.store.repo(body.repo_id)
    if not repo:
        raise HTTPException(404, "repo not found")
    if repo["status"] not in ("ready", "error"):
        # search_code needs the index; other tools still work if indexing failed.
        raise HTTPException(409, f"repo is still {repo['status']}")
    return {"run_id": r.start_run(body.repo_id, body.task), "status": "running"}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = get_runner()
    run = r.store.run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    run["pending_approval"] = r.pending_interrupt(run_id) if run["status"] == "awaiting_approval" else None
    run["approvals"] = r.store.approvals(run_id)
    return run


@app.get("/runs/{run_id}/stream")
async def stream(run_id: str):
    r = get_runner()
    if not r.store.run(run_id):
        raise HTTPException(404, "run not found")

    async def gen():
        last = 0
        while True:
            for e in r.store.events(run_id, after=last):
                last = e["id"]
                yield {"event": e["type"], "id": str(e["id"]), "data": json.dumps(e["data"])}
            status = r.store.run(run_id)["status"]
            if status in ("done", "error") and not r.store.events(run_id, after=last):
                yield {"event": "end", "data": json.dumps({"status": status})}
                return
            await asyncio.sleep(0.4)

    return EventSourceResponse(gen())


@app.post("/runs/{run_id}/approve")
def approve(run_id: str, body: ApproveIn):
    r = get_runner()
    run = r.store.run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    if run["status"] != "awaiting_approval":
        raise HTTPException(409, f"run is {run['status']}, nothing to approve")
    r.approve(run_id, body.approved, body.edited_body)
    return {"run_id": run_id, "status": "running"}
