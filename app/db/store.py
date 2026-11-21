"""SQLite store for repos, runs, streamed events and the approval audit log."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
  id TEXT PRIMARY KEY, url TEXT, slug TEXT, status TEXT, progress INT DEFAULT 0,
  total INT DEFAULT 0, chunks INT DEFAULT 0, error TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, repo_id TEXT, task TEXT, status TEXT, final_answer TEXT,
  tool_calls INT DEFAULT 0, iterations INT DEFAULT 0, error TEXT,
  started_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, type TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, tool TEXT, args TEXT,
  approved INT, edited_body TEXT, decided_at REAL);
"""


class Store:
    def __init__(self, path: Path | str):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def _exec(self, sql: str, args=()):
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _one(self, sql: str, args=()) -> dict | None:
        with self._lock:
            r = self._conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    # repos
    def add_repo(self, url: str, slug: str) -> str:
        rid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO repos (id,url,slug,status,created_at) VALUES (?,?,?,?,?)",
                   (rid, url, slug, "queued", time.time()))
        return rid

    def update_repo(self, rid: str, **kw):
        cols = ",".join(f"{k}=?" for k in kw)
        self._exec(f"UPDATE repos SET {cols} WHERE id=?", (*kw.values(), rid))

    def repo(self, rid: str) -> dict | None:
        return self._one("SELECT * FROM repos WHERE id=?", (rid,))

    # runs
    def add_run(self, repo_id: str, task: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO runs (id,repo_id,task,status,started_at) VALUES (?,?,?,?,?)",
                   (run_id, repo_id, task, "running", time.time()))
        return run_id

    def update_run(self, run_id: str, **kw):
        cols = ",".join(f"{k}=?" for k in kw)
        self._exec(f"UPDATE runs SET {cols} WHERE id=?", (*kw.values(), run_id))

    def run(self, run_id: str) -> dict | None:
        return self._one("SELECT * FROM runs WHERE id=?", (run_id,))

    # events
    def add_event(self, run_id: str, type_: str, data: dict):
        self._exec("INSERT INTO events (run_id,ts,type,data) VALUES (?,?,?,?)",
                   (run_id, time.time(), type_, json.dumps(data, default=str)))

    def events(self, run_id: str, after: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, after)).fetchall()
        return [{"id": r["id"], "type": r["type"], "data": json.loads(r["data"])} for r in rows]

    # approvals audit log
    def log_approval(self, run_id: str, tool: str, args: dict, approved: bool, edited_body: str | None):
        self._exec("INSERT INTO approvals (run_id,tool,args,approved,edited_body,decided_at) "
                   "VALUES (?,?,?,?,?,?)",
                   (run_id, tool, json.dumps(args), int(approved), edited_body, time.time()))

    def approvals(self, run_id: str) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM approvals WHERE run_id=? ORDER BY id", (run_id,)).fetchall()]
