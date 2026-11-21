"""DevAgent UI: add a repo, give a task, watch every step live, approve/reject writes."""
import json
import os
import time

import httpx
import streamlit as st

st.set_page_config(page_title="DevAgent", page_icon="🛠️", layout="wide")


@st.cache_resource
def embedded_api(port: int = 8765) -> str:
    """Single-process mode (e.g. Streamlit Community Cloud): run the FastAPI app in a thread."""
    import sys
    import threading
    from pathlib import Path

    import uvicorn
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    return url


def resolve_api() -> str:
    if os.getenv("DEVAGENT_API"):
        return os.environ["DEVAGENT_API"]
    try:  # a separately started API (uvicorn app.main:app) wins
        httpx.get("http://localhost:8000/health", timeout=1).raise_for_status()
        return "http://localhost:8000"
    except httpx.HTTPError:
        return embedded_api()


API = resolve_api()
st.title("🛠️ DevAgent — GitHub code assistant")
if os.getenv("DRY_RUN_WRITES", "").lower() in ("1", "true", "yes"):
    st.info("Public demo: write actions still pause for approval, but approved issues/comments "
            "are simulated and never sent to GitHub.")

ICONS = {"plan": "🗺️", "tool_call": "🔧", "tool_result": "📄", "draft": "✍️",
         "reflect_feedback": "🔁", "approval_required": "✋", "approval_decision": "✅",
         "final": "🏁", "done": "📊", "error": "❌"}


def api(method, path, **kw):
    r = httpx.request(method, f"{API}{path}", timeout=30, **kw)
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.text}")
        st.stop()
    return r.json()


def render_event(t, d):
    icon = ICONS.get(t, "•")
    if t == "plan":
        st.markdown(f"{icon} **Plan**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(d["steps"], 1)))
    elif t == "tool_call":
        for c in d["calls"]:
            st.markdown(f"{icon} `{c['name']}({json.dumps(c['args'])[:200]})`")
    elif t == "tool_result":
        with st.expander(f"{icon} {d['name']} → {d['chars']} chars"):
            st.code(d["preview"])
    elif t == "reflect_feedback":
        st.warning(f"{icon} Reflect sent work back: {d['text']}")
    elif t == "approval_required":
        st.info(f"{icon} Waiting for approval: **{d['tool']}**")
    elif t == "approval_decision":
        st.markdown(f"{'✅ Approved' if d['approved'] else '🚫 Rejected'} `{d['tool']}`")
    elif t == "final":
        st.success("🏁 Final answer")
        st.markdown(d["text"])
    elif t == "done":
        st.caption(f"📊 {d['tool_calls']} tool calls · {d['iterations']} iterations")
    elif t == "error":
        st.error(d["message"])


with st.sidebar:
    st.header("1. Repository")
    url = st.text_input("Public GitHub URL", "https://github.com/pallets/itsdangerous")
    if st.button("Add & index"):
        st.session_state.repo_id = api("POST", "/repos", json={"url": url})["repo_id"]
    if rid := st.session_state.get("repo_id"):
        s = api("GET", f"/repos/{rid}/status")
        st.write(f"**{s['slug']}** — {s['status']}")
        if s["total"]:
            st.progress(min(s["progress"] / s["total"], 1.0), f"{s['progress']}/{s['total']} files")
        if s["status"] not in ("ready", "error"):
            time.sleep(1)
            st.rerun()
        if s["status"] == "error":
            st.caption(f"Indexing failed ({s['error']}); search_code disabled, other tools work.")

st.header("2. Task")
task = st.text_area("What should the agent do?",
                    "Explain this repo's architecture and where signing happens.", height=90)
examples = st.columns(3)
for col, ex in zip(examples, ["Write a README for this repo",
                              "Find bugs in the main module and open an issue",
                              "Summarise the last 10 commits"]):
    if col.button(ex):
        task = ex
        st.session_state.pending_task = ex

if st.button("▶ Run agent", type="primary", disabled="repo_id" not in st.session_state):
    t = st.session_state.pop("pending_task", task)
    st.session_state.run_id = api("POST", "/runs", json={"repo_id": st.session_state.repo_id, "task": t})["run_id"]

if run_id := st.session_state.get("run_id"):
    st.header("3. Live steps")
    run = api("GET", f"/runs/{run_id}")
    events = []
    with httpx.Client(timeout=None) as c:
        # Read whatever the stream has so far (it replays from the start).
        with c.stream("GET", f"{API}/runs/{run_id}/stream", timeout=3) as s:
            ev = None
            try:
                for line in s.iter_lines():
                    if line.startswith("event:"):
                        ev = line[6:].strip()
                    elif line.startswith("data:") and ev:
                        events.append((ev, json.loads(line[5:].strip())))
                        if ev == "approval_required" and run["status"] == "awaiting_approval":
                            break
            except httpx.ReadTimeout:
                pass
    for t, d in events:
        if t != "end":
            render_event(t, d)

    if run["status"] == "awaiting_approval" and run["pending_approval"]:
        p = run["pending_approval"]
        st.subheader(f"✋ Approve `{p['tool']}`?")
        st.json({k: v for k, v in p["args"].items() if k != "body"})
        body = st.text_area("Body (you can edit before approving)", p["args"].get("body", ""), height=200)
        a, b = st.columns(2)
        if a.button("✅ Approve", type="primary"):
            edited = body if body != p["args"].get("body", "") else None
            api("POST", f"/runs/{run_id}/approve", json={"approved": True, "edited_body": edited})
            st.rerun()
        if b.button("🚫 Reject"):
            api("POST", f"/runs/{run_id}/approve", json={"approved": False})
            st.rerun()
    elif run["status"] == "running":
        time.sleep(1)
        st.rerun()
