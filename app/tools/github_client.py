"""Thin GitHub REST client. The token is held only in memory and never logged."""
from __future__ import annotations

import base64
import re

import httpx

from app.config import get_settings

_URL_RE = re.compile(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


class GitHubError(RuntimeError):
    pass


def parse_repo_url(url: str) -> tuple[str, str]:
    """'https://github.com/owner/name' -> ('owner', 'name')."""
    m = _URL_RE.search(url.strip())
    if not m:
        raise ValueError(f"Not a GitHub repo URL: {url!r}")
    return m.group(1), m.group(2)


class GitHubClient:
    def __init__(self, token: str | None = None, base_url: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        s = get_settings()
        if token is None and s.github_token:
            token = s.github_token.get_secret_value()
        headers = {"Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "devagent"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(base_url=base_url or s.github_api, headers=headers,
                                  timeout=30, transport=transport, follow_redirects=True)

    def _get(self, path: str, **params):
        r = self._http.get(path, params=params or None)
        if r.status_code >= 400:
            # Never echo request headers (they carry the token).
            raise GitHubError(f"GitHub GET {path} -> {r.status_code}: {r.text[:200]}")
        return r

    def _post(self, path: str, json: dict):
        r = self._http.post(path, json=json)
        if r.status_code >= 400:
            raise GitHubError(f"GitHub POST {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()

    # ---- read ----
    def repo(self, owner: str, name: str) -> dict:
        return self._get(f"/repos/{owner}/{name}").json()

    def tree(self, owner: str, name: str, ref: str) -> list[dict]:
        data = self._get(f"/repos/{owner}/{name}/git/trees/{ref}", recursive="1").json()
        return [{"path": t["path"], "type": t["type"], "size": t.get("size", 0)}
                for t in data.get("tree", [])]

    def file(self, owner: str, name: str, path: str, ref: str) -> str:
        data = self._get(f"/repos/{owner}/{name}/contents/{path}", ref=ref).json()
        if isinstance(data, list):
            raise GitHubError(f"{path} is a directory")
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        # Large files (>1MB) come without content; fall back to raw download.
        if data.get("download_url"):
            return httpx.get(data["download_url"], timeout=30).text
        return data.get("content", "")

    def commits(self, owner: str, name: str, n: int = 10) -> list[dict]:
        data = self._get(f"/repos/{owner}/{name}/commits", per_page=str(n)).json()
        return [{"sha": c["sha"][:7],
                 "author": (c.get("commit", {}).get("author") or {}).get("name", "?"),
                 "date": (c.get("commit", {}).get("author") or {}).get("date", ""),
                 "message": c.get("commit", {}).get("message", "").splitlines()[0]}
                for c in data]

    def tarball(self, owner: str, name: str, ref: str) -> bytes:
        return self._get(f"/repos/{owner}/{name}/tarball/{ref}").content

    # ---- write (only ever called after human approval) ----
    def create_issue(self, owner: str, name: str, title: str, body: str) -> dict:
        d = self._post(f"/repos/{owner}/{name}/issues", {"title": title, "body": body})
        return {"number": d.get("number"), "url": d.get("html_url")}

    def comment_on_pr(self, owner: str, name: str, pr: int, body: str) -> dict:
        d = self._post(f"/repos/{owner}/{name}/issues/{pr}/comments", {"body": body})
        return {"id": d.get("id"), "url": d.get("html_url")}
