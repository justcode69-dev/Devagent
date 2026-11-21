"""LangChain tools bound to one repo. Read tools run freely; write tools need approval.

Every read tool wraps repo content in <repo_data> tags. The system prompt tells
the model that anything inside those tags is untrusted data, never instructions.
"""
from __future__ import annotations

import re

from langchain_core.tools import BaseTool, tool

from app.config import get_settings
from app.indexing.fetch_repo import RepoContext
from app.tools import sandbox
from app.tools.code_search import CodeIndex

WRITE_TOOLS = {"create_issue", "comment_on_pr"}


def wrap(source: str, text: str) -> str:
    # Neutralise attempts to close the data block from inside repo content.
    text = text.replace("</repo_data>", "</repo_data_>")
    return f'<repo_data source="{source}">\n{text}\n</repo_data>'


def make_tools(ctx: RepoContext, index: CodeIndex | None = None,
               enable_tests: bool = True) -> tuple[list[BaseTool], list[BaseTool]]:
    """Returns (read_tools, write_tools)."""
    s = get_settings()

    @tool
    def list_files(path: str = "") -> str:
        """List files and directories in the repo under `path` ('' = repo root). Shows the tree two levels deep."""
        prefix = path.strip("/")
        prefix = prefix + "/" if prefix else ""
        rows = []
        for t in ctx.tree:
            if not t["path"].startswith(prefix):
                continue
            rel = t["path"][len(prefix):]
            if rel.count("/") > 1:
                continue
            rows.append(f'{"[dir] " if t["type"] == "tree" else ""}{t["path"]}')
        if not rows:
            return f"No entries under '{path}'."
        extra = f"\n... ({len(rows) - 300} more)" if len(rows) > 300 else ""
        return wrap(f"tree:{path or '/'}", "\n".join(rows[:300]) + extra)

    @tool
    def read_file(path: str, start_line: int = 1) -> str:
        """Read a file from the repo with line numbers. Large files are truncated; pass start_line to page."""
        path = path.strip().lstrip("/")
        if not ctx.has_file(path):
            return f"ERROR: '{path}' does not exist. Use list_files to find the right path."
        lines = ctx.read(path).splitlines()
        start = max(1, start_line)
        chunk = lines[start - 1: start - 1 + s.max_file_lines]
        body = "\n".join(f"{i:>5}| {l}" for i, l in enumerate(chunk, start))
        end = start + len(chunk) - 1
        note = (f"\n[truncated: showing {start}-{end} of {len(lines)} lines; "
                f"call read_file('{path}', start_line={end + 1}) for more]"
                if end < len(lines) else "")
        return wrap(path, body + note)

    @tool
    def grep(pattern: str, path_glob: str = "") -> str:
        """Exact keyword or regex search across the repo's source files. Optional path_glob filters paths by substring (e.g. 'src/')."""
        if len(pattern) > 200:
            return "ERROR: pattern too long."
        try:
            rx = re.compile(pattern)
        except re.error:  # not a valid regex -> treat as a literal string
            rx = re.compile(re.escape(pattern))
        hits = []
        for f in ctx.indexable_files():
            if path_glob and path_glob not in f["path"]:
                continue
            try:
                text = ctx.read(f["path"])
            except Exception:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                # LLMs often pass literals like "len(xs)"; match as regex OR plain text.
                if pattern in line or rx.search(line):
                    hits.append(f'{f["path"]}:{n}: {line.strip()[:160]}')
                    if len(hits) >= 50:
                        return wrap(f"grep:{pattern}", "\n".join(hits) + "\n[stopped at 50 matches]")
        return wrap(f"grep:{pattern}", "\n".join(hits)) if hits else f"No matches for {pattern!r}."

    @tool
    def get_commits(n: int = 10) -> str:
        """Recent commits (sha, author, date, first line of message). n is capped at 30."""
        rows = ctx.client.commits(ctx.owner, ctx.name, min(max(n, 1), 30))
        return wrap("commits", "\n".join(
            f'{c["sha"]} {c["date"][:10]} {c["author"]}: {c["message"]}' for c in rows))

    @tool
    def search_code(query: str, k: int = 5) -> str:
        """Semantic search over the indexed repo code. Use for 'where is X handled?' questions. Returns file paths with line ranges."""
        if index is None or not index.is_indexed(ctx.collection):
            return "search_code unavailable: repo is not indexed. Use grep instead."
        hits = index.search(ctx.collection, query, min(max(k, 1), 10))
        parts = [f'--- {h["path"]}:{h["start_line"]}-{h["end_line"]} (score {h["score"]})\n'
                 f'{h["text"][:800]}' for h in hits]
        return wrap(f"search:{query}", "\n".join(parts) or "No results.")

    @tool
    def run_tests(test_path: str = "") -> str:
        """Run the repo's pytest suite in an offline Docker sandbox (no network, 1 CPU, time limit). Optional test_path limits to one file/dir."""
        return wrap("run_tests", sandbox.run_tests(ctx, test_path))

    @tool
    def create_issue(title: str, body: str) -> str:
        """Create a GitHub issue in the repo. WRITE ACTION: a human must approve it before it runs."""
        if s.dry_run_writes:
            return f"DRY RUN (public demo): approved issue {title!r} was NOT sent to GitHub."
        r = ctx.client.create_issue(ctx.owner, ctx.name, title, body)
        return f"Issue #{r['number']} created: {r['url']}"

    @tool
    def comment_on_pr(pr: int, body: str) -> str:
        """Post a review comment on pull request number `pr`. WRITE ACTION: a human must approve it before it runs."""
        if s.dry_run_writes:
            return f"DRY RUN (public demo): approved comment on PR #{pr} was NOT sent to GitHub."
        r = ctx.client.comment_on_pr(ctx.owner, ctx.name, pr, body)
        return f"Comment posted on PR #{pr}: {r['url']}"

    read = [list_files, read_file, grep, get_commits, search_code]
    if enable_tests:
        read.append(run_tests)
    return read, [create_issue, comment_on_pr]
