"""RepoContext: one repo's tree + lazily cached file contents."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.tools.github_client import GitHubClient, parse_repo_url

INDEXABLE_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".md")
SKIP_DIRS = ("node_modules/", "dist/", "build/", ".venv/", "venv/", "vendor/", "__pycache__/")


@dataclass
class RepoContext:
    owner: str
    name: str
    ref: str
    client: GitHubClient
    tree: list[dict] = field(default_factory=list)
    _cache: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_url(cls, url: str, client: GitHubClient | None = None) -> "RepoContext":
        client = client or GitHubClient()
        owner, name = parse_repo_url(url)
        ref = client.repo(owner, name).get("default_branch", "main")
        ctx = cls(owner=owner, name=name, ref=ref, client=client)
        ctx.tree = client.tree(owner, name, ref)
        return ctx

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def collection(self) -> str:
        return f"repo_{self.owner}_{self.name}".lower().replace("-", "_").replace(".", "_")

    def files(self) -> list[dict]:
        return [t for t in self.tree if t["type"] == "blob"]

    def has_file(self, path: str) -> bool:
        return any(t["path"] == path for t in self.files())

    def read(self, path: str) -> str:
        path = path.strip().lstrip("/")
        if path not in self._cache:
            self._cache[path] = self.client.file(self.owner, self.name, path, self.ref)
        return self._cache[path]

    def indexable_files(self) -> list[dict]:
        s = get_settings()
        out = [f for f in self.files()
               if f["path"].endswith(INDEXABLE_EXT)
               and not any(d in f["path"] for d in SKIP_DIRS)
               and f.get("size", 0) <= s.max_index_file_bytes]
        return out[: s.max_index_files]
