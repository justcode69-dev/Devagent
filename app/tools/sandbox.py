"""run_tests: executes pytest for the repo inside a locked-down Docker container.

Isolation: --network none, CPU/memory/pids limits, all capabilities dropped,
non-root user, no-new-privileges, hard wall-clock timeout. The repo is
downloaded on the host (with network) and mounted into the container, which
itself has no network at all.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

from app.config import get_settings
from app.indexing.fetch_repo import RepoContext


def _safe_extract(data: bytes, dest: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk() or m.isdev():
                continue
            members.append(m)
        tar.extractall(dest, members=members, filter="data")
    roots = [p for p in dest.iterdir() if p.is_dir()]
    return roots[0] if len(roots) == 1 else dest


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def build_command(repo_dir: Path, name: str, test_path: str = "") -> list[str]:
    s = get_settings()
    return ["docker", "run", "--rm", "--name", name,
            "--network", "none", "--cpus", "1", "--memory", "512m", "--pids-limit", "256",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "-v", f"{repo_dir}:/repo", "-w", "/repo",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            s.sandbox_image,
            "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--maxfail=20",
            *([test_path] if test_path else [])]


def run_tests(ctx: RepoContext, test_path: str = "") -> str:
    if test_path and (".." in test_path or test_path.startswith("/")):
        return "ERROR: test_path must be a relative path inside the repo."
    if not docker_available():
        return "SANDBOX_UNAVAILABLE: Docker is not running, tests were not executed."
    s = get_settings()
    name = f"devagent-sbx-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="devagent-") as tmp:
        repo_dir = _safe_extract(ctx.client.tarball(ctx.owner, ctx.name, ctx.ref), Path(tmp))
        repo_dir.chmod(0o777)
        for p in repo_dir.rglob("*"):
            if p.is_dir():
                p.chmod(0o777)
        try:
            r = subprocess.run(build_command(repo_dir, name, test_path), capture_output=True,
                               text=True, timeout=s.sandbox_timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return f"TIMEOUT: tests exceeded {s.sandbox_timeout_s}s and were killed."
    out = (r.stdout + "\n" + r.stderr).strip()
    tail = "\n".join(out.splitlines()[-60:])
    return f"exit_code={r.returncode}\n{tail}"
