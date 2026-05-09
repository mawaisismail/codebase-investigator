"""Clone and manage public GitHub repositories in session-scoped sandboxes."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

WORKSPACES_ROOT = Path(__file__).resolve().parent.parent / "workspaces"
WORKSPACES_ROOT.mkdir(exist_ok=True)

GITHUB_HOSTS = {"github.com", "www.github.com"}


class RepoError(Exception):
    pass


@dataclass
class Repo:
    session_id: str
    url: str
    owner: str
    name: str
    ref: str
    path: Path
    size_mb: float

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}@{self.ref}"


def _normalize_github_url(url: str) -> tuple[str, str, str]:
    """Return (clone_url, owner, repo) from a public GitHub URL.

    Accepts:
      https://github.com/<owner>/<repo>
      https://github.com/<owner>/<repo>.git
      https://github.com/<owner>/<repo>/tree/<branch>/...
    """
    url = url.strip()
    if not url:
        raise RepoError("Empty URL.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.hostname not in GITHUB_HOSTS:
        raise RepoError("Only public github.com URLs are supported.")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise RepoError("URL must include owner and repo, e.g. https://github.com/owner/repo")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not re.match(r"^[A-Za-z0-9._-]+$", owner) or not re.match(r"^[A-Za-z0-9._-]+$", repo):
        raise RepoError("Invalid owner/repo characters.")
    clone_url = f"https://github.com/{owner}/{repo}.git"
    return clone_url, owner, repo


def _dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def clone_repo(url: str, size_cap_mb: float = 80.0) -> Repo:
    """Shallow-clone a public GitHub repo to a session-scoped temp dir."""
    clone_url, owner, name = _normalize_github_url(url)
    session_id = uuid.uuid4().hex[:12]
    target = WORKSPACES_ROOT / session_id
    target.mkdir(parents=True, exist_ok=False)

    try:
        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                clone_url,
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RepoError(
                f"git clone failed: {result.stderr.strip()[:300] or 'unknown error'}"
            )
        # Get the actual default branch we ended up on.
        ref_proc = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        )
        ref = ref_proc.stdout.strip() or "HEAD"
        size = _dir_size_mb(target)
        if size > size_cap_mb:
            shutil.rmtree(target, ignore_errors=True)
            raise RepoError(
                f"Repo is {size:.1f} MB, larger than the {size_cap_mb:.0f} MB cap."
            )
        # Drop .git to save space and prevent tools from grepping it.
        git_dir = target / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)
        return Repo(
            session_id=session_id,
            url=clone_url,
            owner=owner,
            name=name,
            ref=ref,
            path=target,
            size_mb=size,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(target, ignore_errors=True)
        raise RepoError("git clone timed out (>180s).")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def cleanup_repo(repo: Repo) -> None:
    if repo.path.exists():
        shutil.rmtree(repo.path, ignore_errors=True)
