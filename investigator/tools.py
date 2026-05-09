"""Read-only investigation tools.

Each tool is sandboxed: paths must stay inside the repo root. Output is line-
numbered so the agent's citations naturally include line numbers.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

MAX_BYTES_PER_READ = 64_000  # ~16k tokens of source
MAX_LINES_PER_READ = 800
MAX_GREP_MATCHES = 60
MAX_FIND_RESULTS = 200
MAX_TREE_ENTRIES = 800

# Extensions/dirs we never want to peek into.
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".turbo",
    "target",
    "vendor",
    ".cache",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".mov", ".avi", ".webm",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".class", ".jar", ".so", ".dll", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".exe", ".bin",
}


class ToolError(Exception):
    pass


def _safe_resolve(repo_root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` within `repo_root`. Reject anything that escapes it."""
    if rel_path is None:
        raise ToolError("Missing path.")
    rel_path = rel_path.strip().lstrip("/")
    if rel_path in ("", "."):
        return repo_root
    candidate = (repo_root / rel_path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        raise ToolError(f"Path escapes repo root: {rel_path}")
    return candidate


def list_tree(repo_root: Path, path: str = ".", max_depth: int = 2) -> dict[str, Any]:
    """List directories and files under `path` up to `max_depth` levels deep."""
    target = _safe_resolve(repo_root, path)
    if not target.exists():
        raise ToolError(f"Path not found: {path}")
    if not target.is_dir():
        raise ToolError(f"Path is not a directory: {path}")

    max_depth = max(1, min(int(max_depth or 2), 4))
    base = target.resolve()
    entries: list[str] = []
    truncated = False

    for root, dirs, files in os.walk(base):
        rel_root = Path(root).resolve().relative_to(base)
        depth = 0 if str(rel_root) == "." else len(rel_root.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        # Prune excluded directories in-place.
        dirs[:] = sorted([d for d in dirs if d not in EXCLUDED_DIRS])
        for d in dirs:
            entries.append(str((rel_root / d) / "") if str(rel_root) != "." else f"{d}/")
            if len(entries) >= MAX_TREE_ENTRIES:
                truncated = True
                break
        for f in sorted(files):
            entries.append(str(rel_root / f) if str(rel_root) != "." else f)
            if len(entries) >= MAX_TREE_ENTRIES:
                truncated = True
                break
        if truncated:
            break

    rel_target = "" if target == base else str(target.relative_to(base))
    return {
        "path": rel_target or ".",
        "max_depth": max_depth,
        "entry_count": len(entries),
        "truncated": truncated,
        "entries": entries,
    }


def read_file(
    repo_root: Path,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read lines [start_line, end_line] (1-indexed, inclusive) from a text file."""
    target = _safe_resolve(repo_root, path)
    if not target.exists():
        raise ToolError(f"File not found: {path}")
    if not target.is_file():
        raise ToolError(f"Not a file: {path}")
    if target.suffix.lower() in BINARY_EXTS:
        raise ToolError(f"Refusing to read binary file: {path}")

    try:
        size = target.stat().st_size
    except OSError as e:
        raise ToolError(f"Cannot stat {path}: {e}")
    if size > 4 * MAX_BYTES_PER_READ:
        # File is large; we'll still allow line-range reads but warn the agent.
        pass

    try:
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as e:
        raise ToolError(f"Cannot read {path}: {e}")

    total = len(lines)
    start_line = max(1, int(start_line or 1))
    if end_line is None or end_line <= 0:
        end_line = min(total, start_line + MAX_LINES_PER_READ - 1)
    end_line = min(int(end_line), total)
    if start_line > total:
        return {
            "path": str(target.relative_to(repo_root.resolve())),
            "total_lines": total,
            "start_line": start_line,
            "end_line": start_line,
            "content": "",
            "truncated": False,
            "note": f"start_line {start_line} is past end of file ({total} lines).",
        }
    if end_line - start_line + 1 > MAX_LINES_PER_READ:
        end_line = start_line + MAX_LINES_PER_READ - 1

    selected = lines[start_line - 1 : end_line]
    # Render with line numbers — this is what the model sees and cites against.
    width = len(str(end_line))
    rendered = "".join(
        f"{str(start_line + i).rjust(width)}│{line}" if line.endswith("\n")
        else f"{str(start_line + i).rjust(width)}│{line}\n"
        for i, line in enumerate(selected)
    )
    truncated = end_line < total
    byte_len = len(rendered.encode("utf-8"))
    if byte_len > MAX_BYTES_PER_READ:
        rendered = rendered.encode("utf-8")[:MAX_BYTES_PER_READ].decode(
            "utf-8", errors="ignore"
        )
        truncated = True

    return {
        "path": str(target.relative_to(repo_root.resolve())),
        "total_lines": total,
        "start_line": start_line,
        "end_line": end_line,
        "content": rendered,
        "truncated": truncated,
    }


def grep(
    repo_root: Path,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Search for a regex `pattern` in text files under `path`. Returns matches with line numbers."""
    if not pattern:
        raise ToolError("grep: pattern is required.")
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        raise ToolError(f"Invalid regex: {e}")

    target = _safe_resolve(repo_root, path)
    base = repo_root.resolve()
    matches: list[dict[str, Any]] = []
    files_scanned = 0
    truncated = False

    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            full = Path(root) / f
            rel = str(full.relative_to(base))
            if glob and not fnmatch.fnmatch(rel, glob) and not fnmatch.fnmatch(f, glob):
                continue
            if Path(f).suffix.lower() in BINARY_EXTS:
                continue
            try:
                size = full.stat().st_size
            except OSError:
                continue
            if size > 2_000_000:  # skip files over 2 MB
                continue
            files_scanned += 1
            try:
                with full.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if compiled.search(line):
                            matches.append({
                                "path": rel,
                                "line": lineno,
                                "text": line.rstrip("\n")[:300],
                            })
                            if len(matches) >= MAX_GREP_MATCHES:
                                truncated = True
                                break
            except OSError:
                continue
            if truncated:
                break
        if truncated:
            break

    return {
        "pattern": pattern,
        "path": path,
        "glob": glob,
        "case_insensitive": case_insensitive,
        "files_scanned": files_scanned,
        "match_count": len(matches),
        "truncated": truncated,
        "matches": matches,
    }


def find_files(
    repo_root: Path,
    name_glob: str,
    path: str = ".",
) -> dict[str, Any]:
    """Find files whose name matches the glob (e.g. '*.py', 'auth*.ts')."""
    if not name_glob:
        raise ToolError("find_files: name_glob is required.")
    target = _safe_resolve(repo_root, path)
    base = repo_root.resolve()
    results: list[str] = []
    truncated = False
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            if fnmatch.fnmatch(f, name_glob):
                results.append(str((Path(root) / f).relative_to(base)))
                if len(results) >= MAX_FIND_RESULTS:
                    truncated = True
                    break
        if truncated:
            break
    return {
        "name_glob": name_glob,
        "path": path,
        "result_count": len(results),
        "truncated": truncated,
        "results": sorted(results),
    }


# Tool dispatch table — used by both investigator and auditor.
def dispatch(name: str, args: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    if name == "list_tree":
        return list_tree(
            repo_root,
            path=args.get("path", "."),
            max_depth=args.get("max_depth", 2),
        )
    if name == "read_file":
        return read_file(
            repo_root,
            path=args["path"],
            start_line=args.get("start_line", 1),
            end_line=args.get("end_line"),
        )
    if name == "grep":
        return grep(
            repo_root,
            pattern=args["pattern"],
            path=args.get("path", "."),
            glob=args.get("glob"),
            case_insensitive=bool(args.get("case_insensitive", False)),
        )
    if name == "find_files":
        return find_files(
            repo_root,
            name_glob=args["name_glob"],
            path=args.get("path", "."),
        )
    raise ToolError(f"Unknown tool: {name}")
