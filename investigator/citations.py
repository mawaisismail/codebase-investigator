"""Programmatic citation validator.

The investigator is asked to cite specific files and line ranges in a strict
format. This module verifies those citations against the actual repo before
the LLM auditor sees the answer — hallucinated paths or line ranges get
flagged with zero token cost.

Citation format the investigator is instructed to use:

    [path/to/file.py:42-58]

Optional code blocks following a citation are also verified to byte-match
the file contents on those lines:

    [path/to/file.py:42-58]
    ```python
    def foo():
        ...
    ```
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Matches [path:start-end] or [path:line]. Allows reasonable path chars.
CITATION_RE = re.compile(
    r"\[([A-Za-z0-9_./\-]+?\.[A-Za-z0-9]+|[A-Za-z0-9_./\-]+):(\d+)(?:-(\d+))?\]"
)

# Matches ```lang\n<code>\n``` blocks immediately following a citation.
FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]*)\n(.*?)```", re.DOTALL)


@dataclass
class CitationCheck:
    raw: str
    path: str
    start: int
    end: int
    file_exists: bool = False
    range_valid: bool = False
    total_lines: int | None = None
    snippet_present: bool = False
    snippet_matches: bool | None = None  # None if no snippet to verify
    issue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation": self.raw,
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "file_exists": self.file_exists,
            "range_valid": self.range_valid,
            "total_lines": self.total_lines,
            "snippet_present": self.snippet_present,
            "snippet_matches": self.snippet_matches,
            "issue": self.issue,
            "ok": self.is_ok(),
        }

    def is_ok(self) -> bool:
        if not self.file_exists or not self.range_valid:
            return False
        if self.snippet_present and self.snippet_matches is False:
            return False
        return True


@dataclass
class CitationReport:
    checks: list[CitationCheck] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.is_ok())

    @property
    def issues(self) -> list[CitationCheck]:
        return [c for c in self.checks if not c.is_ok()]

    @property
    def all_ok(self) -> bool:
        return self.total > 0 and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ok_count": self.ok_count,
            "issue_count": len(self.issues),
            "all_ok": self.all_ok,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        if self.total == 0:
            return "No citations found."
        if self.all_ok:
            return f"All {self.total} citations verified."
        return (
            f"{self.ok_count}/{self.total} citations verified; "
            f"{len(self.issues)} have issues."
        )


def _read_lines(path: Path) -> list[str] | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.readlines()
    except OSError:
        return None


def _normalize(s: str) -> str:
    """Whitespace-normalize for snippet comparison.

    The agent may indent or wrap differently than the source, so we compare
    lines after stripping trailing whitespace and collapsing runs of internal
    whitespace. This is intentionally lenient: it catches outright fabrication
    while tolerating cosmetic differences.
    """
    return re.sub(r"[ \t]+", " ", s.rstrip())


def _snippet_matches(snippet: str, file_lines: list[str], start: int, end: int) -> bool:
    snippet_lines = [
        _normalize(ln) for ln in snippet.splitlines() if _normalize(ln) != ""
    ]
    if not snippet_lines:
        return True
    file_window = [_normalize(ln) for ln in file_lines[start - 1 : end]]
    file_window = [ln for ln in file_window if ln != ""]
    if not file_window:
        return False
    # Subsequence match: each non-empty snippet line must appear in order in
    # the file window. Tolerates the agent omitting lines for brevity.
    i = 0
    for fl in file_window:
        if i < len(snippet_lines) and snippet_lines[i] == fl:
            i += 1
    return i == len(snippet_lines)


def validate_answer(answer_text: str, repo_root: Path) -> CitationReport:
    """Walk citations in `answer_text` and verify each against `repo_root`."""
    report = CitationReport()
    base = repo_root.resolve()

    for match in CITATION_RE.finditer(answer_text):
        raw = match.group(0)
        rel = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start

        check = CitationCheck(raw=raw, path=rel, start=start, end=end)

        # Resolve safely
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            check.issue = "path escapes repo root"
            report.checks.append(check)
            continue
        if not candidate.exists() or not candidate.is_file():
            check.issue = "file does not exist"
            report.checks.append(check)
            continue
        check.file_exists = True

        lines = _read_lines(candidate)
        if lines is None:
            check.issue = "could not read file"
            report.checks.append(check)
            continue
        check.total_lines = len(lines)
        if start < 1 or end < start or end > len(lines):
            check.issue = (
                f"range {start}-{end} invalid for file with {len(lines)} lines"
            )
            report.checks.append(check)
            continue
        check.range_valid = True

        # Look for an immediately-following code block.
        tail = answer_text[match.end() : match.end() + 4000]
        # Strip leading whitespace/newlines before the fence.
        leading = re.match(r"\s*", tail).group(0)
        fence = FENCE_RE.match(tail[len(leading) :])
        if fence:
            snippet = fence.group(1)
            check.snippet_present = True
            check.snippet_matches = _snippet_matches(snippet, lines, start, end)
            if not check.snippet_matches:
                check.issue = "quoted snippet does not match file contents"

        report.checks.append(check)

    return report
