from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CITATION_RE = re.compile(
    r"\[([A-Za-z0-9_./\-]+?\.[A-Za-z0-9]+|[A-Za-z0-9_./\-]+):(\d+)(?:-(\d+))?\]"
)
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
    snippet_matches: bool | None = None
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
    i = 0
    for fl in file_window:
        if i < len(snippet_lines) and snippet_lines[i] == fl:
            i += 1
    return i == len(snippet_lines)


def validate_answer(answer_text: str, repo_root: Path) -> CitationReport:
    report = CitationReport()
    base = repo_root.resolve()

    for match in CITATION_RE.finditer(answer_text):
        raw = match.group(0)
        rel = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start

        check = CitationCheck(raw=raw, path=rel, start=start, end=end)

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

        tail = answer_text[match.end() : match.end() + 4000]
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
