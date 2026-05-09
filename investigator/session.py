from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from investigator.ledger import ClaimsLedger
from investigator.repo import Repo


@dataclass
class TurnRecord:
    turn: int
    question: str
    answer: str
    audit: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citation_report: dict[str, Any] | None = None
    investigator_elapsed_s: float = 0.0
    auditor_elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "question": self.question,
            "answer": self.answer,
            "audit": self.audit,
            "tool_calls": self.tool_calls,
            "citation_report": self.citation_report,
            "investigator_elapsed_s": self.investigator_elapsed_s,
            "auditor_elapsed_s": self.auditor_elapsed_s,
            "error": self.error,
        }


@dataclass
class Session:
    session_id: str
    repo: Repo
    history: list[types.Content] = field(default_factory=list)
    ledger: ClaimsLedger = field(default_factory=ClaimsLedger)
    turns: list[TurnRecord] = field(default_factory=list)
    turn_count: int = 0

    def repo_meta(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "url": self.repo.url,
            "owner": self.repo.owner,
            "name": self.repo.name,
            "ref": self.repo.ref,
            "size_mb": round(self.repo.size_mb, 2),
            "slug": self.repo.slug,
            "turn_count": self.turn_count,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def add(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.repo_meta() for s in self._sessions.values()]
