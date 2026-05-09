from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Claim:
    text: str
    turn: int

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "turn": self.turn}


@dataclass
class ClaimsLedger:
    claims: list[Claim] = field(default_factory=list)

    def add(self, text: str, turn: int) -> None:
        text = text.strip()
        if not text:
            return
        for c in self.claims:
            if c.text == text:
                return
        self.claims.append(Claim(text=text, turn=turn))

    def add_many(self, texts: list[str], turn: int) -> None:
        for t in texts:
            self.add(t, turn)

    def recent_strings(self, limit: int = 12) -> list[str]:
        return [c.text for c in self.claims[-limit:]]

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.claims]
