"""FastAPI server orchestrating the investigator + auditor pipeline."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, Field

from investigator import auditor as auditor_mod
from investigator import investigator as investigator_mod
from investigator.citations import validate_answer
from investigator.repo import RepoError, clone_repo, cleanup_repo
from investigator.session import Session, SessionStore, TurnRecord

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("investigator")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_AUDITOR_MODEL = os.environ.get("GEMINI_AUDITOR_MODEL", GEMINI_MODEL)
REPO_SIZE_CAP_MB = float(os.environ.get("REPO_SIZE_CAP_MB", "80"))

if not GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY is not set. /chat will fail until you set it.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Codebase Investigator", version="0.1.0")
store = SessionStore()

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- request/response models ----------

class CreateRepoBody(BaseModel):
    url: str = Field(..., description="Public GitHub URL.")


class ChatBody(BaseModel):
    session_id: str
    question: str


class ResetBody(BaseModel):
    session_id: str
    keep_repo: bool = True


# ---------- routes ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": GEMINI_MODEL,
        "auditor_model": GEMINI_AUDITOR_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
        "active_sessions": len(store.list_sessions()),
    }


@app.post("/repo")
def create_repo(body: CreateRepoBody) -> dict[str, Any]:
    try:
        repo = clone_repo(body.url, size_cap_mb=REPO_SIZE_CAP_MB)
    except RepoError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("clone failed")
        raise HTTPException(status_code=500, detail=f"Internal error during clone: {e}")
    session = Session(session_id=repo.session_id, repo=repo)
    store.add(session)
    log.info("Created session %s for %s (%.1f MB)", repo.session_id, repo.slug, repo.size_mb)
    return session.repo_meta()


@app.post("/chat")
def chat(body: ChatBody) -> dict[str, Any]:
    if client is None:
        raise HTTPException(status_code=500, detail="Server has no GEMINI_API_KEY configured.")
    session = store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id (did the server restart?).")
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    session.turn_count += 1
    turn_idx = session.turn_count

    # 1. Investigator
    inv_run, next_history = investigator_mod.run_investigator(
        client=client,
        model=GEMINI_MODEL,
        repo_root=session.repo.path,
        repo_slug=session.repo.slug,
        question=question,
        history=session.history,
        claims_ledger=session.ledger.recent_strings(),
    )
    if inv_run.error and not inv_run.answer:
        # Don't bump history on hard failure.
        log.warning("Turn %d investigator error: %s", turn_idx, inv_run.error)
        return {
            "session_id": session.session_id,
            "turn": turn_idx,
            "error": inv_run.error,
            "tool_calls": [t.to_dict() for t in inv_run.tool_calls],
        }

    # 2. Programmatic citation check
    citation_report = validate_answer(inv_run.answer, session.repo.path)

    # 3. Auditor — separate call, fresh context
    audit = auditor_mod.run_auditor(
        client=client,
        model=GEMINI_AUDITOR_MODEL,
        repo_root=session.repo.path,
        question=question,
        answer=inv_run.answer,
        citation_report=citation_report,
        prior_claims=session.ledger.recent_strings(),
    )

    # 4. Update ledger from audit's extracted claims (only on non-untrustworthy)
    if audit.verdict in {"trust", "caution"} and audit.new_claims:
        session.ledger.add_many(audit.new_claims, turn=turn_idx)

    # 5. Persist updated history (slim — drops intermediate tool calls)
    session.history = next_history

    record = TurnRecord(
        turn=turn_idx,
        question=question,
        answer=inv_run.answer,
        audit=audit.to_dict(),
        tool_calls=[t.to_dict() for t in inv_run.tool_calls],
        citation_report=citation_report.to_dict(),
        investigator_elapsed_s=round(inv_run.elapsed_s, 2),
        auditor_elapsed_s=round(audit.elapsed_s, 2),
        error=inv_run.error,
    )
    session.turns.append(record)

    return {
        "session_id": session.session_id,
        "turn": turn_idx,
        "answer": inv_run.answer,
        "audit": audit.to_dict(),
        "citation_report": citation_report.to_dict(),
        "tool_calls": [t.to_dict() for t in inv_run.tool_calls],
        "investigator_elapsed_s": round(inv_run.elapsed_s, 2),
        "auditor_elapsed_s": round(audit.elapsed_s, 2),
        "ledger": session.ledger.to_list(),
        "investigator_warning": inv_run.error,
    }


@app.post("/reset")
def reset(body: ResetBody) -> dict[str, Any]:
    session = store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    if not body.keep_repo:
        cleanup_repo(session.repo)
        store.remove(body.session_id)
        return {"ok": True, "removed": True}
    session.history = []
    session.ledger.claims.clear()
    session.turns.clear()
    session.turn_count = 0
    return {"ok": True, "removed": False}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    return {
        **session.repo_meta(),
        "ledger": session.ledger.to_list(),
        "turns": [t.to_dict() for t in session.turns],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
