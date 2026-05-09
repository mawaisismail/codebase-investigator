"""Auditor agent — second-opinion review of the investigator's answer.

Design constraints from the brief:
- Audit must come from a SEPARATE context — not self-scoring in the same call.
- We deliberately give the auditor the user's question + the final answer +
  tool access, but NOT the investigator's chain of thought, system prompt,
  or its tool history. It re-derives its own evidence.
- We pre-run a programmatic citation check and pass the result to the auditor
  so cheap deterministic findings don't burn tokens.
- We pass recent prior claims so the auditor can flag self-contradictions.

Output is structured JSON. The UI renders it as a trust badge + bullets.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from investigator import tools as repo_tools
from investigator.citations import CitationReport
from investigator.gemini_tools import build_tool

DEFAULT_MODEL = os.environ.get("GEMINI_AUDITOR_MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
MAX_TOOL_ROUNDTRIPS = int(os.environ.get("AUDITOR_MAX_ROUNDTRIPS", "8"))


SYSTEM_PROMPT = """You are CodeAuditor, an independent reviewer. Your job is to decide whether another engineer's answer about a codebase is trustworthy.

You see ONLY:
- The user's question
- The other engineer's final answer (with its citations)
- A pre-flight programmatic check of those citations
- A short list of prior verified claims from earlier turns (for contradiction-checking)
- The same read-only tools (list_tree, find_files, grep, read_file)

You do NOT see their reasoning or what files they read. You re-verify the claims yourself.

YOUR PROCESS
1. Read the answer. Identify its specific factual claims and any opinions/recommendations.
2. For each non-trivial factual claim, USE THE TOOLS to independently confirm it. Spot-check 2-4 of the most load-bearing claims; you don't need to verify everything.
3. Note any citation issues from the programmatic check — those are facts, not your judgment.
4. Check for contradictions against the prior verified claims list.
5. Look for: hallucinated citations, over-confident claims, suggested fixes that would break something else (consider call sites, callers of the changed function), missing context, reasoning gaps.

OUTPUT — return ONLY valid JSON, no prose, no markdown fences:

{
  "verdict": "trust" | "caution" | "untrustworthy",
  "headline": "one sentence summarizing the audit",
  "verified": ["claim 1 you independently confirmed", "claim 2", ...],
  "concerns": [
    {"severity": "low" | "med" | "high", "issue": "what's wrong", "evidence": "what you found via tools, with [path:lines] where relevant"}
  ],
  "contradictions": ["specific contradiction with a prior claim, or empty list"],
  "suggested_followups": ["short user-facing question or check, or empty list"]
}

VERDICT RUBRIC
- "trust": claims and citations check out, no high-severity concerns, no contradictions.
- "caution": citations mostly fine but one or more medium concerns OR opinions stated as facts OR a suggested fix that overlooks a caller. Most answers should land here.
- "untrustworthy": hallucinated citation, materially wrong factual claim, or direct contradiction with a prior claim that wasn't acknowledged.

Be concise. Each concern's "evidence" must be a SHORT sentence with at least one [path:line] reference where applicable. If you have no concerns, return concerns: [].
"""


@dataclass
class AuditResult:
    verdict: str  # trust | caution | untrustworthy | error
    headline: str
    verified: list[str] = field(default_factory=list)
    concerns: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    suggested_followups: list[str] = field(default_factory=list)
    citation_report: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    raw_response: str | None = None
    error: str | None = None
    new_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "verified": self.verified,
            "concerns": self.concerns,
            "contradictions": self.contradictions,
            "suggested_followups": self.suggested_followups,
            "citation_report": self.citation_report,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }


def _build_audit_user_message(
    question: str,
    answer: str,
    citation_report: CitationReport,
    prior_claims: list[str],
) -> str:
    cit_lines = [f"- {c.raw}: " + ("OK" if c.is_ok() else (c.issue or "issue"))
                 for c in citation_report.checks] or ["- (no citations found in answer)"]
    claims_block = (
        "\n".join(f"- {c}" for c in prior_claims[-12:])
        if prior_claims else "(none)"
    )
    return (
        f"USER QUESTION:\n{question}\n\n"
        f"ENGINEER'S ANSWER:\n{answer}\n\n"
        f"PROGRAMMATIC CITATION CHECK ({citation_report.summary()}):\n"
        + "\n".join(cit_lines)
        + f"\n\nPRIOR VERIFIED CLAIMS (most recent {min(len(prior_claims), 12)}):\n{claims_block}\n\n"
        "Now use the tools to independently verify the most load-bearing claims, then return the JSON."
    )


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove first fence line and trailing fence
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _claims_extraction_prompt(answer: str) -> str:
    return (
        "Extract 1-5 atomic factual claims from the engineer's answer below. Each claim should be a "
        "short, self-contained, verifiable sentence (≤140 chars), e.g. 'Auth tokens are stored in Redis "
        "with a 24h TTL [src/auth/store.py:42-58]'. Skip opinions and recommendations. "
        "Return ONLY a JSON array of strings, no prose.\n\nANSWER:\n" + answer
    )


def _extract_new_claims(client: genai.Client, model: str, answer: str) -> list[str]:
    """A tiny third call to distill claims for the ledger.

    This is intentionally separate from the audit so the audit's "concerns" and
    the ledger's "what was asserted" don't get tangled. Failures here are
    non-fatal — the ledger just doesn't grow.
    """
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=_claims_extraction_prompt(answer))])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()][:5]
    except Exception:
        return []
    return []


def run_auditor(
    *,
    client: genai.Client,
    model: str,
    repo_root: Path,
    question: str,
    answer: str,
    citation_report: CitationReport,
    prior_claims: list[str],
    max_roundtrips: int = MAX_TOOL_ROUNDTRIPS,
) -> AuditResult:
    started = time.time()
    tool = build_tool()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    user_text = _build_audit_user_message(question, answer, citation_report, prior_claims)
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
    ]

    final_text = ""
    error: str | None = None

    for _ in range(max_roundtrips):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error = f"Auditor Gemini call failed: {e}"
            break

        cand = (resp.candidates or [None])[0]
        if cand is None or cand.content is None or not (cand.content.parts or []):
            error = "Auditor produced empty response."
            break
        contents.append(cand.content)

        function_calls = [p.function_call for p in cand.content.parts if p.function_call]
        text_parts = [p.text for p in cand.content.parts if p.text]

        if function_calls:
            response_parts: list[types.Part] = []
            for fc in function_calls:
                args = dict(fc.args or {})
                try:
                    result = repo_tools.dispatch(fc.name, args, repo_root)
                except repo_tools.ToolError as e:
                    result = {"error": str(e)}
                response_parts.append(
                    types.Part.from_function_response(name=fc.name, response=result)
                )
            contents.append(types.Content(role="user", parts=response_parts))
            continue

        final_text = "".join(t for t in text_parts if t).strip()
        break
    else:
        error = f"Auditor hit max_roundtrips ({max_roundtrips})."

    elapsed = time.time() - started

    if error and not final_text:
        return AuditResult(
            verdict="error",
            headline="Audit could not complete.",
            error=error,
            elapsed_s=elapsed,
            citation_report=citation_report.to_dict(),
        )

    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(_strip_json_fences(final_text))
        if not isinstance(parsed, dict):
            raise ValueError("audit response was not a JSON object")
    except Exception as e:
        return AuditResult(
            verdict="error",
            headline="Auditor returned malformed JSON.",
            error=f"{e}: {final_text[:300]}",
            elapsed_s=elapsed,
            raw_response=final_text,
            citation_report=citation_report.to_dict(),
        )

    verdict = str(parsed.get("verdict", "caution")).lower()
    if verdict not in {"trust", "caution", "untrustworthy"}:
        verdict = "caution"

    new_claims = _extract_new_claims(client, model, answer)

    return AuditResult(
        verdict=verdict,
        headline=str(parsed.get("headline", "")).strip() or "Audit complete.",
        verified=[str(x) for x in (parsed.get("verified") or [])],
        concerns=[c for c in (parsed.get("concerns") or []) if isinstance(c, dict)],
        contradictions=[str(x) for x in (parsed.get("contradictions") or [])],
        suggested_followups=[str(x) for x in (parsed.get("suggested_followups") or [])],
        citation_report=citation_report.to_dict(),
        elapsed_s=elapsed,
        raw_response=final_text,
        new_claims=new_claims,
    )
