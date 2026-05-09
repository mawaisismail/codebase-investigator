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
from investigator.gemini_tools import build_tool

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOOL_ROUNDTRIPS = int(os.environ.get("MAX_TOOL_ROUNDTRIPS", "12"))


SYSTEM_PROMPT = """You are CodeInvestigator, a senior software engineer who answers questions about an unfamiliar codebase by reading actual files.

You have these tools, all read-only and sandboxed to the repo:
- list_tree(path, max_depth) — orient yourself in the structure
- find_files(name_glob, path) — locate candidate files
- grep(pattern, path, glob, case_insensitive) — find symbols, imports, patterns
- read_file(path, start_line, end_line) — read the exact lines you'll cite

WORKFLOW
1. Start by orienting (list_tree at root, then drill in). Don't read files until you know which ones matter.
2. Use grep/find_files to locate the entry points relevant to the question.
3. read_file the specific ranges you'll cite. NEVER cite a line you haven't read.
4. Write the answer using citations in the EXACT format below.

CITATION FORMAT (strict — your answer is checked programmatically)
- Inline citation: [path/relative/to/repo:start-end] e.g. [src/auth/login.py:42-58]
- Single line: [path:42]
- A citation MAY be followed by a fenced code block quoting the cited lines, e.g.

  [src/auth/login.py:42-46]
  ```python
  def login(req):
      ...
  ```

- Quoted snippets must match the file. Do not paraphrase inside the fence.
- Do NOT cite files or ranges you did not read with read_file. Hallucinated citations will be flagged.

ANSWER STYLE
- Concrete and specific. Pull the user toward the code, not toward generalities.
- For "how does X work?" answers, walk through the actual call path with citations.
- For evaluation/opinion questions, separate observations from your judgment. Mark opinions as such.
- For multi-turn conversations, you'll see prior assistant messages. Be consistent with what you said before; if you must contradict yourself, say so explicitly.
- Keep answers tight. A reader skimming citations should be able to follow without reading the prose.

If the question can't be answered from the code (e.g., asks about runtime behavior or stakeholders), say so plainly and explain what you can answer instead.
"""


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "ok": self.ok, "summary": self.summary}


@dataclass
class InvestigatorRun:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    raw_history_for_debug: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    roundtrips: int = 0
    error: str | None = None


def _summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    if name == "list_tree":
        return f"{result.get('entry_count', 0)} entries{'(+truncated)' if result.get('truncated') else ''}"
    if name == "read_file":
        return (
            f"{result.get('path')} lines {result.get('start_line')}-{result.get('end_line')} "
            f"of {result.get('total_lines')}"
        )
    if name == "grep":
        return f"{result.get('match_count', 0)} matches across {result.get('files_scanned', 0)} files"
    if name == "find_files":
        return f"{result.get('result_count', 0)} files matching {result.get('name_glob')}"
    return "ok"


def _safe_dispatch(name: str, args: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], ToolCallRecord]:
    try:
        result = repo_tools.dispatch(name, args, repo_root)
        return result, ToolCallRecord(
            name=name, args=args, ok=True, summary=_summarize_tool_result(name, result)
        )
    except repo_tools.ToolError as e:
        err = {"error": str(e)}
        return err, ToolCallRecord(name=name, args=args, ok=False, summary=f"ERROR: {e}")
    except Exception as e:
        err = {"error": f"unexpected: {e}"}
        return err, ToolCallRecord(name=name, args=args, ok=False, summary=f"UNEXPECTED: {e}")


def _build_user_turn(
    question: str,
    repo_slug: str,
    claims_ledger: list[str],
) -> str:
    parts = [f"REPO: {repo_slug}", f"QUESTION: {question}"]
    if claims_ledger:
        ledger_text = "\n".join(f"- {c}" for c in claims_ledger[-12:])
        parts.append(
            "PRIOR VERIFIED CLAIMS (from earlier turns; stay consistent or explicitly correct):\n"
            + ledger_text
        )
    return "\n\n".join(parts)


def run_investigator(
    *,
    client: genai.Client,
    model: str,
    repo_root: Path,
    repo_slug: str,
    question: str,
    history: list[types.Content],
    claims_ledger: list[str],
    max_roundtrips: int = MAX_TOOL_ROUNDTRIPS,
) -> tuple[InvestigatorRun, list[types.Content]]:
    started = time.time()
    tool = build_tool()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    user_text = _build_user_turn(question, repo_slug, claims_ledger)
    user_content = types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
    contents: list[types.Content] = list(history) + [user_content]

    tool_calls: list[ToolCallRecord] = []
    final_text = ""
    error: str | None = None
    roundtrips = 0

    for _ in range(max_roundtrips):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error = f"Gemini call failed: {e}"
            break

        cand = (resp.candidates or [None])[0]
        if cand is None or cand.content is None or not (cand.content.parts or []):
            error = "Empty response from model."
            break

        contents.append(cand.content)

        function_calls = [p.function_call for p in cand.content.parts if p.function_call]
        text_parts = [p.text for p in cand.content.parts if p.text]

        if function_calls:
            roundtrips += 1
            response_parts: list[types.Part] = []
            for fc in function_calls:
                args = dict(fc.args or {})
                result, record = _safe_dispatch(fc.name, args, repo_root)
                tool_calls.append(record)
                response_parts.append(
                    types.Part.from_function_response(name=fc.name, response=result)
                )
            contents.append(types.Content(role="user", parts=response_parts))
            continue

        final_text = "".join(t for t in text_parts if t).strip()
        break
    else:
        error = f"Hit max_roundtrips ({max_roundtrips}) without a final answer."

    if not final_text and not error:
        error = "Model produced no text answer."

    elapsed = time.time() - started

    if final_text:
        next_history = list(history) + [
            user_content,
            types.Content(role="model", parts=[types.Part.from_text(text=final_text)]),
        ]
    else:
        next_history = list(history)

    run = InvestigatorRun(
        answer=final_text,
        tool_calls=tool_calls,
        elapsed_s=elapsed,
        roundtrips=roundtrips,
        error=error,
    )
    return run, next_history
