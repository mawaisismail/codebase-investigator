from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google.genai import types  # noqa: E402

from investigator.repo import clone_repo  # noqa: E402
from investigator import investigator as inv_mod  # noqa: E402
from investigator import auditor as audit_mod  # noqa: E402
from investigator.citations import validate_answer  # noqa: E402
from investigator.ledger import ClaimsLedger  # noqa: E402


def _content(parts):
    return types.Content(role="model", parts=parts)


def _resp_with_parts(parts):
    cand = MagicMock()
    cand.content = _content(parts)
    resp = MagicMock()
    resp.candidates = [cand]
    resp.text = "".join(p.text for p in parts if getattr(p, "text", None))
    return resp


def make_scripted_client(scripted_responses):
    queue = list(scripted_responses)
    client = MagicMock()

    def gen(model, contents, config):
        if not queue:
            raise AssertionError(
                f"Unexpected extra Gemini call (already used {len(scripted_responses)} responses)."
            )
        parts = queue.pop(0)
        return _resp_with_parts(parts)

    client.models = MagicMock()
    client.models.generate_content.side_effect = gen
    return client


def test_pipeline_end_to_end():
    repo = clone_repo("https://github.com/sindresorhus/is-online", size_cap_mb=20)
    print(f"cloned {repo.slug} → {repo.path}")

    investigator_script = [
        [types.Part.from_function_call(name="list_tree", args={"path": ".", "max_depth": 2})],
        [types.Part.from_function_call(name="read_file", args={"path": "package.json", "start_line": 1, "end_line": 30})],
        [types.Part.from_text(
            text=(
                "This package exports an `isOnline` function that pings DNS/HTTPS endpoints to check connectivity.\n\n"
                "[package.json:1-15] declares the package metadata.\n\n"
                "[ghosts/notreal.js:1-3] this citation is hallucinated.\n"
            )
        )],
    ]

    inv_client = make_scripted_client(investigator_script)
    run, history = inv_mod.run_investigator(
        client=inv_client,
        model="gemini-fake",
        repo_root=repo.path,
        repo_slug=repo.slug,
        question="What does this package do?",
        history=[],
        claims_ledger=[],
    )
    assert run.error is None, run.error
    assert "isOnline" in run.answer
    assert len(run.tool_calls) == 2
    assert run.tool_calls[0].name == "list_tree"
    assert run.tool_calls[0].ok is True
    assert run.tool_calls[1].name == "read_file"
    assert run.tool_calls[1].ok is True
    print(f"investigator OK — {run.roundtrips} tool roundtrips, answer={len(run.answer)} chars")

    cit = validate_answer(run.answer, repo.path)
    assert cit.total == 2, cit.to_dict()
    assert cit.ok_count == 1
    assert any(not c.is_ok() and "ghosts/notreal.js" in c.path for c in cit.checks)
    print(f"citation validator caught hallucination: {cit.summary()}")

    audit_script = [
        [types.Part.from_function_call(name="read_file", args={"path": "package.json", "start_line": 1, "end_line": 30})],
        [types.Part.from_text(text=json.dumps({
            "verdict": "untrustworthy",
            "headline": "One citation is hallucinated.",
            "verified": ["package.json declares is-online"],
            "concerns": [
                {"severity": "high", "issue": "hallucinated citation", "evidence": "ghosts/notreal.js does not exist"}
            ],
            "contradictions": [],
            "suggested_followups": []
        }))],
        [types.Part.from_text(text=json.dumps([
            "is-online package exports isOnline() [package.json:1-15]"
        ]))],
    ]
    audit_client = make_scripted_client(audit_script)
    audit = audit_mod.run_auditor(
        client=audit_client,
        model="gemini-fake",
        repo_root=repo.path,
        question="What does this package do?",
        answer=run.answer,
        citation_report=cit,
        prior_claims=[],
    )
    assert audit.verdict == "untrustworthy", audit.to_dict()
    assert audit.concerns and audit.concerns[0]["severity"] == "high"
    assert audit.new_claims == ["is-online package exports isOnline() [package.json:1-15]"]
    print(f"auditor OK — verdict={audit.verdict}, {len(audit.concerns)} concerns, {len(audit.new_claims)} new claims")

    ledger = ClaimsLedger()
    ledger.add_many(audit.new_claims, turn=1)
    assert len(ledger.claims) == 1
    print(f"ledger OK — recent: {ledger.recent_strings()}")

    import shutil
    shutil.rmtree(repo.path, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    test_pipeline_end_to_end()
