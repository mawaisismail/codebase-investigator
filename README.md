# Codebase Investigator

A web app that takes a public GitHub URL, lets you ask questions in plain English, and produces answers grounded in specific files and line ranges. **Every answer ships with an independent audit** that surfaces hallucinated citations, over-confident claims, and contradictions with earlier turns.

> Built for the AgentsAnywhere senior engineer assessment. One-day budget. Stack: Python 3.12 + FastAPI + Gemini 2.5 Flash + a single-file vanilla-JS UI.

---

## What's in the box

```
codebase-investigator/
├── app.py                          # FastAPI server + orchestration
├── investigator/
│   ├── repo.py                     # Shallow-clone GitHub URLs to a sandbox
│   ├── tools.py                    # list_tree / read_file / grep / find_files (read-only, sandboxed)
│   ├── gemini_tools.py             # FunctionDeclarations the agents call
│   ├── investigator.py             # Investigator agent (tool-using loop)
│   ├── auditor.py                  # Independent auditor (separate context)
│   ├── citations.py                # Programmatic citation validator
│   ├── ledger.py                   # Claims ledger (multi-turn coherence)
│   └── session.py                  # In-memory session state
├── static/index.html               # Single-page chat UI
├── tests/test_pipeline_offline.py  # End-to-end pipeline test, no API key needed
├── requirements.txt
├── .env.example
└── README.md
```

---

## Run it

```bash
# 1. Install
pip install --user --break-system-packages -r requirements.txt
# (or in a venv if you have python3-venv)

# 2. Configure
cp .env.example .env
# edit .env and set GEMINI_API_KEY

# 3. Start
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000

# 4. Open
open http://127.0.0.1:8000
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

### Quick sanity check (no API key)

```bash
python3 tests/test_pipeline_offline.py
```

This clones a small public repo, runs the full investigator → citation-check → auditor pipeline against a scripted Gemini stand-in, and asserts the citation validator catches a hallucinated citation.

---

## How it works

```
              ┌──── user question ────────────────────────────────────────┐
              │                                                           │
              ▼                                                           │
   ┌─────────────────────┐    tool calls    ┌────────────────────┐        │
   │   Investigator      │ ───────────────▶ │   list_tree        │        │
   │   (Gemini 2.5 Flash)│ ◀─────────────── │   read_file        │        │
   │                     │    results       │   grep / find_files│        │
   │   Cites [path:L-L]  │                  └────────────────────┘        │
   └─────────────────────┘                                                │
              │ answer with citations                                     │
              ▼                                                           │
   ┌─────────────────────┐                                                │
   │ Programmatic check  │   Verifies every [path:L-L] citation:          │
   │ (citations.py)      │   • file exists                                │
   │                     │   • line range valid for the file              │
   │                     │   • optional fenced snippet matches contents   │
   └─────────────────────┘                                                │
              │ citation report                                           │
              ▼                                                           │
   ┌─────────────────────┐    SEPARATE context, fresh prompt              │
   │   Auditor           │    sees: question + answer + citation report   │
   │   (Gemini 2.5 Flash)│          + recent claims ledger + tools        │
   │                     │    does NOT see investigator's chain of        │
   │   Returns JSON:     │          thought, system prompt, tool history  │
   │   verdict / concerns│                                                │
   └─────────────────────┘                                                │
              │ structured audit                                          │
              ▼                                                           │
   ┌─────────────────────┐                                                │
   │  Claims ledger      │  Atomic verified claims persist across turns. │
   │  (ledger.py)        │  Surfaced to investigator AND auditor on every │
   │                     │  subsequent turn → catches contradictions.     │
   └─────────────────────┘ ─────────────────────────────────────────────▶ │
```

### The two requirements, mapped to code

#### 1. "Every answer ships with its own audit" — and the self-scoring rule

The brief is explicit: a confidence score in the same call that produced the answer is noise, not a signal. We use **three independent checks**, none of which is the investigator scoring itself:

| Check | Mechanism | Source file | What it catches |
|---|---|---|---|
| **Citation validator** | Pure-Python regex parse + filesystem check + byte-match of fenced snippets | `investigator/citations.py` | Hallucinated paths, invalid line ranges, fabricated code blocks |
| **Independent auditor** | Second Gemini call with a fresh prompt, fresh context, no access to investigator's reasoning | `investigator/auditor.py` | Over-confident claims, missing context, fixes that break callers, contradictions |
| **Claims ledger** | Atomic verified claims extracted after each turn, persisted across turns, surfaced to both agents | `investigator/ledger.py` | Self-contradiction across turns ("you said Z earlier") |

The auditor returns structured JSON (`verdict ∈ {trust, caution, untrustworthy}` + concerns + contradictions + verified claims). The UI renders the verdict as a colored pill, lists concerns inline under the answer, and crosses out citations that failed the programmatic check.

#### 2. "Stay sharp over many turns"

- **Conversation history**: the investigator sees prior user turns + its own prior final answers (intermediate tool calls are dropped to keep context lean — Gemini Flash has 1M context but free-tier rate limits make slim history a virtue).
- **Claims ledger**: at the top of every investigator and auditor prompt we inject the last 12 verified claims. The investigator is instructed to stay consistent or explicitly correct itself. The auditor checks each new turn against the ledger and surfaces contradictions in the JSON output.

---

## Key design choices

**No embeddings / RAG.** Repos under ~80 MB are small enough to navigate by structure + grep. This makes citations honest — the model has to actually *find* and *read* the code, not paraphrase a vector-similarity match. It also avoids the failure mode where chunked snippets lose line-number fidelity.

**Strict citation format** (`[path/to/file.py:42-58]`). One regex parses them. The model is instructed to emit exactly this format and is told that hallucinated citations are programmatically flagged. This is the single highest-leverage decision in the project — it makes trust verifiable without involving an LLM.

**Auditor never sees the investigator's reasoning.** It sees the user's question, the final answer, the citation report, and the prior claims ledger — that's it. It re-derives evidence from the repo using the same tools. This is the "different context" requirement of the brief, taken literally.

**Slim history across turns.** We persist the user message and the model's *final* text answer per turn, not the intermediate tool calls. Tool roundtrips on a 12-turn conversation would balloon context fast on free tier; the ledger captures what actually needs to persist.

**Read-only, sandboxed tools.** Every tool resolves paths against the repo root and rejects anything that escapes it. The agent cannot list `/etc`, write files, or shell out.

**Two-call audit (verdict + claims extraction).** A single audit call returning both verdict JSON *and* a claims list invites the model to conflate "what I'm flagging" with "what was asserted." Splitting the calls keeps each one's job legible. The claims-extraction call is cheap (no tools, JSON-mode) and failures are non-fatal.

---

## What I cut for time

- **Persistent storage.** Sessions live in process memory. Restart the server, sessions are gone. For a real product I'd back them with SQLite + a workspaces directory you can clean up on a TTL.
- **Streaming responses.** The UI shows a "Investigating, then auditing…" placeholder. Streaming the investigator's tool calls live would be nicer but adds frontend complexity.
- **Auth, multi-user, rate-limiting.** Single-user local app. Don't expose this to the internet without adding those.
- **More tools.** I considered `read_diff_history` (compare a function against its prior versions in `git log -L`) and `find_callers` (a syntax-aware lookup, not just grep). These would help the auditor catch "this fix would break a caller" — instead I rely on the auditor running `grep` itself for call sites.
- **Per-language smarts.** No tree-sitter, no LSP. Pure file-text. The model's pre-training does the heavy lifting for language structure.
- **A nicer "diff" view of disagreements.** When the auditor disagrees with the investigator, both views are shown side-by-side rather than reconciled. A "challenge" round (investigator gets to respond) is the obvious next step.

---

## What I kept that I'm proud of

- The citation validator is **pure Python with no LLM calls** and runs in milliseconds. It's the deterministic floor on trust — if a citation hallucinates, no LLM judgment is needed.
- The auditor's JSON contract has a real rubric in the system prompt (`trust`/`caution`/`untrustworthy` with explicit criteria) so the verdicts mean something. The default landing zone is `caution` — most non-trivial answers should land there.
- Claims ledger entries are auto-extracted by a tiny third LLM call with `response_mime_type=application/json`. They're scoped, verifiable sentences with their own citations — not a vague conversation summary.
- The frontend shows everything: per-citation verification status, the audit verdict pill, concerns with severity, contradictions, suggested follow-ups, the claims ledger, and a collapsible list of every tool call the investigator made. The reviewer can see exactly what the system did and why.

---

## Tradeoffs I made consciously

- **Free-tier Gemini is rate-limited.** A turn typically takes 20–60 seconds because we make 1 investigator call (with 3–8 tool roundtrips) plus 1 auditor call (with its own tool roundtrips) plus 1 claims-extraction call. For a paid key, this drops considerably. I capped roundtrips at 12 (investigator) and 8 (auditor) so a runaway loop fails fast.
- **Repo size cap of 80 MB** (configurable). This catches most real projects without letting the agent get lost in vendored dependencies. The agent also sees `.git`, `node_modules`, `vendor`, etc. excluded from `list_tree` and `grep`.
- **The auditor's tool budget is smaller than the investigator's** (8 vs 12 roundtrips). Audits should spot-check, not re-investigate. If the auditor needs more than 8 tool calls, the answer was probably under-specified.

---

## Testing the example questions from the brief

Once the server's running, paste a public repo URL and try:

1. *"How does auth work here, and what would you change about it?"* — pick a repo with auth (e.g. a Django sample). Watch the investigator find middleware/decorators and the auditor verify the call paths.
2. *"This signup flow feels off — walk me through it and flag anything risky."* — note that the answer separates description from judgment.
3. *"Is there dead code? What's safe to delete?"* — investigator typically greps for unused exports; auditor often pushes back ("did you check the test files?").
4. *"Why is this function async? Does it need to be?"* — pick a specific function, follow with "actually, what about the caller in X?" — see the ledger keep the prior context alive.
5. *"Suggest a better way to handle errors in the API layer."* — auditor commonly downgrades to `caution` when fixes would break callers.
6. *"Walk me through what this service does. Skip the obvious."* — overview-style; ledger fills up; subsequent turns ride on those facts.

The 8–15 turn conversation arc the brief describes is exactly what the ledger is for.
