# SR&ED Automation Platform — Comprehensive Handoff Document

**Generated:** 2026-02-17  
**Status:** Completed through **Increment 10** (of 12)  
**Test suite:** 84 tests, all passing  
**Tooling:** Python 3.12+, uv, SQLite, SQLModel, Streamlit, OpenAI API, DuckDB

> For the full product vision, optimization goals, and post-Prompt-12 roadmap, see `project.md` in the repo root.

---

## 0. Platform Overview — What This System Does

### The Problem

Canadian companies claiming the **Scientific Research & Experimental Development (SR&ED)** tax credit must assemble complex evidence packages: timesheets, payroll stubs, invoices, Jira exports, technical documents, and more. The evidence is messy, inconsistent, and spread across dozens of files. Preparing a claim today is a manual, error-prone process that takes consultants weeks of effort per client.

### The Solution

This platform is an **AI-powered SR&ED claim preparation system** that automates the heavy lifting while keeping humans in control of every consequential decision. It is **not** a one-pass parser — it is a **reasoning system** that:

1. **Ingests** messy client evidence (CSV timesheets, PDF payroll stubs, DOCX technical docs, images, Jira exports)
2. **Extracts** structured data using OCR/vision models and LLM-driven structured extraction
3. **Builds a world model** — a consistent, queryable representation of people, hours, pay periods, contradictions, and evidence
4. **Resolves ambiguity** through identity resolution (fuzzy name matching), schema mapping hypotheses, and human-in-the-loop review
5. **Validates cross-source consistency** (e.g., payroll vs. timesheet totals) and flags mismatches as blocking contradictions
6. **Computes scenario totals** for claimable labour hours at different confidence thresholds
7. **Produces an auditable export package** with citations, narratives, and evidence provenance — but only after all blocking issues are resolved

### End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. SETUP                                                           │
│     Create a Run → Add People (name, role, hourly rate)             │
│                                                                     │
│  2. INGEST                                                          │
│     Upload files (CSV, PDF, DOCX, TXT, images)                     │
│     → Vision OCR for PDFs/images → ExtractionArtifacts              │
│     → CSV profiling via DuckDB → StagingRows                        │
│     → Text chunking → Segments (searchable)                         │
│                                                                     │
│  3. UNDERSTAND                                                      │
│     Schema mapping: CSV columns → target fields (LLM hypotheses)    │
│     Identity resolution: raw names → canonical Person records       │
│       (fuzzy matching + human confirmation → PersonAlias)           │
│     Hybrid search: FTS5 + vector embeddings + RRF fusion            │
│                                                                     │
│  4. VALIDATE                                                        │
│     Payroll extraction: vision artifacts → structured pay periods    │
│     Cross-validation: payroll hours vs. timesheet hours per period   │
│     Mismatch > 5% → BLOCKING contradiction + ReviewTask             │
│     Missing rates → BLOCKING contradiction                          │
│     Run status → NEEDS_REVIEW until all blockers resolved           │
│                                                                     │
│  5. REVIEW (Human-in-the-loop)                                      │
│     Resolve contradictions → ReviewDecision → DecisionLock          │
│     Confirm/reject alias mappings                                   │
│     Supersede previous decisions if needed                          │
│     Agent respects all locks — cannot reopen resolved issues        │
│                                                                     │
│  6. OPTIMIZE (Increments 9, 11 — in progress)                      │
│     Promote staging rows → LedgerLabourHour (UNSORTED bucket)       │
│     Partial inclusion fractions (0.0–1.0)                           │
│     Scenario totals: Conservative / Balanced / Aggressive           │
│     Group hours into Claim Projects                                 │
│                                                                     │
│  7. EXPORT (Increment 12 — planned)                                 │
│     Narrative drafting with evidence-bounded citations               │
│     Export bundle: JSON + Markdown + CSV + citation index            │
│     Hard-blocked if evidence gaps or BLOCKING contradictions remain  │
└─────────────────────────────────────────────────────────────────────┘
```

### Architecture at a Glance

| Layer | Components | Purpose |
|---|---|---|
| **UI** | Streamlit multipage app (8 pages) | Run management, people intake, file upload, search, CSV tools, agent runner, tasks & gates, payroll validation |
| **Agent** | OpenAI tool-calling loop + 18 registered tools | Autonomous multi-step reasoning with full observability (every LLM call and tool invocation logged to DB) |
| **Tools** | Ingest, search, CSV intelligence, people, aliases, payroll, tasks, contradictions, locks, memory | Safe operations layer — the agent never writes raw SQL or modifies the DB directly |
| **Models** | 17 SQLModel tables + 2 FTS5 virtual tables | Canonical data, staging, world model, observability, search indices |
| **Search** | FTS5 (BM25) + OpenAI embeddings + hybrid RRF fusion | Cross-file coherence without giant context windows |
| **LLM** | OpenAI API (4 model roles) | Vision OCR, embeddings, agent reasoning, structured JSON extraction |
| **Storage** | SQLite + local filesystem | Everything persisted locally — data never leaves the machine except for OpenAI API calls |

### Key Design Principles

- **People-first anchors** — People records are the foundation; rates must be set before export
- **Conservative branching** — Competing hypotheses only when ambiguity is material
- **Human decisions are final** — DecisionLocks prevent the agent from reopening resolved issues
- **Idempotency everywhere** — File dedup by SHA256, staging dedup by row hash, task dedup by issue_key, embedding dedup by entity key
- **Observability built-in** — Every LLM call and tool invocation logged with tokens, duration, args, and results
- **Hard gates before export** — Blocking contradictions and evidence gaps must be resolved before any output is generated
- **Local-first** — All data persisted in SQLite + filesystem; external calls only to OpenAI API

---

## 1. Quick Start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv)
uv sync

# Create .env
echo 'OPENAI_API_KEY=sk-...' > .env

# Initialize database
uv run sred db init

# Run tests
uv run pytest

# Launch UI
uv run streamlit run streamlit_app.py

# CLI health check
uv run sred doctor
```

---

## 2. Repository Structure

```
SRED_v5/
├── .env                          # OpenAI key + config overrides (gitignored)
├── .gitignore
├── pyproject.toml                # uv/hatch project config, pytest settings
├── uv.lock                       # Locked dependencies
├── streamlit_app.py              # Streamlit entrypoint (multipage nav)
├── project.md                    # Full product spec (vision, decisions, roadmap)
├── HANDOFF.md                    # THIS FILE
├── README.md                     # Setup instructions
├── data/                         # Runtime data dir (DB + uploaded files)
│   └── sred.db                   # SQLite database (created by `sred db init`)
├── src/sred/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings (env-driven)
│   ├── db.py                     # SQLite engine, init_db(), get_session()
│   ├── logging.py                # stdlib logging with run_id context var
│   ├── cli.py                    # Typer CLI: doctor, db init, db reindex, db search
│   ├── gates.py                  # Gate logic: blocking checks, run status transitions
│   │
│   ├── models/
│   │   ├── __init__.py           # Re-exports all models
│   │   ├── base.py               # TimestampMixin, ProvenanceMixin
│   │   ├── core.py               # Run, Person, File, Segment (+ enums)
│   │   ├── finance.py            # StagingRow, LedgerLabourHour, PayrollExtract
│   │   ├── hypothesis.py         # Hypothesis, StagingMappingProposal
│   │   ├── artifact.py           # ExtractionArtifact
│   │   ├── vector.py             # VectorEmbedding (BLOB storage)
│   │   ├── memory.py             # MemoryDoc (markdown memory)
│   │   ├── world.py              # Contradiction, ReviewTask, ReviewDecision, DecisionLock
│   │   ├── alias.py              # PersonAlias (identity resolution mappings)
│   │   └── agent_log.py          # ToolCallLog, LLMCallLog
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── registry.py           # Tool registry (register_tool, get_openai_tools_schema)
│   │   ├── runner.py             # OpenAI tool-calling loop (run_agent_loop)
│   │   └── tools.py              # 18 registered tool implementations
│   │
│   ├── ingest/
│   │   ├── csv_intel.py          # DuckDB: csv_profile, csv_query, propose_schema_mapping
│   │   ├── process.py            # process_source_file (PDF/CSV/DOCX/TXT/image dispatcher)
│   │   ├── segment.py            # chunk_text, create_text_segments, process_csv_content
│   │   └── vision.py             # pdf_to_images, vision_extract_pdf/image (OpenAI Vision)
│   │
│   ├── llm/
│   │   └── openai_client.py      # OpenAI client: get_chat_completion, get_vision_completion
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── fts.py                # FTS5 setup, reindex, search_segments
│   │   ├── embeddings.py         # OpenAI embeddings, store_embeddings (cached)
│   │   ├── vector_search.py      # cosine_similarity, batch search
│   │   └── hybrid_search.py      # RRF fusion, hybrid_search, SearchResult dataclass
│   │
│   ├── storage/
│   │   └── files.py              # save_upload, sanitize_filename, compute_sha256
│   │
│   └── ui/
│       ├── state.py              # Streamlit session state helpers (run_id context)
│       ├── validation.py         # Pre-flight checks (schema, data dir, DB connection)
│       └── pages/
│           ├── 1_run.py          # Create/select runs
│           ├── 2_people.py       # Add people, set rates
│           ├── 3_uploads.py      # Upload files, trigger processing
│           ├── 4_dashboard.py    # Run status metrics, readiness checklist
│           ├── 5_search.py       # Hybrid/FTS/Vector search UI
│           ├── 6_csv_tools.py    # CSV profiling, SQL console, schema hypotheses
│           ├── 7_agent.py        # Agent runner UI (run loop, trace, logs)
│           ├── 8_tasks.py        # Tasks & Gates (contradictions, resolve, locks, supersede)
│           └── 9_payroll.py      # Payroll validation (extracts, mismatch breakdown, contradictions)
│
└── tests/
    ├── test_agent.py             # 64 tests: tools, registry, gates, locks, agent loop, aliases, payroll, context
    ├── test_csv_intel.py         # 3 tests: DuckDB profile, query, schema proposal
    ├── test_db.py                # 4 tests: core/finance/artifact/vector models
    ├── test_ingest.py            # 3 tests: chunking, text processing, PDF processing
    ├── test_search_logic.py      # 4 tests: hash, cosine, vector storage, RRF
    ├── test_smoke.py             # 2 tests: settings load, CLI doctor
    └── test_ui_logic.py          # 4 tests: sanitization, hashing, upload, validation
```

---

## 3. Configuration

All settings are in `src/sred/config.py` via `pydantic-settings`. Loaded from `.env` or environment variables.

| Setting | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key (SecretStr) |
| `OPENAI_MODEL_AGENT` | `gpt-5` | Agent planning + tool calling |
| `OPENAI_MODEL_VISION` | `gpt-5-mini` | Vision/OCR extraction |
| `OPENAI_MODEL_STRUCTURED` | `gpt-4o-2024-08-06` | Strict JSON outputs |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embeddings (config only; code uses `text-embedding-3-small` in `embeddings.py`) |
| `PAYROLL_MISMATCH_THRESHOLD` | `0.05` | 5% threshold for blocking contradiction |

**Note:** There is a mismatch between `config.py` (`text-embedding-3-large`) and `search/embeddings.py` (`text-embedding-3-small`). The code in `embeddings.py` hardcodes the model string rather than reading from settings. This should be unified in a future increment.

---

## 4. Data Model (Complete)

### 4.1 Base Mixins (`models/base.py`)

- **TimestampMixin** — `created_at`, `updated_at` (UTC, auto-set)
- **ProvenanceMixin** — `source_file_id`, `page_number`, `row_number`

### 4.2 Core (`models/core.py`)

| Model | Key Fields | Notes |
|---|---|---|
| **Run** | `id`, `name`, `status` | Status enum: `INITIALIZING`, `PROCESSING`, `NEEDS_REVIEW`, `COMPLETED`, `FAILED` |
| **Person** | `run_id`, `name`, `role`, `hourly_rate`, `rate_status` | Rate status: `PENDING` / `SET`. People-first anchor. |
| **File** | `run_id`, `path`, `original_filename`, `mime_type`, `size_bytes`, `content_hash`, `status` | Status: `UPLOADED` / `PROCESSED` / `ERROR`. Deduped by `content_hash` per run. |
| **Segment** | `file_id`, `run_id`, `content`, `status` | Inherits ProvenanceMixin. Chunked text for search. |

### 4.3 Finance (`models/finance.py`)

| Model | Key Fields | Notes |
|---|---|---|
| **StagingRow** | `run_id`, `raw_data` (JSON), `status`, `row_type`, `row_hash`, `normalized_text` | Row types: `UNKNOWN`, `TIMESHEET`, `PAYROLL`, `INVOICE`, `JIRA`. Inherits ProvenanceMixin. |
| **LedgerLabourHour** | `run_id`, `person_id`, `date`, `hours`, `description`, `bucket`, `inclusion_fraction`, `confidence` | Bucket defaults to `UNSORTED`. **Not yet populated by any tool** — this is Increment 9. |
| **PayrollExtract** | `run_id`, `file_id`, `period_start`, `period_end`, `total_hours`, `total_wages`, `currency`, `employee_count`, `confidence`, `raw_json` | Structured payroll data extracted via LLM from vision artifacts. Unique constraint on `(run_id, file_id, period_start, period_end)`. |

### 4.4 Hypothesis (`models/hypothesis.py`)

| Model | Key Fields | Notes |
|---|---|---|
| **Hypothesis** | `run_id`, `type`, `description`, `status`, `parent_id` | Types: `CSV_SCHEMA`, `CLAIM_CLUSTERING`. Status: `ACTIVE`/`REJECTED`/`ACCEPTED`. Self-referencing FK for branching. |
| **StagingMappingProposal** | `hypothesis_id`, `file_id`, `mapping_json`, `confidence`, `reasoning` | CSV column→target field mapping proposals. Has `mapping` property for JSON↔dict. |

### 4.5 World Model (`models/world.py`) — Increment 8

| Model | Key Fields | Notes |
|---|---|---|
| **Contradiction** | `run_id`, `issue_key`, `contradiction_type`, `severity`, `status` | Types: `MISSING_RATE`, `PAYROLL_MISMATCH`, `UNKNOWN_BASIS`, `MISSING_EVIDENCE`, `OTHER`. Severity: `LOW`→`BLOCKING`. Deduped by `issue_key`. |
| **ReviewTask** | `run_id`, `issue_key`, `title`, `description`, `severity`, `status`, `contradiction_id` | Status: `OPEN`/`RESOLVED`/`SUPERSEDED`. Deduped by `issue_key`. |
| **ReviewDecision** | `run_id`, `task_id`, `decision`, `decided_by` | Free-text human resolution. `decided_by`: `HUMAN` or `SYSTEM`. |
| **DecisionLock** | `run_id`, `issue_key`, `decision_id`, `reason`, `active` | Prevents re-opening resolved issues. `active=False` when superseded. |

### 4.6 Identity Resolution (`models/alias.py`) — Increment 9a

| Model | Key Fields | Notes |
|---|---|---|
| **PersonAlias** | `run_id`, `person_id`, `alias`, `source`, `confidence`, `status` | Maps raw name variants to canonical Person records. Status: `PROPOSED`/`CONFIRMED`/`REJECTED`. Unique constraint on `(run_id, alias)`. |

### 4.7 Other Models

| Model | File | Notes |
|---|---|---|
| **ExtractionArtifact** | `models/artifact.py` | Vision extraction outputs. Kinds: `VISION_TEXT`, `VISION_TABLES_JSON`. Links to File + Segments. |
| **VectorEmbedding** | `models/vector.py` | BLOB float32 storage. Unique constraint on `(entity_type, entity_id, model)`. Entity types: `SEGMENT`, `STAGING_ROW`, `MEMORY_MD`. |
| **MemoryDoc** | `models/memory.py` | Markdown memory documents. Keyed by `run_id` + `path`. Idempotent writes via `content_hash`. |
| **ToolCallLog** | `models/agent_log.py` | Every agent tool invocation: name, args, result, success, duration_ms. |
| **LLMCallLog** | `models/agent_log.py` | Every OpenAI API call: model, tokens, finish_reason, prompt_summary. |

---

## 5. Agent System

### 5.1 Tool Registry (`agent/registry.py`)

Global dict-based registry. Each tool has: `name`, `description`, `parameters` (JSON Schema), `handler` (callable).

- `register_tool(name, description, parameters, handler)` — registers a tool
- `get_openai_tools_schema()` — returns OpenAI function-calling format
- `get_tool_handler(name)` — returns the handler callable

### 5.2 Registered Tools (`agent/tools.py`)

All handlers have signature: `handler(session: Session, run_id: int, **kwargs) -> dict`

| Tool | Purpose | Key Behavior |
|---|---|---|
| `ingest_process_file` | Process uploaded file | Idempotent (skips if PROCESSED) |
| `search_hybrid` | FTS + vector search | Returns ranked snippets |
| `csv_profile` | DuckDB CSV profiling | Returns columns, types, sample rows, row count |
| `csv_query` | Read-only SQL on CSV | Table alias `df`, capped at 50 rows |
| `people_list` | List people in run | — |
| `people_get` | Get person details | Run-scoped validation |
| `tasks_list_open` | List open ReviewTasks | Optional severity filter |
| `tasks_create` | Create ReviewTask | **Deduped by issue_key, blocked by DecisionLock** |
| `contradictions_list_open` | List open Contradictions | — |
| `contradictions_create` | Create Contradiction | **Deduped by issue_key, blocked by DecisionLock** |
| `locks_list_active` | List active DecisionLocks | — |
| `memory_write_summary` | Write/update MemoryDoc | Idempotent by path + content_hash |
| `aliases_resolve` | Fuzzy-match raw names to Person | Extracts distinct names from TIMESHEET StagingRows, proposes matches. **Read-only.** |
| `aliases_confirm` | Persist alias→Person mapping | Idempotent (updates if alias exists). Validates person belongs to run. |
| `aliases_list` | List all PersonAlias records | — |
| `payroll_extract` | LLM-extract payroll periods from vision artifacts | Structured JSON extraction via `OPENAI_MODEL_STRUCTURED`. Idempotent per period. |
| `payroll_validate` | Compare payroll vs timesheet totals | >5% mismatch → BLOCKING Contradiction + ReviewTask + gate update. Deduped. |
| `payroll_summary` | List PayrollExtract records | — |

### 5.3 Agent Runner (`agent/runner.py`)

`run_agent_loop(session, run_id, user_message, max_steps=10, context_notes=None) -> AgentResult`

- Uses `OPENAI_MODEL_AGENT` with OpenAI tool-calling
- Multi-step loop: LLM → tool calls → feed results back → repeat
- Stops on: plain text response (`complete`), max steps (`max_steps`), or API error (`error`)
- Logs every LLM call to `LLMCallLog` and every tool call to `ToolCallLog`
- Returns `AgentResult` with full step trace

### 5.4 Dynamic System Prompt

The system prompt is assembled dynamically from three parts:

1. **Static rules** — capabilities list, behavioral rules (never write raw SQL, use tools only, etc.)
2. **`## Current Run State`** — auto-generated by `build_run_context(session, run_id)` on every invocation. Includes: run name/status, people count (+ pending rates), files uploaded/processed, timesheet staging row count, alias counts (confirmed/total), open contradictions/tasks, active decision locks.
3. **`## Immediate Goal`** — optional `context_notes` string passed by the caller (e.g., "We are currently resolving identities for File #12"). Only included when provided.

This ensures the agent always knows its immediate situation without the caller needing to manually describe the run state.

---

## 6. Gate Logic (`gates.py`)

| Function | Purpose |
|---|---|
| `get_blocking_contradictions(session, run_id)` | Returns OPEN + BLOCKING contradictions |
| `get_open_blocking_tasks(session, run_id)` | Returns OPEN + BLOCKING review tasks |
| `has_active_lock(session, run_id, issue_key)` | Checks if a DecisionLock exists for the key |
| `update_run_gate_status(session, run_id)` | Evaluates blockers → sets `NEEDS_REVIEW` or clears back to `PROCESSING` |

**Gate rules:**
- Any OPEN + BLOCKING contradiction or task → `Run.status = NEEDS_REVIEW`
- All blockers resolved → `Run.status = PROCESSING`
- Called automatically after every task/contradiction creation via agent tools

---

## 7. Ingestion Pipeline

### Flow

1. **Upload** (`ui/pages/3_uploads.py`) → `save_upload()` → `File` record (status=UPLOADED)
2. **Process** (`ingest/process.py: process_source_file`) dispatches by MIME type:
   - **PDF** → `vision_extract_pdf()` → per-page `ExtractionArtifact` + `Segment`s
   - **Image** → `vision_extract_image()` → `ExtractionArtifact` + `Segment`s
   - **CSV** → `pd.read_csv()` → `StagingRow` + `Segment` per row
   - **DOCX** → `python-docx` → `Segment`s
   - **TXT/MD/JSON** → raw read → `Segment`s
3. **Caching** — if `ExtractionArtifact` exists for same `content_hash` + `run_id`, reuses cached text
4. **CSV Intelligence** (`ingest/csv_intel.py`):
   - `csv_profile()` — DuckDB in-memory: schema, sample, row count
   - `csv_query()` — arbitrary read-only SQL via `df` view
   - `propose_schema_mapping()` — LLM-driven column→target mapping with conservative branching (max 2 alternatives)

### Chunking (`ingest/segment.py`)

- Splits by `\n\n` (paragraphs), max 1000 chars per chunk
- Force-splits oversized paragraphs by character count
- CSV rows: one `StagingRow` + one `Segment` per row

---

## 8. Search System

### Components

| Component | File | Method |
|---|---|---|
| **FTS5** | `search/fts.py` | `segment_fts` + `memory_fts` virtual tables. BM25 ranking. |
| **Embeddings** | `search/embeddings.py` | OpenAI `text-embedding-3-small`. Cached by `(entity_type, entity_id, model)`. |
| **Vector Search** | `search/vector_search.py` | Brute-force cosine similarity over all run embeddings. |
| **Hybrid** | `search/hybrid_search.py` | Reciprocal Rank Fusion (RRF) of FTS + vector results. |

### Indexing

- FTS is **not** auto-updated. Requires explicit `sred db reindex` CLI command or `reindex_all()`.
- Embeddings are stored on-demand via `store_embeddings()` (checks for existing before calling API).

---

## 9. UI Pages (Streamlit)

Navigation defined in `streamlit_app.py`. All pages use `Session(engine)` directly.

| Page | File | Purpose |
|---|---|---|
| Runs | `1_run.py` | Create/select runs. Sets `st.session_state["run_id"]`. |
| People | `2_people.py` | Add people with name/role/rate. Inline rate editing. |
| Uploads | `3_uploads.py` | Upload files (CSV/PDF/DOCX/TXT). Dedupe by SHA256. Process button per file. |
| Dashboard | `4_dashboard.py` | Metrics (people count, file count, pending rates). Readiness checklist. |
| Search | `5_search.py` | Hybrid/FTS/Vector search with mode selector. |
| CSV Tools | `6_csv_tools.py` | Profile, SQL console, schema hypothesis generation. |
| Agent Runner | `7_agent.py` | Run agent loop with max_steps slider + optional context notes. Full step trace. Tool + LLM call history. |
| Tasks & Gates | `8_tasks.py` | Contradictions list, review tasks with resolve→lock flow, supersede lock UI. Gate status banner. |
| Payroll Validation | `9_payroll.py` | Payroll extracts list, per-period mismatch breakdown table (🔴/🟢), overall summary metrics, payroll contradiction list. |

---

## 10. CLI (`cli.py`)

```bash
uv run sred doctor          # Health check: config, API key, data dir
uv run sred db init          # Create all tables + FTS5 virtual tables
uv run sred db reindex       # Rebuild FTS5 index from source tables
uv run sred db search "query"  # Search segments via FTS5
```

---

## 11. Dependencies (`pyproject.toml`)

```
duckdb>=1.4.4          # CSV intelligence
loguru>=0.7.3          # (imported but logging.py uses stdlib)
numpy>=2.4.2           # Vector math
openai>=2.21.0         # LLM + embeddings + vision
pandas>=2.3.3          # CSV processing
pdf2image>=1.17.0      # PDF → images for vision
pillow>=12.1.1         # Image handling
pydantic>=2.12.5       # Validation
pydantic-settings>=2.13.0  # Env-driven config
python-docx>=1.2.0     # DOCX extraction
python-dotenv>=1.2.1   # .env loading
sqlmodel>=0.0.33       # ORM (SQLAlchemy + Pydantic)
streamlit>=1.54.0      # UI
typer==0.12.5          # CLI

# Dev
pytest>=8.0.0
```

**System dependency:** `poppler` (required by `pdf2image` for PDF→image conversion). Install via `brew install poppler` (macOS) or `apt install poppler-utils` (Linux).

---

## 12. Test Suite

Run: `uv run pytest` (84 tests, ~2s)

| File | Tests | Coverage |
|---|---|---|
| `test_agent.py` | 64 | Tool registry (18 tools), all tool functions, issue_key dedup, lock enforcement, gate logic (6 gate tests), agent loop (simple, tool call, max steps, unknown tool), fuzzy ratio (4), alias tools (resolve/confirm/list — 9), build_run_context (3), dynamic prompt injection (2), payroll tools (extract/validate/summary — 12) |
| `test_csv_intel.py` | 3 | DuckDB profile, query, schema proposal (mocked LLM) |
| `test_db.py` | 4 | Core models, finance models, artifacts, vector embeddings (unique constraint) |
| `test_ingest.py` | 3 | Text chunking, text file processing, PDF processing (mocked vision) |
| `test_search_logic.py` | 4 | SHA256 hash, cosine similarity, vector storage + caching, RRF fusion |
| `test_smoke.py` | 2 | Settings load, CLI doctor command |
| `test_ui_logic.py` | 4 | Filename sanitization, SHA256, file upload, schema validation |

All tests use in-memory SQLite. External APIs (OpenAI) are mocked via `unittest.mock.patch`.

---

## 13. Key Design Patterns

### issue_key Deduplication
Both `Contradiction` and `ReviewTask` use an `issue_key` string (e.g. `MISSING_RATE:person:3`) for deduplication within a run. The agent tools check:
1. Is there an active `DecisionLock` for this key? → refuse with `{"status": "locked"}`
2. Is there an existing OPEN record with this key? → refuse with `{"status": "duplicate"}`
3. Otherwise → create and return `{"status": "created"}`

### DecisionLock Lifecycle
1. Human resolves a `ReviewTask` → creates `ReviewDecision` + `DecisionLock(active=True)`
2. Task status → `RESOLVED`, linked contradiction status → `RESOLVED`
3. Agent cannot create new tasks/contradictions with that `issue_key`
4. Human can **supersede**: old lock `active=False`, new lock `active=True` with new reason

### Gate Behavior
- `update_run_gate_status()` is called after every task/contradiction creation
- BLOCKING severity items trigger `Run.status = NEEDS_REVIEW`
- Resolving all blockers returns status to `PROCESSING`

### Tool Handler Contract
- Signature: `handler(session: Session, run_id: int, **kwargs) -> dict`
- Always returns a dict (JSON-serializable)
- Errors returned as `{"error": "message"}`, never raised
- Run-scoped: tools validate `entity.run_id == run_id`

### Idempotency
- File upload: deduped by `content_hash` per run
- File processing: skips if `status == PROCESSED`
- Embeddings: skips if `(entity_type, entity_id, model)` exists
- Memory docs: skips if `content_hash` unchanged
- Tasks/contradictions: deduped by `issue_key`
- Person aliases: unique constraint on `(run_id, alias)` — confirm updates existing record
- Payroll extracts: unique constraint on `(run_id, file_id, period_start, period_end)` — skips existing periods

---

## 14. What's Built (Increments 1–10)

| # | Increment | Status |
|---|---|---|
| 1 | Project scaffold, config, logging | Done |
| 2 | Core DB schema (Run, Person, File, Segment) | Done |
| 3 | File upload, storage, dedup | Done |
| 4 | Ingestion pipeline (PDF vision, CSV, DOCX, TXT) | Done |
| 5 | Search (FTS5 + embeddings + hybrid RRF) | Done |
| 6 | CSV intelligence (DuckDB profiling + schema mapping hypotheses) | Done |
| 7 | Agent runner (OpenAI tool-calling loop) + tool registry + logs | Done |
| 8 | World model (contradictions, review tasks, decision locks, gates) | Done |
| 9a | Identity resolution (PersonAlias model, fuzzy matching, alias tools) | Done |
| 9b | Dynamic agent context injection (build_run_context, context_notes) | Done |
| 10 | Payroll extraction (structured LLM extraction) + mismatch validation (>5% → BLOCKING) + UI | Done |

---

## 15. What's Next (Increments 9c, 11–12)

### Increment 9c: Labour Hours Ledger + Scenarios (remaining work)
- Promote timesheet-like `StagingRow`s into `LedgerLabourHour` (bucket=`UNSORTED`)
- Partial inclusion support (`inclusion_fraction`)
- Scenario computation: Conservative (conf≥0.8), Balanced (conf≥0.5), Aggressive (conf≥0.2)
- Missing rate → BLOCKING contradiction before export
- Statuses: `DRAFT`, `UNSUBSTANTIATED`, `READY`

**Key models already exist:** `LedgerLabourHour` in `models/finance.py`, `Person.rate_status` in `models/core.py`. Identity resolution (PersonAlias) is now complete, enabling person mapping for promotion.

### Increment 11: Review & Grouping
- Create `ClaimProject` model (not yet created)
- Create `ClaimProjectAllocation` model (not yet created)
- Allocate/reallocate hours from `UNSORTED` into projects with fractions
- Allocation decisions create `DecisionLock`s
- UI for grouping

### Increment 12: Memory Docs + Narrative + Export Pack
- OpenClaw-style markdown memory generation and indexing
- Evidence-bounded narrative drafting per project
- Deterministic citation validation (every claim must cite evidence IDs)
- Export bundle: JSON + MD + CSV + citations
- **Export hard-blocked** if evidence gaps or BLOCKING contradictions remain

---

## 16. Known Issues / Technical Debt

1. **Embedding model mismatch** — `config.py` says `text-embedding-3-large`, `embeddings.py` hardcodes `text-embedding-3-small`. Should read from `settings.OPENAI_EMBEDDING_MODEL`.

2. **FTS not auto-indexed** — New segments are not automatically added to the FTS5 index. Requires manual `sred db reindex`. Consider adding triggers or calling reindex after ingestion.

3. **Dashboard placeholder** — `4_dashboard.py` line 53 still says "Task system not implemented yet". Should be updated to query actual `ReviewTask` / `Contradiction` counts now that Increment 8 is complete.

4. **`loguru` in dependencies but unused** — `logging.py` uses stdlib `logging`, not `loguru`. Either switch to loguru or remove the dependency.

5. **basedpyright lint warnings** — SQLModel's `Optional[int]` PKs cause `int | None` assignment warnings throughout. These are false positives at runtime but noisy in strict type checking.

6. **Vision confidence placeholder** — `vision.py` sets `confidence=1.0` for all pages. OpenAI Vision doesn't expose logprobs for vision, so this is a known limitation.

7. **CSV `process_csv_content`** — Uses `pd.json_normalize` in a hacky way (line 75 of `segment.py`), then immediately overwrites with `json.dumps`. The normalize call is dead code.

8. **~~No identity resolution yet~~** — **Resolved in Increment 9a.** PersonAlias model + `aliases_resolve`/`aliases_confirm`/`aliases_list` tools now handle fuzzy name→Person mapping. Future work: extend to invoice vendor names, Jira usernames, and email extraction (see §16C below).

---

## 17. Conventions for the Next Developer

### Adding a New Model
1. Create in `src/sred/models/your_model.py`
2. Add import to `src/sred/models/__init__.py`
3. Add module import to `src/sred/db.py: init_db()`
4. Run `uv run sred db init` to create tables

### Adding a New Agent Tool
1. Write handler in `src/sred/agent/tools.py` with signature `handler(session, run_id, **kwargs) -> dict`
2. Call `register_tool(name, description, parameters, handler)` immediately after the function
3. The tool is automatically available to the agent runner

### Adding a New UI Page
1. Create `src/sred/ui/pages/N_name.py`
2. Add `st.Page(...)` entry to `streamlit_app.py`

### Running Tests
```bash
uv run pytest              # All tests
uv run pytest tests/test_agent.py -v   # Specific file
uv run pytest -k "test_gate"           # Pattern match
```

### Database Reset
```bash
rm data/sred.db
uv run sred db init
```
