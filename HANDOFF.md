# SR&ED Automation Platform — Comprehensive Handoff Document

**Generated:** 2026-02-16  
**Status:** Completed through **Increment 8** (of 12)  
**Test suite:** 48 tests, all passing  
**Tooling:** Python 3.12+, uv, SQLite, SQLModel, Streamlit, OpenAI API, DuckDB

> For the full product vision, optimization goals, and post-Prompt-12 roadmap, see `project.md` in the repo root.

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
│   │   ├── finance.py            # StagingRow, LedgerLabourHour
│   │   ├── hypothesis.py         # Hypothesis, StagingMappingProposal
│   │   ├── artifact.py           # ExtractionArtifact
│   │   ├── vector.py             # VectorEmbedding (BLOB storage)
│   │   ├── memory.py             # MemoryDoc (markdown memory)
│   │   ├── world.py              # Contradiction, ReviewTask, ReviewDecision, DecisionLock
│   │   └── agent_log.py          # ToolCallLog, LLMCallLog
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── registry.py           # Tool registry (register_tool, get_openai_tools_schema)
│   │   ├── runner.py             # OpenAI tool-calling loop (run_agent_loop)
│   │   └── tools.py              # 12 registered tool implementations
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
│           └── 8_tasks.py        # Tasks & Gates (contradictions, resolve, locks, supersede)
│
└── tests/
    ├── test_agent.py             # 28 tests: tools, registry, gates, locks, agent loop
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

### 4.6 Other Models

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

### 5.3 Agent Runner (`agent/runner.py`)

`run_agent_loop(session, run_id, user_message, max_steps=10) -> AgentResult`

- Uses `OPENAI_MODEL_AGENT` with OpenAI tool-calling
- Multi-step loop: LLM → tool calls → feed results back → repeat
- Stops on: plain text response (`complete`), max steps (`max_steps`), or API error (`error`)
- Logs every LLM call to `LLMCallLog` and every tool call to `ToolCallLog`
- Returns `AgentResult` with full step trace

### 5.4 System Prompt

The agent is instructed to:
- Never write raw SQL or modify DB directly
- Use provided tools only
- Create tasks/contradictions for uncertain situations rather than guessing
- Respect DecisionLocks

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
| Agent Runner | `7_agent.py` | Run agent loop with max_steps slider. Full step trace. Tool + LLM call history. |
| Tasks & Gates | `8_tasks.py` | Contradictions list, review tasks with resolve→lock flow, supersede lock UI. Gate status banner. |

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

Run: `uv run pytest` (48 tests, ~2s)

| File | Tests | Coverage |
|---|---|---|
| `test_agent.py` | 28 | Tool registry, all tool functions, issue_key dedup, lock enforcement, gate logic (6 gate tests), agent loop (simple, tool call, max steps, unknown tool) |
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

---

## 14. What's Built (Increments 1–8)

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

---

## 15. What's Next (Increments 9–12)

### Increment 9: Labour Hours Ledger + Scenarios
- Promote timesheet-like `StagingRow`s into `LedgerLabourHour` (bucket=`UNSORTED`)
- Partial inclusion support (`inclusion_fraction`)
- Scenario computation: Conservative (conf≥0.8), Balanced (conf≥0.5), Aggressive (conf≥0.2)
- Missing rate → BLOCKING contradiction before export
- Statuses: `DRAFT`, `UNSUBSTANTIATED`, `READY`

**Key models already exist:** `LedgerLabourHour` in `models/finance.py`, `Person.rate_status` in `models/core.py`

### Increment 10: Payroll Extraction + Mismatch Validation
- Vision-extract payroll text → structured output (period totals)
- Compare payroll totals vs timesheet totals
- Mismatch > 5% (`PAYROLL_MISMATCH_THRESHOLD`) → BLOCKING contradiction + ReviewTask
- Use `OPENAI_MODEL_STRUCTURED` for strict JSON extraction

**Key infrastructure already exists:** `ContradictionType.PAYROLL_MISMATCH`, `PAYROLL_MISMATCH_THRESHOLD` in config

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

8. **No identity resolution yet** — The project spec (section 9) calls for alias mapping (invoice names → Person, Jira usernames → Person). Not yet implemented.

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
