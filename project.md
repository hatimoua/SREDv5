# SR&ED Automation Platform (PoC v0.1) — Holistic Project Spec
 
**Jurisdiction:** Canada (SR&ED)  
**Primary optimization unit:** **Labour hours**  
**Core stance:** Accuracy + performance > cost (external API calls allowed)  
**Guiding design:** People-first anchors + OpenClaw-style context (Markdown memory + hybrid search)

---

## 1) Vision

Build a local-first (data persisted locally) SR&ED automation platform that can ingest messy client evidence (timesheets, payroll stubs, invoices, Jira exports, PDFs, DOCX, etc.), build a consistent “world model” of the claim, and maximize **eligible labour hours** while remaining **auditable**, **human-controlled**, and **robust to ambiguity**.

This is **not** a one-pass parser. It is a **reasoning system**:
- It forms **competing hypotheses** only when ambiguity is material (**conservative branching**).
- It collects more evidence across files before forcing decisions.
- It computes scenario totals even when evidence is incomplete, but flags them as **Draft/Unsubstantiated**.
- It **hard-blocks final generation/export** when blocking contradictions or evidence gaps remain.
- It persists hypotheses, contradictions, decisions, and evidence provenance so the outcome is explainable and reviewable.

---

## 2) What we optimize for (and what we don’t)

### Optimize for
- **Total SR&ED-claimable labour hours** in Canada.
- Consultant-style optimization supported:
  - Combine high-confidence and low-confidence inclusions **transparently**
  - Support **partial inclusion** (e.g., 10 hours logged → include 0.6 fraction)
  - Support **reallocation** between claim buckets/projects during review (human-approved)

### Not optimizing for (PoC)
- Multi-jurisdiction rules (Canada-only)
- Perfect deterministic replay of every run (exploratory acceptable)
- A beautiful UI (Streamlit basic is fine)

---

## 3) Key product decisions

### People-first anchors (foundation)
The UI starts with **People intake**, and People records act as a persistent **constraint/reference layer** for the entire workflow:
- **Name + Role are mandatory**
- Rate can be **Pending**
- However, **claim generation/export is blocked** if included hours depend on people with pending rates

### Project grouping is post-processing
- During ingestion and early modeling, labour hours land in a default bucket: **`UNSORTED`**
- Users group hours into **Claim Projects** during Review/Grouping (Prompt 11)
- We do not force project selection during ingestion

### Evidence coverage feedback loop
- Scenario totals can compute (Draft/Unsubstantiated allowed)
- But **Generate/Export is hard-blocked** until “evidence coverage gaps” are resolved:
  - Example gap: timesheet hours exist for a bucket but no supporting technical documentation exists

### Payroll mismatch threshold
- **Start at 5%**
- If payroll totals mismatch timesheet totals > 5%, treat as a **blocking contradiction**
- Rationale: >5% often indicates systemic issues (e.g., vacation pay, missing pay codes)

### Human decisions are final
- Human resolutions create **DecisionLocks**
- The agent must respect locks and **must not reopen** locked issues
- A human may explicitly **supersede** a lock (creates a new lock, old lock becomes inactive)

---

## 4) OpenClaw-style context management

We adopt OpenClaw’s “simple, scalable context” approach:
- Maintain compact **Markdown memory docs** (system-generated summaries)
- Use **hybrid retrieval** (BM25/FTS + embeddings stored in SQLite + fusion)
- Avoid hype-heavy GraphRAG; instead rely on:
  - evidence provenance
  - retrieval tools
  - iterative reasoning loops

**Benefits:**
- Handles small and very large projects
- Keeps context windows manageable
- Makes system state visible and inspectable

---

## 5) High-level architecture

### Layers
1. **Streamlit UI**
   - Run management, People intake, file upload, dashboard
   - Task resolution UI (blocking issues)
   - Scenarios + grouping + export

2. **Holistic Orchestrator Agent**
   - Deep loop, conservative branching
   - Uses tools only (no raw SQL, no direct DB writes)

3. **Tools layer (safe operations)**
   - Ingest/extract/segment/index/search
   - CSV intelligence (DuckDB)
   - Identity resolution
   - Hypothesis/contradiction/task/lock management
   - Scenario computation
   - Narrative generation + citation validation

4. **SQLite (canonical + staging + world model)**
   - Stores everything: raw artifacts, segments, staging rows, embeddings, memory docs, hypotheses, contradictions, tasks, decisions

5. **OpenAI API**
   - Vision extraction for OCR/text+tables
   - Embeddings API
   - LLM tool-calling (agent orchestration)
   - Structured outputs for extraction/classification

---

## 6) Data model overview (conceptual)

### Anchor layer (persistent constraints)
- **Person**
  - name, role, rate_status (PENDING/SET), hourly_rate (nullable)
- **Time anchors**
  - timesheet totals, payroll period totals (derived from ingested evidence)
- **Identity map**
  - aliases: “J. Smith”, “John Smith Consulting Inc.”, Jira username, etc.

### Evidence layer (facts + provenance)
- **SourceFile**: stored path, sha256, mime type, size, status
- **ExtractionArtifact**: vision text, extracted tables, confidence, page pointers
- **Segment**: chunked text with provenance (page/row), status (PENDING/DONE/ERROR)
- **StagingRow**: unknown structured rows (CSV/Jira/Invoices/Payroll), normalized_text, hash

### Hours ledger (optimization substrate)
- **LedgerLabourHour**
  - person_id (nullable until mapped)
  - date (nullable)
  - hours
  - description
  - confidence
  - inclusion_fraction (0..1)
  - bucket = `UNSORTED` initially
  - provenance pointers (file_id + row/page)

### Post-processing grouping (review)
- **ClaimProject**
- **ClaimProjectAllocation**
  - project_id, labour_hour_id, fraction, rationale, created_by (HUMAN/SYSTEM)

### World model
- **Hypothesis** + **HypothesisAssumption**
  - conservative branching; store competing interpretations when needed
- **Contradiction**
  - severity: LOW/MEDIUM/HIGH/BLOCKING
  - types: MISSING_RATE, PAYROLL_MISMATCH, UNKNOWN_BASIS, MISSING_EVIDENCE, OTHER
- **ReviewTask** + **ReviewDecision**
- **DecisionLock**
  - records final human decisions to prevent reopening

### Retrieval
- **MemoryDoc** (Markdown “memory”)
- **VectorEmbedding** (BLOB float32, stored locally in SQLite)
- **FTS5** index (Segments + MemoryDocs + optionally staging rows)

### Observability
- **LLMCallLog**
- **ToolCallLog**

---

## 7) Ingestion & extraction strategy

### Always OCR PDFs
Accuracy-first stance:
- Every PDF goes through a **vision extraction** step, even if it appears text-based.
- We store:
  - per-page extracted text
  - extracted tables (if available)
  - confidence + provenance pointers
- This ensures scanned PDFs don’t silently fail.

### File types supported (PoC path)
- CSV: stored + profiled + queried via DuckDB; rows go to staging
- PDF: always vision-extracted to artifacts → segments/staging rows
- DOCX/TXT/MD/JSON: extract text deterministically; segments created
- Jira exports: treated as structured (CSV/JSON) or evidence segments

### Store everything
No discarding. Noise is labeled, not deleted.

---

## 8) Search: Hybrid (OpenClaw-style) with SQLite storage

### Components
- **FTS5** keyword search (BM25)
- **Embeddings** stored locally as `VectorEmbedding` BLOBs
- **Hybrid fusion** via Reciprocal Rank Fusion (RRF) or weighted scoring

### Indexed targets
- Segment text
- MemoryDoc markdown
- (Optional) staging row normalized text

### Why this matters
- Enables cross-file coherence (e.g., “vacation pay” buried in payroll text)
- Keeps the agent from needing giant context windows
- Keeps state transparent via MD memory

---

## 9) Identity resolution (people-first, explicit step)

A dedicated identity resolution step maps:
- invoice names/vendor strings → Person (contractor/employee)
- timesheet person strings → Person
- Jira usernames/emails → Person

Outputs:
- Alias map updates
- Conflicts → ReviewTasks
- Human confirmations → DecisionLocks

Identity resolution happens **before** deep technical reasoning/optimization, because hours allocations and cost basis depend on it.

---

## 10) Scenario computation (hours) + statuses

Scenarios compute **labour hours totals**:
- Conservative: confidence ≥ 0.8
- Balanced: confidence ≥ 0.5
- Aggressive: confidence ≥ 0.2 (flagged)

**Key rule:**
- Scenarios may compute even if evidence is missing → status **DRAFT/UNSUBSTANTIATED**
- Export is blocked until evidence coverage is resolved

Statuses:
- **DRAFT**: missing anchors (e.g., person mapping incomplete)
- **UNSUBSTANTIATED**: hours exist but technical evidence coverage is weak/missing
- **READY**: anchors resolved + evidence coverage OK + blocking contradictions cleared

---

## 11) Evidence coverage loop (Missing Info feedback)

We explicitly check:
- For each bucket/project (or UNSORTED), do we have supporting technical evidence segments?

If not:
- Create a **ReviewTask: MISSING_SUPPORTING_EVIDENCE**
- Do not block scenario totals
- **Do block Generate/Export**

This creates a natural OpenClaw-style “loop back to user to upload / attach more documentation”.

---

## 12) Guardrails against hallucination / wrong optimization

Even though the system is agent-driven:
- LLM never writes DB directly (tools only)
- Staging → promote step validates schema and provenance
- Idempotency via hashes (segments/rows)
- Self-consistency loops for high-impact extraction tasks
- Citation validation for narratives (every claim must cite evidence IDs)
- Human gates for:
  - merges/identity conflicts
  - low-confidence inclusion if required
  - export readiness

---

## 13) Tech stack (PoC v0.1)

### Frontend
- Streamlit (basic pages)

### Backend
- Python 3.11+
- SQLite
- SQLModel + Pydantic
- uv (env/deps)

### Data tooling
- DuckDB (CSV intelligence: schema profiling + SELECT queries)
- NumPy (cosine similarity on vectors)

### LLM + AI
- OpenAI API:
  - vision-capable model for OCR/text+tables extraction
  - embeddings model
  - agent model using tool calling
  - structured output model for strict JSON extraction

### Search
- SQLite FTS5
- local embeddings table in SQLite + python similarity search
- hybrid fusion (RRF)

### Observability
- local DB logs (LLMCallLog, ToolCallLog)
- run_id logging via contextvars

---

## 14) Where we are after Prompt 7 (expected state)

By completing Prompt 7, the platform should have:

1) **Core infrastructure**
- Settings/config
- logging with run_id
- DB engine + init

2) **Core DB schema**
- Runs, People (anchors), Files, Segments
- Artifacts (vision extraction outputs)
- Vector embeddings storage in SQLite
- Memory docs (md memory layer)
- FTS setup + reindex CLI

3) **UI foundation**
- run creation/selection
- people intake
- file upload + dedupe by sha256
- dashboard + basic search page (if implemented by this stage)

4) **Ingestion building blocks**
- file processing entrypoints as tools (vision extraction + segmentation) and persistence
- CSV profiling/query tools via DuckDB

5) **Agent runner**
- OpenAI tool-calling loop
- tool registry + dispatch
- tool/LLM call logs stored locally

At this point, you have the “skeleton nervous system”:
- UI can collect anchors & data
- Agent can call tools
- Retrieval exists
- Everything is logged and stored

---

## 15) Where we should stand after Prompt 12 (definition of done)

After Prompt 12, the PoC should support an end-to-end workflow:

### Prompt 8: World model + gates
- Hypotheses persisted (conservative branching)
- Contradictions ledger
- ReviewTask + ReviewDecision
- DecisionLocks enforced (human decisions final)
- Run pauses when blocking contradictions exist

### Prompt 9: Labour hours ledger + scenarios
- Promote timesheet-like rows into LedgerLabourHour (UNSORTED)
- Partial inclusion supported
- Scenario totals computed + flagged statuses
- Missing rate creates blocking contradiction before export

### Prompt 10: Payroll extraction + mismatch validation
- Vision-extracted payroll text parsed into payroll period totals (structured output)
- Payroll mismatch >5% triggers blocking contradiction/task

### Prompt 11: Review & grouping
- Create Claim Projects
- Allocate/reallocate hours from UNSORTED into projects with fractions
- Allocation decisions create DecisionLocks

### Prompt 12: Memory docs + narrative + export pack
- OpenClaw-style markdown memory generated and indexed
- Evidence-bounded narrative drafting per project
- Deterministic citation validation
- Export bundle (JSON + MD + CSV + citations)
- Export is hard-blocked if evidence gaps or blocking contradictions remain

**End result:** a PoC that can ingest real evidence, build a coherent view, compute hours scenarios, and produce an exportable claim package with auditability and human controls.

---

## 16) Next steps after Prompt 12 (recommended roadmap)

### A) Accuracy & evaluation harness (highest ROI)
- Create “golden runs” with known totals and known issues
- Add regression tests for:
  - payroll mismatch detection
  - identity mapping stability
  - scenario computations
  - export gating correctness

### B) Performance improvements
- Parallelize embeddings and vision extraction (batching + concurrency)
- Incremental indexing (avoid full reindex)
- Vector search acceleration:
  - pre-normalize vectors
  - store norm
  - use approximate top-k if dataset grows

### C) Stronger identity resolution
- More alias sources:
  - email extraction from documents
  - invoice vendor normalization
  - Jira user maps
- Improved conflict UX:
  - show evidence snippets side-by-side
  - one-click lock decisions

### D) Smarter evidence coverage
- Require minimal evidence pack per project:
  - tickets/specs/notes + date alignment with labour
- Provide “evidence request” checklist to the client:
  - “upload architecture notes”, “export Jira epics”, etc.

### E) Optimization techniques (consultant-grade)
- Explicit risk curve per scenario
- Constraint-based inclusion (“include low confidence only if backed by X evidence types”)
- Automated “what to ask next” suggestions to improve readiness fast

### F) Hardening & privacy
- Consent + “external API usage” banners
- Redaction options (strip SIN/address etc. before sending)
- Encryption at rest for stored docs (optional)
- Key management and auditing

---

## 17) Practical notes / operating assumptions

- External calls are allowed, so OCR and extraction may send PDFs/images to OpenAI vision models.
- Data retention and privacy must be handled explicitly in the UI and documentation.
- “Exploratory acceptable” means we prioritize iteration and speed, but we still log enough for auditability.

---

## 18) Quick success checklist (PoC)
A PoC run is “successful” if a user can:
1) Create run + enter people
2) Upload a messy mix of files (CSV + PDFs)
3) Agent ingests + extracts + indexes
4) Scenarios compute and are flagged appropriately
5) UI prompts user to resolve blocking contradictions
6) User groups hours into projects
7) System produces an export bundle that is blocked until gaps are resolved


