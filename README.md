# AutoStream Social-to-Lead Agent

A CLI conversational agent for a fictional SaaS product, **AutoStream** (automated video editing for content creators). The active implementation uses a LangGraph workflow to classify intent, route requests, answer product questions from a local verified knowledge base (direct lookup first, semantic retrieval + grounded generation only when needed), and run a deterministic lead qualification flow. Confirmed leads are persisted to SQLite at the application boundary with idempotent retry behavior.

## Actual architecture and request flow

The active runtime path is:

1. **CLI / application boundary (`app.py`)** loads environment variables (`dotenv`), builds the compiled LangGraph once (`build_graph()`), and runs a turn loop.
2. For each user turn, `app.build_turn_state` merges durable state with the new `input` and clears transient keys (`TRANSIENT_STATE_KEYS` in `agent/state.py`) so stale per-turn values do not leak.
3. **LangGraph (`agent/graph.py`)** executes: `intent` → `route` → one handler node (`greeting` | `business` | `lead` | `fallback`) → `END`.
   - `intent_node` uses deterministic classification (`agent/intent.py`).
   - `route_node` / conditional edges use centralized routing (`agent/routing.py`, `route_from_state` / `resolve_route`) based on `intent` and `lead_state`.
4. **Handlers (`agent/handlers/*`)** produce the response:
   - `greeting`: greetings, thanks, goodbye.
   - `business`: lead-interest fallback text when bypassing routing; otherwise direct KB lookup → (if deferred to RAG) semantic retrieval → grounded generation only when relevant context is found.
   - `lead`: deterministic lead qualification FSM (no LLM, no SQLite).
   - `fallback`: safe clarification response.
5. After the graph returns, **`app.py`** extracts durable state, calls `persist_confirmed_lead_if_needed` (SQLite persistence only for explicitly confirmed, complete leads), appends bounded conversation history, and prints the response.

Routing behavior includes active-lead handling: during lead collection/confirmation, recognizable business questions route to the `business` path so product information can be answered while preserving durable lead state; the next turn resumes the same lead stage. Cancel/confirm/lead-input cases are routed to the `lead` FSM as implemented in `agent/routing.py`.

## Technology stack and responsibilities

| Area | Implementation | Responsibility |
|---|---|---|
| Orchestration | `langgraph`, `agent/graph.py` | StateGraph nodes, conditional routing, compiled workflow |
| State model | `agent/state.py` | Intent labels, lead FSM states, `TRANSIENT_STATE_KEYS`, `AgentState` TypedDict |
| Intent classification | `agent/intent.py` | Deterministic rule/phrase-based intent detection (no I/O, no model calls) |
| Routing | `agent/routing.py` | Centralized route resolution from `intent` + `lead_state` |
| Knowledge base source | `knowledge_base/data.json`, `agent/kb/loader.py` | Verified local JSON facts, validation, cached loading |
| Direct lookup | `agent/kb/lookup.py` | Deterministic answers for supported factual business questions; defers complex/semantic cases |
| Semantic retrieval | `agent/kb/retrieval/*` | KB record building, Sentence Transformers embeddings, NumPy cosine similarity, top-K/threshold retrieval |
| Grounded generation | `agent/kb/retrieval/generation.py`, `agent/llm/gemini_client.py` | Prompt built from retrieved facts only; Gemini called through `google-genai` client |
| Lead FSM | `agent/handlers/lead.py`, `agent/handlers/lead_validation.py` | Deterministic collection, validation, corrections, cancellation, explicit confirmation |
| Persistence | `agent/db/lead_repository.py`, `app.py` | SQLite repository; save attempted only after graph execution for confirmed complete leads |
| CLI | `app.py` | Turn loop, durable/transient state boundary, history, persistence integration |

**Important behavior notes (as implemented):**
- Direct structured/factual lookups that are answered, explicitly not found, or return lookup errors are handled without Gemini.
- If direct lookup defers to RAG (`DEFER_RAG`), semantic retrieval runs first. Gemini generation is used only when relevant retrieval context is found (`retrieval.has_relevant_context` and hits present).
- The lead FSM is deterministic and does not call Gemini and does not perform SQLite operations.
- SQLite persistence is triggered only for `lead_state=confirmed` with a stable `lead_submission_id` and complete name/email/platform, and only when not already marked persisted.
- Gemini model availability and live API connectivity are **not verified** by this README or by the default automated tests.

## Repository structure (active vs retained legacy modules)

```text
social-to-lead-agent/
|-- app.py                          # Active CLI entrypoint and persistence boundary
|-- README.md
|-- requirements.txt
|-- .env.example                    # Placeholder environment template (if present in workspace)
|-- .gitignore
|-- knowledge_base/
|   `-- data.json                   # Verified AutoStream product/policy facts
|-- agent/
|   |-- __init__.py
|   |-- graph.py                    # Active LangGraph workflow
|   |-- intent.py                   # Active deterministic intent classification
|   |-- routing.py                  # Active centralized routing
|   |-- state.py                    # Active state/constants
|   |-- handlers/                   # Active handler layer
|   |   |-- business.py
|   |   |-- fallback.py
|   |   |-- greeting.py
|   |   |-- lead.py
|   |   |-- lead_validation.py
|   |   `-- response.py
|   |-- kb/                         # Active knowledge base, lookup, retrieval, grounding
|   |   |-- loader.py
|   |   |-- lookup.py
|   |   |-- text.py
|   |   `-- retrieval/
|   |       |-- embeddings.py
|   |       |-- generation.py
|   |       |-- records.py
|   |       |-- retriever.py
|   |       `-- similarity.py
|   |-- llm/
|   |   `-- gemini_client.py        # Active google-genai client wrapper
|   |-- db/
|   |   `-- lead_repository.py      # Active SQLite persistence
|   |-- rag.py                      # Retained legacy module (not used by active app/graph path)
|   |-- tools.py                    # Retained legacy module (not used by active app/graph path)
|   `-- workflow.py                 # Retained legacy module (not used by active app/graph path)
`-- tests/                          # unittest suite (unit + integration)
```

**Retained legacy modules:** `agent/rag.py`, `agent/tools.py`, and `agent/workflow.py` are present in the repository but are not part of the active LangGraph CLI execution path traced from `app.py` → `agent.graph.build_graph()` → active handlers/KB/LLM/DB modules. They should not be read as the current lead persistence or conversation architecture.

## Installation and Windows PowerShell virtual environment activation

From the project root in Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If script execution is restricted by local policy, activation may require an appropriate PowerShell execution policy for the current user/session. Use organization-approved settings.

## Dependency installation

With the virtual environment activated:

```powershell
python -m pip install -r requirements.txt
```

The pinned `requirements.txt` reflects the project’s recorded environment. It includes both `google-genai` (used by the active Gemini client implementation) and `google-generativeai` (present in the pinned environment; not used by the active application import path traced in `agent/llm/gemini_client.py`). Dependency changes are outside this documentation phase.

## Environment configuration

Create a `.env` file in the project root (you can copy from `.env.example` if available) and set values as needed:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
LEADS_DB_PATH=data/leads.db
```

| Variable | Purpose | Default / behavior |
|---|---|---|
| `GEMINI_API_KEY` | API key used by the Gemini client when grounded generation needs to call the model | If missing, the Gemini client raises a configuration error when it must initialize; direct lookup paths that do not require Gemini can still operate. |
| `GEMINI_MODEL` | Gemini model name passed to the `google-genai` client | Defaults to `gemini-3.5-flash-lite` in `agent/llm/gemini_client.py` when not provided. |
| `LEADS_DB_PATH` | SQLite database path override | If unset, the repository resolves to `data/leads.db` (resolved relative to the current working directory by the repository). `app.py` uses the env override when set; otherwise it defaults to `<project_root>/data/leads.db`. |

`.env` should not be committed. `.env.example` contains placeholders/example values only.

## Running the application

With the virtual environment activated and environment configured as needed:

```powershell
python app.py
```

The CLI prints a prompt (`You:`). Type messages to interact with the agent; type `exit` to quit.

The exact conversation flow depends on deterministic intent/routing, knowledge-base content, lead FSM state, and (only for deferred RAG cases with relevant context and a configured client/model) grounded generation behavior.

## Knowledge base and direct lookup

- **Source of truth:** `knowledge_base/data.json` (verified AutoStream pricing/policy facts as structured in the current repository).
- **Loader:** `agent/kb/loader.py` resolves the KB path, validates expected structure, and uses in-memory caching/fingerprinting helpers.
- **Direct lookup:** `agent/kb/lookup.py` handles supported factual business questions deterministically (e.g., pricing/plan facts, refund/support where supported by lookup rules). It returns statuses including answered, not found, deferred-to-RAG, or error.
- **No Gemini for resolved direct outcomes:** answered/not-found/error direct-lookup outcomes are returned by the business handler without calling Gemini.

## Semantic retrieval and grounded Gemini generation

When `lookup_business_query` returns `DEFER_RAG`, `agent/handlers/business.py`:

1. Uses a semantic retriever (`agent/kb/retrieval/retriever.py`, default `SemanticRetriever`) to retrieve relevant KB records.
2. Returns an insufficient verified-information response when no relevant context is found (genuine abstention).
3. Returns a temporary retrieval failure response for internal retrieval/model/embedding errors.
4. Calls grounded generation (`agent/kb/retrieval/generation.py`) **only** when relevant retrieval context exists.

Grounded generation:
- Builds a prompt from retrieved facts (not the entire knowledge base).
- Instructs the model to use retrieved facts only and avoid fabrication.
- Uses `agent/llm/gemini_client.py` (`google-genai`, lazy client initialization, configurable model/key).
- Handles generation failures with safe fallback responses.

Default retriever/relevance settings are defined in `agent/kb/retrieval/retriever.py` (e.g., top-K and relevance threshold constants). The embedding implementation uses Sentence Transformers as implemented in `agent/kb/retrieval/embeddings.py`.

## Lead qualification FSM

Lead qualification is implemented by `agent/handlers/lead.py` with validation in `agent/handlers/lead_validation.py`. It is deterministic (no LLM) and does not persist to SQLite.

**Collection stages:** name → email → platform → explicit confirmation (`awaiting_name`, `awaiting_email`, `awaiting_platform`, `awaiting_confirmation`).

**Behavior:**
- **Validation:** field-specific validators run before accepting values; invalid input keeps the current stage and preserves previously collected valid values.
- **Corrections:** deterministic correction commands (e.g., change/update/fix name/email/platform patterns supported by the handler) update only the targeted field after validation and preserve other fields.
- **Cancellation:** explicit cancel intent/commands clear active lead collection fields and move to a cancelled state; cancelled leads are not persisted.
- **Explicit confirmation:** collecting all fields does not auto-confirm. The user must explicitly confirm. Negative/ambiguous confirmation responses do not confirm the lead.
- **Submission ID:** on explicit confirmation, the FSM assigns a stable `lead_submission_id` (if not already present) and marks the lead ready for persistence handling by the application boundary.
- **Interruptions:** during active lead states, business questions can be routed to business handling while durable lead state is preserved; subsequent turns can resume the prior lead stage (subject to routing/intent behavior).

## SQLite persistence

SQLite persistence is implemented in `agent/db/lead_repository.py` and invoked by `app.persist_confirmed_lead_if_needed` **after** graph execution.

**Behavior:**
- **Default location:** `data/leads.db` (repository default). `app.py` defaults to the project-root `data/leads.db` when `LEADS_DB_PATH` is not set.
- **Override:** set `LEADS_DB_PATH` to use a different database path.
- **What gets saved:** only explicitly confirmed, complete leads (`lead_state=confirmed`, stable `submission_id`, name, email, platform) that are not already marked `lead_persisted`.
- **Schema:** `leads` table with `id`, unique `submission_id`, `name`, `email`, `platform`, and `created_at` (UTC). SQL is parameterized.
- **Idempotency/duplicate protection:** `submission_id` is stable for a confirmed submission and is unique in the database. Re-saving the same submission is treated as an idempotent success (no duplicate row).
- **Retry after failure:** if persistence fails, the application preserves confirmed lead details and `submission_id`, keeps the lead eligible for retry, and returns a clear retry-oriented message. It does not claim the lead was saved.

The SQLite database file is a local generated artifact and should not be committed.

## Running automated tests

Run the full `unittest` suite from the project root (virtual environment activated):

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

The test suite uses mocks/patching where appropriate and is designed not to require live Gemini API calls. Automated tests do not use a real application database path; persistence tests use temporary databases or injected repositories.

## Retrieval evaluation (opt-in)

The semantic retrieval evaluation tests in `tests/integration/kb/retrieval/test_semantic_retrieval.py` are opt-in. They are skipped unless the environment variable `RUN_RETRIEVAL_EVAL` is set to `1` and the required dependencies/model/runtime are available.

Example (PowerShell, opt-in only):

```powershell
$env:RUN_RETRIEVAL_EVAL = "1"
python -m unittest tests.integration.kb.retrieval.test_semantic_retrieval -v
```

This README does **not** claim that the six skipped retrieval evaluation tests have already been rerun in this phase. Run them only when intentionally performing that evaluation in an appropriate environment.

## Notes

- The active implementation uses `google-genai` for Gemini integration. The configured default model name is `gemini-3.5-flash-lite` unless overridden by `GEMINI_MODEL`. Live model availability/API connectivity is not verified here.
- `knowledge_base/data.json` remains the verified factual source used by lookup/retrieval flows; responses should be grounded in that data or safe fallback behavior.
- Legacy modules (`agent/rag.py`, `agent/tools.py`, `agent/workflow.py`) are retained in the repository but are not the active runtime architecture.
