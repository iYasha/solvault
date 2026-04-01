# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dependencies
uv sync

# Run server
sol gateway start                        # background daemon
sol gateway start --foreground           # foreground (dev)
sol gateway start --foreground --reload  # foreground with auto-reload
sol gateway stop                         # stop daemon
sol gateway status                       # check if running

# Channels
sol chat                                 # interactive CLI chat
sol telegram                             # start Telegram bot

# Database migrations
sol migrate                                              # apply all pending
uv run alembic revision --autogenerate -m "description"  # create new migration

# Tests
uv run pytest tests/ -v              # all tests
uv run pytest tests/unit/session/ -v # specific directory
uv run pytest tests/unit/session/test_manager.py::TestGetOrCreateSession::test_creates_new_session -v  # single test

# Linting (auto-runs via pre-commit on commit)
uv run ruff check --fix .
uv run ruff format .
uv run pre-commit run --all-files
```

## Architecture

Sol is a privacy-first personal AI assistant. The package structure mirrors the component design in `docs/HIGH_LEVEL_DESIGN.md`:

- **`gateway/`** — FastAPI server. Lifespan in `main.py` handles startup (WAL mode, PID file) and shutdown. API endpoints under `api/v1/`. Dependencies in `dependencies.py` provide `AsyncSession` via FastAPI's `Depends()`.
- **`session/`** — SQLAlchemy ORM models (`Session`, `ChatMessage`), `SessionManager` for CRUD, `token_window.py` for tiktoken-based context compression.
- **`router/`** — `MessageRouter` normalizes channel-specific messages into a common format and resolves cross-channel user identity via config mappings.
- **`channels/`** — Separate processes that connect to the gateway (Telegram via HTTP POST, CLI via WebSocket). Not in-process.
- **`config.py`** — Pydantic `BaseSettings` with custom `YamlSettingsSource` loading from `~/.sol/config.yaml`. Module-level `settings` singleton.
- **`database.py`** — SQLAlchemy async engine (`sqlite+aiosqlite`), `Base` with naming conventions, `async_session` maker.
- **`cli.py`** — Click CLI. Entry point registered as `sol` in pyproject.toml.

Planned but empty: `core/` (agent), `memory/` (RAG), `tools/`, `privacy/`, `skills/`, `scheduler/`.

## Key Patterns

**Config**: All settings come from `~/.sol/config.yaml`. Priority: init args > env vars > YAML > defaults. Nested Pydantic models (`ServerConfig`, `DataConfig`, `LLMConfig`, etc.). Data dir auto-created on load.

**Database**: SQLite with WAL mode (set in lifespan). Alembic migrations with `render_as_batch=True` (required for SQLite ALTER TABLE). Models register by importing in `alembic/env.py`.

**Testing**: In-memory SQLite with session-scoped engine, per-test transaction rollback. API tests use `httpx.AsyncClient` with `ASGITransport`. DB dependency overridden via `app.dependency_overrides[get_db]`.

**WebSocket protocol** (`/v1/ws`): Client sends `{"type": "message", "text": "..."}`, server responds with `{"type": "chunk", "text": "..."}` frames followed by `{"type": "done", "session_id": "..."}`.

## Testing Guidelines

### File naming
Mirror the source path under `tests/unit/`:
- `src/sol/session/manager.py` → `tests/unit/session/test_manager.py`
- `src/sol/router/message_router.py` → `tests/unit/router/test_message_router.py`
- `src/sol/gateway/api/v1/health.py` → `tests/unit/gateway/api/v1/test_health.py`

### Test behavior, not implementation
Test **what** the code does, not **how** it does it internally. Never mock repositories, DB sessions, or internal service methods.

### Docstrings — Given/When/Then
Every test method gets a one-line docstring:
```python
async def test_creates_new_session(self, db_session):
    """Given no existing session, when get_or_create called, then a new session is persisted."""
```

### Assertion messages
Include a message explaining expected behavior:
```python
assert result.scalar_one() == 1, "Exactly one session should exist in DB"
```

### Class-based organization
Group related tests in classes. Class docstring describes the business rule:
```python
class TestGetOrCreateSession:
    """Session lookup should create new sessions or return existing ones."""
```

### Database — real DB, no mocks
Use the `db_session` fixture (auto-rollback per test). Never mock DB sessions or repositories. Verify state with queries:
```python
result = await db_session.execute(sa.select(sa.func.count()).select_from(Session))
assert result.scalar_one() == 1
```

### API endpoints
Use the `client` fixture (`httpx.AsyncClient` with ASGI transport). DB dependency is auto-overridden:
```python
async def test_returns_ok(self, client):
    response = await client.get("/v1/health")
    assert response.status_code == 200
```

### HTTP clients — mock at the boundary
Mock external HTTP clients with `patch` + `AsyncMock`. Only mock the network boundary, not internal code.

### Workflow
1. Present a test plan (list of test cases with Given/When/Then) before writing code
2. Wait for user approval
3. Implement all tests
4. Run with `uv run pytest tests/path/to/test_file.py -v` and fix failures

## Conventions

- Ruff with line-length 120. Rules: ANN, ASYNC, C4, I (isort), SIM, T20 (no print), UP.
- All imports at module top level — no lazy/inline imports.
- `datetime.now(UTC)` not `datetime.utcnow()` (deprecated).
- `contextlib.suppress()` instead of bare `try/except/pass`.
- SQLAlchemy constraint naming: `ix_`, `uq_`, `ck_`, `fk_`, `pk_` prefixes.
