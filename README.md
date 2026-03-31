# Sol

Privacy-first personal AI assistant powered by local LLMs.

Sol runs as a persistent server on your machine, connects to messaging channels (Telegram, CLI), remembers everything about you, and keeps your data under your control. When a task exceeds local model capabilities, Sol can delegate to cloud models — but only with explicit approval and only after stripping all personally identifiable information.

## Quick Start

```bash
# Install dependencies
uv sync

# Run database migrations
sol migrate

# Start the server (background)
sol start

# Start in foreground (for development)
sol start --foreground

# Check status
sol status

# Stop
sol stop
```

## Configuration

Sol reads configuration from `~/.sol/config.yaml`. All fields have defaults — a minimal config to get started:

```yaml
server:
  host: "127.0.0.1"
  port: 8765
  log_level: "INFO"

llm:
  endpoint: "http://localhost:1234/v1"
  model: "gpt-oss:20b"

channels:
  telegram:
    enabled: false
    bot_token: ""

identity:
  mappings:
    - canonical_id: "user:username"
      telegram_id: "123456789"
      cli_user: "username"
```

## API

- `GET /v1/health` — health check
- `POST /v1/messages` — send a message (used by channel bridges)
- `WS /v1/ws` — WebSocket for CLI streaming

## Project Structure

```
src/sol/
├── cli.py              # CLI commands (start, stop, status, migrate)
├── config.py           # Pydantic settings from ~/.sol/config.yaml
├── database.py         # SQLAlchemy async engine
├── gateway/            # FastAPI server
├── channels/           # Channel processes (CLI, Telegram)
├── router/             # Message routing & identity resolution
├── session/            # Session management & token windowing
├── core/               # Agent brain (planned)
├── memory/             # Memory system (planned)
├── tools/              # Tool system (planned)
├── privacy/            # Privacy gateway (planned)
├── skills/             # Skill system (planned)
└── scheduler/          # Scheduler (planned)
```

## Development

```bash
# Run with auto-reload
sol start --foreground --reload

# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
sol migrate
```
