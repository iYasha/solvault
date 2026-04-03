import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class GatewayConfig(BaseModel):
    host: str = Field(default="127.0.0.1", description="Bind address for the gateway server")
    port: int = Field(default=8765, description="Port for the gateway server")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    api_token: str = Field(default="", description="API token for authentication (empty = no auth)")


class DataConfig(BaseModel):
    dir: Path = Field(default=Path("~/.sol/data"), description="Directory for all Sol data (DB, logs)")

    @property
    def db_file(self) -> Path:
        return self.dir / "sol.db"

    @property
    def logs_dir(self) -> Path:
        return self.dir / "logs"

    @property
    def pid_file(self) -> Path:
        return self.dir.parent / "sol.pid"


class LLMConfig(BaseModel):
    endpoint: str = Field(default="http://localhost:1234/v1", description="OpenAI-compatible API endpoint (LM Studio)")
    model: str = Field(default="gpt-oss:20b", description="Model name for the local LLM")
    api_key: str = Field(default="", description="API key (empty if not required)")
    max_context_tokens: int = Field(default=100_000, description="Maximum context window size in tokens")
    response_token_budget: int = Field(default=4096, description="Max tokens reserved for the response")
    system_prompt_file: str = Field(default="", description="Path to custom system prompt file (empty = use default)")


class TelegramConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable the Telegram bot channel")
    bot_token: str = Field(default="", description="Telegram bot token from @BotFather")


class ChannelsConfig(BaseModel):
    telegram: TelegramConfig = TelegramConfig()


class IdentityMapping(BaseModel):
    canonical_id: str = Field(description="Canonical user identifier")
    telegram_id: str | None = Field(default=None, description="Telegram user ID")
    cli_user: str | None = Field(default=None, description="CLI username (e.g. $USER)")


class IdentityConfig(BaseModel):
    mappings: list[IdentityMapping] = Field(default_factory=list, description="Cross-channel user identity mappings")


class EmbeddingConfig(BaseModel):
    endpoint: str = Field(default="http://localhost:1234/v1", description="Embedding model API endpoint")
    model: str = Field(default="nomic-embed-text-v2-moe", description="Embedding model name")
    api_key: str = Field(default="", description="API key for embeddings (empty if not required)")
    dimensions: int = Field(default=768, description="Embedding vector dimensions")


class MemoryConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    search_top_k: int = Field(default=10, description="Number of memories returned by RAG search")
    injection_max_tokens: int = Field(default=5000, description="Max tokens for memory context in system prompt")
    vector_weight: float = Field(default=0.7, description="Weight for vector search in RRF fusion")
    text_weight: float = Field(default=0.3, description="Weight for keyword search in RRF fusion")
    extraction_enabled: bool = Field(default=True, description="Extract facts from conversations automatically")
    decay_half_life_days: int = Field(default=30, description="Half-life in days for temporal decay scoring")
    decaying_types: list[str] = Field(
        default=["work", "conversations", "events"],
        description="Memory types subject to temporal decay",
    )


class PermissionOverrideConfig(BaseModel):
    defaults: dict[str, str] = Field(
        default={
            "shell": "ask",
            "file_read": "auto_allow",
            "file_write": "ask",
            "file_edit": "ask",
            "web_search": "auto_allow",
            "web_fetch": "auto_allow",
            "web_research": "auto_allow",
            "memory_search": "auto_allow",
            "memory_save": "auto_allow",
        },
        description="Default permission per tool (auto_allow, ask, deny)",
    )
    allow: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Glob patterns to auto-allow per tool, e.g. shell: ['git *']",
    )
    deny: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Glob patterns to deny per tool, e.g. shell: ['rm -rf *']",
    )


class ClaudeConfig(BaseModel):
    model: str = Field(default="claude-sonnet-4-6", description="Claude model for web research delegation")
    timeout: int = Field(default=120, description="Timeout in seconds for Claude CLI calls")


class ToolsConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable tool system")
    shell_timeout: int = Field(default=30, description="Default timeout for shell commands in seconds")
    allowed_paths: list[str] = Field(default=["~/**"], description="Glob patterns for allowed file paths")
    web_fetch_max_chars: int = Field(default=20_000, description="Max characters returned by web_fetch")
    web_search_max_results: int = Field(default=5, description="Default number of web search results")
    approval_timeout: float = Field(default=30.0, description="Seconds to wait for user approval before auto-deny")
    claude: ClaudeConfig = ClaudeConfig()
    permissions: PermissionOverrideConfig = PermissionOverrideConfig()


class JsonSettingsSource(PydanticBaseSettingsSource):
    """Load settings from ~/.sol/config.json."""

    def get_field_value(self, field: object, field_name: str) -> tuple[object, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        config_path = Path("~/.sol/config.json").expanduser()
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            data = json.load(f)
        return data if data else {}


class SolSettings(BaseSettings):
    gateway: GatewayConfig = GatewayConfig()
    data: DataConfig = DataConfig()
    llm: LLMConfig = LLMConfig()
    channels: ChannelsConfig = ChannelsConfig()
    identity: IdentityConfig = IdentityConfig()
    memory: MemoryConfig = MemoryConfig()
    tools: ToolsConfig = ToolsConfig()

    @model_validator(mode="after")
    def ensure_data_dir(self) -> "SolSettings":
        expanded = self.data.dir.expanduser()
        expanded.mkdir(parents=True, exist_ok=True)
        self.data.dir = expanded
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            JsonSettingsSource(settings_cls),
        )


settings = SolSettings()
