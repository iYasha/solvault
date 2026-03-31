from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    api_token: str = ""


class DataConfig(BaseModel):
    dir: Path = Path("~/.sol/data")

    @property
    def db_file(self) -> Path:
        return self.dir / "sol.db"

    @property
    def pid_file(self) -> Path:
        return self.dir.parent / "sol.pid"


class LLMConfig(BaseModel):
    endpoint: str = "http://localhost:1234/v1"
    model: str = "gpt-oss:20b"
    api_key: str = ""
    max_context_tokens: int = 100_000
    response_token_budget: int = 4096


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""


class ChannelsConfig(BaseModel):
    telegram: TelegramConfig = TelegramConfig()


class IdentityMapping(BaseModel):
    canonical_id: str
    telegram_id: str | None = None
    cli_user: str | None = None


class IdentityConfig(BaseModel):
    mappings: list[IdentityMapping] = []


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load settings from ~/.sol/config.yaml."""

    def get_field_value(self, field: object, field_name: str) -> tuple[object, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        config_path = Path("~/.sol/config.yaml").expanduser()
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data if data else {}


class SolSettings(BaseSettings):
    server: ServerConfig = ServerConfig()
    data: DataConfig = DataConfig()
    llm: LLMConfig = LLMConfig()
    channels: ChannelsConfig = ChannelsConfig()
    identity: IdentityConfig = IdentityConfig()

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
            YamlSettingsSource(settings_cls),
        )


settings = SolSettings()
