import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from sol.config import SolSettings, YamlSettingsSource


class TestConfigDefaults:
    """When no config file exists, all settings should use sensible defaults."""

    def test_loads_defaults_when_no_config_file(self, tmp_path):
        """Given no config.yaml, when settings load, then all defaults apply."""
        with patch.object(YamlSettingsSource, "__call__", return_value={}):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.server.host == "127.0.0.1", "Default host should be localhost"
        assert s.server.port == 8765, "Default port should be 8765"
        assert s.llm.model == "gpt-oss:20b", "Default model should be gpt-oss:20b"
        assert s.channels.telegram.enabled is False, "Telegram should be disabled by default"

    def test_creates_data_dir_on_load(self, tmp_path):
        """Given a non-existent data dir, when settings load, then the directory is created."""
        data_dir = tmp_path / "new" / "nested" / "data"
        with patch.object(YamlSettingsSource, "__call__", return_value={}):
            s = SolSettings(data={"dir": str(data_dir)})

        assert s.data.dir.exists(), "Data directory should be created automatically"


class TestConfigFromYaml:
    """When a config.yaml exists, values should override defaults."""

    def test_loads_values_from_yaml(self, tmp_path):
        """Given a config.yaml with custom values, when settings load, then values override defaults."""
        config_data = {
            "server": {"host": "0.0.0.0", "port": 9999},
            "llm": {"model": "custom-model"},
            "channels": {"telegram": {"enabled": True, "bot_token": "test-token"}},
            "identity": {
                "mappings": [
                    {"canonical_id": "user:test", "telegram_id": "111"},
                ],
            },
        }
        with patch.object(YamlSettingsSource, "__call__", return_value=config_data):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.server.host == "0.0.0.0", "Host should be overridden from YAML"
        assert s.server.port == 9999, "Port should be overridden from YAML"
        assert s.llm.model == "custom-model", "Model should be overridden from YAML"
        assert s.channels.telegram.enabled is True, "Telegram should be enabled from YAML"
        assert s.channels.telegram.bot_token == "test-token", "Bot token should be set from YAML"
        assert len(s.identity.mappings) == 1, "Should have one identity mapping"
        assert s.identity.mappings[0].canonical_id == "user:test"


class TestMemoryConfig:
    """Memory configuration should have sensible defaults and accept YAML overrides."""

    def test_memory_defaults(self, tmp_path):
        """Given no config, when settings load, then memory defaults are correct."""
        with patch.object(YamlSettingsSource, "__call__", return_value={}):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.memory.search_top_k == 10, "Default search_top_k should be 10"
        assert s.memory.injection_max_tokens == 5000, "Default injection_max_tokens should be 5000"
        assert s.memory.vector_weight == 0.7, "Default vector_weight should be 0.7"
        assert s.memory.text_weight == 0.3, "Default text_weight should be 0.3"
        assert s.memory.extraction_enabled is True, "Extraction should be enabled by default"

    def test_embedding_defaults(self, tmp_path):
        """Given no config, when settings load, then embedding defaults are correct."""
        with patch.object(YamlSettingsSource, "__call__", return_value={}):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.memory.embedding.model == "nomic-embed-text-v2-moe", "Default embedding model"
        assert s.memory.embedding.dimensions == 768, "Default dimensions should be 768"
        assert s.memory.embedding.endpoint == "http://localhost:1234/v1", "Default endpoint"

    def test_memory_from_yaml(self, tmp_path):
        """Given YAML with memory overrides, when loaded, then values are applied."""
        config_data = {
            "memory": {
                "search_top_k": 5,
                "extraction_enabled": False,
                "embedding": {"model": "custom-embed", "dimensions": 512},
            },
        }
        with patch.object(YamlSettingsSource, "__call__", return_value=config_data):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.memory.search_top_k == 5, "search_top_k should be overridden"
        assert s.memory.extraction_enabled is False, "extraction_enabled should be overridden"
        assert s.memory.embedding.model == "custom-embed", "Embedding model should be overridden"
        assert s.memory.embedding.dimensions == 512, "Dimensions should be overridden"


class TestConfigProperties:
    """Derived paths should be computed correctly from the data directory."""

    def test_db_file_derived_from_data_dir(self, tmp_path):
        """Given a data dir, when accessing db_file, then it points to sol.db inside data dir."""
        with patch.object(YamlSettingsSource, "__call__", return_value={}):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.data.db_file == s.data.dir / "sol.db", "DB file should be inside data dir"

    def test_pid_file_derived_from_data_dir(self, tmp_path):
        """Given a data dir, when accessing pid_file, then it points to sol.pid in parent dir."""
        with patch.object(YamlSettingsSource, "__call__", return_value={}):
            s = SolSettings(data={"dir": str(tmp_path / "sol-data")})

        assert s.data.pid_file == s.data.dir.parent / "sol.pid", "PID file should be in parent of data dir"
