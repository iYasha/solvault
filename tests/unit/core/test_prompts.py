from unittest.mock import patch

from sol.core.prompts import DEFAULT_SYSTEM_PROMPT, load_system_prompt


class TestLoadSystemPrompt:
    """System prompt loading should fall back to default when no file is configured."""

    @patch("sol.core.prompts.settings")
    def test_returns_default_when_no_file_configured(self, mock_settings):
        """Given empty system_prompt_file config, when loading, then DEFAULT_SYSTEM_PROMPT is returned."""
        mock_settings.llm.system_prompt_file = ""

        result = load_system_prompt()

        assert result == DEFAULT_SYSTEM_PROMPT, "Should return default prompt when no file configured"

    def test_loads_from_file_when_configured(self, tmp_path):
        """Given a valid system_prompt_file, when loading, then file content is returned."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom system prompt")

        with patch("sol.core.prompts.settings") as mock_settings:
            mock_settings.llm.system_prompt_file = str(prompt_file)
            result = load_system_prompt()

        assert result == "Custom system prompt", "Should return content from the configured file"

    @patch("sol.core.prompts.settings")
    def test_falls_back_to_default_when_file_missing(self, mock_settings):
        """Given a non-existent system_prompt_file, when loading, then DEFAULT_SYSTEM_PROMPT is returned."""
        mock_settings.llm.system_prompt_file = "/nonexistent/prompt.txt"

        result = load_system_prompt()

        assert result == DEFAULT_SYSTEM_PROMPT, "Should fall back to default when file does not exist"
