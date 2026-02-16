import os
import pytest
from typer.testing import CliRunner
from sred.config import Settings
from sred.cli import app

runner = CliRunner()

def test_settings_load():
    """Verify settings can be instantiated with dummy values."""
    # We use a mocked environment or rely on the fact that pydantic-settings 
    # handles missing values by raising validation error if no default/env provided.
    # Since OPENAI_API_KEY is required, we must provide it.
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
    try:
        settings = Settings()
        assert settings.OPENAI_MODEL_AGENT == "gpt-5"
        assert settings.PAYROLL_MISMATCH_THRESHOLD == 0.05
    finally:
        del os.environ["OPENAI_API_KEY"]

from unittest.mock import patch

def test_cli_doctor():
    """Verify the doctor command runs without error."""
    # We mock the settings object in the cli module to ensure it sees the key
    with patch("sred.cli.settings") as mock_settings:
        # Mock the secret value
        mock_settings.OPENAI_API_KEY.get_secret_value.return_value = "sk-test-dummy-key"
        mock_settings.PAYROLL_MISMATCH_THRESHOLD = 0.05
        mock_settings.OPENAI_MODEL_AGENT = "gpt-5"
        mock_settings.OPENAI_MODEL_VISIONS = "gpt-5-mini" # Typo in original code? No, vision is singular.

        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "SRED Automation Doctor" in result.stdout
        assert "OPENAI_API_KEY:           ✅ Set" in result.stdout
