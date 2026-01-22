"""Tests for prompt manager."""

from pathlib import Path

import pytest

from src.inference.prompt import PromptManager
from src.schema.loader import SchemaConfig


class TestPromptManager:
    """Tests for PromptManager."""

    @pytest.fixture
    def prompt_manager(self, test_config_path: Path) -> PromptManager:
        """Create prompt manager with test templates."""
        pm = PromptManager(test_config_path / "prompts")
        pm.load_templates()
        return pm

    @pytest.fixture
    def schema_config(self, test_config_path: Path) -> SchemaConfig:
        """Load test schema config."""
        return SchemaConfig.from_yaml(test_config_path / "schema.yaml")

    def test_load_templates(self, test_config_path: Path) -> None:
        """Test template loading."""
        pm = PromptManager(test_config_path / "prompts")
        pm.load_templates()
        assert pm._analyze_template is not None

    def test_render_analyze_prompt(
        self,
        prompt_manager: PromptManager,
        schema_config: SchemaConfig,
    ) -> None:
        """Test prompt rendering."""
        email_content = "Hello, this is a test email."

        prompt = prompt_manager.render_analyze_prompt(
            email_content=email_content,
            schema_config=schema_config,
        )

        assert email_content in prompt
        assert "test_category_1" in prompt
        assert "formal" in prompt
        assert "en" in prompt

    def test_render_without_loading_raises(self, test_config_path: Path) -> None:
        """Test that rendering without loading raises error."""
        pm = PromptManager(test_config_path / "prompts")
        # Not calling load_templates()

        config = SchemaConfig.from_yaml(test_config_path / "schema.yaml")

        with pytest.raises(RuntimeError, match="not loaded"):
            pm.render_analyze_prompt("Test", config)

    def test_template_not_found(self, tmp_path: Path) -> None:
        """Test error when template file doesn't exist."""
        pm = PromptManager(tmp_path)

        with pytest.raises(Exception):  # Jinja2 TemplateNotFound
            pm.load_templates()
