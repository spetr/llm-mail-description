"""Tests for schema loader."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from src.schema.loader import load_schema, SchemaConfig


class TestSchemaConfig:
    """Tests for SchemaConfig."""

    def test_load_from_yaml(self, test_config_path: Path) -> None:
        """Test loading schema from YAML file."""
        config = SchemaConfig.from_yaml(test_config_path / "schema.yaml")

        assert config.output_language == "en"
        assert "short_summary" in config.fields
        assert "categories" in config.fields

    def test_field_types(self, test_config_path: Path) -> None:
        """Test different field types are parsed correctly."""
        config = SchemaConfig.from_yaml(test_config_path / "schema.yaml")

        # String field
        assert config.fields["short_summary"].type == "string"
        assert config.fields["short_summary"].max_length == 100

        # Array field
        assert config.fields["keywords"].type == "array"
        assert config.fields["keywords"].item_type == "string"

        # Enum field
        assert config.fields["tone"].type == "enum"
        assert "formal" in config.fields["tone"].enum_values  # type: ignore

        # Array of enum
        assert config.fields["categories"].type == "array"
        assert config.fields["categories"].item_type == "enum"


class TestLoadSchema:
    """Tests for load_schema function."""

    def test_creates_pydantic_model(self, test_config_path: Path) -> None:
        """Test that a valid Pydantic model is created."""
        model, config = load_schema(test_config_path / "schema.yaml")

        assert issubclass(model, BaseModel)
        assert config.output_language == "en"

    def test_model_has_expected_fields(self, test_config_path: Path) -> None:
        """Test that model has all expected fields."""
        model, _ = load_schema(test_config_path / "schema.yaml")

        field_names = set(model.model_fields.keys())
        expected = {
            "short_summary",
            "detailed_summary",
            "keywords",
            "categories",
            "tone",
            "detected_language",
        }
        assert field_names == expected

    def test_model_validates_data(self, test_config_path: Path) -> None:
        """Test that model validates data correctly."""
        model, _ = load_schema(test_config_path / "schema.yaml")

        # Valid data
        valid_data = {
            "short_summary": "Test summary",
            "detailed_summary": "Longer summary here",
            "keywords": ["test", "email"],
            "categories": ["test_category_1"],
            "tone": "formal",
            "detected_language": "en",
        }
        instance = model.model_validate(valid_data)
        assert instance.short_summary == "Test summary"

    def test_model_rejects_invalid_enum(self, test_config_path: Path) -> None:
        """Test that model rejects invalid enum values."""
        model, _ = load_schema(test_config_path / "schema.yaml")

        invalid_data = {
            "short_summary": "Test",
            "detailed_summary": "Test",
            "keywords": ["test"],
            "categories": ["invalid_category"],  # Not in enum
            "tone": "formal",
            "detected_language": "en",
        }

        with pytest.raises(Exception):  # Pydantic ValidationError
            model.model_validate(invalid_data)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when schema file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_schema(tmp_path / "nonexistent.yaml")
