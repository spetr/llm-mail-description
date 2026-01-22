"""Dynamic Pydantic model generation from YAML schema."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, create_model


class FieldConfig(BaseModel):
    """Configuration for a single schema field."""

    type: str
    description: str = ""
    max_length: int | None = None
    min_items: int | None = None
    max_items: int | None = None
    item_type: str | None = None
    enum_values: list[str] | None = None
    groups: dict[str, list[str]] | None = None  # Grouped enum values


class SchemaConfig(BaseModel):
    """Schema configuration loaded from YAML."""

    output_language: str = "en"
    fields: dict[str, FieldConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "SchemaConfig":
        """Load schema configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        output_lang = data.get("output", {}).get("language", "en")

        fields = {}
        for name, config in data.get("fields", {}).items():
            # If groups exist, flatten them into enum_values
            if "groups" in config and config["groups"]:
                all_values = []
                for group_values in config["groups"].values():
                    all_values.extend(group_values)
                config["enum_values"] = all_values

            fields[name] = FieldConfig.model_validate(config)

        return cls(output_language=output_lang, fields=fields)

    def get_category_groups(self) -> dict[str, list[str]] | None:
        """Get category groups for prompt formatting."""
        categories_field = self.fields.get("categories")
        if categories_field:
            return categories_field.groups
        return None


def _create_literal_type(values: list[str]) -> Any:
    """Create a Literal type from a list of values."""
    from typing import Literal

    # Dynamically create Literal with all values
    return Literal[tuple(values)]  # type: ignore[valid-type]


def _build_field_definition(config: FieldConfig) -> tuple[Any, Any]:
    """
    Build Pydantic field type and Field() definition from config.

    Returns:
        Tuple of (type_annotation, Field_instance)
    """
    field_kwargs: dict[str, Any] = {}

    if config.description:
        field_kwargs["description"] = config.description

    # Handle different field types
    if config.type == "string":
        field_type: Any = str
        if config.max_length:
            field_kwargs["max_length"] = config.max_length

    elif config.type == "enum":
        if not config.enum_values:
            raise ValueError("enum type requires enum_values")
        field_type = _create_literal_type(config.enum_values)

    elif config.type == "array":
        if config.min_items:
            field_kwargs["min_length"] = config.min_items
        if config.max_items:
            field_kwargs["max_length"] = config.max_items

        # Determine array item type
        if config.item_type == "string":
            field_type = list[str]
        elif config.item_type == "enum":
            if not config.enum_values:
                raise ValueError("array of enum requires enum_values")
            item_type = _create_literal_type(config.enum_values)
            field_type = list[item_type]  # type: ignore[valid-type]
        else:
            field_type = list[str]  # Default to string array

    else:
        raise ValueError(f"Unsupported field type: {config.type}")

    return (field_type, Field(**field_kwargs) if field_kwargs else ...)


def load_schema(path: Path) -> tuple[type[BaseModel], SchemaConfig]:
    """
    Load YAML schema and create dynamic Pydantic model.

    Args:
        path: Path to schema.yaml file

    Returns:
        Tuple of (EmailAnalysis model class, SchemaConfig)

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema is invalid
    """
    config = SchemaConfig.from_yaml(path)

    # Build field definitions for create_model
    field_definitions: dict[str, Any] = {}

    for field_name, field_config in config.fields.items():
        field_definitions[field_name] = _build_field_definition(field_config)

    # Create the dynamic model
    email_analysis_model = create_model(
        "EmailAnalysis",
        __doc__="Dynamically generated email analysis schema.",
        **field_definitions,
    )

    return email_analysis_model, config
