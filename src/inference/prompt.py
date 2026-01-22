"""Prompt template management."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

from src.schema.loader import SchemaConfig

# Qwen3 chat template special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class PromptManager:
    """
    Manages Jinja2 prompt templates.

    Loads templates from disk and renders them with context.
    Formats output for Qwen3 chat template with thinking disabled.
    """

    def __init__(self, prompts_path: Path) -> None:
        self._env = Environment(
            loader=FileSystemLoader(prompts_path),
            autoescape=False,  # Plain text, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._analyze_template: Template | None = None

    def load_templates(self) -> None:
        """Load all required templates."""
        self._analyze_template = self._env.get_template("analyze.jinja")

    def render_analyze_prompt(
        self,
        email_content: str,
        schema_config: SchemaConfig,
    ) -> str:
        """
        Render the analysis prompt for a single email.

        Uses Qwen3 chat template format with thinking mode disabled.
        The /no_think directive at the start of system message disables
        the model's internal reasoning (thinking) mode for faster inference.

        Args:
            email_content: The email text to analyze
            schema_config: Schema configuration with field definitions

        Returns:
            Rendered prompt string in Qwen3 chat format
        """
        if not self._analyze_template:
            raise RuntimeError("Templates not loaded. Call load_templates() first.")

        # Extract enum values for prompt context
        categories: list[str] = []
        category_groups: dict[str, list[str]] = {}
        tones: list[str] = []

        for field_name, field_config in schema_config.fields.items():
            if field_name == "categories":
                if field_config.groups:
                    category_groups = field_config.groups
                if field_config.enum_values:
                    categories = field_config.enum_values
            elif field_name == "tone" and field_config.enum_values:
                tones = field_config.enum_values

        # Render the template (contains system instructions + email)
        raw_prompt = self._analyze_template.render(
            email_content=email_content,
            output_language=schema_config.output_language,
            categories=categories,
            category_groups=category_groups,
            tones=tones,
        )

        # Format as Qwen3 chat template
        # The template already contains /no_think at the start
        return self._format_qwen3_chat(raw_prompt)

    def _format_qwen3_chat(self, system_message: str) -> str:
        """
        Format message using Qwen3 chat template.

        Qwen3 expects:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>assistant

        For single-turn inference with constrained JSON output,
        we put all instructions in system message and let the model
        generate the assistant response directly.
        """
        return (
            f"{IM_START}system\n"
            f"{system_message}{IM_END}\n"
            f"{IM_START}assistant\n"
        )
