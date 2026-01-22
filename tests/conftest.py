"""Shared test fixtures."""

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.config.settings import Settings


@pytest.fixture
def test_config_path(tmp_path: Path) -> Path:
    """Create temporary config directory with test files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # app.yaml
    (config_dir / "app.yaml").write_text("""
server:
  host: "127.0.0.1"
  port: 8000

batching:
  max_batch_size: 4
  max_wait_ms: 10

inference:
  backend: "triton"
  triton_urls:
    - "localhost:8001"
  model_name: "test-model"
  timeout_seconds: 5.0

model:
  hf_repo: "test/model"
  max_input_tokens: 2048
  max_output_tokens: 256

output:
  language: "en"
""")

    # schema.yaml
    (config_dir / "schema.yaml").write_text("""
output:
  language: "en"

fields:
  short_summary:
    type: string
    max_length: 100

  detailed_summary:
    type: string
    max_length: 500

  keywords:
    type: array
    item_type: string
    min_items: 1
    max_items: 5

  categories:
    type: array
    item_type: enum
    min_items: 1
    groups:
      test_group:
        - test_category_1
        - test_category_2

  tone:
    type: enum
    enum_values:
      - formal
      - informal

  detected_language:
    type: string
""")

    # prompts
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "analyze.jinja").write_text("""
Analyze this email:
{{ email_content }}

Categories:
{% for group_name, values in category_groups.items() %}
{{ group_name }}: {{ values | join(", ") }}
{% endfor %}
Tones: {{ tones | join(", ") }}
Output language: {{ output_language }}
""")

    return config_dir


@pytest.fixture
def test_settings(test_config_path: Path) -> Settings:
    """Create test settings instance."""
    return Settings(
        environment="test",
        log_level="DEBUG",
        config_path=test_config_path,
    )


@pytest.fixture
def test_client() -> Iterator[TestClient]:
    """Create test client for API testing."""
    from src.api.app import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
