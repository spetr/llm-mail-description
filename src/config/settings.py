"""Application settings loaded from environment and YAML config."""

from functools import lru_cache
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseModel):
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class BatchingSettings(BaseModel):
    """Micro-batching configuration."""

    max_batch_size: int = Field(default=16, ge=1, le=64)
    max_wait_ms: int = Field(default=50, ge=10, le=1000)


class InferenceSettings(BaseModel):
    """Inference backend configuration."""

    backend: str = "triton"
    triton_urls: list[str] = Field(default_factory=lambda: ["localhost:8001"])
    model_name: str = "llama-3.1-8b"
    timeout_seconds: float = 30.0


class ModelSettings(BaseModel):
    """Model configuration for download and inference."""

    hf_repo: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_tokens: int = 4096
    max_output_tokens: int = 512


class OutputSettings(BaseModel):
    """Output configuration."""

    language: str = "en"


class AppConfig(BaseModel):
    """Application configuration loaded from app.yaml."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate configuration consistency."""
        if not self.inference.triton_urls:
            raise ValueError("At least one Triton URL is required")
        return self


class Settings(BaseSettings):
    """Root settings combining environment variables and YAML config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment variables
    environment: str = "development"
    log_level: str = "INFO"
    config_path: Path = Path("./config")
    hf_token: str | None = None

    # Loaded from YAML
    app: AppConfig = Field(default_factory=AppConfig)

    @model_validator(mode="after")
    def load_yaml_config(self) -> Self:
        """Load app.yaml and merge with defaults."""
        config_file = self.config_path / "app.yaml"

        if config_file.exists():
            with open(config_file) as f:
                yaml_config = yaml.safe_load(f) or {}
            self.app = AppConfig.model_validate(yaml_config)

        return self

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def schema_path(self) -> Path:
        """Path to schema.yaml."""
        return self.config_path / "schema.yaml"

    @property
    def prompts_path(self) -> Path:
        """Path to prompts directory."""
        return self.config_path / "prompts"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance. Call once at startup."""
    return Settings()
