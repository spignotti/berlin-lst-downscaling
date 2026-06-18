"""Configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-level settings loaded from .env and environment variables.

    Values are resolved in order: existing ``os.environ`` → ``.env`` file
    → field default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GCP service account JSON key path
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Weights & Biases API key
    WANDB_API_KEY: str = ""


settings = Settings()
