"""
Chirii AI settings, read from environment variables or a .env file. Secrets never have working defaults.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
DEFAULT_DATA_DIR = (Path(os.environ["LOCALAPPDATA"]) / "Chirii") if os.environ.get("LOCALAPPDATA") else Path.home() / ".local" / "share" / "chirii"
PLACEHOLDER_SECRETS = {"", "change-me", "changeme", "secret", "replace-with-a-long-random-string"}
MIN_SECRET_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHIRII_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # Signs login sessions. Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"
    secret_key: str = Field(..., repr=False)

    # S5: encrypts every user's memory file (AES-256-GCM, a key per user derived from this). Create it with
    #   python -m chirii.app.keys create
    # If it is lost, the data cannot be recovered: keep a copy in a password manager, not next to the data.
    data_key: Optional[str] = Field(None, repr=False)

    # S5: users' encrypted memory files live here, outside the repository and outside OneDrive by default.
    # Environment variables are expanded ("%LOCALAPPDATA%\Chirii").
    data_dir: Path = DEFAULT_DATA_DIR

    # Any OpenAI-compatible server: llama.cpp's llama-server locally, or a GPU endpoint later
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "qwen2.5-1.5b-instruct-q8_0"
    llm_api_key: str = Field("", repr=False)
    llm_timeout_seconds: float = 60.0

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Where fastembed keeps the downloaded embedding model. Ignored by git.
    embed_cache_dir: Path = REPO_ROOT / "models" / "fastembed"
    log_level: str = "INFO"

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, value):
        return Path(os.path.expandvars(os.path.expanduser(str(value))))

    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_real(cls, value: str) -> str:
        if value.strip().lower() in PLACEHOLDER_SECRETS or len(value) < MIN_SECRET_LENGTH:
            raise ValueError(f"CHIRII_SECRET_KEY must be a random string of at least {MIN_SECRET_LENGTH} characters")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
