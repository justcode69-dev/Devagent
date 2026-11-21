"""Central settings. Everything secret comes from the environment / .env only."""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b,qwen/qwen3.8-27b"

    github_token: SecretStr | None = None
    github_api: str = "https://api.github.com"
    # Public demo safety: approved write actions are simulated instead of hitting GitHub.
    dry_run_writes: bool = False

    qdrant_url: str | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    max_iterations: int = 8
    max_file_lines: int = 400
    max_index_file_bytes: int = 200_000
    max_index_files: int = 400
    data_dir: Path = Path("./data")

    sandbox_image: str = "devagent-sandbox:latest"
    sandbox_timeout_s: int = 120

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "devagent.db"

    @property
    def checkpoint_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def redact(text: str) -> str:
    """Strip any configured secret from text before it is logged or streamed."""
    s = get_settings()
    for secret in (s.github_token, s.google_api_key, s.groq_api_key):
        if secret and secret.get_secret_value():
            text = text.replace(secret.get_secret_value(), "***")
    return text
