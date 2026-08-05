import logging
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def read_secret(name: str) -> Optional[str]:
    """Read a Docker secret mounted at /run/secrets/<name>, if present.

    Docker secrets are exposed as files, not environment variables, so they
    never show up in `docker inspect` output or a dumped process environment.
    """
    secret_path = Path("/run/secrets") / name
    if secret_path.exists():
        return secret_path.read_text().strip()
    return None


class Settings(BaseSettings):
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-opus-4-5"
    max_image_size_mb: int = 10
    max_image_side: int = 1024

    class Config:
        # Resolve to project root so .env in Back_Lumiere/ is loaded
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        extra = "ignore"

    @property
    def provider(self) -> str:
        """Return the name of the configured provider: 'anthropic' or 'none'."""
        if self.anthropic_api_key:
            return "anthropic"

        return "none"


settings = Settings()
settings.anthropic_api_key = read_secret("anthropic_api_key") or settings.anthropic_api_key

# Experiment config (set by the training service; harmless defaults here so
# the backend can report the current values even when it's the one booting).
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "lumiere_segformer_unified")
EXPERIMENT_VERSION = os.environ.get("EXPERIMENT_VERSION", "v1")
NUM_EPOCHS = os.environ.get("NUM_EPOCHS", "50")
LEARNING_RATE = os.environ.get("LEARNING_RATE", "5e-5")

def log_startup_config() -> None:
    """Log config/secrets confirmation. Called from main.py after
    setup_logging(), since this module is imported before that runs and a
    logger.info() here would otherwise be silently dropped (no handlers yet)."""
    logger.info(
        "Config loaded — experiment=%s version=%s epochs=%s lr=%s",
        EXPERIMENT_NAME, EXPERIMENT_VERSION, NUM_EPOCHS, LEARNING_RATE,
    )
    logger.info(
        "Secrets resolved from /run/secrets: db_password=%s, anthropic_api_key=%s",
        "***" if read_secret("db_password") else "MISSING",
        "***" if read_secret("anthropic_api_key") else "MISSING",
    )
