from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: Optional[str] = None
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
