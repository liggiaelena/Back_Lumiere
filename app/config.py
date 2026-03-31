from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    max_image_size_mb: int = 10
    max_image_side: int = 1024

    class Config:
        env_file = ".env"

settings = Settings()
