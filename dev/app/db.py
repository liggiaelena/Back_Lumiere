import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from app.config import read_secret

load_dotenv()


def _build_database_url() -> str:
    """Assemble the connection string, reading the password from the
    db_password Docker secret (a file, never an environment variable)
    instead of a pre-built DATABASE_URL with the password embedded."""
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    username = os.getenv("DB_USERNAME", "postgres")
    password = read_secret("db_password") or os.getenv("DB_PASSWORD")
    hostname = os.getenv("DB_HOSTNAME", "db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "lumiere_db")

    if not password:
        raise RuntimeError(
            "Error: no database password found. Provide the db_password Docker "
            "secret, DB_PASSWORD, or set DATABASE_URL directly."
        )
    return f"postgresql://{username}:{password}@{hostname}:{port}/{name}"


DATABASE_URL = _build_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
