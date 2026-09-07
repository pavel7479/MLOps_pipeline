"""Database URL loading and application-level SQLAlchemy resources."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DatabaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatabaseConfigurationError(RuntimeError):
    """Raised when safe PostgreSQL configuration is unavailable."""


@dataclass
class DatabaseRuntime:
    """One Engine and Session factory shared by an application."""

    engine: Engine
    session_factory: sessionmaker[Session]

    def dispose(self) -> None:
        self.engine.dispose()


def get_database_url(settings: DatabaseSettings) -> str:
    """Read the secret connection string from environment, never YAML."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    url = os.getenv(settings.url_env_variable)
    if not url:
        raise DatabaseConfigurationError(
            f"Database URL environment variable is not set: {settings.url_env_variable}"
        )
    if not url.lower().startswith("postgresql+psycopg://"):
        raise DatabaseConfigurationError(
            "DATABASE_URL must use the postgresql+psycopg driver"
        )
    return url


def create_database(settings: DatabaseSettings) -> DatabaseRuntime:
    url = get_database_url(settings)
    engine = create_engine(url, pool_pre_ping=settings.pool_pre_ping)
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    return DatabaseRuntime(engine=engine, session_factory=factory)
