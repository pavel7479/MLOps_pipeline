"""Small PostgreSQL readiness check."""

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def database_is_ready(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            return connection.execute(select(1)).scalar_one() == 1
    except SQLAlchemyError:
        return False
