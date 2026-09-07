"""Check PostgreSQL connectivity and report the current inference schema."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alembic.migration import MigrationContext
from sqlalchemy import inspect, select

from src.config import load_settings
from src.database import create_database


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    database = create_database(settings.database)
    try:
        with database.engine.connect() as connection:
            connection.execute(select(1)).scalar_one()
            revision = MigrationContext.configure(connection).get_current_revision()
            tables = inspect(connection).get_table_names()
        print("PostgreSQL connection: OK")
        print(f"Alembic revision: {revision or 'not migrated'}")
        print(
            "Predictions table: "
            + ("present" if "predictions" in tables else "not migrated")
        )
        if "predictions" not in tables:
            print(r"Run: .\.venv\Scripts\alembic.exe upgrade head")
    finally:
        database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
