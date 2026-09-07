"""Create the inference predictions table.

Revision ID: 20260907_0001
Revises:
Create Date: 2026-09-07
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260907_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column(
            "features",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("prediction", sa.String(length=8), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_alias", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_run_id", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "prediction IN ('BUY', 'HOLD', 'SELL')",
            name="ck_predictions_prediction",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id", name="uq_predictions_request_id"),
    )
    op.create_index(
        "ix_predictions_created_at", "predictions", ["created_at"], unique=False
    )
    op.create_index(
        "ix_predictions_model_version",
        "predictions",
        ["model_version"],
        unique=False,
    )
    op.create_index(
        "ix_predictions_prediction", "predictions", ["prediction"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_prediction", table_name="predictions")
    op.drop_index("ix_predictions_model_version", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
