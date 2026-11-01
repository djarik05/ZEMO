"""add_experiments_table

Revision ID: f7bc2570c04f
Revises: f5a3b2c1d4e6
Create Date: 2026-11-01 17:52:58.017914

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7bc2570c04f'
down_revision: Union[str, None] = 'f5a3b2c1d4e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('dataset_id', sa.String(length=36), nullable=False),
        sa.Column('dataset_version_id', sa.String(length=36), nullable=True),
        sa.Column('task', sa.String(length=50), nullable=False),
        sa.Column('algorithm_id', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('hyperparameters', sa.JSON(), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('artifact_path', sa.String(length=512), nullable=True),
        sa.Column('model_size_bytes', sa.Integer(), nullable=True),
        sa.Column('logs', sa.JSON(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_experiments_dataset_id', 'experiments', ['dataset_id'], unique=False)
    op.create_index('ix_experiments_status', 'experiments', ['status'], unique=False)
    op.create_index('ix_experiments_created_at', 'experiments', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_experiments_created_at', table_name='experiments')
    op.drop_index('ix_experiments_status', table_name='experiments')
    op.drop_index('ix_experiments_dataset_id', table_name='experiments')
    op.drop_table('experiments')
