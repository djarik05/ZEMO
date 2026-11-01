"""add_provider_and_model_capability_models

Revision ID: f5a3b2c1d4e6
Revises: ea8b2f888923
Create Date: 2026-10-21 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f5a3b2c1d4e6'
down_revision = 'ea8b2f888923'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create providers table
    op.create_table(
        'providers',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('provider_type', sa.Enum('openai', 'anthropic', 'google_gemini', 'mistral', 'ollama', 'openai_compatible', 'custom', name='providertype'), nullable=False),
        sa.Column('base_url', sa.String(512), nullable=False),
        sa.Column('auth_method', sa.Enum('api_key', 'bearer_token', 'basic_auth', 'oauth2', 'none', name='authenticationmethod'), nullable=False, server_default='api_key'),
        sa.Column('credential_ref', sa.String(100), nullable=True),
        sa.Column('default_headers', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('extra_config', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('status', sa.Enum('active', 'inactive', 'error', 'testing', name='providerstatus'), nullable=False, server_default='inactive'),
        sa.Column('last_test_at', sa.DateTime(), nullable=True),
        sa.Column('last_test_result', sa.Text(), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', name='uq_providers_name'),
    )
    op.create_index('ix_providers_type_enabled', 'providers', ['provider_type', 'enabled'])
    op.create_index('ix_providers_status', 'providers', ['status'])
    op.create_index('ix_providers_created_at', 'providers', ['created_at'])
    op.create_index('ix_providers_credential_ref', 'providers', ['credential_ref'])

    # Create model_capabilities table
    op.create_table(
        'model_capabilities',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('provider_id', sa.String(36), nullable=False),
        sa.Column('model_identifier', sa.String(255), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('capabilities', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('max_context_tokens', sa.Integer(), nullable=True),
        sa.Column('max_output_tokens', sa.Integer(), nullable=True),
        sa.Column('supports_streaming', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('supports_system_prompt', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('supports_parallel_tool_calls', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('input_modalities', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('output_modalities', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('pricing_input_per_1k', sa.Float(), nullable=True),
        sa.Column('pricing_output_per_1k', sa.Float(), nullable=True),
        sa.Column('is_deprecated', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('metadata', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['provider_id'], ['providers.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('provider_id', 'model_identifier', name='uq_model_capabilities_provider_model'),
    )
    op.create_index('ix_model_capabilities_provider_model', 'model_capabilities', ['provider_id', 'model_identifier'])
    op.create_index('ix_model_capabilities_deprecated', 'model_capabilities', ['is_deprecated'])


def downgrade() -> None:
    op.drop_index('ix_model_capabilities_deprecated', table_name='model_capabilities')
    op.drop_index('ix_model_capabilities_provider_model', table_name='model_capabilities')
    op.drop_table('model_capabilities')

    op.drop_index('ix_providers_credential_ref', table_name='providers')
    op.drop_index('ix_providers_created_at', table_name='providers')
    op.drop_index('ix_providers_status', table_name='providers')
    op.drop_index('ix_providers_type_enabled', table_name='providers')
    op.drop_table('providers')

    # Drop enum types
    op.execute('DROP TYPE IF EXISTS providertype')
    op.execute('DROP TYPE IF EXISTS providerstatus')