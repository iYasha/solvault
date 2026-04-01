"""add memory tables

Revision ID: b1971f10a405
Revises: aa5bedf71c05
Create Date: 2026-04-01 21:41:53.032585

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1971f10a405'
down_revision: Union[str, Sequence[str], None] = 'aa5bedf71c05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('memories',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('confidence', sa.String(), nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('tags_json', sa.String(), nullable=False),
    sa.Column('embedding', sa.LargeBinary(), nullable=True),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_memories'))
    )
    with op.batch_alter_table('memories', schema=None) as batch_op:
        batch_op.create_index('idx_memories_type', ['type'], unique=False)

    # FTS5 virtual table for BM25 keyword search
    op.execute("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            content,
            memory_id UNINDEXED,
            tokenize='porter unicode61'
        )
    """)

    # sqlite-vec virtual table for vector search (768-dim float32)
    op.execute("""
        CREATE VIRTUAL TABLE memories_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding float[768]
        )
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS memories_fts")
    op.execute("DROP TABLE IF EXISTS memories_vec")

    with op.batch_alter_table('memories', schema=None) as batch_op:
        batch_op.drop_index('idx_memories_type')

    op.drop_table('memories')
