import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from sol.config import settings
from sol.database import Base, VecConnection
from sol.memory import models as _memory_models  # noqa: F401 — register models
from sol.session import models as _session_models  # noqa: F401 — register models

target_metadata = Base.metadata

config = context.config
config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{settings.data.db_file}")

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Virtual tables (FTS5, sqlite-vec) are managed manually in migrations.
# Exclude them from autogenerate so Alembic doesn't try to drop/recreate them.
_VIRTUAL_TABLE_PREFIXES = ("memories_fts", "memories_vec")


def include_object(object: object, name: str, type_: str, reflected: bool, compare_to: object) -> bool:  # noqa: ANN001
    if type_ == "table" and any(name.startswith(prefix) for prefix in _VIRTUAL_TABLE_PREFIXES):
        return False
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        include_object=include_object,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        include_object=include_object,
        target_metadata=target_metadata,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"factory": VecConnection},
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)
    if connectable is None:
        asyncio.run(run_async_migrations())
    else:
        do_run_migrations(connectable)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
