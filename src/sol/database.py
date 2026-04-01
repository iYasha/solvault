import sqlite3

import sqlite_vec
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from sol.config import settings


class VecConnection(sqlite3.Connection):
    """sqlite3 Connection that auto-loads the sqlite-vec extension."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.enable_load_extension(True)
        self.load_extension(sqlite_vec.loadable_path())
        self.enable_load_extension(False)


engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.data.db_file}",
    echo=False,
    connect_args={"factory": VecConnection},
)

async_session = async_sessionmaker(engine, expire_on_commit=False)

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)
