import contextlib
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from sol import __version__
from sol.config import settings
from sol.core.agent import create_agent
from sol.database import engine
from sol.gateway.api.router import api_router
from sol.logging_config import configure_logging

log = structlog.get_logger()


def write_pid_file(path: str | os.PathLike) -> None:
    with open(path, "w") as f:
        f.write(str(os.getpid()))


def remove_pid_file(path: str | os.PathLike) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    configure_logging(settings.server.log_level, log_dir=settings.data.logs_dir)

    # Enable SQLite WAL mode
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")

    app.state.agent = create_agent()
    log.info("sol.agent.ready", model=settings.llm.model)

    write_pid_file(settings.data.pid_file)
    log.info("sol.started", host=settings.server.host, port=settings.server.port)

    yield

    log.info("sol.stopping")
    remove_pid_file(settings.data.pid_file)
    await engine.dispose()
    log.info("sol.stopped")


app = FastAPI(title="sol", version=__version__, lifespan=lifespan)
app.include_router(api_router)


def run() -> None:
    uvicorn.run(
        "sol.gateway.main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
        access_log=False,
    )
