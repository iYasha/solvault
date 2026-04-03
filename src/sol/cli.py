import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

from sol import __version__
from sol.config import SolSettings, settings

_SOL_DIR = Path("~/.sol").expanduser()


@click.group()
@click.version_option(version=__version__, prog_name="sol")
def cli() -> None:
    """Sol — privacy-first personal AI assistant."""


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite existing config")
def init(force: bool) -> None:
    """Initialize ~/.sol directory with default config."""
    _SOL_DIR.mkdir(parents=True, exist_ok=True)

    config_file = _SOL_DIR / "config.json"
    if config_file.exists() and not force:
        click.echo(f"Config already exists: {config_file}")
        click.echo("Use --force to overwrite.")
        return

    defaults = SolSettings.model_construct()
    data = json.loads(defaults.model_dump_json())

    with open(config_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")

    click.echo(f"Created {config_file}")
    click.echo("Edit it to configure Sol, then run: sol gateway start")


@cli.group()
def gateway() -> None:
    """Manage the gateway server."""


@gateway.command()
@click.option("--foreground", "-f", is_flag=True, help="Run in the foreground instead of daemonizing")
@click.option("--host", default=None, help="Bind host")
@click.option("--port", default=None, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (implies --foreground)")
def start(foreground: bool, host: str | None, port: int | None, reload: bool) -> None:
    """Start the gateway server."""
    if reload:
        foreground = True

    if foreground:
        import uvicorn

        uvicorn.run(
            "sol.gateway.main:app",
            host=host or settings.gateway.host,
            port=port or settings.gateway.port,
            log_config=None,
            access_log=False,
            reload=reload,
        )
        return

    pid_file = settings.data.pid_file
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            click.echo(f"Sol already running (PID {pid})")
            return
        except ProcessLookupError:
            pid_file.unlink()

    logs_dir = settings.data.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    daemon_log = logs_dir / "daemon.log"
    with open(daemon_log, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "sol.cli", "gateway", "start", "--foreground"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(1)
    if proc.poll() is None:
        click.echo(f"Sol daemon started (PID {proc.pid})")
    else:
        click.echo("Sol failed to start. Check ~/.sol/data/sol.log")


@gateway.command()
def stop() -> None:
    """Stop the running gateway server."""
    pid_file = settings.data.pid_file

    if not pid_file.exists():
        click.echo("Sol is not running")
        return

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sol daemon stopped (PID {pid})")
    except ProcessLookupError:
        click.echo("Sol was not running (stale PID file)")
        pid_file.unlink()


@gateway.command()
def status() -> None:
    """Check if the gateway server is running."""
    pid_file = settings.data.pid_file

    if not pid_file.exists():
        click.echo("Sol is not running")
        return

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
        click.echo(f"Sol is running (PID {pid})")
    except ProcessLookupError:
        click.echo("Sol is not running (stale PID file)")
        pid_file.unlink()


@cli.command()
def chat() -> None:
    """Start an interactive chat session with Sol."""
    from sol.channels.cli.chat import run_chat

    run_chat()


@cli.command()
def telegram() -> None:
    """Start the Telegram bot."""
    import asyncio

    if not settings.channels.telegram.bot_token:
        click.echo("Error: Telegram bot_token not configured in ~/.sol/config.yaml")
        return

    from sol.channels.telegram.bot import start_bot

    asyncio.run(start_bot())


@cli.command()
def migrate() -> None:
    """Run database migrations (alembic upgrade head)."""
    from alembic.config import Config

    from alembic import command

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    click.echo("Migrations applied")


if __name__ == "__main__":
    cli()
