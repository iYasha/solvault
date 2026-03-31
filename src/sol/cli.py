import os
import signal
import subprocess
import sys
import time

import click

from sol import __version__
from sol.config import settings


@click.group()
@click.version_option(version=__version__, prog_name="sol")
def cli() -> None:
    """Sol — privacy-first personal AI assistant."""


@cli.command()
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
            host=host or settings.server.host,
            port=port or settings.server.port,
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

    log_file = settings.data.dir / "sol.log"
    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "sol.cli", "start", "--foreground"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(1)
    if proc.poll() is None:
        click.echo(f"Sol daemon started (PID {proc.pid})")
    else:
        click.echo("Sol failed to start. Check ~/.sol/data/sol.log")


@cli.command()
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


@cli.command()
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
def migrate() -> None:
    """Run database migrations (alembic upgrade head)."""
    from alembic.config import Config

    from alembic import command

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    click.echo("Migrations applied")


if __name__ == "__main__":
    cli()
