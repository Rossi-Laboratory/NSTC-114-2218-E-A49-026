from rich.console import Console
from rich.table import Table
import logging
import os

_console = Console()
_LEVEL = os.getenv("VLA_SIM_LOGLEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, _LEVEL, logging.INFO))
logger = logging.getLogger("vla-sim")
logger.setLevel(getattr(logging, _LEVEL, logging.INFO))

def banner(title: str):
    _console.rule(f"[bold cyan]{title}[/bold cyan]")

def kv(k, v):
    _console.print(f"[bold]{k}[/bold]: {v}")
