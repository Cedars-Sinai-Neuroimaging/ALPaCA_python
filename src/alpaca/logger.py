import logging
from rich.logging import RichHandler
from rich.console import Console

# Global console object for rich printing
console = Console()

# Configure logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, console=console, markup=True)]
)

log = logging.getLogger("alpaca")

__all__ = ["log", "console", "set_log_level"]

def set_log_level(level: str):
    """Set the logging level for the alpaca logger."""
    if level.lower() == "quiet":
        log.setLevel(logging.ERROR)
    elif level.lower() == "verbose":
        log.setLevel(logging.DEBUG)
    else: # standard
        log.setLevel(logging.INFO)
