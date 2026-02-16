import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable to track run_id across async calls
run_id_ctx: ContextVar[Optional[str]] = ContextVar("run_id", default=None)

def get_run_id() -> str:
    """Retrieve the current run_id or generate a new one if not set."""
    rid = run_id_ctx.get()
    if rid is None:
        rid = str(uuid.uuid4())
        run_id_ctx.set(rid)
    return rid

class RunIDFilter(logging.Filter):
    """Injects run_id into log records."""
    def filter(self, record):
        record.run_id = get_run_id()
        return True

def configure_logging(level: str = "INFO"):
    """Configures the root logger with a standard format including run_id."""
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [%(run_id)s] | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    
    # Add filter to inject run_id
    handler.addFilter(RunIDFilter())
    
    logger.addHandler(handler)

    # Silence noisy libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize logging on import with default settings
configure_logging()
logger = logging.getLogger("sred")
