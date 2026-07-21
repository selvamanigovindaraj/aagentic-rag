import logging
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger(__name__)


@contextmanager
def stage(name: str, **attributes):
    started = perf_counter()
    try:
        yield
    finally:
        logger.info(
            "agent_stage",
            extra={"stage": name, "duration_ms": (perf_counter() - started) * 1000, **attributes},
        )
