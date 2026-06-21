import os
import sys
import structlog
import logging
from pathlib import Path
from omegaconf import OmegaConf

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if os.getenv("ENV") == "prod" else structlog.processors.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def load_config():
    config_path = Path("config.yml")
    if not config_path.exists():
        raise FileNotFoundError("config.yml not found")
    
    conf = OmegaConf.load(config_path)
    # Merge with env variables if needed
    return conf

setup_logging()
logger = structlog.get_logger()
config = load_config()
