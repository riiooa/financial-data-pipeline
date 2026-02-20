from inspect import getargs
import logging
import logging.config
import json
import time
import os
from functools import wraps
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from turtle import back
from typing import Any, Dict, Optional
from venv import logger

from sqlalchemy import exc

def setup_logging(config_file: str = "config/logging.conf", default_level=logging.INFO):
    """Setup logging configuration with automatic folder creation"""
    log_dir = "/opt/airflow/logs"
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except:
            pass

    config_path = Path(config_file)
    if config_path.exists():
        logging.config.fileConfig(config_file, disable_existing_loggers=False)
    else:
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    logging.info("Success Logging configured")

class StructuredLogger:
    """Logger that produces JSON output for monitoring system integration"""
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: str, message: str, **kwargs):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.upper(),
            'message': message,
            'logger': self.logger.name,
            **kwargs
        }
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json.dumps(log_entry))

    def info(self, msg, **kwargs): self._log('info', msg, **kwargs)
    def error(self, msg, **kwargs): self._log('error', msg, **kwargs)
    def warning(self, msg, **kwargs): self._log('warning', msg, **kwargs)
    def debug(self, msg, **kwargs): self._log('debug', msg, **kwargs)

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """Decorator retry with cleaner exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            curr_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise

                    logger.warning(f"Attemp {attempt}/{max_attempts} failed: {e}. Retrying in {curr_delay}s...")
                    time.sleep(curr_delay)
                    curr_delay *= backoff

        return wrapper
    return decorator

@contextmanager
def timer(name: str= "operation"):
    """Context manager to measure execution duration"""
    logger = logging.getLogger(__name__)
    start = time.time()
    try:
        yield
    finally:
        logger.info(f"{name} completed in {time.time() - start:.2f} seconds")

@contextmanager
def db_transaction(connection, name: str = "transaction"):
    """Secure database transaction context manager"""
    logger = logging.getLogger(__name__)
    try:
        yield connection
        connection.commit()
        logger.info(f"Success!. {name} committed")
    except Exception as e:
        connection.rollback()
        logger.error(f"ERROR!: {name} rolled back due to error: {e}")
        raise