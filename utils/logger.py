"""Standard logger for the TruthScope system.

This module provides a unified logging interface that logs
messages both to stdout and to an output file.
"""

import logging
import os
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    """Sets up and returns a standard logger for TruthScope.
    
    Args:
        name: The name of the logger instance (e.g., __name__).
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid attaching handlers multiple times if instantiated repeatedly
    if logger.handlers:
        return logger
        
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    
    # Standard output console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler writing to logs/truthscope.log
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "truthscope.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Do not propagate to root logger to prevent duplicate logs
    logger.propagate = False
    
    return logger
