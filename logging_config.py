import logging
import sys
from datetime import datetime


# Define custom PRINT level (between INFO and WARNING)
PRINT_LEVEL = 25
logging.addLevelName(PRINT_LEVEL, 'PRINT')


def print_log(self, message, *args, **kwargs):
    """
    Custom logger method for PRINT level.
    Use: logger.print("message")
    """
    if self.isEnabledFor(PRINT_LEVEL):
        self._log(PRINT_LEVEL, message, args, **kwargs)


# Add print method to Logger class
logging.Logger.print = print_log


class LevelFilter(logging.Filter):
    """
    Filter that only allows logs of a specific level.
    
    This ensures that each log file contains only its designated level,
    without duplication across files.
    """
    def __init__(self, level):
        super().__init__()
        self.level = level
    
    def filter(self, record):
        return record.levelno == self.level


def setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=False):
    """
    Configure logging with separate log files for each level.
    
    Creates the following structure:
    - logs/debug/bnp_TIMESTAMP.log    - Only DEBUG messages
    - logs/info/bnp_TIMESTAMP.log     - Only INFO messages
    - logs/warning/bnp_TIMESTAMP.log  - Only WARNING messages
    - logs/error/bnp_TIMESTAMP.log    - Only ERROR messages
    
    Console output: 
    - If print_all_logs=False: Only shows PRINT level (use logger.print("message"))
    - If print_all_logs=True: Shows all log levels (DEBUG, INFO, WARNING, ERROR, PRINT)
    
    Args:
        base_log_dir: Base directory for log files (default: 'logs')
        enable_console: If True, console will show log messages (default: True)
        print_all_logs: If True, console shows all levels; if False, only PRINT level (default: False)
    
    Returns:
        root_logger: Configured root logger
    """
    import os
    
    # Create formatter for files
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create formatter for console
    if print_all_logs:
        # Full format for console when showing all logs
        console_formatter = logging.Formatter(
            fmt='%(levelname)-8s | %(name)s | %(message)s'
        )
    else:
        # Simple format for PRINT level only
        console_formatter = logging.Formatter(
            fmt='%(message)s'
        )
    
    # Configure root logger (set to DEBUG to capture everything)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if print_all_logs:
            # Show all levels
            console_handler.setLevel(logging.DEBUG)
        else:
            # Only PRINT level
            console_handler.setLevel(PRINT_LEVEL)
            console_handler.addFilter(LevelFilter(PRINT_LEVEL))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Remove the file handlers code entirely
    # No directories or log files will be created
    
    return root_logger


def setup_logging(log_level='INFO', log_to_file=True, log_dir='logs'):
    """
    Configure logging for Branch-and-Price solver (single-file mode).
    
    This is the simple version that writes all logs to a single file.
    Use setup_multi_level_logging() for separate files per level.

    Args:
        log_level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        log_to_file: If True, also write logs to file
        log_dir: Directory for log files
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (removed to prevent creating logs/)
    # (The log_to_file and log_dir arguments are kept for backward compatibility with existing calls,
    # but they do nothing now)

    return root_logger


def get_logger(name):
    """Get a logger for a specific module."""
    return logging.getLogger(name)