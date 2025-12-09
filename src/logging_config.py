#!/usr/bin/env python3
"""
Centralized logging configuration for the NNforHJB repository.

This module provides a single function to configure loguru logging consistently
across all modules in the repository. It should be called once at application startup.
"""

import os
import sys
from loguru import logger


def setup_logging(verbose=True, log_file=None, log_level="DEBUG", rotation="10 MB"):
    """
    Configure loguru logger for the entire repository.
    
    This function should be called once at the start of your application/notebook.
    It sets up both console and file logging with consistent formatting.
    
    Args:
        verbose (bool): Whether to print logs to terminal (default: True)
        log_file (str, optional): Path to log file. If None, uses default location
                                   in log_history/training.log relative to project root
        log_level (str): Minimum log level for file logging (default: "DEBUG")
        rotation (str): Log file rotation size (default: "10 MB")
    
    Example:
        >>> from src.logging_config import setup_logging
        >>> setup_logging(verbose=True)
        >>> from loguru import logger
        >>> logger.info("This will be logged to both console and file")
    """
    # Remove default handler to avoid duplicates
    logger.remove()
    
    if verbose:
        # Add terminal output with colored formatting
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )
    
    # Determine log file path
    if log_file is None:
        # Default: log_history/training.log relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_history_dir = os.path.join(current_dir, "..", "log_history")
        os.makedirs(log_history_dir, exist_ok=True)
        log_file = os.path.join(log_history_dir, "training.log")
    
    # Always log to file (even if verbose=False)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention="30 days",  # Keep logs for 30 days
        compression="zip"  # Compress old logs
    )
    
    if verbose:
        logger.info("Logging configured successfully")


def get_logger():
    """
    Get the configured logger instance.
    
    This is a convenience function that returns the loguru logger.
    You can also directly import: from loguru import logger
    
    Returns:
        Logger: The configured loguru logger instance
    """
    return logger


