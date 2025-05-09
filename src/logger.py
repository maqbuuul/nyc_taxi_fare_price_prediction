import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging():
    """Set up logging with appropriate configuration"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{timestamp}.log")

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Create and get the logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# Initialize logging when module is imported
logger = setup_logging()


def get_logger(name):
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)
