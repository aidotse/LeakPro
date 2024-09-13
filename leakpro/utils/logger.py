"""Module contains the function to setup the logger for the current run."""
import logging
import os


def setup_logger(name: str, save_file: bool=True, save_path:str=None) -> logging.Logger:
    """Generate the logger for the current run.

    Args:
    ----
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.
        save_path (str): Path to save the log file.

    Returns:
    -------
        logging.Logger: Logger object for the current run.

    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    # Console handler for output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    my_logger.addHandler(console_handler)

    if save_file:
        if save_path is None:
            raise ValueError("save_path cannot be None if save_file is True.")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        filename = f"{save_path}.log"
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger


def setup_logger(name: str = "leakpro_log", level: int = logging.INFO) -> logging.Logger:
    """Sets up a common logger with a console handler initially.

    Args:
    ----
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO).

    Returns:
    -------
        logging.Logger: Configured logger with a console handler.

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    # Console handler for output to the console
    if not logger.hasHandlers():  # Ensure no duplicate handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger

def add_file_handler(logger: logging.Logger, log_file_path: str) -> None:
    """Adds a file handler to the logger for logging to a specified file.

    Args:
    ----
        logger (logging.Logger): The logger instance to which the file handler will be added.
        log_file_path (str): The file path for the log file.

    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Create the file handler
    file_handler = logging.FileHandler(log_file_path)

    # Add file handler to logger
    logger.addHandler(file_handler)

logger = setup_logger(name="leakpro")
