#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Module contains the function to setup the logger for the current run."""

import logging
import os


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
