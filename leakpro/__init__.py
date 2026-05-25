#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Init file for leakpro package."""

# make the main class and abstract input handler available
from .input_handler.abstract_input_handler import AbstractInputHandler  # noqa: F401
from .leakpro import LeakPro  # noqa: F401
