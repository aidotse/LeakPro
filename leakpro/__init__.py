"""Init file for leakpro package."""

# make the main class and abstract input handler available
from .leakpro import LeakPro  # noqa: F401
from .user_inputs.abstract_input_handler import AbstractInputHandler  # noqa: F401
