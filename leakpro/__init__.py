"""Init file for leakpro package."""

# make the main class and abstract input handler available
from .leakpro import (
    AbstractInputHandler,  # noqa: F401
    LeakPro,  # noqa: F401
)
