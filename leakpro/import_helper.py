"""Compatibility module for various Python versions."""
import sys
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union  # noqa: F401

if sys.version_info >= (3, 11):
    from typing import Self
else:
<<<<<<< HEAD
    Self = TypeVar("Self")
=======
    Self = TypeVar("Self")
>>>>>>> 2db0b25 (forgot importhelper.py)
