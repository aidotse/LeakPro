"""Compatibility module for various Python versions."""
import sys
from types import ModuleType  # noqa: F401
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union  # noqa: F401

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar("Self")
