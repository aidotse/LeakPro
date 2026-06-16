"""Base class for modality-specific input handler extensions."""

from abc import ABC, abstractmethod

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Self


class AbstractModalityExtension(ABC):
    """Base class for extensions that add modality-specific helpers to a handler."""

    def __init__(self: Self, handler: AbstractInputHandler) -> None:
        self.handler = handler
        self.population = getattr(handler, "population", None)
        self.public_dataset = getattr(handler, "public_dataset", None)
        self.private_dataset = getattr(handler, "private_dataset", None)

    @abstractmethod
    def augmentation(self: Self, data, n_aug: int):
        """Return augmented samples for the input data."""
        pass
