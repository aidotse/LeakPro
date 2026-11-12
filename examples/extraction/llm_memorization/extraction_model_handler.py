"""Model handler for the extraction example.

Extraction never trains or evaluates the target, so ``train``/``eval`` are not used. The tokenizer
comes from the target's ``pretrained_name_or_path``; define ``get_tokenizer(self)`` here to override
it.
"""

from leakpro import AbstractInputHandler


class ExtractionModelHandler(AbstractInputHandler, role="model"):
    """No data or training needed for extraction."""

    def train(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D102
        raise NotImplementedError("Extraction does not train the target")

    def eval(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D102
        raise NotImplementedError("Extraction does not evaluate the target")
