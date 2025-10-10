from typing import Sequence, List, Any, Optional, Union, Dict, Literal
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from contextlib import contextmanager

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

Mode = Literal["raw", "tokenized"]

class TextInputHandler(AbstractInputHandler):
    class UserDataset(AbstractInputHandler.UserDataset):

        def __init__(
            self,
            data: Sequence[str],
            tokenizer: Union[str, PreTrainedTokenizerBase],
            *,
            mode: Mode = "tokenized",
            max_length: int = 256,
            truncation: bool = True,
            pad_to_max_length: bool = True,
            **kwargs: Any
        ) -> None:
            """Dataset for text LLM MIA."""
            
            self.data: List[str] = [str(x) for x in data]
            self.targets: List[int] = [0 for _ in data]

            # tokenizer (lazy for pickle safety)
            self._tokenizer: Optional[PreTrainedTokenizerBase] = None
            if isinstance(tokenizer, str):
                self.tokenizer_id: Optional[str] = tokenizer
            else:
                self._tokenizer = tokenizer
                self.tokenizer_id = getattr(tokenizer, "name_or_path", None)

            # config
            self.mode: Mode = mode
            self.max_length = int(max_length)
            self.truncation = bool(truncation)
            self.pad_to_max_length = bool(pad_to_max_length)

            # store aux fields safely
            for k, v in kwargs.items():
                setattr(self, k, v)

        # ---- tokenizer (lazy) ----
        @property
        def tok(self) -> PreTrainedTokenizerBase:
            if self._tokenizer is None:
                if not self.tokenizer_id:
                    raise RuntimeError("No tokenizer provided (neither object nor tokenizer_id).")
                tok = AutoTokenizer.from_pretrained(self.tokenizer_id, use_fast=True)
                tok.pad_token = tok.eos_token 
                self._tokenizer = tok
            return self._tokenizer

        def __getstate__(self) -> Dict[str, Any]:
            """Exclude tokenizer from state."""
            s = dict(self.__dict__)
            s["_tokenizer"] = None
            return s

        def __setstate__(self, state: Dict[str, Any]) -> None:
            """Restore state without tokenizer."""
            self.__dict__.update(state)
            self._tokenizer = None

        def set_mode(self, mode: Mode) -> None:
            """Set the mode for data retrieval."""
            if mode not in ("raw", "tokenized"):
                raise ValueError("mode must be 'raw' or 'tokenized'")
            self.mode = mode

        @contextmanager
        def as_mode(self, mode: Mode):
            """Temporarily set the mode for data retrieval."""
            old = self.mode
            self.set_mode(mode)
            try:
                yield self
            finally:
                self.mode = old

        def __getitem__(self, index: int):
            text = self.data[index]
            if self.mode == "raw":
                return self.data[index]  # text only
            # tokenized path (fixed shape so default collate works)
            enc = self.tok(
                text,
                padding=("max_length" if self.pad_to_max_length else False),
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            return item

    # ---- fail loudly per your request ----
    def train(self, dataloader, model, criterion, optimizer):
        raise NotImplementedError("TextInputHandler.train is intentionally unimplemented. Training must be handled by the caller.")

    def eval(self, dataloader, model, criterion, device):
        raise NotImplementedError("TextInputHandler.eval is intentionally unimplemented. Evaluation must be handled by the caller.")
