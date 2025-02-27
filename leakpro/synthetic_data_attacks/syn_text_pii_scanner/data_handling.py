"""Module containing LabelSet, EncodedDoc, NERDataset and CollatorWPadding classes to handle text data used in synthetic text PII scanner.""" # noqa: E501
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerFast


class LabelSet:
    """Manages the mapping between string labels and their corresponding IDs.

    Attributes:
        IOB2_FORMAT (bool): if True, LabelSet constructed using IOB2 format, otherwise IO format used.
        labels_to_id (Dict[str, int]): Mapping from label strings to IDs.
        ids_to_label (Dict[int, str]): Mapping from IDs to label strings.
        labels (List[str]): Original labels
        f_labels (List[str]): Final labels

    """

    def __init__(self, *, labels: List[str], IOB2_FORMAT: bool = False) -> None: # noqa: N803
        """Initialize the LabelSet with a list of labels.

        Args:
            labels (List[str]): List of unique label strings.
            IOB2_FORMAT (bool): if True, LabelSet constructed using IOB2 format, otherwise IO format used. Defaults to False.

        """
        #Set labels and IOB2_FORMAT
        self.labels = labels
        self.IOB2_FORMAT = IOB2_FORMAT
        #Initialize labels_to_id and ids_to_label
        self.labels_to_id = {"O": 0}
        self.ids_to_label = {0: "O"}
        #Populate labels_to_id and ids_to_label depending on IOB2_FORMAT flag
        if IOB2_FORMAT:
            for num, (label, s) in enumerate(itertools.product(labels, "BI"), start=1):
                label = f"{s}-{label}"
                self.labels_to_id[label] = num
                self.ids_to_label[num] = label
        else:
            for num, label in enumerate(labels, start=1):
                self.labels_to_id[label] = num
                self.ids_to_label[num] = label
        #Set f_labels
        self.f_labels = list(self.labels_to_id.keys())

class EncodedDoc(BaseModel):
    """Represents a encoded document with its token ids, offsets and labels (optional).

    Attributes:
        input_ids (List[int]): The input token IDs.
        offsets (List[Tuple[int, int]]): The offsets for each token in input text.
        labels (Optional[List[int]]): The labels for each token.

    """

    input_ids: List[int]
    offsets: List[Tuple[int, int]]
    labels: Optional[List[int]]

class NERDataset(Dataset):
    """Dataset for Named Entity Recognition (e.g. token classification) tasks.

    Attributes:
        tokenizer (PreTrainedTokenizerFast): The tokenizer used to process the text.
        label_set (Optional[LabelSet]): The set of labels used to annotate documents.
        label_key (Optional[str]): The label key to review under annotations.
        data (List[EncodedDoc]): Encoded (processed) list of documents.
        max_length (int): The maximum encoded length in data.

    """

    def __init__(
        self,
        input_data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        label_set: Optional[LabelSet] = None,
        label_key: Optional[str] = None
    ) -> None:
        """Initialize the NERDataset.

        Args:
            input_data (List[Dict[str, Any]]): The raw input data (text documents) to process.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
            label_set (Optional[LabelSet]):
                The set of labels used to annotate documents. Use if data is annotated, otherwise None.
            label_key (Optional[str]): The label key to review under annotations. Use if data is annotated, otherwise None.

        """
        # Validate input label_set and label_key
        if label_set is not None or label_key is not None:
            if label_set is None or label_key is None:
                raise Exception("label_set and label_key have to either be both None or both not None.")
            assert isinstance(label_set, LabelSet), "label_set needs to be a LabelSet."
            assert isinstance(label_key, str), "label_key needs to be a string."
        # Set attributes
        self.tokenizer = tokenizer
        self.label_set = label_set
        self.label_key = label_key
        # Process input data
        self.data, self.max_length = self.process_input_data(input_data=input_data)

    def pre_suf_fun(self, x: str) -> str:
        """Function that adds beginining of sequence and end of sequence tokens before and after string."""
        return self.tokenizer.bos_token + x + self.tokenizer.eos_token

    def process_input_data(self, *, input_data: List[Dict[str, Any]]) -> List[EncodedDoc]:
        """Process input data into list of encoded documents (EncodedDoc).

        Args:
            input_data (List[Dict[str, Any]]): The raw input data (text documents) to process.

        Returns:
            List[EncodedDoc]: The encoded (processed) list of documents.
            int: The maximum encoded length document in processed list of documents.

        """
        #Placeholder processed_data and max_length
        processed_data: List[EncodedDoc] = []
        max_length: int = 0
        #Iterate through input documents
        for doc in input_data:
            #Tokenize document (using pre_suf_fun)
            encoding = self.tokenizer(
                self.pre_suf_fun(doc["text"]),
                add_special_tokens = False,
                return_attention_mask = False,
                return_offsets_mapping = True
            )
            #Get aligned labels if self.label_set is not None
            if self.label_set is not None:
                #Get aligned labels
                aligned_labels = self.align_labels(
                    encoding = encoding,
                    annotations = doc["annotations"]
                )
            else:
                aligned_labels = None
            #Append EncodedDoc to processed_data
            processed_data.append(EncodedDoc(
                input_ids = encoding["input_ids"],
                labels = aligned_labels,
                offsets = encoding["offset_mapping"]
            ))
            #Check for max_length
            max_length = max(len(encoding["input_ids"]), max_length)
        return processed_data, max_length

    def align_labels(self, *,
        encoding: BatchEncoding,
        annotations: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[str], List[Dict[str, Tuple[int, int]]]]:
        """Align labels with tokenized input.

        Args:
            encoding (BatchEncoding): The tokenized input.
            annotations (List[Dict[str, Any]]): The annotations for the input.

        Returns:
            List[int]: List of aligned labels

        """
        #Set variables
        offsets_map = np.array(encoding["offset_mapping"])
        len_seq = len(encoding["input_ids"])
        aligned_labels: List[int] = [0] * len_seq  # 0 corresponds to "O" label
        prefix_length = len(self.tokenizer.bos_token)
        #Iterate through annotations
        for anno in annotations:
            #Annotated label key value is in label_set.labels
            if anno[self.label_key] in self.label_set.labels:
                #Calculate start and end token indexes
                start = prefix_length + anno["start_offset"]
                end = prefix_length + anno["end_offset"]
                cond_start = offsets_map[:, 0] >= start
                cond_end = offsets_map[:, 1] <= end
                cond = cond_start & cond_end
                matches = np.nonzero(cond)[0]
                start_tok_idx = matches.min()
                end_tok_idx = matches.max()
                #Iterate through token indexes and set aligned labels depending on IOB2_FORMAT
                for num, token_ix in enumerate(range(start_tok_idx, end_tok_idx+1)):
                    if self.label_set.IOB2_FORMAT:
                        prefix = "B" if num == 0 else "I"
                        aligned_labels[token_ix] = self.label_set.labels_to_id[f"{prefix}-{anno[self.label_key]}"]
                    else:
                        aligned_labels[token_ix] = self.label_set.labels_to_id[anno[self.label_key]]
        return aligned_labels

    def __len__(self) -> int:
        """Get the number of encoded documents.

        Returns:
            int: The number of encoded documents.

        """
        return len(self.data)

    def __getitem__(self, idx: int) -> EncodedDoc:
        """Get a single encoded document by index.

        Args:
            idx (int): The index of the encoded document.

        Returns:
            EncodedDoc: The encoded document at the given index.

        """
        return self.data[idx]

@dataclass
class CollatorWPadding:
    """Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        crossentropy_ignore_index (int): The index to be ignored for CrossEntropyLoss in labels. Default value -100.
        with_labels_flag (bool): True if collator will receive and include labels in batch, False otherwise.

    """

    tokenizer: PreTrainedTokenizerFast
    crossentropy_ignore_index: int = -100

    def __call__(self, features: List[EncodedDoc]) -> Dict[str, torch.tensor]: # noqa: D102
        #Calculate max_length
        max_length = 0
        for feature in features:
            max_length = max(len(feature.input_ids), max_length)
        #Set with_labels_flag depending if features have labels
        with_labels_flag = True
        if features[0].labels is None:
            with_labels_flag = False
        #Set batch_fill_values
        batch_fill_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 1
        }
        if with_labels_flag:
            batch_fill_values["labels"] = self.crossentropy_ignore_index
        #Initiate batch
        batch = {}
        for key, fill_value in batch_fill_values.items():
            batch[key] = torch.full((len(features), max_length), fill_value, dtype=torch.long)
        #Update batch depending on left or right padding
        for i, feature in enumerate(features):
            len_feats = len(feature.input_ids)
            len_pad = max_length - len_feats
            if self.tokenizer.padding_side == "left":
                batch["input_ids"][i, len_pad:] = torch.tensor(feature.input_ids, dtype=torch.long)
                batch["attention_mask"][i, 0:len_pad] = 0
                if with_labels_flag:
                    batch["labels"][i, len_pad:] = torch.tensor(feature.labels, dtype=torch.long)
            else:
                batch["input_ids"][i, 0:len_feats] = torch.tensor(feature.input_ids, dtype=torch.long)
                batch["attention_mask"][i, len_feats:] = 0
                if with_labels_flag:
                    batch["labels"][i, 0:len_feats] = torch.tensor(feature.labels, dtype=torch.long)
        return batch
