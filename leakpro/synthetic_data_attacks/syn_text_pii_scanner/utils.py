"""Module with utility functions and classes used in synthetic text PII scanner."""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

import leakpro.synthetic_data_attacks.syn_text_pii_scanner.data_handling as dh
from leakpro.synthetic_data_attacks.syn_text_pii_scanner.pii_token_classif_models import ner_longformer_model as lgfm
from leakpro.synthetic_data_attacks.syn_text_pii_scanner.sentence_transformers_models.model import model_sen_trans


def get_device() -> str:
    """Auxiliary function that returns the device as a string."""
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    return dev

# Set device to CUDA or MPS if available, otherwise use CPU
device: torch.device = torch.device(get_device())

def load_json_data(*, file_path: str) -> List[Dict[str, Any]]:
    """Function to load data from json file in given file path."""
    with open(file_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    return data

class ModelInputOutput(BaseModel):
    """ModelInputOutput object holding model input and output from forward pass."""

    model_config = ConfigDict(arbitrary_types_allowed = True)
    inputs: np.ndarray
    logits: np.ndarray
    predictions: np.ndarray
    labels: Optional[np.ndarray]
    attention_mask_start: np.ndarray

class SubData(BaseModel):
    """SubData object holding input variables, raw_data, dataset, dataloader and model_input_output objects."""

    model_config = ConfigDict(arbitrary_types_allowed = True)
    model_config["protected_namespaces"] = ()
    path_or_data: Union[str, List[Dict[str, Any]]]
    label_set: Optional[dh.LabelSet] = None
    label_key: Optional[str] = None
    batch_size: int
    shuffle: bool
    num_workers: int
    raw_data: Optional[List[Dict[str, Any]]] = None
    dataset: Optional[dh.NERDataset] = None
    dataloader: Optional[DataLoader] = None
    model_input_output: Optional[ModelInputOutput] = None

class Data(BaseModel):
    """Data object holding original and synthetic SubData."""

    model_config = ConfigDict(arbitrary_types_allowed = True)
    ori: SubData
    syn: SubData

def load_data(*,
    data: Data,
    tokenizer: PreTrainedTokenizerFast,
    crossentropy_ignore_index: int = -100
) -> None:
    """Function to load data original and synthetic data.

    Function additionally appends to input data a dh.NERDataset object
    and a DataLoader.

    Arguments:
        data (Data): Data class holding input variables.
            Loaded objects (data, NERDataset and DataLoader) will be appended to it.
        tokenizer (PreTrainedTokenizerFast): The tokenizer used to encode the data.
        crossentropy_ignore_index (int): Ignore index for cross entropy loss (used in collator function). Defaults to -100.

    """
    #Assert Data class
    assert isinstance(data, Data), "Input data must be of type Data."
    #Load original and synthetic data
    for attr in ["ori", "syn"]:
        sd = getattr(data, attr)
        #Set raw data depending on path_or_data
        if isinstance(sd.path_or_data, str):
            #Load json raw data
            raw_data = load_json_data(file_path=sd.path_or_data)
        else:
            raw_data = sd.path_or_data
        #Get dataset
        dataset = dh.NERDataset(
            input_data = raw_data,
            tokenizer = tokenizer,
            label_set = sd.label_set,
            label_key = sd.label_key
        )
        #Get dataloader
        dataloader = DataLoader(
            dataset,
            collate_fn = dh.CollatorWPadding(
                tokenizer = tokenizer,
                crossentropy_ignore_index = crossentropy_ignore_index
            ),
            batch_size = sd.batch_size,
            shuffle = sd.shuffle,
            num_workers = sd.num_workers
        )
        #Append to data object
        sd.raw_data = raw_data
        sd.dataset = dataset
        sd.dataloader = dataloader

def load_model(*,
    num_labels: int,
    non_0_label_weight: int = 3,
    crossentropy_ignore_index: int = -100,
    path: Optional[str] = None
) -> lgfm.NERLongformerModel:
    """Function to load NERLongformerModel with specific num_labels.

    Arguments:
        num_labels (int): The number of labels.
        non_0_label_weight (int): Weight for non 0 labels. Defaults to 3.
        crossentropy_ignore_index(int): Ignore index for cross entropy loss. Defaults to -100.
        path (Optional[str]): The path of the model weights.
            If not provided (None), no weights are loaded.

    """
    # Set weights_crossentropy
    weights_crossentropy = torch.tensor(
        [1.0] + [non_0_label_weight*1.0 for _ in range(num_labels-1)],
        device = device,
        dtype = torch.float32
    )
    return lgfm.load_model(
        num_labels = num_labels,
        weights_crossentropy = weights_crossentropy,
        crossentropy_ignore_index = crossentropy_ignore_index,
        device = device,
        path = path
    )

def forward_pass(*, # noqa: C901
    data: Data,
    num_labels: int,
    model: lgfm.NERLongformerModel,
    verbose: bool = False
) -> None:
    """Function to do a forward pass on original and synthetic data.

    Function appends to model_input_output in data.

    Arguments:
        data (Data): Data class holding input variables.
            Input and output of model will be appended to model_input_output.
        num_labels (int): The number of labels.
        model (lgfm.NERLongformerModel): Model to perform forward pass with.
        verbose: bool, default is False
            If True, prints progress of forward pass.

    """
    #Assert input
    assert isinstance(data, Data), "Input data must be of type Data."
    assert isinstance(model, lgfm.NERLongformerModel), "Model must be of type NERLongformerModel."
    assert data.ori.raw_data is not None, "Load data must be run before forward pass."
    #Forward pass for original and synthetic input
    for attr in ["ori", "syn"]:
        if verbose:
            print(f"Starting forward_pass with {attr}.") # noqa: T201
        sd = getattr(data, attr)
        #Set with_labels_flag
        with_labels_flag = True
        if sd.label_set is None:
            with_labels_flag = False
        #Set model_limit
        model_limit = 4096 #Longformer model has 4096 limit
        #Get max_length and len_dataset
        max_length = min(sd.dataset.max_length, model_limit)
        len_dataset = len(sd.dataset)
        #Placeholders
        logits = torch.full((len_dataset, max_length, num_labels), -10000.0, dtype=torch.float32, device=device)
        inputs = torch.full((len_dataset, max_length), -100, dtype=torch.long, device=device)
        attention_mask = torch.full((len_dataset, max_length), 0, dtype=torch.long, device=device)
        labels = None
        if with_labels_flag:
            labels = torch.full((len_dataset, max_length), -100, dtype=torch.long, device=device)
        #Forward pass
        with torch.no_grad(), tqdm(total=len(sd.dataloader), disable=not verbose) as pbar:
            for j, X in enumerate(sd.dataloader): # noqa: N806
                #Send vectors to gpu
                for key in ["input_ids", "attention_mask"]:
                    X[key] = X[key].to(device)
                #Get batch size and length
                batch_size, length = X["input_ids"][:, 0:max_length].shape
                #Prepare start and end
                start = j * sd.dataloader.batch_size
                end = start + batch_size
                #Set inputs and attention_mask
                inputs[start:end, (max_length-length):] = X["input_ids"][:, 0:max_length]
                attention_mask[start:end, (max_length-length):] = X["attention_mask"][:, 0:max_length]
                #Set labels
                if with_labels_flag:
                    X["labels"] = X["labels"].to(device)
                    labels[start:end, (max_length-length):] = X["labels"][:, 0:max_length]
                #Forward pass
                Y = model(**X) # noqa: N806
                #Set logits
                logits[start:end, (max_length-length):, :] = Y.logits
                if verbose:
                    pbar.update(1)
        #Convert output to numpy
        inputs = inputs.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()
        logits = logits.float().cpu().numpy()
        predictions = np.argmax(logits, axis=2)
        predictions = np.where(attention_mask==0, 0, predictions)
        attention_mask_start = np.argmax(attention_mask == 1, axis=1)
        if with_labels_flag:
            labels = labels.cpu().numpy()
            labels = np.where(labels==-100, 0, labels)
        #Set input and output to data model_input_output
        sd.model_input_output = ModelInputOutput(
            inputs = inputs,
            logits = logits,
            predictions = predictions,
            labels = labels,
            attention_mask_start = attention_mask_start
        )
        if verbose:
            print(f"Ending forward_pass with {attr}.") # noqa: T201


class PII(BaseModel):
    """PII object holding document number, start and end token indexes, the tokens and text of the PII."""

    doc_nr: int
    start_end_tok_idx: Tuple[int, int]
    tokens: List[int]
    text: str

    def __init__(self, *,
        doc_nr: int,
        input: np.ndarray,
        start: int,
        end: int,
        tokenizer: PreTrainedTokenizerFast
    ) -> None:
        assert end>start, "End must be greater than start in PII."
        input = input[start:end]
        tokens = input.tolist()
        super().__init__(
            doc_nr = doc_nr,
            start_end_tok_idx = (start, end),
            tokens = tokens,
            text = tokenizer.decode(input).strip()
        )

def get_PIIs_01(*, # noqa: N802
    labels: np.ndarray,
    inputs: np.ndarray,
    tokenizer: PreTrainedTokenizerFast
) -> List[PII]:
    """Function to get a list of PIIs from passed labels and inputs.

    Notes: labels and inputs must have the same shape.
    Each row in labels and inputs is a document and will be numbered accordingly in resulting list of PIIs.

    Arguments:
        labels (np.ndarray): Array of 0s or 1s. If 1, then corresponding inputs are a PII.
        inputs (np.ndarray): Array of tokens.
        tokenizer (PreTrainedTokenizerFast): The tokenizer used to encode the data.

    """
    #Assert shapes
    assert labels.shape == inputs.shape, "Labels and inputs must have the same shape."
    #Placeholder for PIIs
    piis = []
    #Iterate through documents
    for doc_nr in range(labels.shape[0]):
        #Get labels_ and input
        labels_ = labels[doc_nr]
        input = inputs[doc_nr]
        # Set start
        start = None
        # Iterate through labels_
        for i, bit in enumerate(labels_):
            if bit == 1 and start is None:
                start = i  # Mark the start of a block of 1s
            elif bit == 0 and start is not None:
                piis.append(PII(
                    doc_nr = doc_nr,
                    input = input,
                    start = start,
                    end = i,
                    tokenizer = tokenizer
                ))
                start = None # Reset start
        # If sequence ends with 1s, close the last block
        if start is not None:
            piis.append(PII(
                doc_nr = doc_nr,
                input = input,
                start = start,
                end = i+1,
                tokenizer = tokenizer
            ))
    return piis

def aux_piis_texts_and_doc_nrs_start_end(*,
    piis: List[PII]
) -> Tuple[List[PII], List[str], Dict[int, Tuple[int, int]]]:
    """Auxiliary function to get ordered piis, texts and document numbers start and end indexes.

    Arguments:
        piis (List[PII]): List of PIIs.

    Returns:
        List[PII]: List of PIIs ordered by doc_nr.
        List[str]: List of texts from PIIs.
        Dict[int, Tuple[int, int]]: Doc_nr dictionary with start and end indexes.

    """
    # Sort piis per doc_nrs
    piis = sorted(piis, key=lambda x: x.doc_nr)
    # Placeholder texts and doc_nrs_start_end
    texts = []
    doc_nrs_start_end = {}
    # Iterate through piis to get texts and doc_nrs_start_end
    start = 0
    doc_nr = None
    for i, pii in enumerate(piis):
        #Append to texts
        texts.append(pii.text)
        if doc_nr != pii.doc_nr:
            if doc_nr is not None:
                # Set end
                end = i
                # Add start end to doc_nrs_start_end
                doc_nrs_start_end[doc_nr] = [start, end]
            #Set start and doc_nr
            start = i
            doc_nr = pii.doc_nr
    # Add final doc_nr
    doc_nrs_start_end[doc_nr] = [start, len(piis)]
    return piis, texts, doc_nrs_start_end

def detect_non_public_pii(*,
    piis: List[PII],
    similarity_threshold: float,
    min_nr_repetitions: int
) -> List[PII]:
    """Function to detect from a list of PIIs the non-public PIIs.

    Function compares each PII against PIIs in other documents. Similar PIIs are selected
    based on similarity_threshold input. If there are more than min_nr_repetitions, the PII
    is deemed public, otherwise non-public.

    Arguments:
        piis (List[PII]): List of PIIs.
        similarity_threshold (float): Cosine similarity threshold to consider PIIs as similar.
        min_nr_repetitions (int): Minimum number of repetitions in different documents for a PII to be deemed as public.

    Returns:
        List[PII]: List of PIIs that are deemed as non-public.

    """
    # Set length of PIIs
    len_piis = len(piis)
    # Get sorted piis, texts and doc_nrs_start_end
    piis, texts, doc_nrs_start_end = aux_piis_texts_and_doc_nrs_start_end(piis=piis)
    # Embed all PIIs in the list
    embeddings = model_sen_trans.encode(texts, convert_to_tensor=True)
    # Calculate similarities
    similarities = model_sen_trans.similarity(embeddings, embeddings)
    # Placeholder list for non-public PIIs
    non_public_piis = []
    # Compare each PII with every other PII in other documents
    for i, pii in enumerate(piis):
        start, end = doc_nrs_start_end[pii.doc_nr]
        other_indices = torch.arange(end, len_piis)
        if start>0:
            other_indices = torch.cat((torch.arange(0, start), other_indices))
        similarities_i = similarities[i, other_indices]
        # Count how many times this PII is repeated with high similarity in other documents
        similar_items = similarities_i >= similarity_threshold
        similar_indices = torch.nonzero(similar_items).flatten()  # Get indices of similar items
        doc_nrs = set()
        for idx in similar_indices:
            ori_index = other_indices[idx.item()] # Original index in piis list
            doc_nr = piis[ori_index].doc_nr
            # Add document number to doc_nrs
            doc_nrs.add(doc_nr)
        # Determine the number of unique documents where this PII is repeated or similar
        repetition_count = len(doc_nrs)
        # Determine if the PII is public based on the repetition count
        is_public = repetition_count >= min_nr_repetitions
        if not is_public:
            non_public_piis.append(pii)
    return non_public_piis

def round_to_6(x: float) -> float:
    """Auxiliary function to round input float to 6 decimals."""
    return round(float(x), 6)

def calc_distribution(*,
    array: np.ndarray,
    percentiles: List[int] = [10, 25, 50, 75, 90, 99] # noqa: B006
) -> None:
    """Auxiliary function to calculate distribution of passed array for given percentiles."""
    array = np.sort(array.reshape(-1))
    length = array.size
    distr = {
        "mean": round_to_6(array.mean()),
        "0": round_to_6(array.min())
    }
    for p in percentiles:
        idx = int(length*p/100)
        distr[str(p)] = round_to_6(array[idx])
    distr["100"] = round_to_6(array.max())
    return distr

def print_distribution(*,
    distr: dict
) -> None:
    """Auxiliary function to print passed distribution."""
    print("Start print_distribution") # noqa: T201
    keys = sorted(distr.keys())
    keys.remove("mean")
    print(f"Mean: {distr['mean']}") # noqa: T201
    for k in keys:
        print(f"{k}th Percentile: {distr[k]}") # noqa: T201
    print("End print_distribution") # noqa: T201

def count_sort_similar_items(*,
    similar_items: Tuple[np.ndarray, np.ndarray],
    ori_piis: List[PII],
    syn_piis: List[PII]
) -> List:
    """Auxiliary function to count and sort similar items."""
    sim_items_dict = {}
    #First iteration
    for x, y in zip(*similar_items):
        x = int(x)
        y = int(y)
        sim_items_dict.setdefault(x, {}).setdefault("ori_doc_nr", ori_piis[x].doc_nr)
        sim_items_dict[x].setdefault("ori_text", ori_piis[x].text)
        sim_items_dict[x].setdefault("syn_items", []).append(y)
        sim_items_dict[x].setdefault("syn_docs", set()).add(syn_piis[y].doc_nr)
    #Second iteration
    for k, v in sim_items_dict.items():
        sim_items_dict[k]["len_syn_items"] = len(v["syn_items"])
        sim_items_dict[k]["syn_docs"] = sorted(v["syn_docs"])
        sim_items_dict[k]["len_syn_docs"] = len(v["syn_docs"])
    return sorted(
        ({"ori_item": k, **v} for k, v in sim_items_dict.items()), # Include new key ori_item and flatten v into the new dict
        key = lambda item: item["len_syn_docs"], # Sort by 'len_syn_docs'
        reverse = True
    )

def compare_piis_lists(*,
    ori_piis: List[PII],
    syn_piis: List[PII],
    similarity_threshold: float,
    ignore_list: List[str] = [], # noqa: B006
    verbose: bool = False
) -> Tuple[int, int, List[Dict], Dict]:
    """Function to compare original and synthetic list of PIIs for similarities.

    Function compares each PII against PIIs in other documents. Similar PIIs are selected
    based on similarity_threshold input.

    Arguments:
        ori_piis (List[PII]): List of original PIIs.
        syn_piis (List[PII]): List of synthetic PIIs.
        similarity_threshold (float): Cosine similarity threshold to consider PIIs as similar.
        ignore_list (List[str]): List of text PIIs to ignore from original list.
        verbose: bool, default is False
            If True, prints distribution of similarities.

    Returns:
        int: Similar number of items.
        int: Total number of items.
        List[Dict]: List of dictionaries holding a counted sort of similarities.
        Dict: Distribution of similarities.

    """
    if verbose:
        print("\nStart compare_piis_lists") # noqa: T201
    #Filter ori_piis with ignore list
    ori_piis = [pii for pii in ori_piis if pii.text not in ignore_list]
    #Get embeddings
    embeddings = []
    for piis in [ori_piis, syn_piis]:
        text = [pii.text for pii in piis]
        embeddings.append(model_sen_trans.encode(text))
    #Get similarities and calculate similarities distribution
    similarities = model_sen_trans.similarity(embeddings[0], embeddings[1]).numpy()
    distr = calc_distribution(array=similarities)
    #Get similar_items
    similar_items = np.where(similarities >= similarity_threshold)
    #Get sorted_sim_items
    sorted_sim_items = count_sort_similar_items(
        similar_items = similar_items,
        ori_piis = ori_piis,
        syn_piis = syn_piis
    )
    #Get similar and total items
    sit = similar_items[0].size
    tot = similarities.size
    if verbose:
        print_distribution(distr=distr)
        print(f"Nr. Similar Items: {sit:,}") # noqa: T201
        print(f"Total Items: {tot:,}") # noqa: T201
        print(f"Percentage: {round_to_6(sit/tot*100)}%") # noqa: T201
        print("End compare_piis_lists") # noqa: T201
    return sit, tot, sorted_sim_items, distr
