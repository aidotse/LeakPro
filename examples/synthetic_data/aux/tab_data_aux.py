"""Module with function to modify TAB dataset to be used with synthetic text PII scanner."""
from typing import Any, Dict, List

from leakpro.synthetic_data_attacks.syn_text_pii_scanner import utils


def tab_data_treatment(*,
    tab_path: str,
    only_first_annotator_flag: bool,
    include_tasks_flag: bool = False
) -> List[Dict[str, Any]]:
    """Function to modify the TAB dataset json file to be used with synthetic text PII scanner.

    All annotators decisions are collapsed consider as equally correct training examples.
    Read more about the TAB dataset (https://github.com/NorskRegnesentral/text-anonymization-benchmark).

    Args:
        tab_path (str): The path to the TAB JSON file.
        only_first_annotator_flag (bool): If True, only the first annotator's annotations will be included.
            If False, all annotators' annotations will be included.
        include_tasks_flag (bool): If True, the tasks will be added to the documents' texts.

    Returns:
        List[Dict[str, Any]]: A list of documents with text and annotations entries.

    """
    # Load the JSON data from the file
    data = utils.load_json_data(file_path=tab_path)
    # Initialize modified_data to return
    modified_data: List[Dict[str, Any]] = []
    # Process each document in the data
    for ann_data in data:
        # Placeholder document
        dct = {
            "split": ann_data["dataset_type"],
            "text": ann_data["text"],
            "doc_id": ann_data["doc_id"],
        }
        doc_id = ann_data["doc_id"]
        if include_tasks_flag:
            text_to_add = "\nDocument:\n"
            dct["task"] = ann_data["task"]
            task_str = "Task: Annotate the document to anonymise the following person: "
            len_ts = len(task_str)
            assert dct["task"][:len_ts] == "Task: Annotate the document to anonymise the following person: "
            dct["text"] = dct["task"] + text_to_add + dct["text"]
            task_span = dct["task"][len_ts:]
            end = len_ts + len(task_span)
            task_annotation = {
                "entity_type": "PERSON",
                "start_offset": len_ts,
                "end_offset": end,
                "span_text": task_span,
                "identifier_type": "DIRECT",
                "confidential_status": "NOT_CONFIDENTIAL",
                "label": "MASK",
                "more_ex_label": "MASK",
                "id": doc_id
            }
            task_offset = len(dct["task"]) + len(text_to_add)
        # Set annotators
        annotators = sorted(ann_data["annotations"].keys())
        if only_first_annotator_flag:
            annotators = annotators[0:1]
        # Collect annotations for each annotator
        for annotator in annotators:
            annotations = []
            if include_tasks_flag:
                annotations.append(task_annotation)
            for annotation in ann_data["annotations"][annotator]["entity_mentions"]:
                annotation["label"] = "MASK" if annotation["identifier_type"] != "NO_MASK" else "NO_MASK"
                annotation["id"] = doc_id
                #More exclusive label
                all_code = annotation["entity_type"]=="CODE"
                all_person = annotation["entity_type"]=="PERSON"
                more_exclusive_label_cond = any((
                    all_code,
                    all_person
                ))
                annotation["more_ex_label"] = "MASK" if more_exclusive_label_cond else "NO_MASK"
                #Add task_offset if include_tasks_flag
                if include_tasks_flag:
                    annotation["start_offset"] += task_offset
                    annotation["end_offset"] += task_offset
                annotations.append(annotation)
            #Set annotations and append dct to modified_data
            dct["annotations"] = annotations
            modified_data.append(dct)
    return modified_data
