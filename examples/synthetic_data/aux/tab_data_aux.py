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

#Predefined ignore PIIs list
tab_predefined_ignore_piis_list = [
    "576", "119",
    "Governor", "Minister", "Mary", "wife",
    "İ.Y.", "İ.Y", "İ.G.", "C.", "W.S", "Mr R.", "Ms S.", "C.K", "M.W.", "M.W", "N.B", "Mr R.G", "D. R.",
    "N.B.", "N. B.", "Z.O.", "R.H", "R.O.", "R.O",
    "A.S.K.", "W.B.", "S.Y", "S.Y.", "Mr M.Y.", "M.Y.", "M.Y", "Ü.K", "Ms L.", "Mr E", "Z.Z",
    "O. M. U", "O. M. U.", "E.P.", "E.P",
    "C.R", "J.L.", "A. J.", "N.C.", "N.C", "Mrs P", "Mr T", "Mr H", "H.Y.", "H.Y",
    "R.P", "R.P.", "Mr A.Ç.", "Mr A. Ç", "Dr M.", "Dr M", "Dr G", "Dr. G.",
    "P. O.", "I", "I.C", "Mrs K", "M.A.Z", "J.W.", "S.S.", "Mr J.C.", "İ.G", "I.",
    "Mr J", "J.B", "Dr. S", "Mrs K.", "J.B.", "W.P.", "W.P", "Mrs Z.", "Mr F", "M.A.Z.", "D.B", "M.U.", "Mr B.Z",
    "J.T.", "Mr H.P", "L.H.", "F.I", "Dr O.", "I.A.", "İ.B", "İ.B.", "Mr Y.I.", "L.H", "İ.D.", "Mr Z.K", "Y.C",
    "Ö.D.", "Mrs M", "Mrs M.", "Ms S.G", "Mr H.K", "Lutz", "Ms B.", "M.N.A", "Mr J.B.", "C.P", "R.B",
    "Dr. P.", "Mr O", "Mr Z.K.", "Dr. L.-K.",
    "K. Ch.", "K.Ch", "K. Ch",
    "Mr Perrin",
    "Mr Gölcüklü",
    "Ms K. Jones",
    "Mrs S. Jaczewska",
    "Lord Justice Aldous",
    "Judge Borrego Borrego",
    "Mr. Durmaz",
    "Mr McGrath",
    "Mr. Lutz",
    "Mr Justice Potts",
    "Mr Justice Barron",
]
assert len(tab_predefined_ignore_piis_list) == len(set(tab_predefined_ignore_piis_list))

def print_ori_syn_cases_fact(*,
    syn_piis: List[utils.PII],
    sorted_sim_items: List[Dict],
    data: utils.Data
) -> None:
    """Auxiliary factory function that returns a function to print original and syn cases."""
    def print_ori_syn_cases_fun(i: int) -> None:
        pii = sorted_sim_items[i]["ori_text"]
        ori_doc_nr = sorted_sim_items[i]["ori_doc_nr"]
        syn_items = sorted_sim_items[i]["syn_items"]
        print("\n### PII", pii,"\n") # noqa: T201
        print("Original court case:") # noqa: T201
        print(data.ori.raw_data[ori_doc_nr]["text"]) # noqa: T201
        for j, syn_item in enumerate(syn_items[0:3]):
            syn_text = syn_piis[syn_item].text
            syn_doc_nr = syn_piis[syn_item].doc_nr
            print("\n########") # noqa: T201
            print(f"Synthetic case {j}, pii: {syn_text}") # noqa: T201
            print(data.syn.raw_data[syn_doc_nr]["text"]) # noqa: T201
    return print_ori_syn_cases_fun
