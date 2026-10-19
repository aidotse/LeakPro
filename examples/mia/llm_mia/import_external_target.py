##
## Copyright 2023-2026 Lindholmen Science Park AB
## SPDX-License-Identifier: Apache-2.0
##
"""Import an already-trained target checkpoint + pre-split member/non-member JSON files.

For auditing a model you *already* fine-tuned (e.g. a WBC-paper-style config with
``load_from_base_dir`` and ``json_train_path``/``json_test_path``), instead of
``prepare_target.py``'s fine-tune-it-yourself flow. Produces exactly the same three artifacts
``prepare_target.py`` does -- ``target_model.pkl``, ``model_metadata.pkl`` under ``--out-dir``, and
the population pickle at ``--data-path`` -- so ``run_audit.py``/``audit.yaml`` need no changes.

    cd wbc  # this script lives one level up; run it from inside the experiment folder
    python ../import_external_target.py \\
        --target-model weights/pythia-2.8b-.../ckpts/checkpoint-625 \\
        --tokenizer EleutherAI/pythia-2.8b \\
        --json-train weights/pythia-2.8b-.../train_subset.json \\
        --json-test weights/pythia-2.8b-.../test_subset.json \\
        --out-dir ./target_pythia_external \\
        --data-path ./data/pythia_external_512.pkl \\
        --max-length 512 --dtype bfloat16

``--target-model`` must be a directory ``AutoModelForCausalLM.from_pretrained`` can load directly
(a standard HF/Trainer checkpoint dir with config.json + weights) -- this script loads its *real*
weights, unlike the example's own HFCausalLMWrapper during fine-tuning, which loads an architecture
and then has LeakPro overwrite it via ``prepare_target.py``'s saved state dict.

JSON format assumed: a list of strings, or a list of ``{"text": ...}`` objects (``--text-field`` to
change the key). Adjust ``load_json_texts`` if your files use a different shape -- this is a
best-effort adapter for the paper's config shape, not a verified-against-real-files import (no such
files were available while writing this).
"""

import argparse
import pickle
from pathlib import Path

import joblib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from hf_wrapper import HFCausalLMWrapper
from json_dataset import load_json_texts, tokenise_one_per_text
from llm_data_handler import LLMDataHandler
from llm_model_handler import LLMModelHandler

from leakpro import LeakPro
from leakpro.signals.token_evidence import CausalLMCollate


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True, help="Directory from_pretrained can load (real trained weights)")
    ap.add_argument("--tokenizer", required=True, help="HF hub id or local dir for AutoTokenizer.from_pretrained")
    ap.add_argument("--json-train", required=True, help="Path to the member-sequences JSON file")
    ap.add_argument("--json-test", required=True, help="Path to the non-member-sequences JSON file")
    ap.add_argument("--text-field", default="text", help="Key holding the text, if the JSON holds objects not strings")
    ap.add_argument("--out-dir", required=True, help="Where to write target_model.pkl / model_metadata.pkl")
    ap.add_argument("--data-path", required=True, help="Where to write the population pickle")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--batch-size", type=int, default=1, help="Only used for the held-out eval pass below")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    print(f"tokenising {args.json_train} (members) and {args.json_test} (non-members)")
    train_chunks = tokenise_one_per_text(load_json_texts(args.json_train, args.text_field), tokenizer, args.max_length)
    test_chunks = tokenise_one_per_text(load_json_texts(args.json_test, args.text_field), tokenizer, args.max_length)
    if not train_chunks or not test_chunks:
        raise ValueError("no chunks produced -- check --text-field / JSON shape against load_json_texts")

    all_chunks = train_chunks + test_chunks
    train_indices = list(range(len(train_chunks)))
    test_indices = list(range(len(train_chunks), len(all_chunks)))
    ids = LLMDataHandler.as_object_array(all_chunks)
    population = LLMDataHandler.UserDataset(ids, ids, pad_token_id=pad_token_id)

    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(population, data_path)
    print(f"saved population to {data_path} ({len(population)} sequences: "
          f"{len(train_indices)} members, {len(test_indices)} non-members)")

    print(f"loading real weights from {args.target_model}")
    model = HFCausalLMWrapper(args.target_model, dtype=args.dtype)

    LLMModelHandler.lora = None
    handler = LLMModelHandler()
    collate = CausalLMCollate(pad_token_id=pad_token_id)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    train_loader = DataLoader(LLMDataHandler.UserDataset(population.data[train_indices], population.targets[train_indices],
                                                         pad_token_id=pad_token_id),
                              batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(LLMDataHandler.UserDataset(population.data[test_indices], population.targets[test_indices],
                                                        pad_token_id=pad_token_id),
                             batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print("evaluating (no training -- this checkpoint is already fine-tuned)")
    train_result = handler.eval(train_loader, model, criterion)
    test_result = handler.eval(test_loader, model, criterion)
    print(f"member next-token acc {train_result.accuracy:.4f}  non-member next-token acc {test_result.accuracy:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    with open(out_dir / "target_model.pkl", "wb") as f:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f)

    from leakpro.schemas import TrainingOutput

    metadata = LeakPro.make_mia_metadata(
        train_result=TrainingOutput(model=model, metrics=train_result),
        optimizer=optim.AdamW(model.parameters(), lr=0.0),  # never stepped -- only its config shape is used
        loss_fn=criterion, dataloader=train_loader, test_result=test_result,
        epochs=1,  # MIAMetaDataSchema requires >= 1; this checkpoint was already fine-tuned externally,
                   # not trained here -- 1 is a schema-satisfying placeholder, not a real epoch count.
        train_indices=train_indices, test_indices=test_indices, dataset_name="external-json-import",
    )
    with open(out_dir / "model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"wrote target to {out_dir}")


if __name__ == "__main__":
    main()
