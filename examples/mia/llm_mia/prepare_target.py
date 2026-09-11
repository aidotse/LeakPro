"""Tokenise a text dataset, fine-tune the target LM, and save what LeakPro's MIA handler needs.

    python prepare_target.py --config train_config.yaml [--finetune-method full|lora]

Writes, under ``run.log_dir``:
    target_model.pkl     state dict of HFCausalLMWrapper (CPU tensors; LoRA merged)
    model_metadata.pkl   MIAMetaDataSchema via LeakPro.make_mia_metadata
and the population pickle at ``data.data_path``.

Mirrors the other examples' target-preparation step (cifar_main.ipynb etc.) for a causal LM.
Members = ``train_indices``, non-members = ``test_indices``; both are disjoint chunks of the same split.
"""

import argparse
import pickle
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from hf_wrapper import HFCausalLMWrapper
from llm_data_handler import LLMDataHandler
from llm_model_handler import LLMModelHandler

from leakpro import LeakPro
from leakpro.signals.token_evidence import CausalLMCollate
from leakpro.utils.seed import seed_everything


def _load_texts(cfg: dict) -> list:
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_dataset"], cfg.get("hf_config"), split=cfg["split"])
    field = cfg["text_field"]
    return [t for t in ds[field] if isinstance(t, str) and t.strip()]


def _tokenise(texts: list, tokenizer, cfg: dict) -> list:
    """Return a list of 1-D int64 arrays. fixed: concatenate and cut into max_length chunks (EZ-MIA §A.1)."""
    max_length = int(cfg["max_length"])
    if cfg["chunking"] == "fixed":
        stream = []
        chunks = []
        for t in texts:
            stream.extend(tokenizer(t)["input_ids"])
            while len(stream) >= max_length:
                chunks.append(np.asarray(stream[:max_length], dtype=np.int64))
                stream = stream[max_length:]
            if len(chunks) >= cfg["n_sequences"]:
                break
        return chunks[: cfg["n_sequences"]]
    seqs = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=max_length)["input_ids"]
        if len(ids) >= 2:
            seqs.append(np.asarray(ids, dtype=np.int64))
        if len(seqs) >= cfg["n_sequences"]:
            break
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--finetune-method", choices=["full", "lora"], default=None,
                    help="Override train.finetune_method (used by the LoRA-vs-full sweep)")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run, train_cfg, data_cfg = config["run"], config["train"], config["data"]
    if args.finetune_method:
        train_cfg["finetune_method"] = args.finetune_method
    log_dir = Path(run["log_dir"].format(finetune_method=train_cfg["finetune_method"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(run["random_seed"])
    random.seed(run["random_seed"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(train_cfg["model_name"])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ---- population -------------------------------------------------------------------------------
    data_path = Path(data_cfg["data_path"])
    if data_path.exists():
        population = joblib.load(data_path)
        print(f"loaded population from {data_path} ({len(population)} sequences)")
    else:
        texts = _load_texts(data_cfg)
        random.shuffle(texts)
        seqs = _tokenise(texts, tokenizer, data_cfg)
        if len(seqs) < data_cfg["n_sequences"]:
            raise ValueError(f"only {len(seqs)} sequences available, need {data_cfg['n_sequences']}")
        if data_cfg["chunking"] == "fixed":
            ids = torch.as_tensor(np.stack(seqs))
        else:
            ids = LLMDataHandler.as_object_array(seqs)
        population = LLMDataHandler.UserDataset(ids, ids, pad_token_id=pad_token_id)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(population, data_path)
        print(f"saved population to {data_path} ({len(population)} sequences)")

    n = len(population)
    n_members, n_nonmembers = data_cfg["n_members"], data_cfg["n_nonmembers"]
    assert n_members + n_nonmembers <= n, "population too small for the requested member/non-member split"
    perm = np.random.RandomState(run["random_seed"]).permutation(n)
    train_indices = perm[:n_members].tolist()
    test_indices = perm[n_members:n_members + n_nonmembers].tolist()

    collate = CausalLMCollate(pad_token_id=pad_token_id)
    train_loader = DataLoader(LLMDataHandler.UserDataset(population.data[train_indices], population.targets[train_indices],
                                                         pad_token_id=pad_token_id),
                              batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate)
    test_loader = DataLoader(LLMDataHandler.UserDataset(population.data[test_indices], population.targets[test_indices],
                                                        pad_token_id=pad_token_id),
                             batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate)

    # ---- target -----------------------------------------------------------------------------------
    model = HFCausalLMWrapper(train_cfg["model_name"], dtype=train_cfg.get("dtype", "float32"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg.get("weight_decay", 0.0))

    LLMModelHandler.pad_token_id = pad_token_id
    LLMModelHandler.lora = train_cfg["lora"] if train_cfg["finetune_method"] == "lora" else None
    handler = LLMModelHandler()
    print(f"fine-tuning {train_cfg['model_name']} ({train_cfg['finetune_method']}) on {n_members} members")
    train_result = handler.train(train_loader, model, criterion, optimizer, epochs=train_cfg["epochs"])
    test_result = handler.eval(test_loader, train_result.model, criterion)
    print(f"held-out next-token acc {test_result.accuracy:.4f}  loss {test_result.loss:.4f}")

    model = train_result.model
    model.to("cpu")
    with open(log_dir / "target_model.pkl", "wb") as f:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f)

    metadata = LeakPro.make_mia_metadata(
        train_result=train_result, optimizer=optimizer, loss_fn=criterion, dataloader=train_loader,
        test_result=test_result, epochs=train_cfg["epochs"], train_indices=train_indices,
        test_indices=test_indices, dataset_name=data_cfg["hf_dataset"],
    )
    with open(log_dir / "model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"wrote target to {log_dir}")


if __name__ == "__main__":
    main()
