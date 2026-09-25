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
from json_dataset import load_json_texts, tokenise_one_per_text
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
    """Return a list of 1-D int64 arrays.

    Three ``chunking`` modes, following the EZ-MIA paper's own reference code's recipe -- it uses a
    different one per dataset, not the same one everywhere. Both ``fixed`` and ``prefix`` follow that
    recipe rather than reproducing it byte-for-byte: the reference counts whitespace-split *words* to
    decide chunk boundaries and only BPE-truncates afterwards (``models.py``'s ``_collate_text``),
    while this counts BPE tokens directly throughout -- same recipe, not identical sequences.

    - ``fixed``: concatenate every text into one continuous token stream and slice it into
      back-to-back max_length chunks, carrying overflow from one text into the next. A single long
      text contributes multiple chunks, and a chunk can straddle two unrelated texts. This is what
      the reference uses for WikiText (``_texts_to_sequences_concat``).
    - ``prefix``: accumulate texts into a buffer; the moment it reaches max_length, emit exactly the
      first max_length tokens as one chunk and discard the rest of the buffer, then start over from
      the next text. Each source text contributes at most one chunk -- an unusually long text (e.g.
      XSum's minute-by-minute football live-text) never gets over-represented by being sliced into
      several chunks of the same narrow content. This is what the reference uses for XSum/ag_news
      (``sample_splits``'s ``_sequence_generator``), and is why train_config_xsum.yaml uses it too.
    - ``variable``: one truncated (not concatenated) sequence per text, dropped if under 2 tokens.
    """
    max_length = int(cfg["max_length"])
    chunking = cfg["chunking"]
    if chunking == "fixed":
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
    if chunking == "prefix":
        buf = []
        chunks = []
        for t in texts:
            buf.extend(tokenizer(t)["input_ids"])
            if len(buf) >= max_length:
                chunks.append(np.asarray(buf[:max_length], dtype=np.int64))
                buf = []
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


def _member_nonmember_val_split(n: int, data_cfg: dict, seed: int) -> tuple:
    """Random, disjoint (members, non-members, validation) index split of a population of size n.

    The validation split is genuinely separate from the non-member split -- unlike reusing the same
    held-out set for both checkpoint selection and audit scoring, which would let the "best epoch"
    be picked by how well it happens to fit the exact sequences later scored as non-members.
    """
    n_members, n_nonmembers = data_cfg["n_members"], data_cfg["n_nonmembers"]
    n_validation = data_cfg.get("n_validation", 0)
    total = n_members + n_nonmembers + n_validation
    assert total <= n, f"population too small for the requested split: need {total}, have {n}"
    perm = np.random.RandomState(seed).permutation(n)
    train_indices = perm[:n_members].tolist()
    test_indices = perm[n_members:n_members + n_nonmembers].tolist()
    val_indices = perm[n_members + n_nonmembers:total].tolist()
    return train_indices, test_indices, val_indices


def _build_population_from_local_json(data_cfg: dict, tokenizer) -> tuple:
    """`data.source: local` -- pre-split member/non-member JSON files (the WBC paper's own config shape).

    Unlike the `hf` path, the member/non-member split is already decided by which file a sequence came
    from -- `train_path` sequences are members, `test_path` sequences are the non-member/held-out set --
    so there is no population-wide permutation here, and `n_members`/`n_nonmembers` do not apply.

    Each JSON entry becomes exactly one (possibly truncated, never concatenated) sequence -- see
    `json_dataset.tokenise_one_per_text` -- so rows are ragged/variable-length, not a dense tensor.
    """
    train_chunks = tokenise_one_per_text(load_json_texts(data_cfg["train_path"], data_cfg.get("text_field", "text")),
                                        tokenizer, int(data_cfg["max_length"]))
    test_chunks = tokenise_one_per_text(load_json_texts(data_cfg["test_path"], data_cfg.get("text_field", "text")),
                                       tokenizer, int(data_cfg["max_length"]))
    if not train_chunks or not test_chunks:
        raise ValueError(f"no chunks produced from {data_cfg['train_path']} / {data_cfg['test_path']} "
                         "-- check text_field / JSON shape against json_dataset.load_json_texts")
    train_indices = list(range(len(train_chunks)))
    test_indices = list(range(len(train_chunks), len(train_chunks) + len(test_chunks)))
    ids = LLMDataHandler.as_object_array(train_chunks + test_chunks)
    return ids, train_indices, test_indices


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
    source = data_cfg.get("source", "hf")
    if data_path.exists():
        population = joblib.load(data_path)
        print(f"loaded population from {data_path} ({len(population)} sequences)")
        # Re-derive the split rather than trusting a second copy of it: for `local` it must match
        # exactly which file each row came from, which only `_build_population_from_local_json` knows.
        if source == "local":
            _, train_indices, test_indices = _build_population_from_local_json(data_cfg, tokenizer)
            val_indices = []
        else:
            train_indices, test_indices, val_indices = _member_nonmember_val_split(
                len(population), data_cfg, run["random_seed"])
    elif source == "local":
        ids, train_indices, test_indices = _build_population_from_local_json(data_cfg, tokenizer)
        val_indices = []
        population = LLMDataHandler.UserDataset(ids, ids, pad_token_id=pad_token_id)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(population, data_path)
        print(f"saved population to {data_path} ({len(population)} sequences: "
              f"{len(train_indices)} members from {data_cfg['train_path']}, "
              f"{len(test_indices)} non-members from {data_cfg['test_path']})")
    else:
        texts = _load_texts(data_cfg)
        if data_cfg["chunking"] == "fixed":
            # `fixed` concatenates consecutive rows into one continuous stream. Some datasets (e.g.
            # WikiText-103's `Salesforce/wikitext`) store one row per paragraph/heading, not one row
            # per document -- shuffling row order first would glue together unrelated paragraphs from
            # different articles into the same chunk, producing incoherent "salad" text that is
            # trivially memorable and inflates membership-inference scores for reasons that have
            # nothing to do with the attack itself. A random rotation keeps every row's neighbours
            # intact (matching the reference code's random-start-offset, sequential-read approach for
            # WikiText) while still giving a seed-dependent slice of the corpus.
            start = random.randrange(len(texts)) if texts else 0
            texts = texts[start:] + texts[:start]
        else:
            random.shuffle(texts)
        seqs = _tokenise(texts, tokenizer, data_cfg)
        if len(seqs) < data_cfg["n_sequences"]:
            raise ValueError(f"only {len(seqs)} sequences available, need {data_cfg['n_sequences']}")
        if data_cfg["chunking"] in ("fixed", "prefix"):  # both always emit exactly max_length tokens
            ids = torch.as_tensor(np.stack(seqs))
        else:
            ids = LLMDataHandler.as_object_array(seqs)
        population = LLMDataHandler.UserDataset(ids, ids, pad_token_id=pad_token_id)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(population, data_path)
        print(f"saved population to {data_path} ({len(population)} sequences)")

        train_indices, test_indices, val_indices = _member_nonmember_val_split(
            len(population), data_cfg, run["random_seed"])

    collate = CausalLMCollate(pad_token_id=pad_token_id)
    train_loader = DataLoader(LLMDataHandler.UserDataset(population.data[train_indices], population.targets[train_indices],
                                                         pad_token_id=pad_token_id),
                              batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate)
    test_loader = DataLoader(LLMDataHandler.UserDataset(population.data[test_indices], population.targets[test_indices],
                                                        pad_token_id=pad_token_id),
                             batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate)
    val_loader = None
    if val_indices:
        val_loader = DataLoader(LLMDataHandler.UserDataset(population.data[val_indices], population.targets[val_indices],
                                                           pad_token_id=pad_token_id),
                                batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate)

    # For `source: local` (WBC's own config shape), there is no separate validation split -- eval
    # reuses test_loader, matching that paper's own protocol (see wbc/train_config.yaml's comment).
    # For an HF-dataset population, a disjoint validation split is required so checkpoint selection
    # is never influenced by the exact sequences later scored as non-members.
    if train_cfg.get("eval_strategy") == "epoch":
        if source == "local":
            eval_dataloader = test_loader
        elif val_loader is not None:
            eval_dataloader = val_loader
        else:
            raise ValueError("train.eval_strategy: epoch requires data.n_validation > 0 "
                             "(a validation split disjoint from both members and non-members)")
    else:
        eval_dataloader = None

    # ---- target -----------------------------------------------------------------------------------
    model = HFCausalLMWrapper(train_cfg["model_name"], dtype=train_cfg.get("dtype", "float32"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg.get("weight_decay", 0.0))

    LLMModelHandler.lora = train_cfg["lora"] if train_cfg["finetune_method"] == "lora" else None
    handler = LLMModelHandler()
    print(f"fine-tuning {train_cfg['model_name']} ({train_cfg['finetune_method']}) on {len(train_indices)} members")
    train_result = handler.train(
        train_loader, model, criterion, optimizer, epochs=train_cfg["epochs"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        eval_dataloader=eval_dataloader,
        checkpoint_dir=str(log_dir / "ckpts") if train_cfg.get("save_strategy") == "epoch" else None,
        save_total_limit=train_cfg.get("save_total_limit", 1),
        load_best_checkpoint_at_end=train_cfg.get("load_best_checkpoint_at_end", False),
    )
    test_result = handler.eval(test_loader, train_result.model, criterion)
    print(f"held-out next-token acc {test_result.accuracy:.4f}  loss {test_result.loss:.4f}")

    model = train_result.model
    model.to("cpu")
    with open(log_dir / "target_model.pkl", "wb") as f:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f)

    metadata = LeakPro.make_mia_metadata(
        train_result=train_result, optimizer=optimizer, loss_fn=criterion, dataloader=train_loader,
        test_result=test_result, epochs=train_cfg["epochs"], train_indices=train_indices,
        test_indices=test_indices, dataset_name=data_cfg.get("hf_dataset", data_cfg.get("train_path", "local")),
    )
    with open(log_dir / "model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"wrote target to {log_dir}")


if __name__ == "__main__":
    main()
