##
## Copyright 2023-2026 Lindholmen Science Park AB
## SPDX-License-Identifier: Apache-2.0
##
"""Build a member/non-member JSON subset from any HuggingFace dataset.

Produces the `train.json`/`test.json` shape `prepare_target.py`'s `data.source: local` and
`import_external_target.py` expect. Self-contained port of the WBC reference codebase's own
data-prep step (github.com/Stry233/WBC, `dataset/prep.py`) so generating this doesn't require
cloning that repo or installing its `requirements.txt` -- which pins a generic/CUDA `torch` build
that risks clobbering an HPU-specific one already installed. This script never loads a model or
imports `torch` at all: it only uses a tokenizer to count token lengths for filtering, so it runs
identically on CPU/GPU/HPU machines. Needs only `datasets`/`transformers` (leakpro's `llm` extra).

    python prepare_json_subset.py \\
        --dataset-name HuggingFaceTB/cosmopedia --config khanacademy \\
        --num-samples 20000 --min-length 512 --tokenizer-name EleutherAI/pythia-2.8b \\
        --output-dir wbc/cosmopedia-khanacademy-subset

Reproduces the WBC paper's own Khan Academy subset command with the values above. Change
--dataset-name/--config for a different HF dataset (any of cosmopedia's other configs -- stanford,
stories, web_samples_v2, wikihow, auto_math_text -- work the same way).

--trust-remote-code defaults to off: it executes arbitrary Python fetched from the dataset/tokenizer
repo on the Hub, which neither HuggingFaceTB/cosmopedia nor EleutherAI/pythia-2.8b's tokenizer needs.
Only pass it for a dataset or tokenizer whose HF page says it requires custom loading code.
"""

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import List, Optional, Tuple


def build_subset(
    dataset_name: str,
    config: Optional[str],
    split: str,
    text_column: str,
    num_samples: int,
    min_length: int,
    member_ratio: float,
    tokenizer_name: str,
    trust_remote_code: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Filter by token length, shuffle (seed 42), and split into (member_rows, non_member_rows)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)

    dataset = load_dataset(dataset_name, config, split=split, trust_remote_code=trust_remote_code)

    if min_length > 0:
        def is_long_enough(example: dict) -> bool:
            return len(tokenizer.encode(example[text_column], add_special_tokens=False)) >= min_length
        dataset = dataset.filter(is_long_enough, num_proc=multiprocessing.cpu_count())

    dataset = dataset.shuffle(seed=42)
    n = min(num_samples, len(dataset))
    if n < num_samples:
        print(f"warning: only {len(dataset)} examples pass min_length={min_length}, using {n} instead of {num_samples}")
    subset = dataset.select(range(n))

    member_size = int(member_ratio * n)
    member_rows = [{"text": row[text_column]} for row in subset.select(range(member_size))]
    non_member_rows = [{"text": row[text_column]} for row in subset.select(range(member_size, n))]
    return member_rows, non_member_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-name", required=True, help="HF hub dataset id, e.g. HuggingFaceTB/cosmopedia")
    ap.add_argument("--config", default=None, help="HF dataset config/subset name, e.g. khanacademy")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--num-samples", type=int, default=20000, help="Total examples kept, split member/non-member")
    ap.add_argument("--min-length", type=int, default=512, help="Minimum token length to keep an entry (0 disables)")
    ap.add_argument("--member-ratio", type=float, default=0.5)
    ap.add_argument("--tokenizer-name", default="gpt2", help="Used only for length filtering, not for training")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--trust-remote-code", action="store_true",
        help="Allow the dataset/tokenizer to run custom code from the Hub. Off by default; only needed "
             "for a dataset or tokenizer whose HF page explicitly requires it.",
    )
    args = ap.parse_args()

    member_rows, non_member_rows = build_subset(
        args.dataset_name, args.config, args.split, args.text_column,
        args.num_samples, args.min_length, args.member_ratio, args.tokenizer_name,
        args.trust_remote_code,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(member_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(non_member_rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(member_rows)} members -> {out_dir / 'train.json'}")
    print(f"wrote {len(non_member_rows)} non-members -> {out_dir / 'test.json'}")


if __name__ == "__main__":
    main()
