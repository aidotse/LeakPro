#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .config import AttackConfig, make_arg_parser, load_attack_config_from_yaml
from .attack import run_attack


def append_result_csv(row: dict):
	output_path = Path(__file__).parent / "results.csv"
	file_exists = output_path.exists()
	with output_path.open("a", newline="") as file_obj:
		writer = csv.writer(file_obj)
		if not file_exists:
			writer.writerow(["dataset","target_model","reference_variant","attack_variant","metric","value","seed"])
		attack_name = "ez_ratio"
		writer.writerow([row["dataset"], row.get("target_model", "gpt2"), row["ref_variant"], attack_name, "AUC", f"{row['auc']:.6f}", row["seed"]])
		if "tpr_at_fpr_0.01" in row:
			writer.writerow([row["dataset"], row.get("target_model", "gpt2"), row["ref_variant"], attack_name, "TPR@1%FPR", f"{row['tpr_at_fpr_0.01']:.3f}", row["seed"]])
		writer.writerow([row["dataset"], row.get("target_model", "gpt2"), row["ref_variant"], attack_name, "TPR@0.1%FPR", f"{row['tpr_at_fpr_0.001']:.3f}", row["seed"]])


def main():
	ap = make_arg_parser()
	args = ap.parse_args()

	if getattr(args, "config", None):
		cfg = load_attack_config_from_yaml(args.config)
		res = run_attack(cfg)
		print(f"{cfg.dataset}, model={cfg.model_name}, ref={cfg.ref_variant}, AUC={res['auc']:.6f}, TPR@0.1%FPR={res['tpr_at_fpr_0.001']:.3f}")
		append_result_csv(res)
		return

	cfg = AttackConfig(
		dataset=args.dataset,
		ref_variant=args.ref_variant,
		seed=args.seed,
		save_artifacts_path=getattr(args, 'save_artifacts_path', None),
		train_total=args.train_total,
		eval_total=args.eval_total,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		sequence_length=args.sequence_length,
		model_name=args.target_model,
		finetune_method=args.finetune_method,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		val_total=args.val_total,
		distil_max_prompts=args.distil_max_prompts,
		distil_completions=args.distil_completions,
		distil_max_new_tokens=args.distil_max_new_tokens,
		distil_temperature=args.distil_temperature,
		distil_top_p=args.distil_top_p,
		distil_input_max_tokens=args.distil_input_max_tokens,
		distil_train_epochs=args.distil_train_epochs,
		distil_train_batch=args.distil_train_batch,
		distil_train_lr=args.distil_train_lr,
		sft_train_epochs=args.sft_train_epochs,
		sft_train_batch=args.sft_train_batch,
		sft_train_lr=args.sft_train_lr,
	)
	res = run_attack(cfg)
	print(f"{args.dataset}, model={cfg.model_name}, ref={cfg.ref_variant}, AUC={res['auc']:.6f}, TPR@0.1%FPR={res['tpr_at_fpr_0.001']:.3f}")
	append_result_csv(res)


if __name__ == "__main__":
	main()
