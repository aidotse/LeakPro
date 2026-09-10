#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from .datasets import sample_splits, sample_validation_texts, sample_prefix_texts, sample_domain_splits
from .models import build_tokenizer, build_model, prepare_lm_dataloader, finetune_target, compute_ez_scores, build_distillation_reference, build_sft_reference, get_device, empty_device_cache
from .metrics import tpr_at_fpr
from .config import AttackConfig


def save_artifacts(
	save_path: str,
	target_model,
	reference_model,
	tokenizer,
	member_texts: List[str],
	nonmember_texts: List[str],
	cfg: AttackConfig,
) -> None:
	"""Save models and data for later use with verification_harness.py."""
	save_dir = Path(save_path)
	save_dir.mkdir(parents=True, exist_ok=True)

	tqdm.write(f"[save] Saving artifacts to {save_dir}...")

	target_model_path = save_dir / "target_model"
	tqdm.write(f"[save] Saving target model to {target_model_path}...")
	target_model.save_pretrained(target_model_path)

	reference_model_path = save_dir / "reference_model"
	tqdm.write(f"[save] Saving reference model to {reference_model_path}...")
	reference_model.save_pretrained(reference_model_path)

	tokenizer_path = save_dir / "tokenizer"
	tqdm.write(f"[save] Saving tokenizer to {tokenizer_path}...")
	tokenizer.save_pretrained(tokenizer_path)

	data_path = save_dir / "data.json"
	tqdm.write(f"[save] Saving data ({len(member_texts)} members, {len(nonmember_texts)} non-members) to {data_path}...")
	data = {
		"member_texts": member_texts,
		"nonmember_texts": nonmember_texts,
	}
	with open(data_path, "w") as f:
		json.dump(data, f, indent=2)

	config_path = save_dir / "config.json"
	tqdm.write(f"[save] Saving config to {config_path}...")
	config_data = {
		"dataset": cfg.dataset,
		"ref_variant": cfg.ref_variant,
		"seed": cfg.seed,
		"model_name": cfg.model_name,
		"sequence_length": cfg.sequence_length,
		"epochs": cfg.epochs,
		"lr": cfg.lr,
		"batch_size": cfg.batch_size,
		"finetune_method": cfg.finetune_method,
		"lora_r": cfg.lora_r,
		"lora_alpha": cfg.lora_alpha,
		"lora_dropout": cfg.lora_dropout,
		"train_total": cfg.train_total,
		"eval_total": cfg.eval_total,
		"num_members": len(member_texts),
		"num_nonmembers": len(nonmember_texts),
	}
	with open(config_path, "w") as f:
		json.dump(config_data, f, indent=2)

	tqdm.write(f"[save] Artifacts saved successfully to {save_dir}")


def run_attack(cfg: AttackConfig) -> Dict[str, Any]:
	def _set_seed(seed: int, device: torch.device) -> None:
		import random
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if device.type == "cuda":
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		elif device.type == "hpu" and hasattr(torch, "hpu"):
			torch.hpu.manual_seed_all(seed)
	device = get_device()
	tqdm.write(f"[device] Using {device}")
	_set_seed(cfg.seed, device)
	if not getattr(cfg, "val_total", None) or int(cfg.val_total) <= 0:
		raise ValueError("val_total must be provided and > 0; validation is mandatory for selecting the best model checkpoint.")
	requested_val_total = int(cfg.val_total)

	tqdm.write(f"[data] Sampling target dataset ({cfg.dataset})...")
	_, _, target_member_examples, target_nonmember_examples, seq_iter = sample_splits(
		cfg.seed, 0, cfg.eval_total, dataset=cfg.dataset, sequence_length=cfg.sequence_length, return_sequence_iter=True
	)

	tqdm.write(f"[data] Sampling domain dataset for {cfg.dataset}...")
	_, domain_nonmember_examples, domain_val_texts = sample_domain_splits(
		cfg.seed,
		cfg.train_total,
		cfg.dataset,
		sequence_length=cfg.sequence_length,
		val_total=requested_val_total,
	)

	target_val_texts = sample_validation_texts(
		cfg.seed,
		requested_val_total,
		dataset=cfg.dataset,
		sequence_length=cfg.sequence_length,
		sequence_iter=seq_iter,
	)
	if len(target_val_texts) < requested_val_total:
		raise ValueError(f"Requested {requested_val_total} target validation texts but only received {len(target_val_texts)}; validation set is mandatory.")

	distil_seed_texts = None
	if cfg.ref_variant == "distillation":
		distil_seed_texts = sample_prefix_texts(cfg.seed, cfg.dataset, max_texts=int(cfg.distil_max_prompts))
		if len(distil_seed_texts) < int(cfg.distil_max_prompts):
			tqdm.write(f"[distil] collected {len(distil_seed_texts)} prefix texts (requested {int(cfg.distil_max_prompts)})")

	tokenizer = build_tokenizer(cfg.model_name)

	if cfg.finetune_method == "lora":
		use_lora = True
	elif cfg.finetune_method == "full":
		use_lora = False
	else:
		use_lora = (cfg.model_name.lower() != "gpt2")

	ref_model = build_model(cfg.model_name, tokenizer, device, use_lora=False)

	ref_model.to("cpu") # Move to cpu to prevent OOM

	if len(domain_val_texts) < requested_val_total:
		raise ValueError(f"Requested {requested_val_total} domain validation texts but only received {len(domain_val_texts)}; validation set is mandatory.")
	target_val_dataloader = prepare_lm_dataloader(tokenizer, target_val_texts, batch_size=cfg.batch_size, device=device, sequence_length=cfg.sequence_length)

	tqdm.write(f"[target] Training target model on {len(target_member_examples)} target members...")
	target_model = build_model(
		cfg.model_name,
		tokenizer,
		device,
		use_lora=use_lora,
		lora_r=cfg.lora_r,
		lora_alpha=cfg.lora_alpha,
		lora_dropout=cfg.lora_dropout,
		lora_target_modules=cfg.lora_target_modules,
	)
	_target_texts = [x.text for x in target_member_examples]
	dl_target = prepare_lm_dataloader(tokenizer, _target_texts, batch_size=cfg.batch_size, device=device, sequence_length=cfg.sequence_length)
	_ = finetune_target(target_model, dl_target, epochs=cfg.epochs, lr=cfg.lr, val_dataloader=target_val_dataloader, load_best_on_val=True)

	target_model.to("cpu") # Move to CPU to prevent OOM
	empty_device_cache(device)

	if cfg.ref_variant == "distillation":
		tqdm.write("[reference] Building reference model via distillation from target model...")
		ref_model = build_distillation_reference(
			tokenizer,
			target_model,
			distil_seed_texts if distil_seed_texts is not None else [x.text for x in domain_nonmember_examples],
			device,
			max_prompts=cfg.distil_max_prompts,
			completions=cfg.distil_completions,
			max_new_tokens=cfg.distil_max_new_tokens,
			temperature=cfg.distil_temperature,
			top_p=cfg.distil_top_p,
			input_max_tokens=cfg.distil_input_max_tokens,
			train_epochs=cfg.distil_train_epochs,
			train_batch=cfg.distil_train_batch,
			train_lr=cfg.distil_train_lr,
			base_model_name=cfg.model_name,
			use_lora_for_ref=use_lora,
			lora_r=cfg.lora_r,
			lora_alpha=cfg.lora_alpha,
			lora_dropout=cfg.lora_dropout,
			lora_target_modules=cfg.lora_target_modules,
			val_texts=domain_val_texts,
			sequence_length=cfg.sequence_length,
		)
	elif cfg.ref_variant == "sft":
		tqdm.write("[reference] Building reference model via SFT on domain non-members...")
		ref_model = build_sft_reference(
			tokenizer,
			device,
			base_model_name=cfg.model_name,
			texts=[x.text for x in domain_nonmember_examples],
			epochs=cfg.sft_train_epochs,
			batch_size=cfg.sft_train_batch,
			lr=cfg.sft_train_lr,
			use_lora=False,
			lora_r=cfg.lora_r,
			lora_alpha=cfg.lora_alpha,
			lora_dropout=cfg.lora_dropout,
			lora_target_modules=cfg.lora_target_modules,
			val_texts=domain_val_texts,
			sequence_length=cfg.sequence_length,
		)

	ref_model.to("cpu") # Move to CPU to prevent OOM
	empty_device_cache(device)

	tqdm.write("[eval] Computing EZ scores from target_model + reference_model on target data...")
	scores_m = compute_ez_scores(tokenizer, target_model, ref_model, [x.text for x in target_member_examples], device, sequence_length=cfg.sequence_length, batch_size=cfg.batch_size)
	scores_nm = compute_ez_scores(tokenizer, target_model, ref_model, [x.text for x in target_nonmember_examples], device, sequence_length=cfg.sequence_length, batch_size=cfg.batch_size)

	y_eval = np.array([1]*len(scores_m) + [0]*len(scores_nm), dtype=np.int64)
	scores = np.array(scores_m + scores_nm, dtype=np.float32)
	scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

	auc = float(roc_auc_score(y_eval, scores))
	tpr001 = tpr_at_fpr(y_eval, scores, 0.01)
	tpr0001 = tpr_at_fpr(y_eval, scores, 0.001)

	if cfg.save_artifacts_path:
		save_artifacts(
			save_path=cfg.save_artifacts_path,
			target_model=target_model,
			reference_model=ref_model,
			tokenizer=tokenizer,
			member_texts=[x.text for x in target_member_examples],
			nonmember_texts=[x.text for x in target_nonmember_examples],
			cfg=cfg,
		)

	return {
		"dataset": cfg.dataset,
		"ref_variant": cfg.ref_variant,
		"auc": auc,
		"tpr_at_fpr_0.01": float(tpr001),
		"tpr_at_fpr_0.001": float(tpr0001),
		"seed": cfg.seed,
		"target_model": cfg.model_name,
		"train_total": cfg.train_total,
		"eval_total": cfg.eval_total,
		"epochs": cfg.epochs,
		"save_artifacts_path": cfg.save_artifacts_path,
	}
