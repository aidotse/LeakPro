#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Memorization extraction: generate, rank and verify candidate training sequences.

The base attack (Phases 1 and 2) only queries the model, so it runs against a black-box target.
Phases 3 and 4 are an optional, stronger attack that needs the weights.

**Phase 1, initial extraction** (always runs):

1. *Generate* with each prompting strategy in ``initial`` (``baseline`` BOS-only sampling,
   ``temp_decay`` sampling, ``conditional`` greedy continuation of prefixes).
2. *Filter*: drop texts under ``min_words`` words and exact duplicates, score the rest with the zlib
   and lowercase membership scores, and keep the top ``top_percent`` under each score (per strategy).
3. *Verify* the continuations of the selected candidates with the attacker verifier (30-word windows,
   stride 5, first match). The confirmed candidates form **S0**.

**Phase 2, expansion** (runs by default with an ``attacker_verifier``; ``expansion: null`` skips it):

1. Cut anchors of up to 50 tokens from S0, starting at each sequence's first verified window.
2. Prompt with a random 30-50 token prefix of each anchor and sample 150 continuations per anchor
   (temperature decaying from 3 to 1).
3. Filter and verify prefix + continuation (all windows) -> **S1**.
4. Keep the S1 sequences with at least one window that the exhaustive (all-window) verification of S0
   did not find -> **S1-novel**.

**Phase 3, fine-tuning** (``finetune`` set): LoRA continual pre-training on S0 ∪ S1-novel (texts of at
least 20 words, normalized-deduplicated, shuffled).

**Phase 4, re-extraction** (``finetune`` set): Phase 1's strategies on the fine-tuned model. A
generation whose generated text (without the prompt) shares any 30-token window with the fine-tuning
corpus is dropped *before* filtering (both sides lowercased and whitespace-collapsed);
filtering scores with the base model; verification as in Phase 1 -> **S_new**.

**Evaluation** (auditor only): an optional audit oracle checks the selected candidates of every phase
against the real training data. Its verdicts are reported and never fed back into the attack.

**References.** The four-phase pipeline (initial extraction, expansion, fine-tuning, re-extraction) and
its defaults are those of Matacuta, *Anything You Say Can and Will Be Used Against You: Iterative
Memorization Extraction and Fine-Tuning in Large Language Models* (Master's thesis, University of
Gothenburg, 2026, https://hdl.handle.net/2077/93397): 150 generations per anchor, temperature 3 -> 1
over 20 tokens in Phase 2 with ``top_p=0.9`` and repetition penalty 1.1, LoRA ``r=16``, ``alpha=32``,
dropout 0.1, 5 epochs, learning rate 1e-4, effective batch size 8. Phase 1's scores, prefix strategy
and temperature decay follow Carlini et al., *Extracting Training Data from Large Language Models*
(USENIX Security 2021); the fine-tuning phase is influenced by Nasr et al., *Scalable Extraction of
Training Data from Aligned, Production Language Models* (2025).

Where this implementation differs, so the numbers are not read as a reproduction of either work:

- *Sampling* (``baseline``, ``temp_decay``): ``top_k=50`` and ``top_p=0.95``, as in the thesis.
  Carlini et al. sample with top-n 40 and no nucleus filter.
- *Prefixes* (``conditional``): continued greedily, as in the thesis. Carlini et al. sample
  (top-n) from the prefixes.
- *Fine-tuning*: plain LoRA on the unquantized model with a 10-step warm-up. The thesis used QLoRA
  (4-bit base model).
- *Phase 4 overlap*: only the generated text is compared with the fine-tuning corpus, so a prefix that
  happens to be fine-tuning text does not drop the generation. The thesis compared the whole sequence.

Every stage is cached under ``<output_dir>/artifacts`` keyed by its settings and its inputs, so an
interrupted run resumes and a changed setting recomputes only the stages it affects.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtractionAttack
from leakpro.attacks.utils.extraction.elasticsearch_verifier import ElasticsearchVerifier, connect
from leakpro.attacks.utils.extraction.expansion import AnchorGenerator, build_anchors, known_windows, novel_candidates
from leakpro.attacks.utils.extraction.finetune import (
    TokenWindowIndex,
    adapter_disabled,
    attach_lora,
    build_finetune_corpus,
    detach_lora,
    load_lora_state,
    lora_state,
    param_dtypes,
    train_lora,
)
from leakpro.attacks.utils.extraction.generation import StrategyGenerator
from leakpro.attacks.utils.extraction.progress import Progress
from leakpro.attacks.utils.extraction.records import Candidate, Windowing
from leakpro.attacks.utils.extraction.scoring import SCORE_NAMES, drop_short_and_duplicates, score_candidates, select_top_union
from leakpro.attacks.utils.extraction.store import ArtifactStore, stage_key
from leakpro.attacks.utils.extraction.verification import InMemoryVerifier, Verifier, verify_candidates
from leakpro.reporting.extraction_result import ExtractionResult
from leakpro.signals.token_evidence import CausalLMModel
from leakpro.utils.device import get_device
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.save_load import fingerprint_model

PHASE_INITIAL = "initial"
PHASE_EXPANSION = "expansion"
PHASE_FINETUNED = "finetuned"

LLAMA_LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class GenStrategyConfig(BaseModel):
    """One prompting strategy."""

    type: Literal["baseline", "temp_decay", "conditional"] = Field(..., description="Prompting strategy")
    name: Optional[str] = Field(default=None, description="Label for reports; defaults to `type`")
    n: int = Field(default=10_000, ge=1, description="Generations (conditional: prefixes used, capped by the pool)")
    max_tokens: int = Field(default=256, ge=1, description="New tokens (conditional: total tokens incl. the prefix)")
    batch_size: Optional[int] = Field(default=None, ge=1, description="Prompts per batch; default 16, conditional 8")
    temperature: float = Field(default=1.0, gt=0, description="Sampling temperature (baseline, temp_decay)")
    top_k: Optional[int] = Field(default=50, ge=0, description="Top-k filter when sampling; None/0 disables")
    top_p: float = Field(default=0.95, gt=0, le=1, description="Nucleus threshold when sampling")
    t_start: float = Field(default=10.0, gt=0, description="temp_decay: first-token temperature")
    t_end: float = Field(default=1.0, gt=0, description="temp_decay: temperature after the decay")
    decay_steps: int = Field(default=20, ge=1, description="temp_decay: tokens over which the temperature decays")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _defaults(self: Self) -> Self:
        if self.name is None:
            self.name = self.type
        if self.batch_size is None:
            self.batch_size = 8 if self.type == "conditional" else 16
        return self


class WindowingConfig(BaseModel):
    """Verification windows, in words."""

    size: int = Field(default=30, ge=1, description="Window length in words")
    stride: int = Field(default=5, ge=1, description="Step between windows in words")
    min_span: int = Field(default=30, ge=1,
                          description="Shortest window; set below `size` to also check texts shorter than `size` words")

    model_config = ConfigDict(extra="forbid")


class FilterConfig(BaseModel):
    """Ranking and selection of candidates before verification (every phase)."""

    metrics: List[Literal["zlib", "lowercase"]] = Field(default=list(SCORE_NAMES), min_length=1)
    top_percent: float = Field(default=0.05, gt=0, le=1, description="Fraction kept under each metric")
    per_strategy: bool = Field(default=True, description="Select within each strategy (True) or over all of them")
    min_words: int = Field(default=10, ge=0, description="Texts with fewer words are dropped before scoring")
    max_length: int = Field(default=512, ge=2, description="Tokens scored per text")
    batch_size: int = Field(default=4, ge=1, description="Texts per scoring forward pass")

    model_config = ConfigDict(extra="forbid")


class VerifierConfig(BaseModel):
    """A corpus to verify against: a JSON-lines file held in memory, or an Elasticsearch index."""

    type: Literal["in_memory", "elasticsearch"] = Field(default="in_memory", description="Verifier backend")
    path: Optional[str] = Field(default=None, description="in_memory: JSON-lines corpus, one document per line")
    text_field: str = Field(default="text", description="in_memory: field holding the document text")
    id_field: str = Field(default="doc_id", description="in_memory: field holding the document id (else line number)")
    index: Optional[str] = Field(default=None, description="elasticsearch: index built with `index_corpus`")
    host: Optional[str] = Field(default=None, description="elasticsearch: URL; default $ES_HOST or https://localhost:9200")
    user: Optional[str] = Field(default=None, description="elasticsearch: user; default $ES_USER")
    password_env: str = Field(default="ES_PASSWORD", description="elasticsearch: environment variable holding the password")
    ca_certs: Optional[str] = Field(default=None, description="elasticsearch: CA certificate file; default $ES_CA_CERT")
    corpus_version: Optional[str] = Field(default=None, description="elasticsearch: label of the indexed data (cache key)")
    hits_per_query: int = Field(default=5, ge=1, description="elasticsearch: hits re-checked per window")
    mode: Literal["first", "all"] = Field(default="first", description="Stop at the first verified window, or check all")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check(self: Self) -> Self:
        if self.type == "in_memory" and not self.path:
            raise ValueError("An in_memory verifier needs `path`")
        if self.type == "elasticsearch" and not self.index:
            raise ValueError("An elasticsearch verifier needs `index`")
        return self


class ExpansionConfig(BaseModel):
    """Phase 2: expansion."""

    anchor_tokens: int = Field(default=50, ge=1, description="Anchor length in tokens, from the first verified window on")
    min_anchor_tokens: int = Field(default=10, ge=1, description="Shorter anchors are dropped")
    max_anchors: Optional[int] = Field(default=None, ge=1, description="Use only the first anchors (None: all)")
    prefix_lengths: List[int] = Field(default=[30, 35, 40, 45, 50], min_length=1, description="Prefix lengths, one per batch")
    gens_per_anchor: int = Field(default=150, ge=1, description="Generations per anchor")
    batch_size: int = Field(default=50, ge=1, description="Generations per batch (one prefix length per batch)")
    max_tokens: int = Field(default=256, ge=2, description="Total tokens per generation, prefix included")
    temperature: float = Field(default=1.0, gt=0)
    top_k: Optional[int] = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0)
    t_start: float = Field(default=3.0, gt=0)
    t_end: float = Field(default=1.0, gt=0)
    decay_steps: int = Field(default=20, ge=1)

    model_config = ConfigDict(extra="forbid")


class FinetuneConfig(BaseModel):
    """Phases 3 and 4: fine-tuning and re-extraction."""

    corpus_min_words: int = Field(default=20, ge=0, description="Fine-tuning texts with fewer words are dropped")
    corpus_seed: int = Field(default=42, description="Shuffle seed of the fine-tuning corpus")
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.1, ge=0, lt=1)
    target_modules: Union[List[str], str] = Field(default_factory=lambda: list(LLAMA_LORA_MODULES),
                                                  description="Modules given LoRA adapters, or 'all-linear'")
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=8, ge=1)
    lr: float = Field(default=1e-4, gt=0)
    warmup_steps: int = Field(default=10, ge=0, description="Linear warm-up steps, then a constant learning rate")
    max_length: int = Field(default=256, ge=2, description="Tokens per training text")
    overlap_window: int = Field(default=30, ge=1,
                                description="Phase 4: drop generations whose generated text shares a window this long (tokens)")
    strategies: Optional[List[GenStrategyConfig]] = Field(default=None, description="Phase 4 strategies; None: `initial`")

    model_config = ConfigDict(extra="forbid")


class MemorizationExtractionConfig(BaseModel):
    """Configuration of :class:`AttackMemorizationExtraction`."""

    initial: List[GenStrategyConfig] = Field(
        default_factory=lambda: [GenStrategyConfig(type=t) for t in ("baseline", "temp_decay", "conditional")],
        min_length=1, description="Phase 1 prompting strategies",
    )
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    attacker_verifier: Optional[VerifierConfig] = Field(default=None, description="Attacker-side corpus; feeds the attack")
    audit_oracle: Optional[VerifierConfig] = Field(default=None, description="Real training data; only reported")
    expansion: Optional[ExpansionConfig] = Field(
        default_factory=ExpansionConfig,
        description="Phase 2, part of the attack (needs `attacker_verifier`); null runs Phase 1 only",
    )
    finetune: Optional[FinetuneConfig] = Field(default=None, description="Optional Phases 3-4; set to include them")
    generation_backend: Literal["auto", "hf", "torch"] = Field(default="auto", description="Generation backend")
    store_text: Literal["full", "hash", "none"] = Field(default="full", description="Extracted text written to results")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check(self: Self) -> Self:
        phase4 = self.finetune.strategies if self.finetune is not None and self.finetune.strategies else []
        for strategies in (self.initial, phase4):
            names = [s.name for s in strategies]
            if len(set(names)) != len(names):
                raise ValueError(f"Strategy names must be unique, got {names}; set `name` to tell them apart")
        if self.attacker_verifier is None:
            # Phases 2-4 build on S0, which only the attacker verifier produces.
            if "expansion" in self.model_fields_set and self.expansion is not None or self.finetune is not None:
                raise ValueError("`expansion` and `finetune` build on S0, which needs an `attacker_verifier`")
            self.expansion = None  # the default Phase 2 has nothing to expand: Phase 1 only
        return self

    def phases(self: Self) -> List[int]:
        """The phases this config runs, in order: 1, then 2 (unless off), then 3 and 4 (if `finetune`)."""
        return [1] + ([2] if self.expansion is not None else []) + ([3, 4] if self.finetune is not None else [])


def build_verifier(cfg: Optional[VerifierConfig]) -> Optional[Verifier]:
    """Instantiate the verifier described by ``cfg`` (None stays None)."""
    if cfg is None:
        return None
    if cfg.type == "elasticsearch":
        client = connect(cfg.host, cfg.user, cfg.password_env, cfg.ca_certs)
        return ElasticsearchVerifier(client, cfg.index, hits_per_query=cfg.hits_per_query,
                                     corpus_version=cfg.corpus_version)
    return InMemoryVerifier.from_jsonl(cfg.path, text_field=cfg.text_field, id_field=cfg.id_field)


def _extracted_record(cand: Candidate) -> Dict[str, Any]:
    record = cand.to_dict()
    record.pop("prompt_ids")
    record.pop("gen_ids")
    return record


def _copy(cands: Sequence[Candidate]) -> List[Candidate]:
    return [Candidate.from_dict(c.to_dict()) for c in cands]


class AttackMemorizationExtraction(AbstractExtractionAttack):
    """Memorization extraction (Phases 1-2, optionally 3-4). See the module docstring."""

    AttackConfig = MemorizationExtractionConfig

    def description(self: Self) -> dict:
        """Return a description of the attack."""
        return {
            "title_str": "Memorization extraction",
            "reference": "Matacuta, A.-C. Anything You Say Can and Will Be Used Against You: Iterative Memorization "
                         "Extraction and Fine-Tuning in Large Language Models. Master's thesis, University of "
                         "Gothenburg (2026). https://hdl.handle.net/2077/93397. "
                         "Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., "
                         "Brown, T., Song, D., Erlingsson, U., Oprea, A., Raffel, C. Extracting Training Data from Large "
                         "Language Models. USENIX Security (2021).",
            "summary": "Black-box extraction: sample many generations, rank them with zlib and lowercase membership "
                       "scores, verify the top-ranked ones word for word against a reference corpus, and expand the "
                       "verified sequences by re-prompting with them. Optionally fine-tune on the extracted text "
                       "and extract again.",
            "detailed": "Phase 1 samples from BOS-only, temperature-decay and prefix-conditional prompts, ranks the "
                        "candidates by zlib entropy over log-perplexity and by the lowercase log-perplexity ratio, "
                        "and verifies the top fraction with 30-word windows (S0). Phase 2 re-prompts with anchors "
                        "cut from S0 and keeps verified sequences with windows S0 did not contain (S1-novel). "
                        "Phases 3-4 (optional, white-box) LoRA-fine-tune on S0 and S1-novel and repeat Phase 1, "
                        "dropping generations that overlap the fine-tuning data. An optional audit oracle reports "
                        "how many outputs occur in the real training data without influencing the attack.",
        }

    def prepare_attack(self: Self) -> None:
        """Resolve the device, tokenizer, verifiers and artifact store."""
        cfg = self.configs
        self.device = get_device()
        self.tokenizer = self.handler.tokenizer
        self.store = ArtifactStore(f"{self.output_dir}/artifacts")
        self.windowing = Windowing(**cfg.windowing.model_dump())
        self.attacker_verifier = build_verifier(cfg.attacker_verifier)
        self.audit_oracle = build_verifier(cfg.audit_oracle)
        self.model_fingerprint = fingerprint_model(self.target_model)
        self.tokenizer_id = str(getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__))
        self._peft = None
        self._reset_run()
        self.target_model.to(self.device)
        self.target_model.eval()

    # ------------------------------------------------------------------------------------- stages

    def _run_generator(self: Self, generator: Any, key: str, stage_name: str) -> Dict[str, Any]:  # noqa: ANN401
        """Generate (or resume) every batch of ``generator``; returns its stage key and candidates."""
        stage = self.store.stage(stage_name, key)
        if stage.is_done():
            logger.info(f"Reusing cached generations ({stage_name})")
            return {"key": key, "candidates": stage.read()}
        done = stage.completed_batches()
        total = generator.num_batches()
        logger.info(f"Generating {generator.total} sequences ({stage_name}, {total - len(done)} of {total} batches left; "
                    "progress is logged every minute)")
        progress = Progress(stage_name, total, "batch", done=len(done))
        for batch_index in range(total):
            if batch_index not in done:
                batch = generator.generate_batch(batch_index, self.target_model, self.seed, self.device,
                                                 self.configs.generation_backend)
                stage.write_batch(batch_index, batch)
                progress.update()
        candidates = stage.read_batches()
        stage.write(candidates)
        return {"key": key, "candidates": candidates}

    def _generate(self: Self, index: int, strategy: GenStrategyConfig, phase: str, model_key: str) -> Dict[str, Any]:
        """Generate one prompting strategy with the current model (``model_key`` identifies it)."""
        prefixes = self.handler.prefixes() if strategy.type == "conditional" else None
        generator = StrategyGenerator(strategy.name, index, strategy, self.tokenizer, prefixes, phase=phase)
        prefix_digest = stage_key(generator.prefixes) if prefixes is not None else None
        key = stage_key("generate", phase, index, strategy.model_dump(), self.seed, model_key,
                        self.tokenizer_id, self.configs.generation_backend, prefix_digest)
        return self._run_generator(generator, key, f"{phase}-generate-{strategy.name}")

    def _drop_overlap(self: Self, generated: Dict[str, Any], index: TokenWindowIndex, index_key: str,
                      name: str, phase: str) -> Dict[str, Any]:
        """Phase 4: drop generations whose generated text shares a token window with the fine-tuning corpus.

        Only the continuation is checked, not the prompt: for conditional prompts the prefix is corpus
        text the model did not generate. Cached.
        """
        key = stage_key("overlap", generated["key"], index_key, index.window, "continuation")
        stage = self.store.stage(f"{phase}-overlap-{name}", key)
        if stage.is_done():
            return {"key": key, "candidates": stage.read()}
        kept = [c for c in generated["candidates"] if not index.overlaps(c.continuation)]
        removed = len(generated["candidates"]) - len(kept)
        logger.info(f"{name}: dropped {removed} generations overlapping the fine-tuning data")
        stage.write(kept)
        return {"key": key, "candidates": kept}

    def _score(self: Self, generated: Dict[str, Any], name: str, phase: str, text_of: str) -> Dict[str, Any]:
        """Drop short texts and duplicates, then score the rest with the base model (cached)."""
        fcfg = self.configs.filter
        key = stage_key("score", generated["key"], text_of, fcfg.min_words, fcfg.max_length, self.model_fingerprint)
        stage = self.store.stage(f"{phase}-score-{name}", key)
        if stage.is_done():
            logger.info(f"Reusing cached scores ({phase}, {name})")
            return {"key": key, "candidates": stage.read()}
        kept = drop_short_and_duplicates(generated["candidates"], text_of, fcfg.min_words)
        logger.info(f"Scoring {len(kept)} of {len(generated['candidates'])} {phase}/{name} generations")
        with adapter_disabled(self._peft):
            score_candidates(kept, CausalLMModel(self.target_model, device=self.device), self.tokenizer,
                             text_of=text_of, batch_size=fcfg.batch_size, max_length=fcfg.max_length,
                             progress_label=f"{phase}-score-{name}")
        stage.write(kept)
        return {"key": key, "candidates": kept}

    def _verify(
        self: Self,
        candidates: List[Candidate],
        upstream_key: str,
        verifier: Verifier,
        mode: str,
        field: str,
        stage_name: str,
        text_of: str,
    ) -> Dict[str, Any]:
        """Verify ``candidates`` and store the matches on ``field`` (cached); returns the stage key and candidates."""
        key = stage_key("verify", upstream_key, field, verifier.corpus_id, mode, text_of, self.windowing.__dict__)
        stage = self.store.stage(stage_name, key)
        if stage.is_done():
            logger.info(f"Reusing cached verification ({stage_name})")
            return {"key": key, "candidates": stage.read()}
        logger.info(f"Verifying {len(candidates)} candidates ({stage_name}, {mode} windows)")
        verify_candidates(candidates, verifier, self.windowing, mode, text_of, field,
                          progress=Progress(stage_name, len(candidates), "candidate"))
        stage.write(candidates)
        return {"key": key, "candidates": candidates}

    def _select_and_verify(
        self: Self,
        scored: Dict[str, List[Candidate]],
        score_keys: List[str],
        phase: str,
        text_of: str,
        attacker_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select from the scored pools, verify with the attacker verifier, then evaluate with the oracle."""
        cfg, fcfg = self.configs, self.configs.filter
        pools = scored if fcfg.per_strategy else {"all": [c for cands in scored.values() for c in cands]}
        selected = [c for pool in pools.values() for c in select_top_union(pool, fcfg.top_percent, fcfg.metrics)]
        upstream_key = stage_key("select", score_keys, fcfg.metrics, fcfg.top_percent, fcfg.per_strategy)
        logger.info(f"{phase}: selected {len(selected)} candidates for verification")
        if self.attacker_verifier is not None:
            verified = self._verify(selected, upstream_key, self.attacker_verifier,
                                    attacker_mode or cfg.attacker_verifier.mode, "attacker_matches",
                                    f"{phase}-verify", text_of)
            selected, upstream_key = verified["candidates"], verified["key"]
        attacker_key = upstream_key
        if self.audit_oracle is not None:
            # Evaluation only: runs on the selected candidates and is reported, never read by an attack stage.
            # Keyed on the attacker stage too, because the cached records carry the attacker's matches.
            selected = self._verify(selected, upstream_key, self.audit_oracle, cfg.audit_oracle.mode,
                                    "oracle_matches", f"{phase}-oracle", text_of)["candidates"]
        return {"key": attacker_key, "selected": selected}

    # ------------------------------------------------------------------------------------ phases

    def _extraction_phase(
        self: Self,
        phase: str,
        strategies: Sequence[GenStrategyConfig],
        model_key: str,
        overlap: Optional[TokenWindowIndex] = None,
        overlap_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Phase 1 (and Phase 4 with ``overlap``): generate, [drop overlap,] score, select, verify."""
        text_of = "continuation"
        generated, after_overlap, scored, score_keys = {}, {}, {}, []
        for index, strategy in enumerate(strategies):
            gen = self._generate(index, strategy, phase, model_key)
            generated[strategy.name] = gen["candidates"]
            if overlap is not None:
                gen = self._drop_overlap(gen, overlap, overlap_key, strategy.name, phase)
                after_overlap[strategy.name] = gen["candidates"]
            sco = self._score(gen, strategy.name, phase, text_of)
            scored[strategy.name] = sco["candidates"]
            score_keys.append(sco["key"])
        out = self._select_and_verify(scored, score_keys, phase, text_of)
        extra = None
        if overlap is not None:
            extra = {name: {"overlap_removed": len(generated[name]) - len(after_overlap[name])} for name in generated}
        out["metrics"] = self._metrics(generated, scored, out["selected"], extra)
        return out

    def _expansion_phase(self: Self, s0: List[Candidate], s0_key: str) -> Dict[str, Any]:
        """Phase 2: anchors from S0, expansion generation, filtering, exhaustive verification, novelty."""
        ecfg, phase = self.configs.expansion, PHASE_EXPANSION
        anchors = build_anchors(s0, self.tokenizer, "continuation", ecfg.anchor_tokens, ecfg.min_anchor_tokens)
        if ecfg.max_anchors is not None:
            anchors = anchors[:ecfg.max_anchors]
        logger.info(f"Expansion: {len(anchors)} anchors from {len(s0)} S0 sequences")
        generator = AnchorGenerator(anchors, ecfg, self.tokenizer, phase=phase)
        key = stage_key("generate", phase, ecfg.model_dump(), s0_key, self.seed, self.model_fingerprint,
                        self.tokenizer_id, self.configs.generation_backend, [a["tokens"] for a in anchors])
        gen = self._run_generator(generator, key, f"{phase}-generate")
        sco = self._score(gen, generator.name, phase, "full_text")
        # S1 is verified on every window: the novelty test needs all of them.
        out = self._select_and_verify({generator.name: sco["candidates"]}, [sco["key"]], phase, "full_text",
                                      attacker_mode="all")
        s1 = [c for c in out["selected"] if c.attacker_matches]
        # Exhaustive verification of S0, on copies so Phase 1's records keep their first-match results.
        s0_all = self._verify(_copy(s0), s0_key, self.attacker_verifier, "all", "attacker_matches",
                              f"{PHASE_INITIAL}-verify-all", "continuation")["candidates"]
        s1_novel = novel_candidates(s1, known_windows(s0_all))
        out["metrics"] = self._metrics({generator.name: gen["candidates"]}, {generator.name: sco["candidates"]},
                                       out["selected"], {generator.name: {"anchors": len(anchors), "novel": len(s1_novel)}})
        out["s1_novel"] = s1_novel
        return out

    def _finetune(self: Self, corpus: List[str]) -> str:
        """Phase 3: attach LoRA and train it on ``corpus`` (or load the cached adapter); returns its key."""
        ftc = self.configs.finetune
        train_cfg = ftc.model_dump(exclude={"strategies", "overlap_window"})
        key = stage_key("finetune", train_cfg, stage_key(corpus), self.seed, self.model_fingerprint, self.tokenizer_id)
        stage = self.store.stage("finetune", key)
        self._peft = attach_lora(self.target_model, ftc.lora_r, ftc.lora_alpha, ftc.lora_dropout, ftc.target_modules)
        self._peft.to(self.device)
        adapter_path = stage.path / "adapter.pt"
        if stage.is_done():
            logger.info("Reusing the cached LoRA adapter")
            load_lora_state(self._peft, torch.load(adapter_path, map_location=self.device))
        else:
            logger.info(f"Fine-tuning on {len(corpus)} texts")
            losses = train_lora(self._peft, corpus, self.tokenizer, self.device, epochs=ftc.epochs,
                                batch_size=ftc.batch_size, grad_accum=ftc.grad_accum, lr=ftc.lr,
                                warmup_steps=ftc.warmup_steps, max_length=ftc.max_length, seed=self.seed,
                                progress_label="finetune-lora")
            stage.path.mkdir(parents=True, exist_ok=True)
            torch.save(lora_state(self._peft), adapter_path)
            stage.write([], info={"corpus_size": len(corpus), "losses": losses})
        logger.info(f"LoRA: base model {param_dtypes(self._peft, trainable=False)}, "
                    f"adapter weights {param_dtypes(self._peft, trainable=True)}")
        self._peft.eval()
        self.target_model.eval()
        return key

    # ----------------------------------------------------------------------------------- the run
    #
    # run_attack() runs every configured phase. run_phase1() ... run_phase4() run one phase each, in
    # order, so that a notebook can give every phase its own cell; finish() then builds the result.
    # The run state lives on the attack object, and every stage is cached, so running a phase again
    # (also after a kernel restart, from run_phase1() on) reads the finished stages back.

    def _reset_run(self: Self) -> None:
        """Forget the phases run so far (the cached stages stay on disk)."""
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._s0: Optional[List[Candidate]] = None
        self._s0_key: Optional[str] = None
        self._s1_novel: Optional[List[Candidate]] = None
        self._ft_corpus: Optional[List[str]] = None
        self._adapter_key: Optional[str] = None

    def _release_adapter(self: Self) -> None:
        """Remove the LoRA adapter, if attached, so that the base model is queried again."""
        if getattr(self, "_peft", None) is not None:
            detach_lora(self._peft)
            self._peft = None
            self.target_model.eval()

    def _keep(self: Self, phase: str, out: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
        """Store a phase's metrics and extracted records (replacing an earlier run of it)."""
        records = [_extracted_record(c) for c in out["selected"] if c.attacker_matches or c.oracle_matches]
        self._outputs[phase] = {"metrics": out["metrics"], "extracted": records}
        return out["metrics"]

    def _phase_header(self: Self, number: int, title: str) -> None:
        phases = self.configs.phases()
        logger.info(f"===== Phase {number} ({phases.index(number) + 1} of {len(phases)}): {title} =====")

    def _require(self: Self, done: bool, message: str) -> None:
        if not done:
            raise RuntimeError(message)

    def run_phase1(self: Self) -> Dict[str, Dict[str, Optional[float]]]:
        """Phase 1, initial extraction: generate, filter and verify -> S0. Returns the phase's metrics."""
        self._release_adapter()
        self._reset_run()
        self._phase_header(1, "initial extraction")
        logger.info(f"Target model: {param_dtypes(self.target_model)} on {self.device}")
        p1 = self._extraction_phase(PHASE_INITIAL, self.configs.initial, self.model_fingerprint)
        self._s0 = [c for c in p1["selected"] if c.attacker_matches]
        self._s0_key = p1["key"]
        logger.info(f"Phase 1 done: S0 = {len(self._s0)} verified sequences")
        return self._keep(PHASE_INITIAL, p1)

    def run_phase2(self: Self) -> Dict[str, Dict[str, Optional[float]]]:
        """Phase 2, expansion: re-prompt with anchors cut from S0 -> S1-novel. Returns the phase's metrics."""
        self._require(self.configs.expansion is not None,
                      "Phase 2 is off: the config sets `expansion: null` or has no `attacker_verifier`")
        self._require(self._s0 is not None, "Run run_phase1() first")
        self._release_adapter()
        self._phase_header(2, "expansion")
        p2 = self._expansion_phase(self._s0, self._s0_key)
        self._s1_novel = p2["s1_novel"]
        logger.info(f"Phase 2 done: S1-novel = {len(self._s1_novel)} sequences")
        return self._keep(PHASE_EXPANSION, p2)

    def run_phase3(self: Self) -> Dict[str, Any]:
        """Phase 3, fine-tuning: LoRA on S0 and S1-novel. The adapter stays attached for Phase 4."""
        ftc = self.configs.finetune
        self._require(ftc is not None, "Phases 3-4 are optional and not configured: add `finetune` to the config")
        self._require(self._s0 is not None, "Run run_phase1() first")
        self._require(self.configs.expansion is None or self._s1_novel is not None, "Run run_phase2() first")
        self._release_adapter()
        self._phase_header(3, "LoRA fine-tuning")
        s1_novel = self._s1_novel or []
        self._ft_corpus = build_finetune_corpus([c.continuation for c in self._s0] + [c.full_text for c in s1_novel],
                                                ftc.corpus_min_words, ftc.corpus_seed)
        if not self._ft_corpus:
            logger.warning("Fine-tuning skipped: S0 and S1-novel hold no text of the minimum length")
            return {"finetune_corpus": 0}
        try:
            self._adapter_key = self._finetune(self._ft_corpus)
        except BaseException:
            self._release_adapter()
            raise
        logger.info(f"Phase 3 done: adapter trained on {len(self._ft_corpus)} texts")
        return {"finetune_corpus": len(self._ft_corpus)}

    def run_phase4(self: Self) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
        """Phase 4, re-extraction with the fine-tuned model -> S_new. Returns the phase's metrics.

        The adapter is removed afterwards. If it is not attached (Phase 4 was interrupted, or the
        kernel restarted after run_phase1-3), the trained adapter is read back from the cache.
        """
        ftc = self.configs.finetune
        self._require(ftc is not None, "Phases 3-4 are optional and not configured: add `finetune` to the config")
        self._require(self._ft_corpus is not None, "Run run_phase3() first")
        if not self._ft_corpus:
            logger.warning("Phase 4 skipped: Phase 3 had no fine-tuning text")
            return None
        self._phase_header(4, "re-extraction with the fine-tuned model")
        try:
            if self._peft is None:
                self._adapter_key = self._finetune(self._ft_corpus)  # reads the cached adapter
            overlap = TokenWindowIndex(self._ft_corpus, self.tokenizer, ftc.overlap_window)
            out = self._extraction_phase(PHASE_FINETUNED, ftc.strategies or self.configs.initial,
                                         self._adapter_key, overlap, stage_key(self._ft_corpus))
        finally:
            self._release_adapter()
        out["metrics"]["total"]["finetune_corpus"] = len(self._ft_corpus)
        logger.info(f"Phase 4 done: S_new = {out['metrics']['total']['verified']} verified sequences")
        return self._keep(PHASE_FINETUNED, out)

    def finish(self: Self) -> ExtractionResult:
        """Build the result from the phases run so far."""
        self._require(PHASE_INITIAL in self._outputs, "Run run_phase1() first")
        order = (PHASE_INITIAL, PHASE_EXPANSION, PHASE_FINETUNED)
        metrics = {phase: self._outputs[phase]["metrics"] for phase in order if phase in self._outputs}
        extracted = [r for phase in order if phase in self._outputs for r in self._outputs[phase]["extracted"]]
        result = ExtractionResult("memorization_extraction", self.attack_id, self.configs.model_dump(), metrics,
                                  extracted, store_text=self.configs.store_text)
        logger.info("Memorization extraction finished\n" + result.summary())
        return result

    def run_attack(self: Self) -> ExtractionResult:
        """Run the configured phases and return the result."""
        phases = {1: self.run_phase1, 2: self.run_phase2, 3: self.run_phase3, 4: self.run_phase4}
        for number in self.configs.phases():
            phases[number]()
        return self.finish()

    def _metrics(
        self: Self,
        generated: Dict[str, Sequence[Candidate]],
        scored: Dict[str, Sequence[Candidate]],
        selected: Sequence[Candidate],
        extra: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Counts per strategy and in total (``extra`` adds per-strategy counts, summed into the total)."""
        has_attacker, has_oracle = self.attacker_verifier is not None, self.audit_oracle is not None

        def _row(gen: int, kept: int, sel: Sequence[Candidate]) -> Dict[str, Optional[float]]:
            verified = sum(1 for c in sel if c.attacker_matches) if has_attacker else None
            oracle = sum(1 for c in sel if c.oracle_matches) if has_oracle else None
            return {
                "generated": gen,
                "kept": kept,
                "selected": len(sel),
                "verified": verified,
                "extraction_rate": verified / gen if verified is not None and gen else None,
                "oracle_verified": oracle,
                "oracle_rate": oracle / gen if oracle is not None and gen else None,
            }

        rows = {
            name: _row(len(generated[name]), len(scored[name]), [c for c in selected if c.strategy == name])
            for name in generated
        }
        rows["total"] = _row(sum(len(v) for v in generated.values()), sum(len(v) for v in scored.values()), selected)
        for name, counts in (extra or {}).items():
            rows[name].update(counts)
            for k, v in counts.items():
                rows["total"][k] = rows["total"].get(k, 0) + v
        return rows
