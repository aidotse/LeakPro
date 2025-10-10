"""Implementation of the CAMIA attack."""


import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


@dataclass
class SignalConfig:
    # CE streaming
    max_len_ce: int = 1024
    ce_slice_len: int = 256
    ce_batch_size: int = 8
    use_fp16_ce: bool = True

    # truncations (e.g. {"T": None, "200": 200, "300": 300})
    cut_tprimes: Dict[str, Optional[int]] = None

    # feature families
    cb_tau_list: Sequence[float] = (2.5, 3.0, 3.5)
    lz_bin_list: Sequence[int] = (16, 32)

    # optional expensive signals
    slope_tprimes: Sequence[int] = (256, 512)
    apen_tprimes: Sequence[int] = (256,)
    store_full_seq: bool = False
    elbow_store_i: bool = False  # kept for compatibility (placeholder)

    # flat / elbow params (placeholders if you later plug in your elbow code)
    elbow_skip_left: int = 16
    elbow_skip_right_frac: float = 0.2
    flat_skip_left: int = 16
    flat_skip_right: int = 16
    flat_eps_init: float = 0.03
    flat_eps_mul: float = 2.0
    flat_eps_max_factor: float = 8.0

    # neighbors
    nei_k: int = 0  # 0 disables neighbor delta

    def __post_init__(self):
        if self.cut_tprimes is None:
            self.cut_tprimes = {"T": None, "200": 200, "300": 300}


@dataclass
class CombinerConfig:
    clip: float = 1e-5                  # p-value clipping in (clip, 1-clip)
    nei_weight: float = 1.0             # default weight for 'nei_delta'
    default_weight: float = 1.0         # weight for other features
    directions: Dict[str, str] = None   # per-feature tail: "lower" or "higher"

    def __post_init__(self):
        if self.directions is None:
            self.directions = {}


# ================================== Signals ==================================

class SignalExtractor:
    """Efficient feature generator for text-based MIA signals (no LR)."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[str] = None,
        config: Optional[SignalConfig] = None,
    ):
        self.model = model.eval()
        self.tok = tokenizer
        self.cfg = config or SignalConfig()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    # ----------------- helpers -----------------
    @staticmethod
    def _safe_mean(x: Sequence[float]) -> float:
        return float(np.mean(x)) if len(x) else 0.0

    @staticmethod
    def _diff(a: float, b: float) -> float:
        return (a - b) if (np.isfinite(a) and np.isfinite(b)) else 0.0

    @staticmethod
    def _replicate_if_short(seq: Sequence[float], desired: int) -> List[float]:
        if desired is None or not seq or len(seq) >= desired:
            return list(seq)
        out = list(seq)
        while len(out) < desired:
            need = desired - len(out)
            out.extend(seq[:need])
        return out

    # ----------------- CE streaming -----------------
    @torch.no_grad()
    def _ce_losses(self, texts: Sequence[str]) -> List[List[float]]:
        """Token-level CE per text; returns list of CE sequences."""
        if not texts:
            return []

        enc = self.tok(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_len_ce,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        use_amp = self.cfg.use_fp16_ce and (self.device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = self.model(
                enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False
            ).logits[:, :-1]  # [B, T-1, V]

        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:]
        B, T, _ = logits.shape

        slabs = []
        for s in range(0, T, self.cfg.ce_slice_len):
            e = min(s + self.cfg.ce_slice_len, T)
            slab = logits[:, s:e].float()  # [B, S, V]
            tgt_sl = tgt[:, s:e]
            logp = slab.log_softmax(dim=-1)
            ce = -logp.gather(-1, tgt_sl.unsqueeze(-1)).squeeze(-1)  # [B, S]
            slabs.append(ce)
            del slab, logp, ce
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        ce_full = torch.cat(slabs, dim=1)  # [B, T]
        return [ce_full[i, : int(mask[i].sum())].tolist() for i in range(B)]

    @torch.no_grad()
    def _ce_stream(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        bs = self.cfg.ce_batch_size
        for i in range(0, len(texts), bs):
            out.extend(self._ce_losses(texts[i:i + bs]))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return out

    # ----------------- primitive signals -----------------
    @staticmethod
    def _count_below(seq, thr, Tprime):
        sub = seq if Tprime is None else seq[:Tprime]
        return (sum(v <= thr for v in sub) / len(sub)) if sub else 0.0

    def _count_below_mean(self, seq, L=None):
        sl = seq if L is None else seq[:L]
        if not sl:
            return 0.0
        m = self._safe_mean(sl)
        return sum(v <= m for v in sl) / len(sl)

    @staticmethod
    def _count_below_prev_mean(seq, L=None):
        sl = seq if L is None else seq[:L]
        if len(sl) <= 1:
            return 0.0
        below, csum = 0, sl[0]
        for i, v in enumerate(sl[1:], 1):
            below += v <= csum / i
            csum += v
        return below / (len(sl) - 1)

    @staticmethod
    def _lz_complexity(seq, bins, Tprime):
        sub = seq if Tprime is None else seq[:Tprime]
        if not sub:
            return 0.0
        arr = np.asarray(sub)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return 1.0
        edges = np.linspace(lo, hi + 1e-9, bins)
        symbols = "".join(chr(65 + np.digitize(v, edges)) for v in arr)
        i, c = 0, 1
        while i < len(symbols) - 1:
            l = 1
            while symbols[i:i + l] in symbols[:i] and i + l < len(symbols):
                l += 1
            i += l
            c += 1
        return float(c)

    def _slope_signal(self, loss_seq, Tprime):
        if not loss_seq:
            return 0.0
        y = np.asarray(self._replicate_if_short(loss_seq, Tprime)[:Tprime])
        x = np.arange(len(y))
        var = np.mean((x - x.mean()) ** 2)
        if var == 0:
            return 0.0
        cov = np.mean((x - x.mean()) * (y - y.mean()))
        return float(cov / var)  # negative for members

    def _approximate_entropy(self, seq, Tprime, m=8, r=0.8):
        seq = self._replicate_if_short(seq, Tprime)[:Tprime]
        if len(seq) < m + 2:
            return 0.0
        vals = np.asarray(seq)

        def _phi(mm):
            N = len(vals) - mm + 1
            if N <= 0:
                return 0.0
            patterns = np.array([vals[i:i + mm] for i in range(N)])
            C = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) <= r, axis=0) / N
            return np.mean(np.log(C + 1e-12))

        return float(_phi(m) - _phi(m + 1))

    def token_diversity(self, text: str) -> float:
        toks = self.tok.tokenize(text)
        return len(set(toks)) / len(toks) if toks else 0.0

    @torch.no_grad()
    def _mean_ce_many(self, texts: Sequence[str], bs: int = 4) -> np.ndarray:
        out: List[float] = []
        for i in range(0, len(texts), bs):
            losses = self._ce_losses(texts[i:i + bs])
            out.extend(self._safe_mean(s) for s in losses)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return np.asarray(out, dtype=np.float32)

    # ----------------- neighbor delta -----------------
    def add_nei_delta(self, base_dicts: List[Dict[str, float]], neighbour_table):
        if self.cfg.nei_k == 0 or neighbour_table is None:
            return base_dicts
        neighbour_table = neighbour_table[:len(base_dicts)]
        flat = list(itertools.chain.from_iterable(neighbour_table))
        nei_ce = self._mean_ce_many(flat).reshape(len(base_dicts), self.cfg.nei_k)
        out = []
        for d, nei in zip(base_dicts, nei_ce):
            base_ce = self._safe_mean(d["loss_seq_orig"]) if "loss_seq_orig" in d else d["cut_T"]
            nd = d.copy()
            nd["nei_delta"] = float(base_ce - float(nei.mean()))
            out.append(nd)
        return out

    # ----------------- public: batch / dataset -----------------
    @torch.no_grad()
    def gather_signals_for_batch(self, texts: Sequence[str], neighbour_rows=None) -> List[Dict[str, float]]:
        """Computes the feature dicts for a batch of texts."""
        # 3× truncations × 3× repetitions
        variants = {(tag, r): [] for tag in self.cfg.cut_tprimes for r in (0, 1, 2)}

        def _reps(txt: str, tp: Optional[int]):
            if tp is None:
                base = txt
            else:
                ids = self.tok(txt)["input_ids"][:tp]
                base = self.tok.decode(ids, skip_special_tokens=True)
            return base, f"{base} {base}", f"{base} {base} {base}"

        for t in texts:
            for tag, tp in self.cfg.cut_tprimes.items():
                b, r1, r2 = _reps(t, tp)
                variants[(tag, 0)].append(b)
                variants[(tag, 1)].append(r1)
                variants[(tag, 2)].append(r2)

        # batched CE
        losses = {k: self._ce_stream(v) for k, v in variants.items()}

        feats: List[Dict[str, float]] = []
        for i, orig in enumerate(texts):
            L = lambda tag, rp: losses[(tag, rp)][i]
            d: Dict[str, float] = {}

            # 1) cut / cal / ppl / calppl
            for tag, tp in self.cfg.cut_tprimes.items():
                if tp is None:
                    txt_trim = orig
                else:
                    ids_trim = self.tok(orig)["input_ids"][:tp]
                    txt_trim = self.tok.decode(ids_trim, skip_special_tokens=True)
                div = max(self.token_diversity(txt_trim), 1e-12)
                lb, lr1, lr2 = L(tag, 0), L(tag, 1), L(tag, 2)
                vb, vr1, vr2 = map(self._safe_mean, (lb, lr1, lr2))
                for fam, post in (
                    ("cut", lambda x: x),
                    ("cal", lambda x: x / div),
                    ("ppl", lambda x: math.exp(x)),
                    ("calppl", lambda x: math.exp(x) / div),
                ):
                    vb_, vr1_, vr2_ = map(post, (vb, vr1, vr2))
                    d[f"{fam}_{tag}"] = vb_
                    d[f"{fam}_{tag}_rep1_diff"] = self._diff(vb_, vr1_)
                    d[f"{fam}_{tag}_rep2_diff"] = self._diff(vb_, vr2_)

            # 2) count-below CONST (invert) on "200"
            if "200" in self.cfg.cut_tprimes:
                for tau in self.cfg.cb_tau_list:
                    lb, lr1, lr2 = L("200", 0), L("200", 1), L("200", 2)
                    b, r1, r2 = (self._count_below(x, tau, 200) for x in (lb, lr1, lr2))
                    b, r1, r2 = (1 - b, 1 - r1, 1 - r2)
                    k = f"cb_const{tau}"
                    d[k] = b
                    d[f"{k}_rep1_diff"] = self._diff(b, r1)
                    d[f"{k}_rep2_diff"] = self._diff(b, r2)

            # 3) count-below MEAN / prev-mean (invert)
            for tag in self.cfg.cut_tprimes:
                lb, lr1, lr2 = L(tag, 0), L(tag, 1), L(tag, 2)
                base, r1, r2 = (self._count_below_mean(x) for x in (lb, lr1, lr2))
                base, r1, r2 = (1 - base, 1 - r1, 1 - r2)
                k = f"cbm_{tag}"
                d[k] = base
                d[f"{k}_rep1_diff"] = self._diff(base, r1)
                d[f"{k}_rep2_diff"] = self._diff(base, r2)
                d[f"cbpm_{tag}"] = 1 - self._count_below_prev_mean(lb)

            # 4) LZ on 200 + slope/ApEn on full
            for bins in self.cfg.lz_bin_list:
                lb, lr1, lr2 = L("200", 0), L("200", 1), L("200", 2)
                b, r1, r2 = (self._lz_complexity(x, bins, 200) for x in (lb, lr1, lr2))
                k = f"lz_bins{bins}"
                d[k] = b
                d[f"{k}_rep1_diff"] = self._diff(r1, b)
                d[f"{k}_rep2_diff"] = self._diff(r2, b)

            if self.cfg.slope_tprimes or self.cfg.apen_tprimes or self.cfg.store_full_seq or self.cfg.elbow_store_i:
                lb_full = L("T", 0)
                for tp in self.cfg.slope_tprimes:
                    d[f"slope_{tp}"] = -self._slope_signal(lb_full, tp)
                for tp in self.cfg.apen_tprimes:
                    d[f"apen_{tp}"] = -self._approximate_entropy(lb_full, tp)
                if self.cfg.store_full_seq:
                    d["loss_seq_orig"] = lb_full

            feats.append(d)

        # neighbor delta
        if self.cfg.nei_k and neighbour_rows is not None:
            feats = self.add_nei_delta(feats, neighbour_rows)

        return feats

    @torch.no_grad()
    def gather_signals_for_dataset(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        neighbour_table: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for s in range(0, len(texts), batch_size):
            e = s + batch_size
            out.extend(
                self.gather_signals_for_batch(
                    texts[s:e],
                    neighbour_rows=None if neighbour_table is None else neighbour_table[s:e],
                )
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return out

class Combiner:
    """
    Combine feature dicts into a single score using **empirical p-values**.
    Methods: 'edgington' (sum p), 'fisher' (sum -2 log p), 'pearson' (sum -2 log (1-p)), 'george' (sum logit(p)).
    """

    def __init__(self, config: Optional[CombinerConfig] = None):
        self.cfg = config or CombinerConfig()

    # ---- reference from non-members ----
    @staticmethod
    def build_reference(nonmember_dicts: Sequence[Dict[str, float]]) -> Dict[str, np.ndarray]:
        ref: Dict[str, List[float]] = {}
        for d in nonmember_dicts:
            for k, v in d.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    ref.setdefault(k, []).append(float(v))
        return {k: np.sort(np.asarray(vals, dtype=np.float32)) for k, vals in ref.items()}

    # ---- p-values ----
    def _pvalue(self, x: float, ref_arr: np.ndarray, tail: str = "lower") -> float:
        n = ref_arr.size
        if n == 0:
            return 0.5
        # lower tail = P(ref <= x)
        rank = np.searchsorted(ref_arr, x, side="right")  # count <= x
        if tail == "higher":
            p = (n - rank + 1) / (n + 1)
        else:
            p = rank / (n + 1)
        c = self.cfg.clip
        return float(min(max(p, c), 1.0 - c))

    @staticmethod
    def _combine_weighted(pvals: List[Tuple[float, float]], method: str) -> float:
        if not pvals:
            return 0.0
        if method == "edgington":
            return sum(w * p for p, w in pvals)
        if method == "fisher":
            return sum(-2.0 * w * math.log(p) for p, w in pvals)
        if method == "pearson":
            return sum(-2.0 * w * math.log(1.0 - p) for p, w in pvals)
        if method == "george":
            return sum(w * math.log(p / (1.0 - p)) for p, w in pvals)
        raise ValueError(f"Unknown combine method: {method}")

    # ---- scoring ----
    def score_batch(
        self,
        dicts: Sequence[Dict[str, float]],
        ref: Dict[str, np.ndarray],
        method: str = "fisher",
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        W = weights or {}
        out: List[float] = []
        for d in dicts:
            pvals: List[Tuple[float, float]] = []
            for k, v in d.items():
                if k not in ref or not np.isfinite(v):
                    continue
                tail = self.cfg.directions.get(k, "lower")  # default "lower"
                p = self._pvalue(float(v), ref[k], tail=tail)
                w = W.get(k, self.cfg.nei_weight if k == "nei_delta" else self.cfg.default_weight)
                pvals.append((p, w))
            out.append(self._combine_weighted(pvals, method))
        return out



class CAMIAConfig(BaseModel):
    """Configuration for the CAMIA attack."""

    model_config = ConfigDict(extra="forbid")
    extractor: SignalConfig = Field(default_factory=SignalConfig)
    combiner:  CombinerConfig = Field(default_factory=CombinerConfig)

    def build_extractor(self, model, tokenizer, device=None):
        return SignalExtractor(model, tokenizer, device, config=self.extractor)

    def build_combiner(self):
        return Combiner(config=self.combiner)

class AttackCAMIA(AbstractMIA):
    """Implementation of the BASE attack."""

    AttackConfig = CAMIAConfig # required config for attack

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the BASE attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the BASE attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = CAMIAConfig() if configs is None else CAMIAConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.aug = handler.modality_extension
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.extractor = self.configs.build_extractor(self.handler.target_model,
                                                      self.population._tokenizer,
                                                      device=device)
        self.combiner  = self.configs.build_combiner()


    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "CAMIA attack"
        reference_str = ""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        # STEPS:
        # 1) Generate loss sequences for all data
        dataset = self.handler.population


        # reference from non-members
        non_feats = self.extractor.gather_signals_for_dataset(dataset.data[0], batch_size=32)
        ref = self.combiner.build_reference(non_feats)

        dataset.set_mode("raw")
        for i, d in enumerate(dataset):
            if i == len(dataset)-1:
                augs = self.aug.swap_words(text=d,
                                           word_indices=list(range(0, 10)),
                                           want=3)

        # 2) Generate neighbors for audit data
        # 3) Generate loss sequence for audit data and neighbors
        # 4) Compute features for the data


    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """

        # pick out the in-members and out-members signals
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)),np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # compute ROC, TP, TN etc
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="BASE",
                                    metadata=self.configs.model_dump())
