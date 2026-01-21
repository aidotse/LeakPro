"""Text Modality Extensions."""

import math
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler


class TextAugmentor:
    """Generate neighbours for text sequences attack."""

    def __init__(self, handler: AbstractInputHandler, mlm_name:str="bert-base-uncased") -> None:  # noqa: ARG002

        self.device_mlm   = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.mlm_tok = AutoTokenizer.from_pretrained(mlm_name, use_fast=True)
        self.mlm     = AutoModelForMaskedLM.from_pretrained(mlm_name).to(self.device_mlm).eval()

        self.eps = 1e-12
        self.max_len = int(getattr(self.mlm.config, "max_position_embeddings", 512)) - 2  # allow for special tokens

        self.key_mode = "word"   # "word" or "token" (this is for diversity over words or tokens in pruning global beams)

    def _log1mexp(self, logx: torch.Tensor) -> torch.Tensor:
        """Stable computation of log(1 - exp(logx)) for logx <= 0."""

        return torch.where(
            logx < -0.6931471805599453, torch.log1p(-torch.exp(logx)),
            torch.log(-torch.expm1(logx))
        )

    def _topk_from_masked_row(self,
                              row: torch.Tensor,
                              mask: torch.Tensor,
                              k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top-k values/indices from row with boolean mask."""

        v = row.numel()
        vals, tids = torch.topk(row.masked_fill(~mask, -1e9), k=min(k, v))
        return vals, tids

    def _prune_diverse_with_counter(self,
                                    cands: List[Tuple[float, str, torch.Tensor, torch.Tensor, object]],
                                    k: int,
                                    usage: Counter,
                                    lam: float) -> List[Tuple[float, str, torch.Tensor, torch.Tensor]]:
        """Prune candidates to top-k with diversity using a usage counter and lookahead.

        Args:
        ----
            cands: list of (score, string, ids_tensor, attn_tensor, diversity_key)
            k: number of items to select
            usage: Counter object tracking usage of diversity keys
            lam: diversity penalty weight (higher = more diverse)

        Returns:
        -------
            pruned list of (score, string, ids_tensor, attn_tensor)

        """

        cands = sorted(cands, key=lambda x: x[0], reverse=True) # sort by score descending
        selected = []
        local_use = usage.copy() # work on a local copy so scoring within this prune is consistent

        while cands and len(selected) < k:
            best_i = best_adj = None # index and adjusted score of best candidate
            # consider only the top `lookahead` candidates for efficiency
            for i, (sc, _, _, _, key) in enumerate(cands):
                adj = sc - (lam * local_use[key] if key is not None else 0.0) # adjusted score
                if best_adj is None or adj > best_adj: # find best candidate
                    best_adj, best_i = adj, i
            item = cands.pop(best_i) # remove from candidates
            selected.append(item) # add to selected
            key = item[-1] # get the diversity key
            if key is not None:
                local_use[key] += 1 # update local usage

        usage.update(local_use)  # update the original counter
        return [(sc, s, ids, attn) for (sc, s, ids, attn, _) in selected]

    def _div_key(self, seq_tokens: list[int], decoded: str) -> object:
        """Return a diversity key for a candidate based on the current key mode.

        Args:
        ----
            seq_tokens: list of token ids (without special tokens)
            decoded: decoded string (without special tokens)

        Returns:
        -------
            diversity key (str or tuple of ints)

        """
        return decoded.lower() if self.key_mode == "word" else tuple(seq_tokens)

    @torch.no_grad()
    def swap_words(  # noqa: C901, PLR0915
        self,
        text: str,
        word_indices: Sequence[int],                 # 0-based word indices to change (in surface text)
        want: int,                                   # number of outputs to return
        mlm: AutoModelForMaskedLM = None,
        tok: AutoTokenizer = None,  # must be a fast tokenizer to access word_ids()  # noqa: N803
        *,
        Lmax: int = 6,                               # hard cap on WordPiece length for a replacement  # noqa: N803
        k_per_step: int = 6,                         # top-k for token choices at each step
        beam_width_word: int = 6,                   # beam size inside a word
        beam_width_global: int = 6,                 # beam size across words
        length_penalty: float = 0.1,                 # >0 encourages shorter words (applied to STOP)
        diversity_penalty: float = 100.0,              # >0 encourages diverse outputs (in global beam)
    ) -> List[str]:
        """Variable-length whole-word substitutions with sequential/global beams and additive scoring.

        (i) Map given word indices -> token spans.
        (ii) Process words left->right; when decoding a word, previously chosen words are already spliced.
        (iii) Per-slot additive score: log p(choice) - log(1 - p(original_slot_if_exists)).
            For STOP at step t>=1: log p_stop - log(1 - p(original_next_slot_if_exists)).
        (iv) Local (per-word) constrained beam + global beam over words.

        Returns up to `want` unique decoded strings.
        """
        device = self.device_mlm
        global_usage = Counter()  # <-- counter to ensure diverse outputs across all words


        tok = tok or self.mlm_tok # use provided or default
        mlm = mlm or self.mlm # use provided or default
        mlm = mlm.to(device).eval() # ensure on correct device
        # ensure fast tokenizer
        if not getattr(tok, "is_fast", False):
            raise ValueError("Use a fast tokenizer to access word_ids().")

        # one-shot tokenization to get initial word_id mapping
        def tok_with_word_ids(s: str) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[int]]]:
            """Tokenize and return (ids, attn, word_ids)."""
            enc = tok(s, return_tensors="pt", add_special_tokens=True) # add_special_tokens to align with MLM
            ids = enc["input_ids"][0].to(device) # [T]
            attn = enc["attention_mask"][0].to(device) # [T]
            wids = enc.word_ids(0) # List[Optional[int]] word indices per token
            return ids, attn, wids

        def word_to_tok_span(widx: int, word_ids: List[Optional[int]]) -> List[int]:
            """Return list of token indices for the given surface word index (may be empty)."""
            return [i for i, wid in enumerate(word_ids) if wid == widx]

        # constraints (head vs continuation)
        vocab = tok.get_vocab() # get vocab dict {str: int}
        inv_vocab = {v: k for k, v in vocab.items()} # inverse {int: str}

        def is_head_id(tid: int) -> bool:
            """A head piece does not start with ##."""
            return not inv_vocab[tid].startswith("##")

        head_mask_vec = None  # lazily built per forward (depends on device)
        cont_mask_vec = None

        def get_head_cont_masks(v: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """Return boolean masks for head/continuation pieces of size v."""
            nonlocal head_mask_vec, cont_mask_vec # cache
            # check if we need to rebuild mask
            if head_mask_vec is None or head_mask_vec.numel() != v or head_mask_vec.device != self.device_mlm:
                # we can use range as hugging face vocab ids are 0..V-1
                head_mask_vec = torch.tensor([is_head_id(i) for i in range(v)], dtype=torch.bool, device=self.device_mlm)
                cont_mask_vec = ~head_mask_vec
            return head_mask_vec, cont_mask_vec

        def forward_logits_at(ids: torch.Tensor, attn: torch.Tensor, pos_list: List[int]) -> torch.Tensor:
            """Forward MLM and return logits at specified positions."""

            out = mlm(input_ids=ids.unsqueeze(0), attention_mask=attn.unsqueeze(0)).logits[0]  # [T, V]
            idx = torch.tensor(pos_list, device=out.device, dtype=torch.long) # pick positions to return
            return out.index_select(0, idx)  # [len(pos_list), V]

        # Local variable-length beam for ONE word, given current context.
        # It edits a copy of ids by splicing tokens at the word span start.
        def decode_one_word(  # noqa: C901, PLR0915
            ids_ctx: torch.Tensor,
            attn_ctx: torch.Tensor,
            span_tok_idxs: List[int],
        ) -> List[Tuple[float, List[int]]]:
            """Returns list of (score, seq_token_ids) for replacement using variable-length with STOP after at least one piece.

            Args:
                ids_ctx: [T] current full token ids with original span present
                attn_ctx: [T] current attention mask
                span_tok_idxs: list of token indices (in ids_ctx) that form the original word
            Returns:
                List of (score, seq_token_ids) for the replacement candidates (up to beam_width
                inside the word). The seq_token_ids do NOT include special tokens (CLS/SEP).

            """

            # We decode by repeatedly placing a single [MASK] at the (start of span + t),
            # forwarding, and expanding the beam. We remove the original span upfront to make room.
            start, end = span_tok_idxs[0], span_tok_idxs[-1] + 1 # set the start/end token indices for the span

            # remove original span [start:end]
            base = ids_ctx.tolist()
            base_ids = base[:start] + base[end:]
            base_attn = attn_ctx.tolist()
            base_attn = base_attn[:start] + base_attn[end:]

            # original tokens in the span (for scoring)
            orig_span_ids = ids_ctx[span_tok_idxs].tolist()

            # beam state for a word: (score, seq_ids, curr_ids_list)
            # - score is the accumulated log-prob score for the word so far
            # - seq_ids is the list of token ids chosen so far for this word
            # - curr_ids_list is the current full token ids with a single [MASK] at
            init_ids = base_ids[:start] + [tok.mask_token_id] + base_ids[start:] # insert initial [MASK]
            init_attn = base_attn[:start] + [1] + base_attn[start:] # attend to the mask
            # hypothesis: score, seq, ids, attn
            beam_word: List[Tuple[float, List[int], List[int], List[int]]] = [(0.0, [], init_ids, init_attn)]

            v = mlm.get_input_embeddings().num_embeddings  # vocab size
            head_mask_proto, cont_mask_proto = get_head_cont_masks(v) # boolean masks for head/cont pieces

            finished: List[Tuple[float, List[int]]] = []  # (score, seq)

            t = 0
            while t < Lmax and beam_word:
                nxt: List[Tuple[float, List[int], List[int], List[int]]] = []

                # loop over current beam
                for sc, seq, cur_ids, cur_attn in beam_word:
                    # locate current [MASK] position (insertion cursor)
                    cur_ids_tensor = torch.tensor(cur_ids, device=device, dtype=torch.long)
                    cur_attn_tensor = torch.tensor(cur_attn, device=device, dtype=torch.long)
                    try:
                        pos = cur_ids.index(tok.mask_token_id) # position of the [MASK]
                    except ValueError:
                        # no mask found (should not happen)
                        continue

                    # truncate if we exceed max length
                    if len(cur_ids) > self.max_len:
                        cur_ids = cur_ids[:self.max_len]
                        cur_attn = cur_attn[:self.max_len]

                    # forward only at this position
                    row = forward_logits_at(cur_ids_tensor, cur_attn_tensor, [pos])[0]  # [V] logits at pos
                    logp_row = torch.log_softmax(row, dim=-1) # [V] log-probs at pos

                    # calibration term for this slot: log(1 - p(orig_t)) if exists
                    if t < len(orig_span_ids):
                        logp_orig_t = logp_row[orig_span_ids[t]]
                        log1m_p0_t = self._log1mexp(logp_orig_t)
                    else:
                        log1m_p0_t = torch.tensor(0.0, device=device) #we have no token to compare to so set p_orig = 0

                    # constraints and expansions
                    if t == 0:
                        # must choose a headpiece; STOP not allowed at t=0
                        vals, tids = self._topk_from_masked_row(logp_row, head_mask_proto, k_per_step)
                        for v, tid in zip(vals.tolist(), tids.tolist()):
                            # write token tid at pos, then insert a *new* mask after it for potential continuation
                            new_ids = cur_ids[:pos] + [tid, tok.mask_token_id] + cur_ids[pos+1:]
                            new_attn = cur_attn[:pos] + [1, 1] + cur_attn[pos+1:]
                            new_seq  = seq + [tid]
                            new_sc   = sc + (float(v) - float(log1m_p0_t)) # p-(1-p_orig)
                            nxt.append((new_sc, new_seq, new_ids, new_attn))
                    else:
                        # t >= 1: CONTINUE with continuation piece(s) OR STOP
                        # CONTINUE branch
                        vals_c, tids_c = self._topk_from_masked_row(logp_row, cont_mask_proto, k_per_step)
                        for v, tid in zip(vals_c.tolist(), tids_c.tolist()):
                            new_ids = cur_ids[:pos] + [tid, tok.mask_token_id] + cur_ids[pos+1:]
                            new_attn = cur_attn[:pos] + [1, 1] + cur_attn[pos+1:]
                            new_seq  = seq + [tid]
                            new_sc   = sc + (float(v) - float(log1m_p0_t))
                            nxt.append((new_sc, new_seq, new_ids, new_attn))

                        # STOP branch
                        # continuation mass
                        cont_prob = torch.softmax(row, dim=-1)[cont_mask_proto].sum().item() # sum continuation token probs
                        p_stop = max(1e-12, 1.0 - cont_prob)
                        # length prior (encourage stopping as t grows)
                        if length_penalty > 0.0:
                            # logit transform + penalty
                            logit = math.log(p_stop) - math.log(max(1e-12, 1.0 - p_stop))
                            logit = logit + length_penalty * t
                            p_stop = 1.0 / (1.0 + math.exp(-logit))
                            p_stop = min(max(p_stop, 1e-12), 1.0 - 1e-12)
                        stop_score = math.log(p_stop)
                        new_sc = sc + (stop_score - float(log1m_p0_t))
                        # Materialize a FINISHED candidate: remove the cursor [MASK]
                        finished.append((new_sc, seq))  # seq is the wordpiece list

                # prune
                nxt.sort(key=lambda x: x[0], reverse=True)
                beam_word = nxt[:beam_width_word]
                t += 1

            # If nothing STOPped (e.g., hit Lmax), close beams by removing trailing mask
            for sc, seq, _, _ in beam_word:
                finished.append((sc, seq))

            # drop "no-change" candidate (exact match to original span)
            finished = [(sc, seq) for (sc, seq) in finished if seq != orig_span_ids[:len(seq)]]
            # ensure at least something to proceed with
            finished.sort(key=lambda x: x[0], reverse=True)
            return finished[:beam_width_word]

        # --------- GLOBAL BEAM (left -> right over requested words) ----------
        # Initial context
        ids0, attn0, _ = tok_with_word_ids(text)

        # Sort and uniquify requested surface-word indices
        req_words = sorted({int(w) for w in word_indices})

        # Global hypotheses: (score, text_string, ids_tensor, attn_tensor)
        global_beam: List[Tuple[float, str, torch.Tensor, torch.Tensor]] = [(0.0, text, ids0, attn0)]

        for w_i, w in enumerate(req_words):
            nxt_global: List[Tuple[float, str, torch.Tensor, torch.Tensor, object]] = []

            for sc_g, s_g, ids_g, attn_g in tqdm(global_beam, desc=f"Beam search, word {w_i}/{len(word_indices)}", unit="hyp"):
                # Retokenize this hypothesis to map surface word idx -> token span
                _, _, wids = tok_with_word_ids(s_g) # get the token indices for each word
                span = word_to_tok_span(w, wids) # get the token indices for word position w
                if not span:
                    # word index not present (punctuation/changed earlier) -> carry hypothesis forward unchanged
                    nxt_global.append((sc_g, s_g, ids_g, attn_g, None))
                    continue

                # Decode candidates for this word under current context
                span_cands = decode_one_word(ids_g, attn_g, span)  # list of (word_score, seq_token_ids)
                for sc_w, seq in span_cands:
                    # Splice seq at span start
                    start, end = span[0], span[-1] + 1
                    ids_list = ids_g.tolist()
                    new_ids_list = ids_list[:start] + seq + ids_list[end:]
                    new_attn_list = attn_g.tolist()
                    new_attn_list = new_attn_list[:start] + [1] * len(seq) + new_attn_list[end:]
                    new_ids = torch.tensor(new_ids_list, device=device, dtype=torch.long)
                    new_attn = torch.tensor(new_attn_list, device=device, dtype=torch.long)
                    # Decode to text
                    s_new = tok.decode(new_ids, skip_special_tokens=True)
                    key   = self._div_key(seq, s_new) # diversity key for this candidate
                    nxt_global.append((sc_g + sc_w, s_new, new_ids, new_attn, key))

            # prune global
            global_beam = self._prune_diverse_with_counter(
                nxt_global,
                k=beam_width_global,
                usage=global_usage,
                lam=diversity_penalty
            )

            if not global_beam:
                break

        # --------- finalize top-k unique strings ----------
        out, seen = [], set()
        for _, s, _, _ in global_beam:
            if s not in seen:
                seen.add(s)
                out.append(s)
                if len(out) >= want:
                    break
        return out
