#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.signals.token_evidence.

Tests cover, against a tiny pure-torch causal LM (no transformers dependency):
- the shift convention: logprob[i, j] scores token_ids[i, j] == input_ids[i, j + 1]
- argmax and mask/lengths agree with a direct log_softmax computation
- padded and unpadded encodings of the same sequence give identical per-sequence evidence
- vocabulary moments (need_moments) match a NumPy reference
- duck typing: a model returning a bare logits tensor works like one returning .logits
- CausalLMCollate pads ragged tensor / ndarray / list inputs and rejects 1-token sequences
- TokenEvidence.concatenate pads batches of different T and preserves row order
- get_loss is the masked mean NLL; the dense-logit methods raise NotImplementedError
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader

from leakpro.signals.token_evidence import CausalLMCollate, CausalLMModel, EvidenceRequest, TokenEvidence

VOCAB = 11
DIM = 6


class TinyCausalLM(nn.Module):
    """Embedding → causal prefix-sum → linear head. Causal by construction, returns .logits."""

    def __init__(self, vocab: int = VOCAB, dim: int = DIM, bare_tensor: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.bare_tensor = bare_tensor

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> object:
        """Return logits for every position from the (masked) prefix sum of embeddings."""
        emb = self.embed(input_ids)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        hidden = torch.cumsum(emb, dim=1)
        logits = self.head(hidden)
        return logits if self.bare_tensor else SimpleNamespace(logits=logits)


@pytest.fixture
def tiny_lm() -> TinyCausalLM:
    """Deterministic tiny LM on CPU."""
    torch.manual_seed(0)
    return TinyCausalLM()


@pytest.fixture
def wrapped(tiny_lm: TinyCausalLM) -> CausalLMModel:
    """The tiny LM behind the CausalLMModel wrapper, pinned to CPU."""
    return CausalLMModel(tiny_lm, device=torch.device("cpu"))


def _reference(model: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> dict:
    """Direct computation of what token_evidence should return."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    logp = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:]
    return {
        "logprob": logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).numpy(),
        "argmax": logits.argmax(-1).numpy(),
        "mask": attention_mask[:, 1:].bool().numpy(),
        "logp_mean": logp.mean(-1).numpy(),
        "logp_std": logp.std(-1).numpy(),
    }


def test_shift_convention_and_values(wrapped: CausalLMModel, tiny_lm: TinyCausalLM) -> None:
    """logprob[i, j] is log p(input_ids[i, j+1] | input_ids[i, :j+1]); argmax/mask/lengths agree."""
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (3, 7))
    mask = torch.ones_like(ids)
    ev = wrapped.token_evidence(ids, mask)
    ref = _reference(tiny_lm, ids, mask)

    assert ev.logprob.shape == (3, 6)
    np.testing.assert_allclose(ev.logprob, ref["logprob"], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ev.argmax, ref["argmax"])
    np.testing.assert_array_equal(ev.token_ids, ids[:, 1:].numpy())
    np.testing.assert_array_equal(ev.mask, ref["mask"])
    np.testing.assert_array_equal(ev.lengths, np.full(3, 6))
    assert ev.logprob.dtype == np.float32
    assert ev.logp_mean is None
    assert not ev.has_moments


def test_padding_is_invisible_per_sequence(wrapped: CausalLMModel) -> None:
    """The same sequence encoded alone and right-padded inside a batch yields identical evidence."""
    torch.manual_seed(2)
    short = torch.randint(0, VOCAB, (1, 4))
    long_ = torch.randint(0, VOCAB, (1, 9))
    alone = wrapped.token_evidence(short, torch.ones_like(short))

    ids = torch.zeros((2, 9), dtype=torch.long)
    att = torch.zeros((2, 9), dtype=torch.long)
    ids[0, :4], att[0, :4] = short[0], 1
    ids[1], att[1] = long_[0], 1
    batched = wrapped.token_evidence(ids, att)

    assert batched.lengths.tolist() == [3, 8]
    np.testing.assert_allclose(batched.logprob[0, :3], alone.logprob[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(batched.argmax[0, :3], alone.argmax[0])
    # Padded positions are masked and zero/-1 filled regardless of pad id.
    assert not batched.mask[0, 3:].any()
    assert np.all(batched.logprob[0, 3:] == 0.0)
    assert np.all(batched.argmax[0, 3:] == -1)
    assert np.all(batched.token_ids[0, 3:] == -1)


def test_moments_match_reference(wrapped: CausalLMModel, tiny_lm: TinyCausalLM) -> None:
    """need_moments returns the vocab-mean and vocab-std of log p at each position."""
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (2, 5))
    mask = torch.ones_like(ids)
    ev = wrapped.token_evidence(ids, mask, EvidenceRequest(need_moments=True))
    ref = _reference(tiny_lm, ids, mask)
    assert ev.has_moments
    np.testing.assert_allclose(ev.logp_mean, ref["logp_mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ev.logp_std, ref["logp_std"], rtol=1e-5, atol=1e-6)


def test_bare_tensor_output_is_accepted() -> None:
    """A model that returns the logits tensor directly is handled like one returning .logits."""
    torch.manual_seed(0)
    with_attr = CausalLMModel(TinyCausalLM(bare_tensor=False), device=torch.device("cpu"))
    torch.manual_seed(0)
    bare = CausalLMModel(TinyCausalLM(bare_tensor=True), device=torch.device("cpu"))
    ids = torch.randint(0, VOCAB, (2, 6))
    mask = torch.ones_like(ids)
    np.testing.assert_array_equal(with_attr.token_evidence(ids, mask).logprob, bare.token_evidence(ids, mask).logprob)


def test_collate_pads_mixed_inputs() -> None:
    """Tensors, ndarrays and lists of different lengths collate to a padded batch and mask."""
    collate = CausalLMCollate(pad_token_id=7)
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])),
        (np.array([4, 5]), np.array([4, 5])),
        ([6, 8, 9, 10], [6, 8, 9, 10]),
    ]
    ids, att = collate(batch)
    assert ids.shape == att.shape == (3, 4)
    assert ids.dtype == torch.long
    np.testing.assert_array_equal(ids.numpy(), [[1, 2, 3, 7], [4, 5, 7, 7], [6, 8, 9, 10]])
    np.testing.assert_array_equal(att.numpy(), [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])


def test_collate_rejects_single_token_sequence() -> None:
    """A 1-token sequence has no prediction to score."""
    with pytest.raises(ValueError, match="at least two tokens"):
        CausalLMCollate(pad_token_id=0)([([1], [1]), ([1, 2], [1, 2])])


def test_evidence_from_loader_matches_single_batch(wrapped: CausalLMModel) -> None:
    """Batching through a DataLoader + collate gives the same rows as one big batch, in order."""
    torch.manual_seed(4)
    seqs = [torch.randint(0, VOCAB, (n,)) for n in (5, 3, 8, 4, 6)]
    dataset = [(s, s) for s in seqs]
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=CausalLMCollate(pad_token_id=0))
    streamed = wrapped.evidence_from_loader(loader)

    ids, att = CausalLMCollate(pad_token_id=0)(dataset)
    whole = wrapped.token_evidence(ids, att)

    assert streamed.num_sequences == 5
    assert streamed.logprob.shape == whole.logprob.shape == (5, 7)
    np.testing.assert_allclose(streamed.logprob, whole.logprob, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(streamed.mask, whole.mask)
    np.testing.assert_array_equal(streamed.lengths, [4, 2, 7, 3, 5])


def test_concatenate_pads_and_validates() -> None:
    """Concatenate right-pads to the longest T and rejects mismatched moments."""
    a = TokenEvidence(
        logprob=np.ones((1, 2), np.float32), argmax=np.zeros((1, 2), np.int64), token_ids=np.zeros((1, 2), np.int64),
        mask=np.ones((1, 2), bool), lengths=np.array([2]),
    )
    b = TokenEvidence(
        logprob=np.ones((1, 4), np.float32), argmax=np.zeros((1, 4), np.int64), token_ids=np.zeros((1, 4), np.int64),
        mask=np.array([[True, True, True, False]]), lengths=np.array([3]),
    )
    cat = TokenEvidence.concatenate([a, b])
    assert cat.logprob.shape == (2, 4)
    np.testing.assert_array_equal(cat.mask, [[True, True, False, False], [True, True, True, False]])
    np.testing.assert_array_equal(cat.argmax[0, 2:], [-1, -1])
    np.testing.assert_array_equal(cat.lengths, [2, 3])

    with_moments = TokenEvidence(
        logprob=a.logprob, argmax=a.argmax, token_ids=a.token_ids, mask=a.mask, lengths=a.lengths,
        logp_mean=np.zeros((1, 2), np.float32), logp_std=np.ones((1, 2), np.float32),
    )
    with pytest.raises(ValueError, match="moments"):
        TokenEvidence.concatenate([a, with_moments])


def test_token_evidence_validates_shapes() -> None:
    """Inconsistent shapes or lengths are rejected at construction."""
    ok = {"logprob": np.zeros((2, 3), np.float32), "argmax": np.zeros((2, 3), np.int64),
          "token_ids": np.zeros((2, 3), np.int64), "mask": np.ones((2, 3), bool), "lengths": np.array([3, 3])}
    TokenEvidence(**ok)
    with pytest.raises(ValueError, match="lengths must equal"):
        TokenEvidence(**{**ok, "lengths": np.array([3, 2])})
    with pytest.raises(ValueError, match="argmax has shape"):
        TokenEvidence(**{**ok, "argmax": np.zeros((2, 4), np.int64)})
    with pytest.raises(TypeError, match="mask must be bool"):
        TokenEvidence(**{**ok, "mask": np.ones((2, 3), np.int64)})


def test_get_loss_is_masked_mean_nll(wrapped: CausalLMModel) -> None:
    """get_loss averages -logprob over valid positions per sequence."""
    torch.manual_seed(5)
    ids = torch.randint(0, VOCAB, (2, 6))
    att = torch.ones_like(ids)
    att[1, 4:] = 0
    ev = wrapped.token_evidence(ids, att)
    expected = -(ev.logprob * ev.mask).sum(1) / ev.lengths
    np.testing.assert_allclose(wrapped.get_loss(ids, att, per_point=True), expected, rtol=1e-6)
    np.testing.assert_allclose(wrapped.get_loss(ids, att, per_point=False), expected.mean(), rtol=1e-6)


def test_dense_logit_methods_are_not_supported(wrapped: CausalLMModel) -> None:
    """The (B, T, vocab)-materialising Model methods raise rather than silently blow up memory."""
    ids = torch.zeros((1, 3), dtype=torch.long)
    with pytest.raises(NotImplementedError):
        wrapped.get_logits(ids)
    with pytest.raises(NotImplementedError):
        wrapped.get_grad(ids, ids)
    with pytest.raises(NotImplementedError):
        wrapped.get_intermediate_outputs([0], ids)
