"""Tests for ner_longformer_model module."""
import pytest
import torch
from transformers import LongformerTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

from leakpro.synthetic_data_attacks.syn_text_pii_scanner.pii_token_classif_models import ner_longformer_model as lgfm


def test_get_tokenizer() -> None:
    """Assert results of get_tokenizer function."""
    tokenizer = lgfm.get_tokenizer()
    assert isinstance(tokenizer, LongformerTokenizerFast)
    assert tokenizer.clean_up_tokenization_spaces

def test_NERLongformerModel() -> None: # noqa: N802
    """Test class NERLongformerModel init and forward pass methods)."""
    #Case weights_crossentropy not dim=1
    with pytest.raises(Exception, match="weights_crossentropy.dim must be 1 in ner_longformer_model.py"):
        lgfm.NERLongformerModel(num_labels=1, weights_crossentropy=torch.tensor([[1]], dtype=torch.float32))
    #Case weights_crossentropy size not equal num_labels dim=1
    with pytest.raises(Exception, match="weights_crossentropy.numel and num_labels do not match in ner_longformer_model.py"):
        lgfm.NERLongformerModel(num_labels=2, weights_crossentropy=torch.tensor([1], dtype=torch.float32))
    #Case no weights
    num_labels = 3
    model = lgfm.NERLongformerModel(num_labels=num_labels)
    assert isinstance(model, lgfm.NERLongformerModel)
    assert model.num_labels == num_labels
    assert model.loss_fun.ignore_index == -100
    assert torch.equal(model.loss_fun.weight, torch.ones(num_labels, dtype=torch.float32))
    #Case with weights
    num_labels = 2
    weights_crossentropy = torch.tensor([1, 10], dtype=torch.float32)
    crossentropy_ignore_index = -50
    model = lgfm.NERLongformerModel(
        num_labels = num_labels,
        weights_crossentropy = weights_crossentropy,
        crossentropy_ignore_index = crossentropy_ignore_index
    )
    assert isinstance(model, lgfm.NERLongformerModel)
    assert model.num_labels == num_labels
    assert model.loss_fun.ignore_index == crossentropy_ignore_index
    assert torch.equal(model.loss_fun.weight, weights_crossentropy)
    #Forward pass
    input_ids = torch.tensor([[1,2,3]], dtype=torch.long)
    attention_mask = torch.tensor([[1,1,1]], dtype=torch.long)
    for labels in [None, torch.tensor([[0,1,0]], dtype=torch.long)]:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert isinstance(output, TokenClassifierOutput)
        assert isinstance(output.logits, torch.Tensor)
        assert output.logits.shape == (1, 3, 2)
        if labels is None:
            assert output.loss is None
        else:
            assert isinstance(output.loss.item(), float)
