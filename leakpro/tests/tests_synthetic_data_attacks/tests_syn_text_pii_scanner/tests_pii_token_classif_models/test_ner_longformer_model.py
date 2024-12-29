"""Tests for ner_longformer_model module."""
import os
from random import random

import pytest
import torch
from safetensors.torch import save_file
from transformers import LongformerTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

from leakpro.synthetic_data_attacks.syn_text_pii_scanner.pii_token_classif_models import ner_longformer_model as lgfm


def test_get_tokenizer() -> None:
    """Assert results of get_tokenizer function."""
    tokenizer = lgfm.get_tokenizer()
    assert isinstance(tokenizer, LongformerTokenizerFast)
    assert tokenizer.clean_up_tokenization_spaces

def test_load_model() -> None:
    """Test tests load_model function and implicitly tests class NERLongformerModel (init, state_dict, load_state_dict and forward pass methods).""" # noqa: E501
    #Case weights_crossentropy not dim=1
    with pytest.raises(Exception, match="weights_crossentropy.dim must be 1 in ner_longformer_model.py"):
        lgfm.load_model(num_labels=1, weights_crossentropy=torch.tensor([[1]], dtype=torch.float32))
    #Case weights_crossentropy size not equal num_labels dim=1
    with pytest.raises(Exception, match="weights_crossentropy.numel and num_labels do not match in ner_longformer_model.py"):
        lgfm.load_model(num_labels=2, weights_crossentropy=torch.tensor([1], dtype=torch.float32))
    #Case no weights
    model = lgfm.load_model(num_labels=1)
    assert isinstance(model, lgfm.NERLongformerModel)
    assert model.num_labels == 1
    assert model.loss_fun.ignore_index == -100
    assert model.loss_fun.weight is None
    #Case with weights
    num_labels = 2
    weights_crossentropy = torch.tensor([1, 10], dtype=torch.float32)
    crossentropy_ignore_index = -50
    model = lgfm.load_model(
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
    ##Provide path case
    #Modify and get first weight
    param = next(model.longformer.parameters())
    with torch.no_grad():
        param[0][0] = random() # noqa: S311
    fw_model = param[0][0].item()
    #Save current model to a .safetensors file
    path = "./test_load_model.safetensors"
    save_file(model.state_dict(), path)
    #Load new model with no path provided and assert first weight is not equal
    weights_crossentropy2 = torch.tensor([1, 10, 1], dtype=torch.float32)
    assert not torch.equal(weights_crossentropy, weights_crossentropy2)
    model2 = lgfm.load_model(
        num_labels = num_labels+1,
        weights_crossentropy = weights_crossentropy2,
        crossentropy_ignore_index = crossentropy_ignore_index*2
    )
    fw_model2 = next(model2.longformer.parameters())[0][0].item()
    assert not (fw_model - fw_model2)**2 < 0.000001
    #Load new model with path provided and assert first weight is equal and other attributes the same
    model3 = lgfm.load_model(
        num_labels = num_labels+1,
        weights_crossentropy = torch.tensor([1, 10, 1], dtype=torch.float32),
        crossentropy_ignore_index = -50,
        path = path
    )
    fw_model3 = next(model3.longformer.parameters())[0][0].item()
    assert (fw_model - fw_model3)**2 < 0.000001
    assert model.num_labels == model3.num_labels
    assert model.loss_fun.ignore_index == model3.loss_fun.ignore_index
    assert torch.equal(model.loss_fun.weight, model3.loss_fun.weight)
    #Clean up
    os.remove(path)
    assert not os.path.exists(path)
