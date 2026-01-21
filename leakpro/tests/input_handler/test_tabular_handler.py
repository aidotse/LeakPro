"""Test the tabular handler module."""

from copy import deepcopy

import numpy as np
from torch import equal
from torch.utils.data import SequentialSampler

from leakpro.tests.constants import get_tabular_handler_config
from leakpro.tests.input_handler.tabular_input_handler import TabularInputHandler
from leakpro.input_handler.mia_handler import MIAHandler

def test_abstract_handler_setup_tabular(tabular_handler:TabularInputHandler) -> None:
    """Test the initialization of the abstract tabular handler."""
    parameters = get_tabular_handler_config()
    assert tabular_handler is not None
    assert isinstance(tabular_handler, MIAHandler)

    # Check that correct model is instantiated
    assert tabular_handler.configs.target.model_class == tabular_handler.target_model.__class__.__name__

    # Check get/set methods + metadata
    assert tabular_handler.target_model_metadata is not None
    assert tabular_handler.train_indices is not None
    assert len(tabular_handler.train_indices) == parameters.train_data_points
    assert tabular_handler.test_indices is not None
    assert len(tabular_handler.test_indices) == parameters.test_data_points
    assert len(tabular_handler.train_indices) + len(tabular_handler.test_indices) <= parameters.data_points

    assert tabular_handler.target_model_metadata.optimizer is not None
    assert tabular_handler.target_model_metadata.optimizer.name == parameters.optimizer
    assert tabular_handler.target_model_metadata.optimizer.params["lr"] == parameters.learning_rate

    assert tabular_handler.target_model_metadata.criterion is not None
    assert tabular_handler.target_model_metadata.criterion.name == parameters.loss.lower()

    assert tabular_handler.target_model_metadata.epochs == parameters.epochs
    assert tabular_handler.target_model_metadata.data_loader.params["batch_size"] == parameters.batch_size
    assert tabular_handler.population is not None

    # Check data-related methods
    subset_of_population = tabular_handler.get_dataset(np.arange(parameters.data_points // 2))
    assert len(subset_of_population) == parameters.data_points // 2

    # Check label extraction
    labels = tabular_handler.get_labels(np.arange(parameters.data_points))
    assert len(labels) == parameters.data_points
    assert np.all(labels <= parameters.num_classes)
    assert np.all(labels >= 0)


def test_tabular_input_handler(tabular_handler:TabularInputHandler) -> None:
    """Test the CIFAR10 input handler."""
    parameters = get_tabular_handler_config()
    # get dataloader for training
    train_loader = tabular_handler.get_dataloader(tabular_handler.train_indices)
    assert train_loader is not None
    assert len(train_loader.dataset) == parameters.train_data_points
    assert train_loader.batch_size == parameters.batch_size
    # Check that shuffle = false in dataloader
    assert isinstance(train_loader.sampler, SequentialSampler)

    before_weights =  deepcopy(tabular_handler.target_model.to("cpu").state_dict())
    # train the model
    train_dict = tabular_handler.train(train_loader,
                                      tabular_handler.target_model,
                                      tabular_handler.get_criterion(),
                                      tabular_handler.get_optimizer(tabular_handler.target_model),
                                      parameters.epochs)
    # move back to cpu
    after_weights = train_dict.model.to("cpu").state_dict()
    weights_changed = [equal(before_weights[key], after_weights[key]) for key in before_weights]
    assert any(weights_changed) is False
