"""Test the image handler module."""

from copy import deepcopy

import numpy as np
from torch import equal
from torch.utils.data import SequentialSampler

from leakpro.tests.constants import get_image_handler_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.input_handler.mia_handler import MIAHandler


def test_abstract_handler_setup(image_handler:ImageInputHandler) -> None:
    """Test the initialization of the image handler."""
    parameters = get_image_handler_config()
    assert image_handler is not None
    assert isinstance(image_handler, MIAHandler)

    # Check that correct model is instantiated
    assert image_handler.configs.target.model_class == image_handler.target_model.__class__.__name__

    # Check get/set methods + metadata
    assert image_handler.target_model_metadata is not None
    assert image_handler.train_indices is not None
    assert len(image_handler.train_indices) == parameters.train_data_points
    assert image_handler.test_indices is not None
    assert len(image_handler.test_indices) == parameters.test_data_points
    assert len(image_handler.train_indices) + len(image_handler.test_indices) < parameters.data_points

    assert image_handler.target_model_metadata.optimizer is not None
    assert image_handler.target_model_metadata.optimizer.name == parameters.optimizer
    assert image_handler.target_model_metadata.optimizer.params["lr"] == parameters.learning_rate

    assert image_handler.target_model_metadata.criterion is not None
    assert image_handler.target_model_metadata.criterion.name == parameters.loss

    assert image_handler.target_model_metadata.epochs == parameters.epochs
    assert image_handler.target_model_metadata.data_loader.params["batch_size"] == parameters.batch_size
    assert image_handler.population is not None

    # Check data-related methods
    population = image_handler.get_dataset(np.arange(parameters.data_points))

    assert len(population) == parameters.data_points
    unique_labels = np.zeros(parameters.num_classes)
    for _, (data, label) in enumerate(population):
        assert data.shape == parameters.img_size
        assert label < parameters.num_classes
        unique_labels[label] += 1
    assert len(unique_labels > 0) == parameters.num_classes
    assert np.sum(unique_labels) == parameters.data_points
    for label in unique_labels:
        assert label == parameters.images_per_class

    subset_of_population = image_handler.get_dataset(np.arange(parameters.data_points // 2))
    assert len(subset_of_population) == parameters.data_points // 2

    # Check label extraction
    labels = image_handler.get_labels(np.arange(parameters.data_points))
    assert len(labels) == parameters.data_points
    assert np.all(labels < parameters.num_classes)
    assert np.all(labels >= 0)
    assert np.issubdtype(labels.dtype, np.integer)

def test_cifar10_input_handler(image_handler:ImageInputHandler) -> None:
    """Test the CIFAR10 input handler."""
    parameters = get_image_handler_config()
    # get dataloader for training
    train_loader = image_handler.get_dataloader(image_handler.train_indices)
    assert train_loader is not None
    assert len(train_loader.dataset) == parameters.train_data_points
    assert train_loader.batch_size == parameters.batch_size
    # Check that shuffle = false in dataloader
    assert isinstance(train_loader.sampler, SequentialSampler)

    before_weights =  deepcopy(image_handler.target_model.state_dict())
    # train the model
    train_dict = image_handler.train(train_loader,
                                      image_handler.target_model,
                                      image_handler.get_criterion(),
                                      image_handler.get_optimizer(image_handler.target_model),
                                      parameters.epochs)
    after_weights = train_dict.model.state_dict()
    weights_changed = [equal(before_weights[key], after_weights[key]) for key in before_weights]
    assert any(weights_changed) is False
