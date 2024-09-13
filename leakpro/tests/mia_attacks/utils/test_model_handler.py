"""Test the model handler module."""

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.tests.input_handler.cifar10_input_handler import Cifar10InputHandler


def test_model_handler(image_handler:Cifar10InputHandler) -> None:
    """Test the initialization of the model handler."""
    model_handler = ModelHandler(image_handler)
    assert model_handler.handler is not None
    assert model_handler.init_params is not None

    configs = image_handler.configs.target

    model_handler._import_model_from_path(configs.module_path, configs.model_class)
    assert model_handler.model_blueprint.__name__ == configs.model_class

    meta_data_path = f"{configs.target_folder}/model_metadata.pkl"
    meta_data = model_handler._load_metadata(meta_data_path)
    assert meta_data == image_handler.target_model_metadata

    model_handler._get_optimizer_class(meta_data["optimizer"]["name"])
    assert model_handler.optimizer_class.__name__.lower() == meta_data["optimizer"]["name"]

    model_handler._get_criterion_class(meta_data["loss"]["name"])
    assert model_handler.criterion_class.__name__.lower() == meta_data["loss"]["name"]

    model_path = f"{configs.target_folder}/target_model.pkl"
    new_model, new_criterion = model_handler._load_model(model_path)
    assert new_model.__class__.__name__ == image_handler.target_model.__class__.__name__ # Check that the model is the same class
    assert id(new_model) != id(image_handler.target_model) # Check that the model is not the same instance
    assert new_criterion.__class__.__name__ == image_handler.get_criterion().__class__.__name__

    meta_data["optimizer"].pop("name")
    model_handler.optimizer_config = meta_data["optimizer"]
    (m, o, c) = model_handler._get_model_criterion_optimizer()
    assert m.__class__.__name__ == new_model.__class__.__name__
    assert o.__class__.__name__ == new_criterion.__class__.__name__
    assert c.__class__.__name__ == model_handler.optimizer_class.__name__
