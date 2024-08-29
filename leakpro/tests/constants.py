from dotmap import DotMap

STORAGE_PATH = "./leakpro/tests/tmp"

# User input handler for images
parameters = DotMap()
parameters.epochs = 10
parameters.batch_size = 64
parameters.learning_rate = 0.001
parameters.optimizer = "sgd"
parameters.loss = "crossentropyloss"
parameters.data_points = 130
parameters.train_data_points = 20
parameters.test_data_points = 20
parameters.img_size = (3, 32, 32)
parameters.num_classes = 13
parameters.images_per_class = parameters.data_points // parameters.num_classes

# Shadow model configuration for images
shadow_model_config = DotMap()
shadow_model_config.module_path = "./leakpro/tests/input_handler/image_utils.py"
shadow_model_config.model_class = "ConvNet"
shadow_model_config.storage_path = "./leakpro/tests/tmp/model_handler_output"
shadow_model_config.batch_size = 32
shadow_model_config.epochs = 1
shadow_model_config.optimizer = {"name": "sgd", "lr": 0.001}
shadow_model_config.loss = {"name": "crossentropyloss"}

# Lira parameters
lira_params = DotMap()
lira_params.training_data_fraction = 0.1
lira_params.num_shadow_models = 3
lira_params.online = False
lira_params.fixed_variance = True