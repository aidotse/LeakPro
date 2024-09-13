from dotmap import DotMap

STORAGE_PATH = "./leakpro/tests/tmp"

# User input handler for images


def get_image_handler_config():
    parameters = DotMap()
    parameters.target_folder = "./leakpro/tests/tmp/"
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
    return parameters

def get_audit_config():
    #audit configuration
    audit_config = DotMap()
    audit_config.output_dir = STORAGE_PATH
    audit_config.attack_type = "mia"
    # Lira parameters
    audit_config.attack_list.lira.training_data_fraction = 0.1
    audit_config.attack_list.lira.num_shadow_models = 3
    audit_config.attack_list.lira.online = False
    audit_config.attack_list.lira.fixed_variance = True
    return audit_config



# Shadow model configuration for images
def get_shadow_model_config():
    shadow_model_config = DotMap()
    shadow_model_config.module_path = "./leakpro/tests/input_handler/image_utils.py"
    shadow_model_config.model_class = "ConvNet"
    shadow_model_config.batch_size = 32
    shadow_model_config.epochs = 1
    shadow_model_config.optimizer = {"name": "sgd", "lr": 0.001}
    shadow_model_config.loss = {"name": "crossentropyloss"}
    return shadow_model_config


