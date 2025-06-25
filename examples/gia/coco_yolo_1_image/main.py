"""Hyperparameter tuning with optuna on evaluating."""
from coco import get_coco_detection_loader
from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaImageYoloExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import trainyolo
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
from model import yolo_v8_n_basicblock, ComputeLoss
from torch import cuda, device

if __name__ == "__main__":
    seed_everything(1234)
    # This is a modified version of the yolo_v8_nano, where ALL CSP modules have been exchanged for resnet basicblock layers.
    # With these design changes we can get a small amount of information leakage from the model.
    model = yolo_v8_n_basicblock()
    configs = InvertingConfig()
    configs.optimizer = MetaSGD(lr=0.1)
    configs.data_extension = GiaImageYoloExtension()
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    client_loader, data_mean, data_std = get_coco_detection_loader(start_idx=108000, num_images=1, batch_size=1, aug=False)
    model.to(gpu_or_cpu)
    configs.criterion = ComputeLoss(model)
    # To enable training in eval model which is needed for the inverting attack
    model.head.simulate_train_on_eval = True
    # Trial data for optuna to find vulnerable data point.
    trial_data = []
    for i in range(5):
        client_loader, _, _ = get_coco_detection_loader(start_idx=108000+i, num_images=1, batch_size=1, aug=False)
        trial_data.append(client_loader)
    # Initialize attack object and run with optuna.
    attack_object = InvertingGradients(model, client_loader, data_mean, data_std, configs=configs, train_fn=trainyolo,optuna_trial_data=trial_data)
    optuna_config = OptunaConfig()
    optuna_config.n_trials = 100
    attack_object.run_with_optuna(optuna_config=optuna_config)
