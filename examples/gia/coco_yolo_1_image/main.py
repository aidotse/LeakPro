"""Hyperparameter tuning with optuna on evaluating."""
import time
from coco import get_coco_detection_loader
from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaImageYoloExtension
from leakpro.fl_utils.gia_train import trainyolo
from leakpro.run import run_gia_attack
from model import yolo_v8_n, ComputeLoss
from torch import cuda, device
from traineval import test_eval, test_train

if __name__ == "__main__":
    model = yolo_v8_n()
    pre_train_loader, data_mean, data_std = get_coco_detection_loader(start_idx=150, num_images=1, batch_size=1, aug=False)
    # TEST WITH PRE TRAINING ALSO USING TRAINVAL MEtHODS.
    
    # attack setup
    configs = InvertingConfig()
    configs.top10norms = False
    configs.median_pooling = False
    configs.at_iterations = 10000
    configs.tv_reg = 1.0e-06
    configs.attack_lr = 0.1
    configs.data_extension = GiaImageYoloExtension()
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    # need to put model on cuda before initializing the loss function
    client_loader, _, _ = get_coco_detection_loader(start_idx=65000, num_images=1, batch_size=1)
    model.to(gpu_or_cpu)
    configs.criterion = ComputeLoss(model)
    model.head.simulate_train_on_eval = True
    attack_object = InvertingGradients(model, client_loader, data_mean, data_std, configs=configs, train_fn=trainyolo)
    run_gia_attack(attack_object, "TestInverting")