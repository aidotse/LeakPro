"""Hyperparameter tuning with optuna on evaluating."""
from coco import get_coco_detection_loader
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_inverting
from model import yolo_v8_n, ComputeLoss

if __name__ == "__main__":
    model = yolo_v8_n()
    client_loader, data_mean, data_std = get_coco_detection_loader()
    train_fn = train
    # baseline config
    configs = InvertingConfig()
    configs.at_iterations = 50
    configs.median_pooling = True
    configs.top10norms = True
    configs.tv_reg = 1.0e-07
    configs.criterion = ComputeLoss(model)
    result = run_inverting(model, client_loader, train_fn, data_mean, data_std, configs)
    

