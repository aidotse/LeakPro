"""Hyperparameter tuning with optuna on evaluating."""
from collections import OrderedDict
import time
from coco import get_coco_detection_loader
from leakpro.attacks.gia_attacks.huang import Huang, HuangConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaImageDetectionExtension
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_gia_attack
from model import yolo_v8_n, ComputeLoss
from torch import cuda, device
from torch.optim.sgd import SGD
from torch.autograd import grad
from traineval import test_eval, test_train

if __name__ == "__main__":
    model = yolo_v8_n()
    client_loader, data_mean, data_std = get_coco_detection_loader(start_idx=0, num_images=64, batch_size=32)
    train_loader, data_mean, data_std = get_coco_detection_loader(start_idx=1000, num_images=6400, batch_size=32)
    test_train(model, train_loader, client_loader)
    # map50, meanap = test_eval(model, client_loader)
    # time.sleep(10000)
    # baseline config
    configs = InvertingConfig()
    configs.data_extension = GiaImageDetectionExtension()
    configs.top10norms = False
    configs.median_pooling = True
    configs.at_iterations = 32000
    configs.tv_reg = 1.0e-06
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    configs.criterion = ComputeLoss(model)
    attack_object = InvertingGradients(model, client_loader, train, data_mean, data_std, configs)
    run_gia_attack(attack_object, "TestInvertingOneR")
    # attack_object.run_with_optuna()