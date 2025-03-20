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
    # pre training. This currently gives 0 map50 and meanap, needs to be fixed.
    # cant know if attack works or not before that is fixed.
    # issue likely originates from rescaling images (and bounding boxes coordinates)
    # model architecture is built for 640x640 images, that might also be causing issues. 
    # Some parameters have changed to fit 256x256 images better but might still be causing issues.
    # see top of model.py for original repository.
    model = yolo_v8_n()
    pre_train_loader, data_mean, data_std = get_coco_detection_loader(start_idx=0, num_images=320, batch_size=32)
    train_loader, _, _ = get_coco_detection_loader(start_idx=700, num_images=64000, batch_size=32)
    test_train(model, train_loader, pre_train_loader)
    map50, meanap = test_eval(model, pre_train_loader)
    print(map50, meanap)


    
    # attack setup
    configs = InvertingConfig()
    configs.data_extension = GiaImageDetectionExtension()
    configs.top10norms = False
    configs.median_pooling = True
    configs.at_iterations = 24000
    configs.tv_reg = 1.0e-06
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    # need to put model on cuda before initializing the loss function
    client_loader, _, _ = get_coco_detection_loader(start_idx=65000, num_images=1, batch_size=1)
    model.to(gpu_or_cpu)
    configs.criterion = ComputeLoss(model)
    attack_object = InvertingGradients(model, client_loader, train, data_mean, data_std, configs)
    run_gia_attack(attack_object, "TestInverting")