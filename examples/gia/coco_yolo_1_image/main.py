"""Hyperparameter tuning with optuna on evaluating."""
import time

import torch
from coco import get_coco_detection_loader
from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaImageYoloExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import trainyolo
from leakpro.run import run_gia_attack
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
from model import ResNet, YOLOResNet18, yolo_v8_n, ComputeLoss, yolo_v8_n_class
from torch import cuda, device
from torchvision.models.resnet import BasicBlock
from traineval import test_eval, test_train
import torchvision

if __name__ == "__main__":
    seed_everything(1234)
    model = yolo_v8_n()
    # model= yolo_v8_n_class()
    # model = torchvision.models.resnet18(weights=None)
    # model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10)
    # model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    # model = YOLOResNet18()
    # attack setup
    configs = InvertingConfig()
    configs.optimizer = MetaSGD(lr=0.1)
    configs.top10norms = False
    configs.median_pooling = False
    configs.at_iterations = 25000
    configs.tv_reg = 1e-07
    configs.attack_lr = 10.0
    configs.data_extension = GiaImageYoloExtension()
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    # need to put model on cuda before initializing the loss function 118287
    client_loader, data_mean, data_std = get_coco_detection_loader(start_idx=98000, num_images=1, batch_size=1, aug=False)#, img_size=224)
    # train_loader, data_mean, data_std = get_coco_detection_loader(start_idx=0, num_images=96000, batch_size=32)
    # test_train(model, train_loader, client_loader)
    # test_eval(model, client_loader)

    # 2. Load the saved EMA module
    ckpt = torch.load('weights/best.pt', map_location='cpu', weights_only=False)
    ema_module = ckpt['model']            # nn.Module in .half()

    # 3. Copy weights into your fresh 
    model.load_state_dict(ema_module.state_dict())
    model.to(gpu_or_cpu)
    configs.criterion = ComputeLoss(model)
    try:
        model.head.simulate_train_on_eval = True
    except Exception as e:
        pass
    attack_object = InvertingGradients(model, client_loader, data_mean, data_std, configs=configs, train_fn=trainyolo)
    optuna_config = OptunaConfig()
    optuna_config.n_trials = 300
    attack_object.run_with_optuna(optuna_config=optuna_config)
    # run_gia_attack(attack_object, "TestInverting")
