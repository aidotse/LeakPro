"""Gradient inversion attack on DETR with a single COCO image."""
from coco import get_coco_detr_loader
from model import ComputeLoss, detr_resnet50
from torch import cuda, device

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaImageDetrExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train_detr
from leakpro.run import run_gia_attack
from leakpro.utils.seed import seed_everything

if __name__ == "__main__":
    seed_everything(1234)
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")

    model = detr_resnet50(pretrained=True)
    model.eval().to(gpu_or_cpu)

    client_loader, data_mean, data_std = get_coco_detr_loader(num_images=1, img_size=256, start_idx=0, batch_size=1)

    configs = InvertingConfig()
    configs.optimizer = MetaSGD(lr=0.1)
    configs.criterion = ComputeLoss(model)
    configs.data_extension = GiaImageDetrExtension()
    configs.at_iterations = 8000
    configs.tv_reg = 0.02
    configs.attack_lr = 0.1
    configs.median_pooling = True

    attack_object = InvertingGradients(model, client_loader, data_mean, data_std,
                                       train_fn=train_detr, configs=configs)
    result = run_gia_attack(attack_object, experiment_name="detr_gia")
