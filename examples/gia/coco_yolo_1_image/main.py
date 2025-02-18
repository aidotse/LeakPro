"""Hyperparameter tuning with optuna on evaluating."""
from collections import OrderedDict
import time
from coco import get_coco_detection_loader
from leakpro.attacks.gia_attacks.huang import HuangConfig
from leakpro.fl_utils.data_utils import GiaImageDetectionExtension
from leakpro.fl_utils.gia_train import train2
from leakpro.run import run_inverting, run_huang
from model import yolo_v8_n, ComputeLoss
from torch import cuda, device
from torch.optim.sgd import SGD
from torch.autograd import grad

if __name__ == "__main__":
    model = yolo_v8_n()
    client_loader, data_mean, data_std = get_coco_detection_loader()
    # baseline config
    configs = HuangConfig()
    configs.data_extension = GiaImageDetectionExtension()
    configs.at_iterations = 50
    configs.median_pooling = True
    configs.top10norms = True
    configs.tv_reg = 1.0e-07
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    configs.criterion = ComputeLoss(model)
    ext = GiaImageDetectionExtension()
    # rec, label, loader = ext.get_at_data(client_loader=client_loader)
    # for data, label in client_loader:
    #     print(type(label))
    #     print(len(label))
    # for data,label in loader:
    #     print(type(label))
    #     print(len(label))
    # train2(model, client_loader, configs.optimizer, configs.criterion, 1)
    # train2(model, client_loader, configs.optimizer, configs.criterion, 1)
    # train2(model, client_loader, configs.optimizer, configs.criterion, 1)
    # time.sleep(10000)
    # opt = SGD(model.parameters())
    # print("hej")
    # for inputs, labels in client_loader:
    #     inputs = inputs.to(gpu_or_cpu, non_blocking=True)
    #     outputs = model(inputs)
    #     loss = configs.criterion(outputs, labels).sum()
        
    #     # This does not work..
    #     grads = grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True, create_graph=True, only_inputs=True)
        
        # This works!
        # opt.step()
    results = run_huang(model, client_loader, train2, data_mean, data_std, configs)
    # result = run_inverting(model, client_loader, train2, data_mean, data_std, configs)
