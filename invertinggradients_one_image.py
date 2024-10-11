"""Geiping on a single image."""
from examples.gia.gia_cifar10_invertinggradients.model import ResNet, ConvNet
from torchvision.models.resnet import BasicBlock

from examples.gia.gia_cifar10_invertinggradients.cifar import get_cifar10_tensor
from leakpro.fl_utils.gia_train import train
from leakpro.run import run_geiping

# will move this to examples folder after package is working
if __name__ == "__main__":
    model = ConvNet(10)
    # model = ResNet(BasicBlock, [5, 5, 5], num_classes=10, base_width=16 * 10)
    client_dataloader, data_mean, data_std = get_cifar10_tensor(num_images=1, batch_size=1, num_workers=2)
    train_fn = train
    configs = {"at_iterations": 1000}
    result = run_geiping(model, client_dataloader, train_fn, data_mean, data_std, configs)
