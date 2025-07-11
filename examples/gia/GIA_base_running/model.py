"""ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027

This implementation is adapted from GradAttack. Original source: https://github.com/Princeton-SysML/GradAttack.
"""
from torch import nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_factor=1):
        super(ResNet, self).__init__()
        self.width_factor = width_factor
        base_width = 64 * width_factor

        self.in_planes = base_width

        self.conv1 = conv3x3(3, base_width)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(block, 64 * width_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * width_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width_factor, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width_factor, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * width_factor * block.expansion, num_classes)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        norm_layer = nn.BatchNorm2d
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride,
                            downsample, groups=1, base_width=64, dilation=1, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes,
                                groups=1, base_width=64, dilation=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
