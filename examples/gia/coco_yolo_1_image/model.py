"""Re-implemented code to fit new standard COCO structure and GIA, from repository: https://github.com/jahongir7174/YOLOv8-pt, reference to Ultralytics"""
import math
import time
import torch
from torch.nn.functional import cross_entropy, one_hot
from torch.nn.utils.rnn import pad_sequence
import yaml
from torch import nn
import os
from typing import Optional

import torch
import torchvision
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from leakpro.utils.import_helper import Self

def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))
    
class YOLOClassifier(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        # Backbone and neck
        self.net = ResDarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        # Classification head: use the deepest FPN feature (width[5] channels)
        c5 = width[5]
        self.global_avg = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),              # -> (B, c5)
            torch.nn.Linear(c5, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes)  # -> (B, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.net(x)
        p3, p4, p5 = self.fpn(features)

        # Pool and classify using the deepest scale
        z = self.global_avg(p5)       # (B, c5, 1, 1)
        logits = self.classifier(z)   # (B, num_classes)
        return logits

    def fuse(self):
        # Fuse Conv+Norm for faster inference
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8_n_class(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLOClassifier(width, depth, num_classes)

def _make_resnet_layer(block, in_planes, out_planes, blocks, stride=1):
    """
    block: the block class (BasicBlock)
    in_planes: number of channels coming in
    out_planes: number of channels for this stage
    blocks: how many blocks to stack (ResNet-18 uses 2)
    stride: stride for the *first* block in this stage
    """
    downsample = None
    # if we need to change #channels or spatial resolution, apply a 1×1 conv
    if stride != 1 or in_planes != out_planes:
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_planes),
        )

    layers = []
    # first block: may downsample / change channels
    layers.append(block(in_planes, out_planes, stride=stride, downsample=downsample))
    # remaining blocks: keep same size
    for _ in range(1, blocks):
        layers.append(block(out_planes, out_planes))

    return torch.nn.Sequential(*layers)

class ResNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3,
                                 stride=2,
                                 padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ResDarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.p1 = ResNetStem(in_channels=width[0], out_channels=width[1])
        self.layer1 = _make_resnet_layer(BasicBlock,
                                    in_planes=width[1],
                                    out_planes=width[2],
                                    blocks=2,
                                         stride=1)

        # Stage 2: 64→128 channels, downsample once
        self.layer2 = _make_resnet_layer(BasicBlock,
                                         in_planes=width[2],
                                         out_planes=width[3],
                                         blocks=2,
                                         stride=2)

        # Stage 3: 128→256 channels, downsample once
        self.layer3 = _make_resnet_layer(BasicBlock,
                                         in_planes=width[3],
                                         out_planes=width[4],
                                         blocks=2,
                                         stride=2)

        # Stage 4: 256→512 channels, downsample once
        self.layer4 = _make_resnet_layer(BasicBlock,
                                         in_planes=width[4],
                                         out_planes=width[5],
                                         blocks=2,
                                         stride=2)

    def forward(self, x):
        x = self.p1(x)          # 7×7 stem
        x = self.layer1(x)      # 64
        p3 = self.layer2(x)      # 128
        p4 = self.layer3(p3)      # 256
        p5 = self.layer4(p4)      # 512
        return p3, p4, p5        

class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6

class ResDarkFPNShallow(nn.Module):
    def __init__(self, width, depth):
        """
        width: list of channel dims, e.g. [C0, C1, C2, C3, C4, C5]
        This version uses only one BasicBlock per stage to reduce depth,
        while preserving input/output shapes.
        """
        super().__init__()
        # upsample by factor 2 (nearest by default)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # Each residual layer now has just one BasicBlock
        self.h1 = _make_resnet_layer(
            BasicBlock,
            in_planes=width[4] + width[5],
            out_planes=width[4],
            blocks=1,
            stride=1
        )

        self.h2 = _make_resnet_layer(
            BasicBlock,
            in_planes=width[3] + width[4],
            out_planes=width[3],
            blocks=1,
            stride=1
        )

        # plain convs for downsampling
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h5 = Conv(width[4], width[4], 3, 2)

        self.h4 = _make_resnet_layer(
            BasicBlock,
            in_planes=width[3] + width[4],
            out_planes=width[4],
            blocks=1,
            stride=1
        )

        self.h6 = _make_resnet_layer(
            BasicBlock,
            in_planes=width[4] + width[5],
            out_planes=width[5],
            blocks=1,
            stride=1
        )

    def forward(self, x):
        # x: tuple of feature maps (p3, p4, p5)
        p3, p4, p5 = x

        # top-down path
        h1 = self.h1(torch.cat([self.up(p5), p4], dim=1))
        h2 = self.h2(torch.cat([self.up(h1), p3], dim=1))

        # bottom-up path
        h4 = self.h4(torch.cat([self.h3(h2), h1], dim=1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], dim=1))

        return h2, h4, h6


class ResDarkFPN(nn.Module):
    def __init__(self, width, depth):
        """
        width: list of channel dims, e.g. [C0, C1, C2, C3, C4, C5]
        depth: list of #blocks for each Res-layer, e.g. [n1, n2, n3, n4]
        """
        super().__init__()
        # upsample by factor 2 (nearest by default)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # replace CSPs with ResNet BasicBlock stacks
        # h1: from (C4 + C5) -> C4
        self.h1 = _make_resnet_layer(
            BasicBlock, 
            in_planes=width[4] + width[5], 
            out_planes=width[4], 
            blocks=2, 
            stride=1
        )

        # h2: from (C3 + C4) -> C3
        self.h2 = _make_resnet_layer(
            BasicBlock, 
            in_planes=width[3] + width[4], 
            out_planes=width[3], 
            blocks=2, 
            stride=1
        )

        # keep these as plain convs
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h5 = Conv(width[4], width[4], 3, 2)

        # h4: from (C3 + C4) -> C4
        self.h4 = _make_resnet_layer(
            BasicBlock, 
            in_planes=width[3] + width[4], 
            out_planes=width[4], 
            blocks=2, 
            stride=1
        )

        # h6: from (C4 + C5) -> C5
        self.h6 = _make_resnet_layer(
            BasicBlock, 
            in_planes=width[4] + width[5], 
            out_planes=width[5], 
            blocks=2, 
            stride=1
        )

    def forward(self, x):
        p3, p4, p5 = x

        # top-down path
        h1 = self.h1(torch.cat([self.up(p5), p4], dim=1))
        h2 = self.h2(torch.cat([self.up(h1), p3], dim=1))

        # bottom-up path
        h4 = self.h4(torch.cat([self.h3(h2), h1], dim=1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], dim=1))

        return h2, h4, h6

class HalfResDarkFPN(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # Swap two CSP layers (h1, h4) with ResNet BasicBlock stacks
        self.r1 = _make_resnet_layer(
            BasicBlock,
            in_planes=width[4] + width[5],
            out_planes=width[4],
            blocks=depth[0],
            stride=1
        )
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.r4 = _make_resnet_layer(
            BasicBlock,
            in_planes=width[3] + width[4],
            out_planes=width[4],
            blocks=depth[0],
            stride=1
        )
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        # swapped layer r1 in place of h1
        h1 = self.r1(torch.cat([self.up(p5), p4], dim=1))
        h2 = self.h2(torch.cat([self.up(h1), p3], dim=1))
        # swapped layer r4 in place of h4
        h4 = self.r4(torch.cat([self.h3(h2), h1], dim=1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], dim=1))
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)
        self.simulate_train_on_eval = False

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        if self.training:
            return x
        if self.simulate_train_on_eval:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (256 / s) ** 2)

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = ResDarkNet(width, depth)
        self.fpn = ResDarkFPNShallow(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8_n_basicblock(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)

def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

class ComputeLoss:
    def __init__(self, model):
        super().__init__()
        with open(os.path.join('args.yaml'), errors='ignore') as f:
            params = yaml.safe_load(f)
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device
        self.params = params

        # task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)

    def __call__(self, outputs, targets):
        x = outputs[1] if isinstance(outputs, tuple) else outputs
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)

        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

        # targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 5, device=self.device)
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))

        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # boxes
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)

        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor)

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        # print(
        #     f"YOLO losses → cls {loss_cls.item():.4f}, "
        #     f"box {loss_box.item():.4f}, "
        #     f"dfl {loss_dfl.item():.4f}, "
        #     f"anchors pos {fg_mask.sum().item()}"
        # )
        loss_cls *= self.params['cls']
        loss_box *= self.params['box']
        loss_dfl *= self.params['dfl']
        return loss_cls + loss_box + loss_dfl  # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        """
        Task-aligned One-stage Object Detection assigner
        """
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)
        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))

        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)

        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)
        align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)
        bs, n_boxes, _ = true_bboxes.shape
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        metrics = align_metric * mask_in_gts
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        target_labels = true_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = one_hot(target_labels, self.nc)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
    
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Assuming you already have these from your YOLO codebase:
#   Conv, DFL, make_anchors, Head

class ResNetBackbone(nn.Module):
    """ResNet-18 truncated to return the three feature maps at strides 8, 16, 32."""
    def __init__(self, weights=None):
        super().__init__()
        resnet = resnet18(weights=weights)
        # Stem: conv1→bn1→relu→maxpool
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        # Each layer i doubles the stride (after pool):
        # layer1: stride stays 4,  channels=64
        # layer2: stride→8,    channels=128
        # layer3: stride→16,   channels=256
        # layer4: stride→32,   channels=512
        self.layer1 = resnet.layer1   # → (B, 64,  H/4,  W/4)
        self.layer2 = resnet.layer2   # → (B,128,  H/8,  W/8)
        self.layer3 = resnet.layer3   # → (B,256, H/16, W/16)
        self.layer4 = resnet.layer4   # → (B,512, H/32, W/32)

    def forward(self, x):
        x = self.stem(x)
        _ = self.layer1(x)      # stride=4, 64 channels (we won’t use this one)
        f8  = self.layer2(_ )   # stride=8, 128 ch
        f16 = self.layer3(f8)   # stride=16,256 ch
        f32 = self.layer4(f16)  # stride=32,512 ch
        return [f8, f16, f32]

class YOLOResNet18(nn.Module):
    """Combine ResNet-18 backbone + your YOLO Head."""
    def __init__(self, num_classes=80, weights=None):
        super().__init__()
        # 1) Backbone
        self.backbone = ResNetBackbone(weights=None)
        # 2) YOLO head: filters must match the channels of [f8, f16, f32]
        filters = (128, 256, 512)
        self.head = Head(nc=num_classes, filters=filters)

    def forward(self, x):
        # 1) get the 3 feature maps
        feats = self.backbone(x)
        # 2) run through head
        outputs = self.head(feats)
        return outputs

class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self: Self, block: BasicBlock, layers: list, num_classes: int=10, zero_init_residual: bool=False,  # noqa: C901
                 groups: int=1, base_width: int=64, replace_stride_with_dilation: list=None,
                 norm_layer: Optional[nn.Module]=None, strides: list=[1, 2, 2, 2], pool: str="avg") -> None:  # noqa: B006
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == "avg" else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self: Self, x: torch.Tensor) -> None:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
