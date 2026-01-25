import kornia.augmentation as K
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class VGG16(nn.Module):
    def __init__(self, num_classes,  return_feature=True):
        super(VGG16, self).__init__()

        self.init_params = {"num_classes": num_classes,
                            "return_feature": return_feature}

        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        self.return_feature = return_feature  # whether to return feature or not

    def forward(self, x, **kwargs):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        # Important to resolve the issue of returning feature or not
        if self.return_feature:
            return [feature, res]
        else:
            return res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out

    def resize(self, x):
        return K.Resize((64, 64))(x)