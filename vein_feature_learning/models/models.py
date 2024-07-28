import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
import math


class ResNets(torch.nn.Module):
    def __init__(self, backbone, head_type, num_classes=None, **kwargs):
        super(ResNets, self).__init__()
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            ValueError(f'{backbone} is not supported')

        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        if head_type == "simclr":
            self.head = SingleLayerHead(in_channels=resnet.fc.in_features, projection_size=512)
        elif head_type == "byol":
            self.head = MLPHead(in_channels=resnet.fc.in_features, mlp_hidden_size=512, projection_size=128)
        elif head_type == "cls_norm":
            self.head = NormLinear(resnet.fc.in_features, num_classes)
        elif head_type == "cls":
            self.head = nn.Linear(resnet.fc.in_features, num_classes)
        else:
            ValueError(f'{head_type} not supported.')

    def forward(self, x):
        h = torch.squeeze(self.backbone(x))
        return h, self.head(h)


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size=128):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class SingleLayerHead(nn.Module):
    def __init__(self, in_channels, projection_size=512):
        super(SingleLayerHead, self).__init__()

        # We use a single-layer projection head for simclr, which yields better performance.
        self.net = nn.Sequential(
            nn.Linear(in_channels, projection_size),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.net(x)


class NormLinear(nn.Module):
    def __init__(self, input, output):
        super(NormLinear, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.reset_parameters()

    def forward(self, input):
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        input_normalized = F.normalize(input, p=2, dim=1)
        output = input_normalized.matmul(weight_normalized.t())
        return output

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))