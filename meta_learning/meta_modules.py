import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class MetaBase(nn.Module):
    @abstractmethod
    def arrange_parameter(self, weight):
        raise NotImplemented("")

    @abstractmethod
    def allocate_parameter(self):
        raise NotImplemented("")


class MetaConv3d(MetaBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bias = bias

    def arrange_parameter(self, weight):
        if self.bias:
            self.layer.weight.data = weight[0].view(self.layer.weight.data.shape)
            self.layer.bias.data = weight[1].view(self.layer.bias.data.shape)
        else:
            self.layer.weight.data = weight.view(self.layer.weight.data.shape)

    def allocate_parameter(self):
        if self.bias:
            return {"total": self.layer.weight.data.numel(), "type": "conv"}, {"total": self.layer.bias.data.numel(),
                                                                               "type": "bias"}
        else:
            return {"total": self.layer.weight.data.numel(), "type": "conv"}

    def forward(self, x):
        return self.layer(x)


class MetaBatchNorm3d(MetaBase):
    def __init__(self, input):
        super().__init__()
        self.layer = nn.BatchNorm3d(input)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0].view(self.layer.weight.data.shape)
        self.layer.bias.data = weight[1].view(self.layer.weight.data.shape)

    def allocate_parameter(self):
        return {"total": self.layer.weight.data.numel(), "type": "bn"}, {"total": self.layer.bias.data.numel(),
                                                                         "type": "bias"}

    def forward(self, x):
        return self.layer(x)


class MetaLinear(MetaBase):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=out_features)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0].view(self.layer.weight.data.shape)
        self.layer.bias.data = weight[1].view(self.layer.bias.data.shape)

    def allocate_parameter(self):
        return {"total": self.layer.weight.data.numel(), "type": "fc"}, {"total": self.layer.bias.data.numel(),
                                                                         "type": "bias"}
