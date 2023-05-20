import torch.nn as nn
from abc import abstractmethod


class MetaBase(nn.Module):
    @abstractmethod
    def arrange_parameter(self, weight):
        raise NotImplemented("")

    @abstractmethod
    def allocate_parameter(self):
        raise NotImplemented("")


class MetaConv3d(MetaBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0].view(self.layer.weight.data.shape)

    def allocate_parameter(self):
        return {"total": self.layer.weight.data.numel(), "type": "conv"}
        return (self.layer.weight.data.numel())

    def forward(self, x):
        return self.layer(x)


class MetaBatchNorm3d(MetaBase):
    def __init__(self, input):
        super().__init__()
        self.layer = nn.BatchNorm3d(input)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0][0:self.layer.weight.data.numel()].view(self.layer.weight.data.shape)
        self.layer.bias.data = weight[0][self.layer.weight.data.numel():].view(self.layer.weight.data.shape)

    def allocate_parameter(self):
        return {"total": self.layer.weight.data.numel() + self.layer.bias.data.numel(), "type": "bn"}

    def forward(self, x):
        return self.layer(x)


class MetaLinear(MetaBase):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=out_features)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0][0:self.layer.weight.data.numel()].view(self.layer.weight.data.shape)
        self.layer.bias.data = weight[0][self.layer.weight.data.numel():].view(self.layer.weight.data.shape)

    def allocate_parameter(self):
        return {"total": self.layer.weight.data.numel() + self.layer.bias.data.numel(), "type": "fc"}
