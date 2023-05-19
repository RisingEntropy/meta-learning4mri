from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaBase(nn.Module):
    @abstractmethod
    def arrange_parameter(self, weight):
        raise NotImplemented("")
    @abstractmethod
    def report_parameter(self):
        raise NotImplemented("")


class MetaConv3d(MetaBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0][0:self.layer.weight.data.numel()].view(self.layer.weight.data.shape)
        self.layer.bias.data = weight[0][self.layer.weight.data.numel():].view(self.layer.bias.data.shape)

    def report_parameter(self):
        return (self.layer.weight.data.numel()+self.layer.bias.data.numel(),)

    def forward(self, x):
        return self.layer(x)


class MetaBatchNorm3d(MetaBase):
    def __init__(self, input):
        super().__init__()
        self.layer = nn.BatchNorm3d(input)

    def arrange_parameter(self, weight):
        self.layer.weight.data = weight[0][0:self.layer.weight.data.numel()].view(self.layer.weight.data.shape)
        self.layer.bias.data = weight[0][self.layer.weight.data.numel():].view(self.layer.weight.data.shape)

    def report_parameter(self):
        return (self.layer.weight.data.numel()+self.layer.bias.data.numel(),)

    def forward(self, x):
        return self.layer(x)


def conv_layer(in_channels, out_channels):
    return MetaConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)


def down_conv_layer(in_channels, out_channels):
    return MetaConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)


class MetaResBlock(MetaBase):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            conv_layer(in_channels=channels, out_channels=channels),
            MetaBatchNorm3d(channels),
            nn.ReLU(),
            conv_layer(in_channels=channels, out_channels=channels),
            MetaBatchNorm3d(channels)]
        )

    def arrange_parameter(self, weight):
        self.layers[0].arrange_parameter(weight[0])
        self.layers[1].arrange_parameter(weight[1])
        self.layers[3].arrange_parameter(weight[2])
        self.layers[4].arrange_parameter(weight[3])

    def report_parameter(self):
        return (self.layers[0].report_parameter(), self.layers[1].report_parameter(),
                self.layers[3].report_parameter(), self.layers[4].report_parameter())

    def forward(self, x):
        residual = x
        for model in self.layers:
            x = model(x)
        return F.relu(residual + x)


class MetaResnet(MetaBase):
    def __init__(self, net_architecture):
        super().__init__()
        layers = []
        for component in net_architecture:
            if component[0] == "conv7":
                layers.append(
                    nn.Conv3d(in_channels=component[1]["in_channels"], out_channels=component[1]["out_channels"],
                              kernel_size=7, stride=(1, 2, 2), padding=3))
            elif component[0] == "resblock":
                layers.extend([MetaResBlock(component[1]["channels"]) for _ in range(component[1]["repeat"])])
            elif component[0] == "down_conv":
                layers.append(
                    down_conv_layer(in_channels=component[1]["in_channels"], out_channels=component[1]["out_channels"]))
            elif component[0] == "max_pool":
                layers.append(nn.MaxPool3d(kernel_size=component[1]["kernel_size"], stride=component[1]["stride"],
                                           padding=component[1]["padding"]))
            elif component[0] == "flatten":
                layers.append(nn.Flatten())
            elif component[0] == "adaptive_max_pool":
                layers.append(nn.AdaptiveAvgPool3d(output_size=component[1]["output_size"]))
            elif component[0] == "FC":
                layers.append(nn.Linear(in_features=component[1]["in"], out_features=component[1]["out"]))
            elif component[0] == "dropout":
                layers.append(nn.Dropout(p=component[1]["p"]))
        self.layers = nn.ModuleList(layers)
        self.architecture = net_architecture

    def arrange_parameter(self, weight):
        idx = 0
        for model in self.layers:
            if isinstance(model, MetaBatchNorm3d):
                model.arrange_parameter(weight[idx])
            if isinstance(model, MetaConv3d):
                model.arrange_parameter(weight[idx])
            if isinstance(model, MetaResBlock):
                model.arrange_parameter(weight[idx])
            idx += 1
    def report_parameter(self):
        paras = []
        for model in self.layers:
            if isinstance(model, MetaBatchNorm3d):
                paras.append(model.report_parameter())
            if isinstance(model, MetaConv3d):
                paras.append(model.report_parameter())
            if isinstance(model, MetaResBlock):
                paras.append(model.report_parameter())
        return tuple(paras)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x
