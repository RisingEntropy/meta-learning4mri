import torch
import torch.nn as nn
import torchsummary

from meta_learning.meta_resnet import MetaResnet
from meta_learning.meta_transformer import MetaEncoder
from meta_learning.resnet_architecture import resnet34

net = nn.BatchNorm3d(64)
print(net.bias.data.shape)