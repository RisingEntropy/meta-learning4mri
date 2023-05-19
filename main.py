import torch
import torch.nn as nn
import torchsummary

from meta_learning.resnet import MetaResblock, ResBlock
from meta_learning.resnet_architecture import resnet34

# net = ResNet(net_architecture=resnet34).cuda()
# torchsummary.summary(net, (3, 40, 224, 224))
# # for model in net.modules():
# #     print(model)
net = ResBlock(10)
net(torch.rand(1,10,3,4,5))