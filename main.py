import torch
import torch.nn as nn

from meta_learning.meta_resnet import MetaResnet
from meta_learning.resnet_architecture import resnet34





net = MetaResnet(net_architecture=resnet34)
paras = []
for module in net.parameters():
    paras.append(module)
para_allocate_tree = net.allocate_parameter()
para_tree, _ = build_parameter_tree(para_allocate_tree, paras)
net.arrange_parameter(para_tree)
