from abc import abstractmethod

import torch
import torch.nn as nn

from meta_learning.meta_resnet import MetaResnet
from meta_learning.meta_task import Task
from meta_learning.meta_transformer import MetaEncoder


def leaf_node_counter(node):
    """

    :param node:
    :return:
    """
    if isinstance(node, dict):
        return node["total"]
    nodes = []
    for item in node:
        if isinstance(item, dict):
            nodes.append(item)
        elif isinstance(item, tuple) or isinstance(item, list):
            nodes.extend(leaf_node_counter(item))  # recursive expansion
    return nodes


def build_parameter_tree(nodes, para_list, index=0):
    """

    :param nodes:
    :param para_list:
    :param index:
    :return:
    """
    if isinstance(nodes, dict):
        return para_list[index], index + 1
    para_tree = []
    for node in nodes:
        if isinstance(node, dict):
            para_tree.append(para_list[index])
            index += 1
        elif isinstance(node, tuple) or isinstance(node, list):
            to_add, index = build_parameter_tree(node, para_list, index)
            para_tree.append(to_add)
            index = index
    return para_tree, index


class MetaLearningBase:
    @abstractmethod
    def net_state_dicts(self):
        raise NotImplemented("")

    @abstractmethod
    def load_state_dicts(self, state_dict):
        raise NotImplemented("")

    @abstractmethod
    def get_optim_parameters(self):
        raise NotImplemented("")

    @abstractmethod
    def train_one_epoch_for_loss(self, tasks: list or tuple, sub_lr):
        raise NotImplemented("")


class TransformerResnetMAML(MetaLearningBase):
    def __init__(self, resnet_architecture, kernel_size=3, default_channel=256):
        """

        :param resnet_architecture: The architecture of expected resnet
        :param total_feature: How many features you expect to learn in the transformer
        :param kernel_size: Convolution kernel size
        :param default_channel: default kernel parameter channel. The output of the transformer is treat as kernel_size*kernel_size*kernel_size*default_channel. Later a 1x1 convolution is used to adjust the shape to fit the network convolution kernel size
        """
        super().__init__()
        self.resnet = MetaResnet(net_architecture=resnet_architecture)
        self.para_allocate_tree = self.resnet.allocate_parameter()
        self.sub_net_layers = leaf_node_counter(self.para_allocate_tree)
        self.model_dim = kernel_size * kernel_size * kernel_size * default_channel
        self.transformer = MetaEncoder(subnet_layers=len(self.sub_net_layers),
                                       total_feature=len(self.sub_net_layers),
                                       model_dim=self.model_dim, ffn_dim=4 * self.model_dim)
        self.intermediate_nets = nn.ModuleList()
        for out_feature in self.sub_net_layers:
            assert isinstance(out_feature, dict)
            if out_feature["type"] == "conv":
                self.intermediate_nets.append(nn.Conv3d(in_channels=default_channel,
                                                        out_channels=out_feature["in_channels"] * out_feature[
                                                            "out_channels"], kernel_size=1, padding=0, stride=1))
            elif out_feature["type"] == "fc" or out_feature["type"] == "bias" or out_feature["type"] == "bn":
                self.intermediate_nets.append(nn.Linear(in_features=self.model_dim, out_features=out_feature["total"]))

    def net_state_dicts(self):
        return {
            "transformer": self.transformer.state_dict(),
            "intermediate": self.intermediate_nets.state_dict(),
        }

    def get_optim_parameters(self):
        return list(self.transformer.parameters()) + list(self.intermediate_nets.parameters())

    def load_state_dicts(self, state_dict):
        assert "transformer" in state_dict and "intermediate" in state_dict
        self.transformer.load_state_dict(state_dict["transformer"])
        self.intermediate_nets.load_state_dict(state_dict["intermediate"])

    def train_one_epoch_for_loss(self, tasks: list or tuple, sub_lr):
        """

        :param tasks:
        :return: the loss for optimizer to optimize and intermediate report ( in form of a list)
        """
        final_loss = 0
        report = []
        for task in tasks:
            if not isinstance(task, Task):
                raise ValueError("Elements in the tasks list mush be sub-classes of meta_learning.meta_task.Task")
            paras = []
            transformer_out = self.transformer(
                task.meta_learner_input(len(self.sub_net_layers)).unsqueeze(dim=0))  # suppose the transformer
            # input is `not` batched
            for i in range(transformer_out.shape[1]):
                if self.sub_net_layers[i]["type"] == "conv":
                    paras.append(self.intermediate_nets[i](transformer_out[0][i].view(-1, 3, 3, 3)))
                else:
                    paras.append(self.intermediate_nets[i](transformer_out[0][i]))
            self.resnet.arrange_parameter(build_parameter_tree(self.para_allocate_tree, paras)[0])
            loss = task.tune_once(self.resnet)
            intermediate_grad = torch.autograd.grad(loss, self.resnet.parameters())
            intermediate_weight = list(
                map(lambda p: p[1] - sub_lr * p[0], zip(intermediate_grad, self.resnet.parameters())))
            self.resnet.arrange_parameter(build_parameter_tree(self.para_allocate_tree, intermediate_weight)[0])
            final_loss += task.tune_once(self.resnet)
            report.append(task.get_feedback())
        final_loss /= len(tasks)
        return final_loss, report
