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
    if isinstance(node, int):
        return node
    nodes = []
    for item in node:
        if isinstance(item, int):
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
    if isinstance(nodes, int):
        return para_list[index], index + 1
    para_tree = []
    for node in nodes:
        if isinstance(node, int):
            para_tree.append(para_list[index])
            index += 1
        elif isinstance(node, tuple) or isinstance(node, list):
            to_add, index = build_parameter_tree(node, para_list, index)
            para_tree.append(to_add)
    return para_tree, index


class MetaLearningBase:
    @abstractmethod
    def net_state_dicts(self):
        raise NotImplemented("")

    @abstractmethod
    def load_state_dicts(self, state_dict):
        raise NotImplemented("")

    @abstractmethod
    def train_one_epoch_for_loss(self, tasks: list or tuple, sub_lr):
        raise NotImplemented("")


class TransformerResnetMAML(MetaLearningBase):
    def __init__(self, resnet_architecture, feature_per_layer, model_dim):
        super().__init__()
        self.resnet = MetaResnet(net_architecture=resnet_architecture)
        self.para_allocate_tree = self.resnet.allocate_parameter()
        self.sub_net_layers = leaf_node_counter(self.para_allocate_tree)
        self.transformer = MetaEncoder(subnet_layers=len(self.sub_net_layers), features_per_layer=feature_per_layer,
                                       model_dim=model_dim, ffn_dim=4 * model_dim)
        self.intermediate_nets = nn.ModuleList()
        for out_feature in self.sub_net_layers:
            self.intermediate_nets.append(nn.Linear(in_features=model_dim, out_features=out_feature))

    def net_state_dicts(self):
        return {
            "transformer": self.transformer.state_dict(),
            "intermediate": self.intermediate_nets.state_dict(),
        }

    def load_state_dicts(self, state_dict):
        assert "transformer" in state_dict and "intermediate" in state_dict
        self.transformer.load_state_dict(state_dict["transformer"])
        self.intermediate_nets.load_state_dict(state_dict["intermediate"])

    def train_one_epoch_for_loss(self, tasks: list or tuple, sub_lr):
        """

        :param tasks:
        :return: the loss for optimizer to optimize and intermediate report ( in form of a list)
        """
        for task in tasks:
            if not isinstance(task, Task):
                raise ValueError("Elements in the tasks list mush be sub-classes of meta_learning.meta_task.Task")
        paras = []
        transformer_out = self.transformer()
        for i in range(transformer_out.shape[0]):
            paras.append(self.intermediate_nets[i](transformer_out[i]))
        # fill resnet with parameters from the transformer
        self.resnet.arrange_parameter(build_parameter_tree(self.para_allocate_tree, paras))
        sub_net_loss = 0
        report = []
        for task in tasks:
            sub_net_loss += task.tune_once(self.resnet)
            report.append(task.get_feedback())
        sub_net_loss /= len(tasks)
        intermediate_grad = torch.autograd.grad(sub_net_loss, self.resnet.parameters())
        intermediate_weight = list(
            map(lambda p: p[1] - sub_lr * p[0], zip(intermediate_grad, self.resnet.parameters())))
        # fill resnet with optim-once parameter
        self.resnet.arrange_parameter(build_parameter_tree(self.para_allocate_tree, intermediate_weight))
        final_loss = 0
        for task in tasks:
            final_loss += task.tune_once(self.resnet)
        final_loss /= len(tasks)
        return final_loss, report
