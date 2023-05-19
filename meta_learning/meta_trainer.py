import torch
import torch.nn as nn

from meta_learning.resnet import MetaResnet
from meta_learning.resnet_architecture import resnet34


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


def distribute_parameters(nodes, para_list, index):
    """

    :param para_tree:
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
            to_add, index = distribute_parameters(node, para_list, index)
            para_tree.append(to_add)
    return para_tree, index

# net = MetaResnet(resnet34)
# print(len(leaf_node_counter(net.report_parameter())))
