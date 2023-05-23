from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Task:
    @abstractmethod
    def tune_once(self, net: nn.Module):
        """

        :param net:
        :return: the loss
        """
        raise NotImplemented("")

    @abstractmethod
    def get_feedback(self):
        """
        report feedback, for example, the accuracy on validate set on classification tasks.
        :return:
        """
        raise NotImplemented()

    @abstractmethod
    def meta_learner_input(self, num_layers):
        raise NotImplemented()


class TestTask(Task):
    def tune_once(self, net: nn.Module):
        input = torch.rand(3, 3, 40, 224, 224)
        gt = torch.as_tensor([0, 1, 2])
        return F.cross_entropy(net(input), gt)

    def get_feedback(self):
        return "ababa"

    def meta_learner_input(self, num_layers):
        return torch.arange(0, num_layers, num_layers)
