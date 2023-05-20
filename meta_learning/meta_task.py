from abc import abstractmethod
import torch.nn as nn


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
        raise NotImplemented

# class ClassificationTask(Task):
#     def