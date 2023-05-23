import torch
import torch.nn as nn
import torchsummary
from torch.optim import Adam
from tqdm import tqdm

from meta_learning.meta_resnet import MetaResnet
from meta_learning.meta_task import TestTask
from meta_learning.meta_trainer import TransformerResnetMAML
from meta_learning.resnet_architecture import resnet34

trainer = TransformerResnetMAML(resnet_architecture=resnet34)
test_task = TestTask()
optim = Adam(params=trainer.get_optim_parameters(),lr=0.0001)
for epoch in tqdm(range(10)):
    loss, log = trainer.train_one_epoch_for_loss([test_task], 0.001)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(log)