#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:53:59 2020

@author: bertran
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from train_callbacks import train_net, TBWriter, TrainState, Callback

from models.vanilla_cnn import VanillaNet
from forward.simple_model import SimpleLayerDataset, SimpleLayerProblem, SimpleLayerModel


# Create Model and Dataset

model = SimpleLayerModel([],[],source_pos = [75],detector_pos=np.arange(0,3200,100),
                     sample_period=0.01,duration=512*0.01,pulse_width=0.02)

problem = SimpleLayerProblem(model,n_samples=256,interval=10,thickness=500,speed=(200,500))

train_dataset = SimpleLayerDataset(problem,n_samples=100000)
val_dataset = SimpleLayerDataset(problem,n_samples=10)

# Intialize Net
net = VanillaNet(1)

# Set up

class CyclicScheduler(Callback):
    def __init__(self,optimizer,base_lr=0.001,max_lr=0.1):
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr)
    def on_batch_end(self,s):
        self.scheduler.step()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

tbw = TBWriter()
sched = CyclicScheduler(optimizer)


# Train

state = TrainState(
    net,
    train_dataset,
    val_dataset,
    optimizer=optimizer,
    batch_size = 50,
    )

try:
    train_net(
        state,
        [tbw,sched]
             )

except KeyboardInterrupt:
    torch.save(net.state_dict(), 'saved/INTERRUPTED.pth')
    print('Saved interrupt')
