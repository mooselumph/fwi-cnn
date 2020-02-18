# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:54:37 2020

@author: mooselumph
"""

from forward.simple_model import SimpleLayerDataset, SimpleLayerModel
import numpy as np
import matplotlib.pyplot as plt

m = SimpleLayerModel([],[],source_pos = [75],detector_pos=np.arange(0,2500,10),
                     sample_period=0.01,duration=[],pulse_width=0.02)

d = SimpleLayerDataset(m,n_samples=100,interval=5,thickness=100,speed=(1000,3000))

a,s = d.__iter__()

plt.figure(figsize=(8,8))

plt.imshow(a[:,:,0],cmap='gray',aspect='auto')

plt.figure()

plt.plot(s)