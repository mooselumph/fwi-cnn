# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:54:37 2020

@author: mooselumph
"""

from forward.simple_model import SimpleLayerDataset, SimpleLayerProblem, SimpleLayerModel
import numpy as np
import matplotlib.pyplot as plt

from utils.data_vis import plot_speeds

m = SimpleLayerModel([],[],source_pos = [75],detector_pos=np.arange(0,3200,100),
                     sample_period=0.01,duration=512*0.01,pulse_width=0.02)

p = SimpleLayerProblem(m,n_samples=256,interval=10,thickness=500,speed=(200,500))

d = SimpleLayerDataset(p,n_samples=2)

for batch in d:
    a = batch['amplitudes']
    s = batch['speeds']
    
plt.figure(figsize=(8,8))

plt.imshow(a[0,:,:],cmap='gray',aspect='auto')

plt.figure()

plt.plot(s)

img = plot_speeds(s,s/2)

nimg = img.detach().cpu().numpy().squeeze().transpose(1,2,0)

plt.figure()
plt.imshow(nimg)