#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:33:47 2020

@author: bertran
"""

import numpy as np
import matplotlib.pyplot as plt

# %%

nx = 128
dx = 5
nz = 128
dz = 5

# %%

from scipy.spatial.distance import pdist, squareform
  # this is an NxD matrix, where N is number of items and D its dimensionalites
n = 2*nx

X = np.arange(n).reshape((n,1))
pairwise_dists = squareform(pdist(X, 'euclidean'))
s = 15
K = np.exp(-pairwise_dists ** 2 / s ** 2)


# %%

# Generate Random Curve
pts = 50*np.random.multivariate_normal(np.zeros(n),K)

# Generate Random Spacings

t_avg = 100
nr_avg = int(nz*dz/t_avg)

t = np.random.exponential(t_avg,2*nr_avg)
depths = np.cumsum(t)

# Generate Random Speeds
s_hi = 300
s_low = 100
s = np.random.uniform(s_low,s_hi,2*nr_avg)
speeds = np.cumsum(s)

# Generate Random Fault Line
fm = np.random.uniform(-1,1)
fx = np.random.uniform(0,1)*nx*dx

# Shift Terrain random amount along Fault Direction
# - For each position in image, check which side of the fault it is on. 
# - Determine which region shifted pixel is in. 
# - If within fault width, set to fault speed

x,z = np.meshgrid(np.arange(nx)*dx,np.arange(nz)*dz)

#x = fm*z + fx
side = x >= fm*z + fx

slide = 30
# dz^2 + (fm*dz)^2 = d^2 : dz = sqrt(d^2/(1-fm^2))
slidez = slide/np.sqrt(1-fm**2)
slidex = fm*slidez

x[side] = (x-slidex)[side]
z[side] = (z-slidez)[side]

i = (x / dx + nx/2).astype('int')

model = np.zeros((nx,nz))

for d,s in zip(depths,speeds):

    model[z >= pts[i] + d] = s
    
plt.imshow(model)

# Compare with https://arxiv.org/pdf/1811.07875.pdf