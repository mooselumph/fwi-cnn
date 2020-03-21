#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:33:47 2020

@author: bertran
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

def gen_vmodel(
        nx = 128,
        dx = 5,
        nz = 128,
        dz = 5,
        terrain_var = 50,
        terrain_clength = 15,
        t_avg = 100,
        s_start = 2000,
        s_hi = 300,
        s_low = 100,
        slide_mean = 70,
        fault_width = 10,
        ):
    """
    Generates a random velocity model. c.f. https://arxiv.org/pdf/1811.07875.pdf
    """
    
    n = 2*nx
    
    # this is an NxD matrix, where N is number of items and D its dimensionalites
    X = np.arange(n).reshape((n,1))
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = np.exp(-pairwise_dists ** 2 / terrain_clength ** 2)

    # Generate Random Curve
    pts = terrain_var*np.random.multivariate_normal(np.zeros(n),K)
    
    # Generate Random Spacings
    
    nr_avg = int(nz*dz/t_avg)
    
    t = np.random.exponential(t_avg,2*nr_avg)
    depths = np.cumsum(t)
    
    # Generate Random Speeds
    
    s = np.random.uniform(s_low,s_hi,2*nr_avg)
    speeds = np.cumsum(s)+s_start
    
    # Generate Random Fault Line
    fm = np.random.uniform(-1,1)
    fx = np.random.uniform(0,1)*nx*dx
    
    # Shift Terrain random amount along Fault Direction
    # - For each position in image, check which side of the fault it is on. 
    # - Determine which region shifted pixel is in. 
    # - If within fault width, set to fault speed
    
    x,z = np.meshgrid(np.arange(nx)*dx,(np.arange(nz)-nz/2)*dz)
    
    #x = fm*z + fx
    side = x >= fm*z + fx
    fault = np.abs(x - fm*z - fx) <= fault_width
    
    # dz^2 + (fm*dz)^2 = d^2 : dz = sqrt(d^2/(1+fm^2))
    slide = np.random.normal(slide_mean,slide_mean)*(-1)**(np.random.random()>=0.5)
    slidez = slide/np.sqrt(1+fm**2)
    slidex = fm*slidez
    
    x[side] = (x-slidex)[side]
    z[side] = (z-slidez)[side]
    z += nz/2*dz
    
    i = (x / dx + nx/2).astype('int')
    
    model = np.ones((nx,nz))*speeds[0]
    
    for d,s in zip(depths[:-1],speeds[1:]):
    
        model[z >= pts[i] + d] = s
        
    model[fault] = s_start    
        
    return model

# %%

model = gen_vmodel()

plt.imshow(model)
plt.colorbar()
    