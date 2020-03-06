#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:05:02 2020

@author: bertran
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, pool_size=2, kernel_size=3, dim=1):
        super().__init__()
                            
        if dim == 1:
            pool_size = (pool_size,1)
            kernel_size = (kernel_size,1)
            padding = (1,0)
        else:
            assert dim == 2, "dim must be 1 or 2"
            padding = 1
        
        self.down = nn.Sequential(
                        
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels, kernel_size=scale_factor, stride=scale_factor)

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
 
        return self.double_conv(self.up(x))
    

class VanillaNet(nn.Module):
    
    def __init__(self, n_channels):
        super(VanillaNet, self).__init__()
        
        self.n_channels = n_channels
                
        # 512 x 32 x 32
        # 128 x 32 x 64  / 4,1
        # 32  x 32 x 128 / 4,1
        # 8   x 8  x 256 / 4,4
        # 1   x 1  x 512 / 8,8

        self.down = nn.Sequential(
                DoubleConv(n_channels, 32), # N x 1 x 512 x 32
                Down(32,64),
                Down(64, 128),
                Down(128, 256, dim=2),
                nn.Conv2d(256, 512, kernel_size=(8,8), padding=0), # N x 512 x 1 x 1
            )
        
        # N x 256 x 8
        # N x 128 x 16
        # N x 64  x 32
        # N x 32  x 64
        # N x 16  x 128
        # N x 1   x 256
        
        self.up =  nn.Sequential(
                Up(512,256,scale_factor=8),     # N x 256 x 8
                Up(256,128),                    
                Up(128,64),                     
                Up(64,32),                      
                Up(32,16),
                Up(16,1), # N x 1   x 256
            )

    def forward(self, x):
        
        x = self.down(x) # 
        
        x = torch.flatten(x,start_dim=2) # N x 512 x 1
        
        x = self.up(x)
        
        return x