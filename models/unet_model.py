""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import numpy as np

from models.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class BottleNet(nn.Module):
    
    def __init__(self, n_channels, input_shape, n_speeds):
        super(BottleNet, self).__init__()
        
        self.n_channels = n_channels
                
        # self.bottle = nn.Sequential(
        #         DoubleConv(n_channels, 64),
        #         Down(64,128),
        #         Down(128, 256),
        #         Down(256, 512),
        #         Down(512, 512),
        #     )
        
        self.bottle = nn.Sequential(
                DoubleConv(n_channels, 64),
                Down(64,128),
                Down(128, 256),
                Down(256, 256),
                Down(256, 256),
            )
        
        
        reduced_shape = np.array(input_shape) // 2**4
        n_flattened = np.prod(reduced_shape)*256
        
        n_intermediate = 5
        
        self.fc1 = nn.Linear(n_flattened,n_intermediate*n_speeds)
        self.fc2 = nn.Linear(n_intermediate*n_speeds,n_speeds)
        

    def forward(self, x):
        x = self.bottle(x)
        x = torch.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)