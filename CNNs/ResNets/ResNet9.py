import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResNet9(nn.Module): 
    class Res_Block(nn.Module):
        def __init__(self, in_channel, out_channel, stride=1): 
            self.left = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel)
                )
            self.right_side = nn.Sequential()
            # 1x1 for channel size change if needed
            if stride != 1 or in_channel != out_channel:
                self.right_side = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channel)
                )
        def forward(self, x): 
            out = self.left(x)
            out = out + self.right_side(x)
            out = F.relu(out)
            return out
    
    def __init__(self, channel_arch=[3, 16, 32, 512]): 
        if(len(channel_arch) == 0): 
            # raise error here
            pass
        

    def forwars(): 
        pass
        
