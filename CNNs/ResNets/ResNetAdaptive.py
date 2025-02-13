import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResNetAdaptive(nn.Module): 
    def __init__(self,num_classes, c_architecture=[3, 16, 32, 512]): 
        super(ResNetAdaptive, self).__init__()
        if(len(c_architecture) == 0): 
            return

        in_channels = c_architecture[0]
        self.layers = nn.Sequential(*[
            Res_Block(in_channels, out_channels, stride=2)
            for in_channels, out_channels in zip(c_architecture[:-1], c_architecture[1:])
        ])
        self.fc = nn.Linear(c_architecture[-1], num_classes)
    

    def forward(self, x): 
        out = self.layers(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Res_Block, self).__init__()
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
