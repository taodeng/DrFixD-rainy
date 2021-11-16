# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:42:03 2019

@author: th
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import math

class late_fusion(nn.Module):
    def __init__(self):
        super(late_fusion, self).__init__()
        self.fusion = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 8, kernel_size=3, padding = 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 1, kernel_size=1, padding = 0)
                                    )
        self.final = nn.Sigmoid()

    def forward(self, f, g):
        fused = torch.cat((f,g), dim = 1)
        fused = self.fusion(fused)
        fused = self.final(fused)
        return fused




if __name__ == '__main__':
    model = late_fusion()
    tensor1 = Variable(torch.randn(10,1,14,14))
    tensor2 = Variable(torch.randn(10,1,224,224))
    output = model(tensor1, tensor2)
    #print(output.size())