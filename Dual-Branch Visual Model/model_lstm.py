# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:37:41 2019

@author: th
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:06:34 2019

@author: th
"""

#coding=utf-8 
import os
import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    
class lstm(nn.Module):
    def __init__(self,in_planes):
        super(lstm, self).__init__()
        self.in_planes=in_planes
        self.rnn = nn.LSTM(
            input_size=in_planes,
            #hidden_size=in_planes//2,
            hidden_size=in_planes,
            num_layers=2,
            # batch_first=False  # (time_step,batch,input)
            batch_first = True,   # (batch,time_step,input)
            #dropout=args.dropout, 
            bidirectional=False
            )
    def forward(self, x):
        time_step=6
        x = x.view(-1,time_step,self.in_planes)  #(N,time_step,input)
        rout, (hn, hc) = self.rnn(x, None)#rout.size())   #torch.size([N,time_step,output(64)])
        #print(rout.size())
        rout = rout[:, -1, :] #((rout[:, -1, :]).size()) #torch.size([N(4),output(64)])取最后一时间层
        #print(rout.size())
        return rout


class Model(nn.Module):
    def __init__(self):
        n, m = 8, 3

        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(4, 4)

        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd4 = conv3x3(4*n, 8*n)
        self.convd5 = conv3x3(8*n, 16*n)
        self.convd6 = conv3x3(16*n, 32*n)
        self.convd7 = conv3x3(32*n, 64*n)
        self.convd8 = nn.Conv2d(64*n, 128*n, (3,5), 1, 0)##参数修改        
        
        self.lstm0 = lstm(128*n)##参数 
        
        self.dconv5 = nn.ConvTranspose2d(128*n, 64*n,  kernel_size=(3,5), stride=1, padding=0,output_padding=0, bias= False)
        self.convu4 = conv3x3(64*n, 32*n)
        self.convu3 = conv3x3(32*n, 16*n)
        self.convu2 = conv3x3(16*n, 2*n)
        self.convu1 = conv3x3(2*n, 1*n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        x1 = x
        x1=x1.view(-1,3,192,320)
        x1 = self.convd1(x1)
        #print(x1.size())


        x2 = self.convd2(x1)
        #print(x2.size())

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)
        #print(x3.size())

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)
        #print(x4.size())
        
        x5 = self.maxpool(x4)
        x5 = self.convd5(x5)
        #print(x5.size())

        x6 = self.maxpool(x5)
        x6 = self.convd6(x6)
        #print(x6.size())
        
        x7 = self.maxpool(x6)
        x7 = self.convd7(x7)
        #print(x7.size())

        x8 = self.maxpool(x7)
        x8 = self.convd8(x8)
        #print("cnnout:",x8.size())
        x8 = x8.view(-1,1024)

   #加入lstm
        z0 = self.lstm0(x8)
#        print(z0.size())   
        # z0=z0.contiguous().view_as(x8)
        z0=z0.view(-1,1024,1,1)
        #print("lstmout:",z0.size()) 
        
        #y5 = torch.cat([x5, y5], 1)
        y5 = self.dconv5(z0)
        y5 = self.upsample(y5)
        #print(y5.size())
        
        #y4 = torch.cat([x4, y4], 1)
        y4 = self.convu4(y5)
        y4 = self.upsample(y4)
        #print(y4.size())
        
        
        #y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y4)
        y3 = self.upsample(y3)
        #print(y3.size())
        
        
        #y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y3)
        y2 = self.upsample(y2)
        #print(y2.size()) 
        
        #y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y2)
        y1 = self.upsample2(y1)
        #print(y1.size())
        
        y0 = self.convu0(y1)
        y0 = self.sigmoid(y0)
        #print(y0.size())
        # exit(0)
        return y0