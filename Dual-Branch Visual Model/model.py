#coding=utf-8 
import os
import torch.nn as nn
import torch
from model_lf import  late_fusion 
from model_lstm import Model as lstm 
from model_cdnn import Model as cdnn 



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lf =late_fusion()
        self.lstm=lstm()
        self.cdnn=cdnn()


    def forward(self, x, y):
        #print(x.size(),y.size())
        cdnnout = self.cdnn(x)
        print('cdnnout:',cdnnout.size())
        lstmout = self.lstm(y)
        print('lstmout:',lstmout.size())
        lfout = self.lf(lstmout,cdnnout)
        print('lfout:',lfout.size())

        return lfout