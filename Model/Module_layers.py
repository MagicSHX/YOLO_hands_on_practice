import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x

class Conv_Layer_box(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, activation_func, batch_normalization):
        super().__init__()
        padding = (int((kernel_size - 1)/2), int((kernel_size - 1)/2))
        dict_activation_func = {"ReLU": nn.ReLU(inplace=False),
                                "linear": nn.ReLU(inplace=False),
                                "leaky": nn.LeakyReLU(0.1, inplace=False),
                                "mish": Mish()
                               }
        
        if batch_normalization == True:
            bias = False
        else:
            bias = True
        self.conv_box = nn.ModuleList()
        self.conv_box.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias = bias))
        if batch_normalization == True:
            self.conv_box.append(nn.BatchNorm2d(out_channel))
        if activation_func != "linear":
            self.conv_box.append(dict_activation_func[activation_func])
        
    def forward(self, x):
        for layer in self.conv_box:
            x = layer(x)
        return x

class Maxpool_pad_Layer_box(nn.Module):
    def __init__(self, maxpool_size):
        super().__init__()
        self.maxpool_size = maxpool_size
        #why there are 2 padding??????????????
        self.pad_1 = int((self.maxpool_size - 1) / 2)
        self.pad_2 = self.pad_1
    def forward(self, x):
        x = F.pad(x, (self.pad_1, self.pad_2, self.pad_1, self.pad_2), mode = 'replicate')
        x = F.max_pool2d(x, self.maxpool_size, stride = 1)
        return x

class Upsample_layer(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        
    def forward(self, x):
        batch, channel, height, width = x.data.size()
        x = x.view(batch, channel, height, 1, width, 1).expand(batch, channel, height, self.stride, width, self.stride).clone()
        x = x.contiguous().view(batch, channel, height * self.stride, width * self.stride).clone()
        return x

class shortcut(nn.Module):
    def __init__(self):
        super().__init__()
        
class route(nn.Module):
    def __init__(self):
        super().__init__()