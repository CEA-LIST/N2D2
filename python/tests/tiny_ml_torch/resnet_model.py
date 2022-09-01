'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

ResNetV1 (with fewer residual stacks) model definition using Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from operator import __add__


class Conv2dSame(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSame, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class ResNetV1(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10, num_filters=16): 
        super(ResNetV1, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.conv0 = Conv2dSame(in_channels=input_shape[0], out_channels=self.num_filters, kernel_size=3, stride=1, bias=True)
        self.batchnorm0 = nn.BatchNorm2d(self.num_filters)
        
        self.conv1_0 = Conv2dSame(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1, bias=True)
        self.batchnorm1_0 = nn.BatchNorm2d(self.num_filters)
        self.conv1_1 = Conv2dSame(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1, bias=True)
        self.batchnorm1_1 = nn.BatchNorm2d(self.num_filters)

        self.conv2_0 = Conv2dSame(in_channels=self.num_filters, out_channels=self.num_filters*2, kernel_size=3, stride=2, bias=True)
        self.batchnorm2_0 = nn.BatchNorm2d(self.num_filters*2)
        self.conv2_1 = Conv2dSame(in_channels=self.num_filters*2, out_channels=self.num_filters*2, kernel_size=3, stride=1, bias=True)
        self.batchnorm2_1 = nn.BatchNorm2d(self.num_filters*2)
        self.conv2_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters*2, kernel_size=1, stride=2, bias=True)

        self.conv3_0 = Conv2dSame(in_channels=self.num_filters*2, out_channels=self.num_filters*4, kernel_size=3, stride=2, bias=True)
        self.batchnorm3_0 = nn.BatchNorm2d(self.num_filters*4)
        self.conv3_1 = Conv2dSame(in_channels=self.num_filters*4, out_channels=self.num_filters*4, kernel_size=3, stride=1, bias=True)
        self.batchnorm3_1 = nn.BatchNorm2d(self.num_filters*4)
        self.conv3_2 = nn.Conv2d(in_channels=self.num_filters*2, out_channels=self.num_filters*4, kernel_size=1, stride=2, bias=True)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(self.num_filters*4, self.num_classes)

    def forward(self, x):

        # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = F.relu(x)

        # First stack
        y = self.conv1_0(x)
        y = self.batchnorm1_0(y)
        y = F.relu(y)

        y = self.conv1_1(y)
        y = self.batchnorm1_1(y)

        # Overall residual, connect weight layer and identity paths
        x = torch.add(x, y)
        x = F.relu(x)
        
        # Second stack
        y = self.conv2_0(x)
        y = self.batchnorm2_0(y)
        y = F.relu(y)
        
        y = self.conv2_1(y)
        y = self.batchnorm2_1(y)
        
        # Adjust for change in dimension due to stride in identity
        x = self.conv2_2(x)
        
        # Overall residual, connect weight layer and identity paths
        x = torch.add(x, y)
        x = F.relu(x)
        
        # Third stack
        y = self.conv3_0(x)
        y = self.batchnorm3_0(y)
        y = F.relu(y)

        y = self.conv3_1(y)
        y = self.batchnorm3_1(y)

        # Adjust for change in dimension due to stride in identity
        x = self.conv3_2(x)

        # Overall residual, connect weight layer and identity paths
        x = torch.add(x, y)
        x = F.relu(x)
        
        # Final classification layer.
        pool_size = int(np.amin(x.shape[1:3]))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        y = self.flatten(x)
        
        x = self.fc(y)
        x = F.softmax(x, dim=-1)

        return x
