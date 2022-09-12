'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

KWS (key word spotting) model definition using Pytorch
'''

import torch.nn as nn
from numpy import ceil

class depthwise_pointwise_separable_conv(nn.Module):
    def __init__(self, cin, cout,BNorm):
        super(depthwise_pointwise_separable_conv, self).__init__()
        self.BNorm=BNorm
        # DW: cin=cout
        cout_dw=cin
        self.dw = nn.Conv2d(cin, cout_dw, kernel_size=(3,3), stride=1, padding=1, padding_mode='zeros', bias=True, groups=cin)
        if self.BNorm:
            self.dw_bn = nn.BatchNorm2d(cout_dw)
        self.dw_relu = nn.ReLU()
        
        # PW: cin=cout_dw
        self.pw = nn.Conv2d(cout_dw, cout, kernel_size=(1,1), stride=1, bias=True)
        if self.BNorm:
            self.pw_bn = nn.BatchNorm2d(cout)
        self.pw_relu = nn.ReLU()

    def forward(self, x):

        x = self.dw(x)
        if self.BNorm:
            x= self.dw_bn(x)
        x= self.dw_relu(x)
        
        x = self.pw(x)
        if self.BNorm:
            x= self.pw_bn(x)
        x= self.pw_relu(x)

        return x
    

NBCIN=64
class KWS_Net(nn.Module):
    def __init__(self , nb_words, BNorm=True, image_size=(49,10), DropOut=0.1):
        # Definition of all modules
        super(KWS_Net, self).__init__()
        self.BNorm=BNorm
        self.dim_h, self.dim_w=image_size
        self.DropOut=DropOut

       
        self.add_module('conv_0', nn.Conv2d(1, NBCIN, kernel_size=(10,4), stride=2, padding=(5,1), padding_mode='zeros', bias=True))
        self.dim_h=int(ceil(self.dim_h/float(2)))
        self.dim_w=int(ceil(self.dim_w/float(2)))

        if BNorm:
            self.bn_0 = nn.BatchNorm2d(NBCIN)
        self.relu_0 = nn.ReLU()

        self.dwpw1=depthwise_pointwise_separable_conv(NBCIN,NBCIN,self.BNorm)
        self.dwpw2=depthwise_pointwise_separable_conv(NBCIN,NBCIN,self.BNorm)
        self.dwpw3=depthwise_pointwise_separable_conv(NBCIN,NBCIN,self.BNorm)
        self.dwpw4=depthwise_pointwise_separable_conv(NBCIN,NBCIN,self.BNorm)
        

        # count_include_pad=False to avoid unecessary "pad" module in onnx output
        self.avgpool = nn.AvgPool2d(kernel_size=(self.dim_h,self.dim_w), count_include_pad=False)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(NBCIN, nb_words)
        self.DO=nn.Dropout(self.DropOut)


    def forward(self, x):
        # How modules are linked to each other
      
        x = self.conv_0(x)
        if self.BNorm:
            x = self.bn_0(x)
        x = self.relu_0(x)
        x=self.dwpw1(x)
        x=self.dwpw2(x)
        x=self.dwpw3(x)
        x=self.dwpw4(x)
        x= self.avgpool(x)
        x= self.flatten(x)
        x= self.DO(x)
        x=self.fc(x)
        

        return x
