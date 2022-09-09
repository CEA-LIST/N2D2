'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

MobileNetV1 model definition using Pytorch
'''

import torch.nn as nn
from functools import reduce
from operator import __add__


class Conv2dSame(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSame, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


# Define model architecture
def mobilenet_v1(small=False, use_last_bias=True, image_size=96, grayscale=False):
    '''
    MobilenetV1, alpha=0.25 with 96x96 input (color or grayscale) images
    
    Parameters:
        small : Bool, if True, remove last filter multiplication. Default is False
        use_last_bias : Bool, if false, remove bias from last pointwise layer. Default is True
        image_size: int, size of input images
        grayscale: bool, if True, use grayscale images instead of color images
    '''
    
    # Mobilenet parameters
    if grayscale:
        input_shape = [1, image_size, image_size] 
    else:
        input_shape = [3, image_size, image_size] 
    
    # We have two classes: person and non-person
    num_classes = 2

    num_filters = 8

    layers = []

    # 1st layer, pure conv
    layers.append(Conv2dSame(in_channels=input_shape[0], out_channels=num_filters, kernel_size=3, stride=2))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 2nd layer, depthwise separable conv
    # Filter size is always doubled before the pointwise conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    num_filters = 2*num_filters
    layers.append(Conv2dSame(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 3rd layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=2, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    num_filters = 2*num_filters
    layers.append(Conv2dSame(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 4th layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 5th layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=2, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    num_filters = 2*num_filters
    layers.append(Conv2dSame(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 6th layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 7th layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=2, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    num_filters = 2*num_filters
    layers.append(Conv2dSame(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 8th-12th layers, identical depthwise separable convs
    # 8th layer
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 9th layer
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 10th layer
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 11th layer
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 12th layer
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 13th layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=2, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    if small == True:
        layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1))
    else:
        num_filters = 2*num_filters
        layers.append(Conv2dSame(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # 14th layer, depthwise separable conv
    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, groups=num_filters))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    layers.append(Conv2dSame(in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1, bias=use_last_bias))
    # To avoid bias addition during onnx export, we need to remove batchnorm too
    if use_last_bias == True:
        layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())

    # Average pooling, max polling may be used also
    layers.append(nn.AvgPool2d(kernel_size=3))

    # Flatten, FC layer and classify
    layers.append(nn.Flatten())
    layers.append(nn.Linear(num_filters, num_classes))
    layers.append(nn.Softmax(dim=-1))

    model = nn.Sequential(*layers)

    return model
