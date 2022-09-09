'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

Anomaly model definition using Pytorch
'''

import torch.nn as nn


def get_model(inputDim, BIAS=True):
    """
    define the torch model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    layers = []

    layers.append(nn.Linear(inputDim, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 8, bias=BIAS))
    layers.append(nn.BatchNorm1d(8))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(8, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, 128, bias=BIAS))
    layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())

    layers.append(nn.Linear(128, inputDim, bias=BIAS))

    model = nn.Sequential(*layers)

    return model
