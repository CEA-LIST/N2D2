"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

from n2d2.utils import ConfigSection
from n2d2.cells.nn import Fc, Conv, Pool2d, BatchNorm2d, Dropout
from n2d2.cells import Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.filler import Normal, Xavier


def conv_def():
    weights_filler = Xavier(variance_norm='FanOut', scaling=1.0)
    return ConfigSection(activation=Linear(), no_bias=True, weights_filler=weights_filler)

def fc_def():
    weights_filler = Normal(mean=0.0, std_dev=0.01)
    return ConfigSection(no_bias=True, weights_filler=weights_filler)

def bn_def():
    return ConfigSection(activation=Rectifier())



class LeNet(Sequence):
    def __init__(self, nb_outputs=10):
        Sequence.__init__(self, [
            Conv(1, 6, kernel_dims=[5, 5], **conv_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(6, 16, kernel_dims=[5, 5], **conv_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(16, 120, kernel_dims=[5, 5], **conv_def()),
            Fc(120, 84, activation=Rectifier(), **fc_def()),
            Dropout(dropout=0.5),
            Fc(84, nb_outputs, activation=Linear(), **fc_def()),
        ])


class LeNetBN(Sequence):
    def __init__(self, nb_outputs=10):
        Sequence.__init__(self, [
            Conv(1, 6, kernel_dims=[5, 5], **conv_def()),
            BatchNorm2d(6, **bn_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(6, 16, kernel_dims=[5, 5], **conv_def()),
            BatchNorm2d(16, **bn_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(16, 120, kernel_dims=[5, 5], **conv_def()),
            BatchNorm2d(120, **bn_def()),
            Fc(120, 84, activation=Rectifier(), **fc_def()),
            Dropout(dropout=0.5),
            Fc(84, nb_outputs, activation=Linear(), **fc_def()),
        ])




