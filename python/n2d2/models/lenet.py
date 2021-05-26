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
from n2d2.cells.nn import Fc, Conv, Softmax, Pool2d
from n2d2.cells import Sequence
from n2d2.activation import Rectifier
from n2d2.solver import SGD
from n2d2.filler import Normal, He
import n2d2.global_variables

solver_config = ConfigSection(learning_rate=0.05, momentum=0.9, decay=0.0005, learning_rate_decay=0.993)

def conv_def():
    weights_filler = He()
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation=Rectifier(), weights_solver=weights_solver, bias_solver=bias_solver,
                           no_bias=True, weights_filler=weights_filler)

def fc_def():
    weights_filler = Normal(mean=0.0, std_dev=0.01)
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(weights_solver=weights_solver, bias_solver=bias_solver,
                           no_bias=True, weights_filler=weights_filler)

def bn_def():
    scale_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation=Rectifier(), scale_solver=scale_solver, bias_solver=bias_solver)

class LeNet(Sequence):
    def __init__(self, nb_outputs=10):
        conv2_mapping = n2d2.Tensor([6, 16], datatype="bool")
        conv2_mapping.set_values([
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]])

        Sequence.__init__(self, [
            Conv(1, 6, kernel_dims=[5, 5], **conv_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(6, 16, kernel_dims=[5, 5], mapping=conv2_mapping, **conv_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(16, 120, kernel_dims=[5, 5], **conv_def()),
            Fc(120, 84, **fc_def()),
            Fc(84, nb_outputs, **fc_def()),
            Softmax(with_loss=True),
        ])




