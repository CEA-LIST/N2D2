"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
from n2d2.cell import Fc, Conv, Softmax, Pool
from n2d2.deepnet import Layer, Sequence
from n2d2.activation import Rectifier
from n2d2.solver import SGD
from n2d2.filler import He


def conv_def(nb_outputs, **config_parameters):
    net = Conv(NbOutputs=nb_outputs, ActivationFunction=Rectifier(), WeightsFiller=He(), **config_parameters)
    return net

def residual_block():
    net = Sequence([
        Fc(NbOutputs=50, Name='fc1'),
        Layer([Fc(NbOutputs=50, Name='fc2'), Fc(NbOutputs=50, Name='fc3')]),
        Sequence([
            Fc(NbOutputs=50, Name='fc4', NoBias=True),
            Fc(NbOutputs=10, Name='fc5')
        ]),
        Softmax(NbOutputs=10, Name='soft1')
    ])
    return net


def resnet18():
    alpha = 1.0
    learning_rate = 0.1
    max_iterations = 100
    solver_config = ConfigSection(Momentum=0.9, LearningRatePolicy='PolyDecay', Power=1.0, MaxIterations=max_iterations)
    weights_solver = SGD
    weights_solver_config = ConfigSection(LearningRate=learning_rate, Decay=0.0001, **solver_config)
    bias_solver = SGD
    bias_solver_config = ConfigSection(LearningRate=2*learning_rate, Decay=0.0, **solver_config)

    common_config = ConfigSection(NoBias=True)

    # TODO: Mapping?

    stem = Sequence([
        conv_def(int(64*alpha), KernelDims=[7, 7], StrideDims=[2, 2], PaddingDims=[3, 3]),
        Pool(NbOutputs=int(64*alpha), PoolDims=[3, 3], StrideDims=[2, 2], Pooling='Max')
    ])


    net =  Sequence([
        stem
    ])

    return net