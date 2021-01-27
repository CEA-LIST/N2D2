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
from n2d2.cell import Fc, Conv, Softmax, Pool, ElemWise
from n2d2.deepnet import Layer, Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant


def conv_def(nb_outputs, **config_parameters):

    if 'ActivationFunction' in config_parameters:
        act = config_parameters.pop('ActivationFunction')
    else:
        act = Rectifier()
    if 'WeightsFiller' in config_parameters:
        filler = config_parameters.pop('WeightsFiller')
    else:
        filler = He()
    net = Conv(NbOutputs=nb_outputs, ActivationFunction=act, WeightsFiller=filler, NoBias=True, **config_parameters)
    return net

# Residual block generator
def residual_block(nb_outputs, stride, L, projection_shortcut=True, residual_input=None):
    print("Build ResNet block")
    seq = Sequence([
        conv_def(nb_outputs, KernelDims=[3, 3], StrideDims=[stride, stride],
                 WeightsFiller=He(Scaling=(L**(-1.0/(2*2-2)) if L > 0 else 1.0)), PaddingDims=[1, 1]),
        conv_def(nb_outputs,  KernelDims=[3, 3], ActivationFunction=Linear(),
                 WeightsFiller=He(Scaling=(0.0 if L > 0 else 1.0)), StrideDims=[1, 1], PaddingDims=[1, 1]),
    ])
    # Second padding in INI files is [0,0]?

    if projection_shortcut:
        projection = conv_def(nb_outputs, KernelDims=[1, 1], StrideDims=[stride, stride], PaddingDims=[0, 0])
        net = Sequence([
            Layer([seq, projection]),
            ElemWise(nb_outputs, Operation='Sum', ActivationFunction=Rectifier()),
        ])
    elif residual_input is not None:
        net = Sequence([
            seq,
            ElemWise(nb_outputs, Operation='Sum', ActivationFunction=Rectifier(), Inputs=residual_input),
        ])
    else:
        raise RuntimeError("No residual input")

    return net


def resnet18(output_size=1000):
    L = 8
    alpha = 1.0
    learning_rate = 0.1
    max_iterations = 100
    # TODO: Attack solvers to cells using Sequence.get_cells()
    solver_config = ConfigSection(Momentum=0.9, LearningRatePolicy='PolyDecay', Power=1.0, MaxIterations=max_iterations)
    weights_solver = SGD
    weights_solver_config = ConfigSection(LearningRate=learning_rate, Decay=0.0001, **solver_config)
    bias_solver = SGD
    bias_solver_config = ConfigSection(LearningRate=2*learning_rate, Decay=0.0, **solver_config)

    # TODO: Mapping?
    import N2D2
    mapping = N2D2.Tensor_bool(dims=[int(64*alpha),int(64*alpha)])
    for i in range(int(64*alpha)):
        mapping[i+i*int(64*alpha)] = True


    stem = Sequence([
        conv_def(int(64*alpha), KernelDims=[7, 7], StrideDims=[2, 2], PaddingDims=[3, 3]),
        Pool(NbOutputs=int(64*alpha), PoolDims=[3, 3], StrideDims=[2, 2], Pooling='Max', Mapping=mapping)
    ])
    print(stem)

    blocks = []

    blocks.append(residual_block(int(64 * alpha), 1, L, True))
    blocks.append(residual_block(int(64 * alpha), 1, L, False, blocks[0].get_last()))
    blocks.append(residual_block(int(128 * alpha), 2, L, True))
    blocks.append(residual_block(int(128 * alpha), 1, L, False, blocks[2].get_last()))
    blocks.append(residual_block(int(256 * alpha), 2, L, True))
    blocks.append(residual_block(int(256 * alpha), 1, L, False, blocks[4].get_last()))
    blocks.append(residual_block(int(512 * alpha), 2, L, True))
    blocks.append(residual_block(int(512 * alpha), 1, L, False, blocks[6].get_last()))

    body = Sequence(blocks)

    mapping = N2D2.Tensor_bool(dims=[int(512 * alpha), int(512 * alpha)])
    for i in range(int(512 * alpha)):
        mapping[i + i * int(512 * alpha)] = True

    # TODO: Automatic PoolDims setting dependent on input size
    head = Sequence([
        Pool(NbOutputs=int(512 * alpha), PoolDims=[7, 7], StrideDims=[1, 1], Pooling='Average', Mapping=mapping),
        Fc(NbOutputs=output_size, ActivationFunction=Linear(), WeightsFiller=Xavier(Scaling=(0.0 if L > 0 else 1.0)), BiasFiller=Constant(Value=0.0))
    ])
    print("Head")

    net = Sequence([
        stem,
        body,
        head,
        Softmax(NbOutputs=output_size)
    ])
    print('net')

    print(net.get_cells())

    return net