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
from n2d2.cell import Fc, Conv, Softmax, Pool2D, ElemWise
from n2d2.deepnet import Layer, Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant


def conv_def(nb_outputs, **config_parameters):

    if 'activationFunction' in config_parameters:
        act = config_parameters.pop('activationFunction')
    else:
        act = Rectifier()
    if 'weightsFiller' in config_parameters:
        filler = config_parameters.pop('weightsFiller')
    else:
        filler = He()
    net = Conv(nbOutputs=nb_outputs, activationFunction=act, weightsFiller=filler, noBias=True, **config_parameters)
    return net

# Residual block generator
def residual_block(nb_outputs, stride, L, projection_shortcut=True, residual_input=None):
    print("Build ResNet block")
    seq = Sequence([
        conv_def(nb_outputs, kernelDims=[3, 3], strideDims=[stride, stride],
                 weightsFiller=He(scaling=(L**(-1.0/(2*2-2)) if L > 0 else 1.0)), paddingDims=[1, 1]),
        conv_def(nb_outputs,  kernelDims=[3, 3], activationFunction=Linear(),
                 weightsFiller=He(scaling=(0.0 if L > 0 else 1.0)), strideDims=[1, 1], paddingDims=[1, 1]),
    ])

    if projection_shortcut:
        projection = conv_def(nb_outputs, kernelDims=[1, 1], strideDims=[stride, stride], paddingDims=[0, 0])
        net = Sequence([
            Layer([seq, projection]),
            ElemWise(nb_outputs, operation='Sum', activationFunction=Rectifier()),
        ])
    elif residual_input is not None:
        net = Sequence([
            seq,
            ElemWise(nb_outputs, operation='Sum', activationFunction=Rectifier(), inputs=residual_input),
        ])
    else:
        raise RuntimeError("No residual input")

    return net


def resnet18(output_size=1000):
    L = 8
    alpha = 1.0
    learning_rate = 0.1
    max_iterations = 100

    stem = Sequence([
        conv_def(int(64*alpha), kernelDims=[7, 7], strideDims=[2, 2], paddingDims=[3, 3]),
        Pool2D(nbOutputs=int(64*alpha), poolDims=[3, 3], strideDims=[2, 2], pooling='Max')
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

    # TODO: Automatic PoolDims setting dependent on input size
    head = Sequence([
        Pool2D(nbOutputs=int(512 * alpha), poolDims=[7, 7], strideDims=[1, 1], pooling='Average'),
        Fc(nbOutputs=output_size, activationFunction=Linear(), weightsFiller=Xavier(scaling=(0.0 if L > 0 else 1.0)), biasFiller=Constant(value=0.0))
    ])
    print("Head")

    net = Sequence([
        stem,
        body,
        head,
        Softmax(nbOutputs=output_size)
    ])

    print("Add solvers")

    solver_config = ConfigSection(momentum=0.9, learningRatePolicy='PolyDecay', power=1.0, maxIterations=max_iterations)
    weights_solver = SGD
    weights_solver_config = ConfigSection(learningRate=learning_rate, decay=0.0001, **solver_config.get())
    bias_solver = SGD
    bias_solver_config = ConfigSection(learningRate=2 * learning_rate, decay=0.0, **solver_config.get())

    for name, cell in net.get_cells().items():
        print(name)
        if isinstance(cell, Conv) or isinstance(cell, Fc):
            cell.set_weights_solver(weights_solver(**weights_solver_config.get()))
            cell.set_bias_solver(bias_solver(**bias_solver_config.get()))

    return net