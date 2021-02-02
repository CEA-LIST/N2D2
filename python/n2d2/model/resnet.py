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
from n2d2.cell import Fc, Conv, Softmax, Pool2D, ElemWise, BatchNorm
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






class ResNetStem(Sequence):
    def __init__(self, alpha):
        Sequence.__init__(self, [
            Conv(int(64*alpha), kernelDims=[7, 7], strideDims=[2, 2], paddingDims=[3, 3], noBias=True,
                 activationFunction=Rectifier(), weightsFiller=He(), name="conv1"),
            Pool2D(nbOutputs=int(64*alpha), poolDims=[3, 3], strideDims=[2, 2], pooling='Max', name="pool1")
        ], name="stem")


class ResNetBottleneckBlock(Sequence):
    def __init__(self, bottleneck_size, stride, l, projection_shortcut, no_relu, block_name=""):

        self._projection_shortcut = projection_shortcut

        seq = Sequence([
            Conv(nbOutputs=bottleneck_size, kernelDims=[1, 1], strideDims=[1, 1], noBias=True,
                 activationFunction=Rectifier(), weightsFiller=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0),
                 name=block_name+"_1x1"),
            Conv(nbOutputs=bottleneck_size, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[stride, stride], noBias=True,
                 activationFunction=Rectifier(), weightsFiller=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0),
                 name=block_name+"_3x3"),
            Conv(nbOutputs=4 * bottleneck_size, kernelDims=[1, 1], strideDims=[1, 1], noBias=True,
                 activationFunction=Linear(), weightsFiller=He(scaling=0.0 if l > 0 else 1.0),
                 name=block_name+"_1x1_x4")
        ], name="main_branch")

        if no_relu:
            elem_wise = ElemWise(4 * bottleneck_size, operation='Sum', name=block_name+"_sum")
        else:
            elem_wise = ElemWise(4 * bottleneck_size, operation='Sum', activationFunction=Rectifier(), name=block_name+"_sum")

        if projection_shortcut:
            projection = Conv(nbOutputs=4 * bottleneck_size, kernelDims=[1, 1], strideDims=[stride, stride], noBias=True,
                              activationFunction=Linear(),
                              name=block_name+"_1x1_proj")
            block = [Layer([seq, projection]), elem_wise]
        else:
            block = [seq, elem_wise]

        Sequence.__init__(self, block, name=block_name)

    # Override Sequence method
    def add_input(self, inputs):
        Sequence.add_input(self, inputs)
        if not self._projection_shortcut:
            # Connect input directly to ElemWise cell
            self.get_last().add_input(inputs)


class ResNet50BNBody(Sequence):
    def __init__(self, alpha, size, l):

        self.scales = {
            str(size // 4) + "x" + str(size // 4): Sequence([
                ResNetBottleneckBlock(int(64 * alpha), 1, l, True, False, "conv2.1"),
                ResNetBottleneckBlock(int(64 * alpha), 1, l, False, False, "conv2.2"),
                ResNetBottleneckBlock(int(64 * alpha), 1, l, False, True, "conv2.3"),
                BatchNorm(4 * int(64 * alpha), activationFunction=Rectifier(), name="bn2")
            ], name=str(size // 4) + "x" + str(size // 4)),
            str(size // 8) + "x" + str(size // 8): Sequence([
                ResNetBottleneckBlock(int(128 * alpha), 2, l, True, False, "conv3.1"),
                ResNetBottleneckBlock(int(128 * alpha), 1, l, False, False, "conv3.2"),
                ResNetBottleneckBlock(int(128 * alpha), 1, l, False, False, "conv3.3"),
                ResNetBottleneckBlock(int(128 * alpha), 1, l, False, True, "conv3.4"),
                BatchNorm(4 * int(128 * alpha), activationFunction=Rectifier(), name="bn3")
            ], name=str(size // 8) + "x" + str(size // 8)),
            str(size // 16) + "x" + str(size // 16): Sequence([
                ResNetBottleneckBlock(int(256 * alpha), 2, l, True, False, "conv4.1"),
                ResNetBottleneckBlock(int(256 * alpha), 1, l, False, False, "conv4.2"),
                ResNetBottleneckBlock(int(256 * alpha), 1, l, False, False, "conv4.3"),
                ResNetBottleneckBlock(int(256 * alpha), 1, l, False, False, "conv4.4"),
                ResNetBottleneckBlock(int(256 * alpha), 1, l, False, False, "conv4.5"),
                ResNetBottleneckBlock(int(256 * alpha), 1, l, False, True, "conv4.6"),
                BatchNorm(4 * int(256 * alpha), activationFunction=Rectifier(), name="bn4")
            ], name=str(size // 16) + "x" + str(size // 16)),
            str(size // 32) + "x" + str(size // 32): Sequence([
                ResNetBottleneckBlock(int(512 * alpha), 2, l, True, False, "conv5.1"),
                ResNetBottleneckBlock(int(512 * alpha), 1, l, False, False, "conv5.2"),
                ResNetBottleneckBlock(int(512 * alpha), 1, l, False, True, "conv5.3"),
                BatchNorm(4 * int(512 * alpha), activationFunction=Rectifier(), name="bn5")
            ], name=str(size // 32) + "x" + str(size // 32))
        }

        Sequence.__init__(self, [scale[1] for scale in self.scales.items()], name="body")


class ResNetHead(Sequence):
    def __init__(self, alpha, size):
        Sequence.__init__(self, [
            Pool2D(nbOutputs=4 * int(512 * alpha), poolDims=[size // 32, size // 32], strideDims=[1, 1], pooling='Average', name="pool"),
        ], name="head")


class ResNetClassifier(Sequence):
    def __init__(self, output_size, l):
        Sequence.__init__(self, [
            Fc(nbOutputs=output_size, activationFunction=Linear(), weightsFiller=Xavier(scaling=(0.0 if l > 0 else 1.0)), biasFiller=Constant(value=0.0), name="fc"),
            Softmax(nbOutputs=output_size, name="softmax")
        ], name="classifier")

"""
Abstract ResNet class
"""
class ResNet(Sequence):

    body = None
    _with_batchnorm = False

    def __init__(self, output_size=1000, alpha=1.0, size=224, l=0):
        self.stem = ResNetStem(alpha)
        self.head = ResNetHead(alpha, size)
        self.classifier = ResNetClassifier(output_size, l)

        if self.body is None:
            raise RuntimeError("No body defined. Did you try to create and abstract ResNet?")

        Sequence.__init__(self, [self.stem, self.body, self.head, self.classifier])

    def set_ILSVRC_solvers(self, max_iterations):
        print("Add solvers")
        learning_rate = 0.1

        solver_config = ConfigSection(momentum=0.9, learningRatePolicy='PolyDecay', power=1.0,
                                      maxIterations=max_iterations)
        weights_solver = SGD
        weights_solver_config = ConfigSection(learningRate=learning_rate, decay=0.0001, **solver_config.get())
        bias_solver = SGD
        bias_solver_config = ConfigSection(learningRate=2 * learning_rate, decay=0.0, **solver_config.get())

        if self._with_batchnorm:
            bn_solver = SGD
            bn_solver_config = ConfigSection(learningRate=learning_rate, decay=0.0001, **solver_config.get())

        for name, cell in self.get_cells().items():
            print(name)
            if isinstance(cell, Conv) or isinstance(cell, Fc):
                cell.set_weights_solver(weights_solver(**weights_solver_config.get()))
                cell.set_bias_solver(bias_solver(**bias_solver_config.get()))

            if self._with_batchnorm and isinstance(cell, BatchNorm):
                cell.set_scale_solver(bn_solver(**bn_solver_config.get()))
                cell.set_bias_solver(bn_solver(**bn_solver_config.get()))


class ResNet50BN(ResNet):

    def __init__(self, output_size=1000, alpha=1.0, size=224, l=0):
        self._with_batchnorm = True
        self.body = ResNet50BNBody(alpha, size, l)
        ResNet.__init__(self, output_size, alpha, size, l)

"""
class ResNet50(ResNet):

    def __init__(self, output_size=1000, alpha=1.0, size=224, l=0):
        self.body = ResNet50Body(alpha, l)
        ResNet.__init__(self, output_size, alpha, size, l)
"""



