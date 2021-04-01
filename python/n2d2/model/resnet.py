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
from n2d2.cell import Fc, Conv, Softmax, Pool2d, ElemWise, BatchNorm2d
from n2d2.deepnet import Group
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.deepnet
import n2d2.global_variables
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, SliceExtraction, Flip, Composite
from n2d2.model.ILSVRC_outils import ILSVRC_preprocessing

"""
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
    seq = Group([
        conv_def(nb_outputs, kernelDims=[3, 3], strideDims=[stride, stride],
                 weightsFiller=He(scaling=(L**(-1.0/(2*2-2)) if L > 0 else 1.0)), paddingDims=[1, 1]),
        conv_def(nb_outputs,  kernelDims=[3, 3], activationFunction=Linear(),
                 weightsFiller=He(scaling=(0.0 if L > 0 else 1.0)), strideDims=[1, 1], paddingDims=[1, 1]),
    ])

    if projection_shortcut:
        projection = conv_def(nb_outputs, kernelDims=[1, 1], strideDims=[stride, stride], paddingDims=[0, 0])
        net = Group([
            Layer([seq, projection]),
            ElemWise(nb_outputs, operation='Sum', activationFunction=Rectifier()),
        ])
    elif residual_input is not None:
        net = Group([
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

    stem = Group([
        conv_def(int(64*alpha), kernelDims=[7, 7], strideDims=[2, 2], paddingDims=[3, 3]),
        Pool2d(nbOutputs=int(64*alpha), poolDims=[3, 3], strideDims=[2, 2], pooling='Max')
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

    body = Group(blocks)

    # TODO: Automatic PoolDims setting dependent on input size
    head = Group([
        Pool2d(nbOutputs=int(512 * alpha), poolDims=[7, 7], strideDims=[1, 1], pooling='Average'),
        Fc(nbOutputs=output_size, activationFunction=Linear(), weightsFiller=Xavier(scaling=(0.0 if L > 0 else 1.0)), biasFiller=Constant(value=0.0))
    ])
    print("Head")

    net = Group([
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

"""




class ResNetStem(Group):
    def __init__(self, inputs,  alpha):
            conv = Conv(inputs, int(64*alpha), kernelDims=[7, 7], strideDims=[2, 2], paddingDims=[3, 3], noBias=True,
                 activationFunction=Rectifier(), weightsFiller=He(), name="conv1")
            pool = Pool2d(conv, nbOutputs=int(64 * alpha), poolDims=[3, 3], strideDims=[2, 2], pooling='Max', name="pool1")
            Group.__init__(self, [conv, pool], name="stem")


class ResNetBottleneckBlock(Group):
    def __init__(self, inputs, bottleneck_size, stride, l, projection_shortcut, no_relu, block_name=""):

        self._projection_shortcut = projection_shortcut

        seq = Group([ ], name="main_branch")
        seq.add(Conv(inputs, bottleneck_size, kernelDims=[1, 1], strideDims=[1, 1], noBias=True,
                 activationFunction=Rectifier(), weightsFiller=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0),
                 name=block_name+"_1x1"))
        seq.add(Conv(seq.get_last(), nbOutputs=bottleneck_size, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[stride, stride], noBias=True,
                 activationFunction=Rectifier(), weightsFiller=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0),
                 name=block_name+"_3x3"))
        seq.add(Conv(seq.get_last(), nbOutputs=4 * bottleneck_size, kernelDims=[1, 1], strideDims=[1, 1], noBias=True,
                 activationFunction=Linear(), weightsFiller=He(scaling=0.0 if l > 0 else 1.0),
                 name=block_name+"_1x1_x4"))

        if projection_shortcut:
            shortcut = Conv(inputs, nbOutputs=4 * bottleneck_size, kernelDims=[1, 1], strideDims=[stride, stride], noBias=True,
                              activationFunction=Linear(),
                              name=block_name+"_1x1_proj")
        else:
            shortcut = inputs

        if no_relu:
            elem_wise = ElemWise([seq.get_last(), shortcut], 4 * bottleneck_size, operation='Sum',
                                 name=block_name + "_sum")
        else:
            elem_wise = ElemWise([seq.get_last(), shortcut], 4 * bottleneck_size, operation='Sum',
                                 activationFunction=Rectifier(), name=block_name + "_sum")

        block = Group([seq, shortcut])

        Group.__init__(self, [block, elem_wise], name=block_name)


class ResNet50BNBody(Group):
    def __init__(self, inputs, alpha, l):

        # TODO: Fix scales taking into account stem
        seq = Group([])
        seq.add(ResNetBottleneckBlock(inputs, int(64 * alpha), 1, l, True, False, "conv2.1"))
        seq.add(ResNetBottleneckBlock(seq.get_last(), int(64 * alpha), 1, l, False, False, "conv2.2"))
        seq.add(ResNetBottleneckBlock(seq.get_last(), int(64 * alpha), 1, l, False, True, "conv2.3"))
        seq.add(BatchNorm2d(seq.get_last(), 4 * int(64 * alpha), activationFunction=Rectifier(), name="bn2"))

        seq1 = Group([])
        seq1.add(ResNetBottleneckBlock(seq.get_last(), int(128 * alpha), 2, l, True, False, "conv3.1"))
        seq1.add(ResNetBottleneckBlock(seq1.get_last(), int(128 * alpha), 1, l, False, False, "conv3.2"))
        seq1.add(ResNetBottleneckBlock(seq1.get_last(), int(128 * alpha), 1, l, False, False, "conv3.3"))
        seq1.add(ResNetBottleneckBlock(seq1.get_last(), int(128 * alpha), 1, l, False, True, "conv3.4"))
        seq1.add(BatchNorm2d(seq1.get_last(), 4 * int(128 * alpha), activationFunction=Rectifier(), name="bn3"))

        seq2 = Group([])
        seq2.add(ResNetBottleneckBlock(seq1.get_last(), int(256 * alpha), 2, l, True, False, "conv4.1"))
        seq2.add(ResNetBottleneckBlock(seq2.get_last(), int(256 * alpha), 1, l, False, False, "conv4.2"))
        seq2.add(ResNetBottleneckBlock(seq2.get_last(), int(256 * alpha), 1, l, False, False, "conv4.3"))
        seq2.add(ResNetBottleneckBlock(seq2.get_last(), int(256 * alpha), 1, l, False, False, "conv4.4"))
        seq2.add(ResNetBottleneckBlock(seq2.get_last(), int(256 * alpha), 1, l, False, False, "conv4.5"))
        seq2.add(ResNetBottleneckBlock(seq2.get_last(), int(256 * alpha), 1, l, False, True, "conv4.6"))
        seq2.add(BatchNorm2d(seq2.get_last(), 4 * int(256 * alpha), activationFunction=Rectifier(), name="bn4"))

        seq3 = Group([])
        seq3.add(ResNetBottleneckBlock(seq2.get_last(), int(512 * alpha), 2, l, True, False, "conv5.1"))
        seq3.add(ResNetBottleneckBlock(seq3.get_last(), int(512 * alpha), 1, l, False, False, "conv5.2"))
        seq3.add(ResNetBottleneckBlock(seq3.get_last(), int(512 * alpha), 1, l, False, True, "conv5.3"))
        seq3.add(BatchNorm2d(seq3.get_last(), 4 * int(512 * alpha), activationFunction=Rectifier(), name="bn5"))

        self.scales = {}
        name = str(inputs.get_outputs().dimX()) + "x" + str(inputs.get_outputs().dimX())
        self.scales[name] = Group([seq], name=name)
        name = str(seq.get_last().get_outputs().dimX()) + "x" + str(seq.get_last().get_outputs().dimY())
        self.scales[name] = Group([seq], name=name)
        name = str(seq1.get_last().get_outputs().dimX()) + "x" + str(seq1.get_last().get_outputs().dimY())
        self.scales[name] = Group([seq, seq1], name=name)
        name = str(seq2.get_last().get_outputs().dimX()) + "x" + str(seq2.get_last().get_outputs().dimY())
        self.scales[name] = Group([seq, seq1, seq2], name=name)
        name = str(seq3.get_last().get_outputs().dimX()) + "x" + str(seq3.get_last().get_outputs().dimY())
        self.scales[name] = Group([seq, seq1, seq2, seq3], name=name)

        Group.__init__(self, [seq, seq1, seq2, seq3], name="body")


class ResNetHead(Group):
    def __init__(self, inputs, alpha):
        Group.__init__(self, [
            Pool2d(inputs, 4 * int(512 * alpha),
                   poolDims=[inputs.get_last().get_outputs().dimX(), inputs.get_last().get_outputs().dimY()],
                   strideDims=[1, 1], pooling='Average', name="pool"),
        ], name="head")


class ResNetClassifier(Group):
    def __init__(self, inputs, output_size, l):
        fc = Fc(inputs, output_size, activationFunction=Linear(), weightsFiller=Xavier(scaling=(0.0 if l > 0 else 1.0)), biasFiller=Constant(value=0.0), name="fc")
        softmax = Softmax(fc, output_size, withLoss=True,  name="softmax")
        Group.__init__(self, [fc, softmax], name="classifier")


"""
Abstract ResNet class. TODO: Make true abstract class?
"""
class ResNet(Group):

    _with_batchnorm = False
    stem = None
    body = None
    head = None
    classifier = None

    def __init__(self):
        if self.stem is None or self.body is None or self.head is None or self.classifier is None:
            raise RuntimeError("Missing elements. Did you try to create and abstract ResNet?")
        Group.__init__(self, [self.stem, self.body, self.head, self.classifier])

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

            if self._with_batchnorm and isinstance(cell, BatchNorm2d):
                cell.set_scale_solver(bn_solver(**bn_solver_config.get()))
                cell.set_bias_solver(bn_solver(**bn_solver_config.get()))


class ResNet50BN(ResNet):

    def __init__(self, inputs, output_size=1000, alpha=1.0, l=0):
        self._with_batchnorm = True

        self.stem = ResNetStem(inputs, alpha)
        self.body = ResNet50BNBody(self.stem.get_last(), alpha, l)
        self.head = ResNetHead(self.body.get_last(), alpha)
        self.classifier = ResNetClassifier(self.head.get_last(), output_size, l)

        ResNet.__init__(self)

"""
class ResNet50(ResNet):

    def __init__(self, output_size=1000, alpha=1.0, size=224, l=0):
        self.body = ResNet50Body(alpha, l)
        ResNet.__init__(self, output_size, alpha, size, l)
"""



def load_from_ONNX(inputs, resnet_type, version='pre_act', dims=None, batch_size=1, path=None, download=False):
    if dims is None:
        dims = [224, 224, 3]
    #if not dims == [224, 224, 3]:
    #    raise ValueError("This method does not support other dims than [224, 224, 3] yet")
    allowed_types = ['18', '34', '50', '101', '152']
    if not resnet_type in allowed_types:
        raise ValueError("ResNet type must be one of these: '18', '34', '50', '101', '152'!")
    if version == 'pre_act':
        v = "v1"
    elif version == 'post_act':
        v = "v2"
    else:
        raise ValueError("ResNet version must be either 'pre_act' or 'post_act'!")
    resnet_name = "resnet-" + resnet_type + "-" + v

    print("Loading " + version + " ResNet"+str(resnet_type)+
          " from ONNX with dims " + str(dims) + " and batch size " + str(batch_size))
    if path is None and not download:
        raise RuntimeError("No path specified")
    elif not path is None and download:
        raise RuntimeError("Specified at same time path and download=True")
    elif path and not download:
        path = n2d2.global_variables.model_cache + "/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
    else:
        n2d2.utils.download_model(
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/"+"resnet"+resnet_type+v+"/"+"resnet"+resnet_type+v+".onnx",
            n2d2.global_variables.model_cache + "/ONNX/",
            resnet_name)
        path = n2d2.global_variables.model_cache + "/ONNX/"+resnet_name+"/"+"resnet"+resnet_type+v+".onnx"
    model = n2d2.deepnet.DeepNet.load_from_ONNX(inputs, path)
    return model




def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)


def ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin, keepAspectRatio=False, ResizeToFit=False),
        PadCrop(width=size, height=size),
        RangeAffine(firstOperator='Divides', firstValue=[255.0]),
        ColorSpace(colorSpace='RGB'),
        RangeAffine(firstOperator='Minus', firstValue=[0.485, 0.456, 0.406], secondOperator='Divides', secondValue=[0.229, 0.224, 0.225]),
    ])

    return trans
