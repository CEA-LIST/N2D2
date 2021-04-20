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
from n2d2.tensor import Interface
from n2d2.cells.nn import Fc, Conv, Pool2d, GlobalPool2d, ElemWise, BatchNorm2d
from n2d2.cells import Sequence, Block
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.deepnet
import n2d2.global_variables
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, SliceExtraction, Flip, Composite
from n2d2.models.ILSVRC_outils import ILSVRC_preprocessing

"""
def conv_def(nb_outputs, **config_parameters):

    if 'activation_function' in config_parameters:
        act = config_parameters.pop('activation_function')
    else:
        act = Rectifier()
    if 'weights_filler' in config_parameters:
        filler = config_parameters.pop('weights_filler')
    else:
        filler = He()
    net = Conv(nb_outputs, activation_function=act, weights_filler=filler, no_bias=True, **config_parameters)
    return net

# Residual block generator
def residual_block(nb_outputs, stride, L, projection_shortcut=True, residual_input=None):
    print("Build ResNet block")
    seq = Group([
        conv_def(nb_outputs, kernel_dims=[3, 3], stride_dims=[stride, stride],
                 weights_filler=He(scaling=(L**(-1.0/(2*2-2)) if L > 0 else 1.0)), padding_dims=[1, 1]),
        conv_def(nb_outputs,  kernel_dims=[3, 3], activation_function=Linear(),
                 weights_filler=He(scaling=(0.0 if L > 0 else 1.0)), stride_dims=[1, 1], padding_dims=[1, 1]),
    ])

    if projection_shortcut:
        projection = conv_def(nb_outputs, kernel_dims=[1, 1], stride_dims=[stride, stride], padding_dims=[0, 0])
        net = Group([
            Layer([seq, projection]),
            ElemWise(nb_outputs, operation='Sum', activation_function=Rectifier()),
        ])
    elif residual_input is not None:
        net = Group([
            seq,
            ElemWise(nb_outputs, operation='Sum', activation_function=Rectifier(), inputs=residual_input),
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
        conv_def(int(64*alpha), kernel_dims=[7, 7], stride_dims=[2, 2], padding_dims=[3, 3]),
        Pool2d(int(64*alpha), pool_dims=[3, 3], stride_dims=[2, 2], pooling='Max')
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
        Pool2d(int(512 * alpha), pool_dims=[7, 7], stride_dims=[1, 1], pooling='Average'),
        Fc(output_size, activation_function=Linear(), weights_filler=Xavier(scaling=(0.0 if L > 0 else 1.0)), bias_filler=Constant(value=0.0))
    ])
    print("Head")

    net = Group([
        stem,
        body,
        head,
        Softmax(output_size)
    ])

    print("Add solvers")

    solver_config = ConfigSection(momentum=0.9, learningRatePolicy='PolyDecay', power=1.0, maxIterations=max_iterations)
    weights_solver = SGD
    weights_solver_config = ConfigSection(learningRate=learning_rate, decay=0.0001, **solver_config.get())
    bias_solver = SGD
    bias_solver_config = ConfigSection(learningRate=2 * learning_rate, decay=0.0, **solver_config.get())

    for name, cells in net.get_cells().items():
        print(name)
        if isinstance(cells, Conv) or isinstance(cells, Fc):
            cells.set_weights_solver(weights_solver(**weights_solver_config.get()))
            cells.set_bias_solver(bias_solver(**bias_solver_config.get()))

    return net

"""




class ResNetBottleneckBlock(Block):
    def __init__(self, inputs_size, bottleneck_size, stride, l, projection_shortcut, no_relu, block_name=""):

        self._main_branch = Sequence([
            Conv(inputs_size, bottleneck_size, kernel_dims=[1, 1], stride_dims=[1, 1], no_bias=True,
                         activation_function=Rectifier(),
                         weights_filler=He(scaling=l ** (-1.0 / (2 * 3 - 2)) if l > 0 else 1.0),
                         name=block_name + "_1x1"),
            Conv(bottleneck_size, bottleneck_size, kernel_dims=[3, 3], padding_dims=[1, 1],
                         stride_dims=[stride, stride], no_bias=True,
                         activation_function=Rectifier(),
                         weights_filler=He(scaling=l ** (-1.0 / (2 * 3 - 2)) if l > 0 else 1.0),
                         name=block_name + "_3x3"),
            Conv(bottleneck_size, 4 * bottleneck_size, kernel_dims=[1, 1], stride_dims=[1, 1],
                         no_bias=True,
                         activation_function=Linear(), weights_filler=He(scaling=0.0 if l > 0 else 1.0),
                         name=block_name + "_1x1_x4"),
        ], name=block_name+"main_branch")
       

        if projection_shortcut:
            self._projection_shortcut = Conv(inputs_size, 4 * bottleneck_size, kernel_dims=[1, 1], stride_dims=[stride, stride], no_bias=True,
                              activation_function=Linear(),
                              name=block_name+"_1x1_proj")
        else:
            self._projection_shortcut = None

        if no_relu:
            self._elem_wise = ElemWise(operation='Sum', name=block_name + "_sum")
        else:
            self._elem_wise = ElemWise(operation='Sum', activation_function=Rectifier(), name=block_name + "_sum")

        if self._projection_shortcut:
            Block.__init__(self, [self._main_branch, self._projection_shortcut, self._elem_wise], block_name)
        else:
            Block.__init__(self, [self._main_branch, self._elem_wise], block_name)


    def __call__(self, x):
        x.get_deepnet().begin_group(name=self._name)

        if self._projection_shortcut:
            shortcut = self._projection_shortcut(x)
        else:
            shortcut = x
        x = self._main_branch(x)
        x = self._elem_wise(Interface([x, shortcut]))

        x.get_deepnet().end_group()

        return x


class ResNet50BnExtractor(Sequence):
    def __init__(self, alpha=1.0, l=0):

        bottleneck_size = int(64 * alpha)

        self.stem = Sequence([
            Conv(3, bottleneck_size, kernel_dims=[7, 7], stride_dims=[2, 2], padding_dims=[3, 3], no_bias=True,
                 activation_function=Rectifier(), weights_filler=He(), name="conv1"),
            Pool2d(pool_dims=[3, 3], stride_dims=[2, 2], pooling='Max', name="pool1")
        ], name="stem")

        self.div4 = Sequence([
            ResNetBottleneckBlock(bottleneck_size, bottleneck_size, 1, l, True, False, "conv2.1"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv2.2"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, True, "conv2.3"),
            BatchNorm2d(4 * bottleneck_size, activation_function=Rectifier(), name="bn2")
        ], name="div4")

        bottleneck_size = 2 * bottleneck_size
        self.div8 = Sequence([
            ResNetBottleneckBlock(2 * bottleneck_size, bottleneck_size, 2, l, True, False, "conv3.1"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv3.2"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv3.3"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, True, "conv3.4"),
            BatchNorm2d(4 * bottleneck_size, activation_function=Rectifier(), name="bn3")
        ], name="div8")

        bottleneck_size = 2 * bottleneck_size
        self.div16 = Sequence([
            ResNetBottleneckBlock(2 * bottleneck_size, bottleneck_size, 2, l, True, False, "conv4.1"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv4.2"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv4.3"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv4.4"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv4.5"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, True, "conv4.6"),
            BatchNorm2d(4 * bottleneck_size, activation_function=Rectifier(), name="bn4")
        ], name="div16")

        bottleneck_size = 2 * bottleneck_size
        self.div32 = Sequence([
            ResNetBottleneckBlock(2 * bottleneck_size, bottleneck_size, 2, l, True, False, "conv5.1"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, False, "conv5.2"),
            ResNetBottleneckBlock(4 * bottleneck_size, bottleneck_size, 1, l, False, True, "conv5.3"),
            BatchNorm2d(4 * bottleneck_size, activation_function=Rectifier(), name="bn5")
        ], name="div32")


        """
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
        """

        Sequence.__init__(self, [self.stem, self.div4, self.div8, self.div16, self.div32], name="extractor")


class ResNetHead(Sequence):
    def __init__(self, output_size=1000, alpha=1.0, l=0):
        input_size = 4 * int(alpha*512)
        Sequence.__init__(self, [
            GlobalPool2d(pooling='Average', name="pool"),
            Fc(input_size, output_size, activation_function=Linear(),
               weights_filler=Xavier(scaling=(0.0 if l > 0 else 1.0)), bias_filler=Constant(value=0.0), name="fc")
        ], name="head")


class ResNet50Bn(Sequence):
    def __init__(self, output_size=1000, alpha=1.0, l=0):

        self.extractor = ResNet50BnExtractor(alpha, l)
        self.head = ResNetHead(output_size, alpha, l)

        Sequence.__init__(self, [self.extractor, self.head], name="resnet50bn")


"""
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
    model = n2d2.cells.DeepNetCell.load_from_ONNX(inputs, path)
    return model




def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)


def ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin, keep_aspect_ratio=False, resize_to_fit=False),
        PadCrop(width=size, height=size),
        RangeAffine(first_operator='Divides', first_value=[255.0]),
        ColorSpace(color_space='RGB'),
        RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides', second_value=[0.229, 0.224, 0.225]),
    ])

    return trans
