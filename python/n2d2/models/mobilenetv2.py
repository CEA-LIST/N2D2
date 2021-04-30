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
from n2d2.cells.nn import Fc, Conv, ConvDepthWise, ConvPointWise, GlobalPool2d, ElemWise
from n2d2.cells import Sequence, Block
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.deepnet
import n2d2.global_variables
import os
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, SliceExtraction, Flip, Composite
from n2d2.models.ILSVRC_outils import ILSVRC_preprocessing


#TODO: Works and executes, however training does not converge stable

class ReLU6(Rectifier):
    def __init__(self):
        Rectifier.__init__(self, clipping=6.0)


class MobileNetBottleneckBlock(Block):
    def __init__(self, input_size, expansion, output_size, stride, l, residual, block_name=""):

        expansion_size = input_size*expansion

        self._main_branch = Sequence([
            ConvPointWise(input_size, expansion_size, activation=ReLU6(), no_bias=True,
                weights_filler=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0) if residual else He(),
                name=block_name+"_1x1"),
            ConvDepthWise(expansion_size, kernel_dims=[3, 3], padding_dims=[1, 1],
                          stride_dims=[stride, stride], no_bias=True, activation=ReLU6(),
                          weights_filler=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0) if residual else He(),
                          name=block_name+"_3x3"),
            ConvPointWise(expansion_size, output_size, activation=Linear(), no_bias=True,
                 weights_filler=He(scaling=0.0 if l > 0 else 1.0) if residual else He(),
                 name=block_name+"_1x1_linear")
        ], name=block_name+"_main_branch")

        if residual:
            self._elem_wise = ElemWise(operation='Sum', name=block_name+"_sum")
            Block.__init__(self, [self._main_branch, self._elem_wise], name=block_name)

        else:
            self._elem_wise = None
            Block.__init__(self, [self._main_branch], name=block_name)

    def __call__(self, inputs):
        inputs.get_deepnet().begin_group(self.get_name())
        x = self._main_branch(inputs)
        if self._elem_wise:
            x = self._elem_wise([x, inputs])
        inputs.get_deepnet().end_group()
        return x


class Mobilenetv2(Sequence):
    def __init__(self, output_size=1000, alpha=1.0, size=224, l=10, expansion=6):

        self.stem = Sequence([
            Conv(3, int(32 * alpha), kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], no_bias=True, name="conv1"),
            MobileNetBottleneckBlock(int(32 * alpha), 1, int(16 * alpha), 1, l, False, "conv2.1")
        ], name="stem")

        self.body = Sequence([
            Sequence([
                MobileNetBottleneckBlock(int(16 * alpha), expansion, int(24 * alpha), 2, l, False, "conv3.1"),
                MobileNetBottleneckBlock(int(24 * alpha), expansion, int(24 * alpha), 1, l, True, "conv3.2"),
            ], name=str(size // 4) + "x" + str(size // 4)),
            Sequence([
                MobileNetBottleneckBlock(int(24 * alpha), expansion, int(32 * alpha), 2, l, False, "conv4.1"),
                MobileNetBottleneckBlock(int(32 * alpha), expansion, int(32 * alpha), 1, l, True, "conv4.2"),
                MobileNetBottleneckBlock(int(32 * alpha), expansion, int(32 * alpha), 1, l, True, "conv4.3"),
            ], name=str(size // 8) + "x" + str(size // 8)),
            Sequence([
                MobileNetBottleneckBlock(int(32 * alpha), expansion, int(64 * alpha), 2, l, False, "conv5.1"),
                MobileNetBottleneckBlock(int(64 * alpha), expansion, int(64 * alpha), 1, l, True, "conv5.2"),
                MobileNetBottleneckBlock(int(64 * alpha), expansion, int(64 * alpha), 1, l, True, "conv5.3"),
                MobileNetBottleneckBlock(int(64 * alpha), expansion, int(64 * alpha), 1, l, True, "conv5.4"),
                MobileNetBottleneckBlock(int(64 * alpha), expansion, int(96 * alpha), 1, l, False, "conv6.1"),
                MobileNetBottleneckBlock(int(96 * alpha), expansion, int(96 * alpha), 1, l, True, "conv6.2"),
                MobileNetBottleneckBlock(int(96 * alpha), expansion, int(96 * alpha), 1, l, True, "conv6.3"),
            ], name=str(size // 16) + "x" + str(size // 16)),
            Sequence([
                MobileNetBottleneckBlock(int(96 * alpha), expansion, int(160 * alpha), 2, l, False, "conv7.1"),
                MobileNetBottleneckBlock(int(160 * alpha), expansion, int(160 * alpha), 1, l, True, "conv7.2"),
                MobileNetBottleneckBlock(int(160 * alpha), expansion, int(160 * alpha), 1, l, True, "conv7.3"),
                MobileNetBottleneckBlock(int(160 * alpha), expansion, int(320 * alpha), 1, l, False, "conv8.1"),
            ], name=str(size // 32) + "x" + str(size // 32))
        ], name="body")

        self.head = Sequence([
            ConvPointWise(int(320 * alpha), max(1280, int(1280 * alpha)),
                          activation=ReLU6(), weights_filler=He(), no_bias=True, name="conv9"),
            GlobalPool2d(pooling='Average', name="pool"),
            Fc(max(1280, int(1280 * alpha)), output_size, activation=Linear(),
               weights_filler=Xavier(scaling=0.0 if l > 0 else 1.0), bias_filler=Constant(value=0.0), name="fc"),
        ], name="head")

        Sequence.__init__(self, [self.stem, self.body, self.head])

    """
    def set_ILSVRC_solvers(self, max_iterations):
        print("Add solvers")
        learning_rate = 0.1

        solver_config = ConfigSection(momentum=0.9, learningRatePolicy='PolyDecay', power=1.0, maxIterations=max_iterations)
        weights_solver = SGD
        weights_solver_config = ConfigSection(learningRate=learning_rate, decay=0.0001, **solver_config.get())
        bias_solver = SGD
        bias_solver_config = ConfigSection(learningRate=2 * learning_rate, decay=0.0, **solver_config.get())

        for name, cells in self.get_cells().items():
            print(name)
            if isinstance(cells, Conv) or isinstance(cells, Fc):
                cells.set_weights_solver(weights_solver(**weights_solver_config.get()))
                cells.set_bias_solver(bias_solver(**bias_solver_config.get()))
    """



def load_from_ONNX(inputs, dims=None, batch_size=1, path=None, download=False):
    print("Loading MobileNet_v2 from ONNX with dims " + str(dims) + " and batch size " + str(batch_size))
    if path is None and not download:
        raise RuntimeError("No path specified")
    elif not path is None and download:
        raise RuntimeError("Specified at same time path and download=True")
    elif path and not download:
        path = n2d2.global_variables.model_cache + "/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
    else:
        n2d2.utils.download_model("https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
            n2d2.global_variables.model_cache+"/ONNX/",
            "mobilenetv2")
        path = n2d2.global_variables.model_cache+"/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
    model = n2d2.cells.DeepNetCell.load_from_ONNX(inputs, path)
    return model


"""
def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)
"""


def ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin),
        PadCrop(width=size, height=size),
        RangeAffine(first_operator='Divides', first_value=[255.0]),
        ColorSpace(color_space='RGB'),
        RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides', second_value=[0.229, 0.224, 0.225]),
    ])

    return trans



