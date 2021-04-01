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
from n2d2.cell import Fc, Conv, ConvDepthWise, Softmax, Pool2d, BatchNorm2d, ElemWise
from n2d2.deepnet import Group
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.deepnet
import n2d2.global_variables
import os
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, SliceExtraction, Flip, Composite
from n2d2.model.ILSVRC_outils import ILSVRC_preprocessing


class ReLU6(Rectifier):
    def __init__(self):
        Rectifier.__init__(self, clipping=6.0)

class ConvDepthWise(ConvDepthWise):
    def __init__(self, nb_outputs, stride, **config_parameters):
        ConvDepthWise.__init__(self, nbOutputs=nb_outputs, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[stride, stride], noBias=True, **config_parameters)

class ConvElemWise(Conv):
    def __init__(self, nb_outputs, **config_parameters):
        Conv.__init__(self, nbOutputs=nb_outputs, kernelDims=[1, 1], strideDims=[1, 1], noBias=True, **config_parameters)


class MobileNetBottleneckBlock(Group):
    def __init__(self, output_size, expansion_size, stride, l, residual, block_name=""):

        self._residual = residual

        seq = [
            ConvElemWise(expansion_size, activationFunction=ReLU6(),
                weightsFiller=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0) if self._residual else He(),
                name=block_name+"_1x1"),
            ConvDepthWise(expansion_size, stride=stride, activationFunction=ReLU6(),
                weightsFiller=He(scaling=l**(-1.0/(2*3-2)) if l > 0 else 1.0) if self._residual else He(),
                name=block_name+"_3x3"),
            ConvElemWise(output_size, activationFunction=Linear(),
                 weightsFiller=He(scaling=0.0 if l > 0 else 1.0) if self._residual else He(),
                 name=block_name+"_1x1_linear")
        ]

        if self._residual:
            seq.append(ElemWise(output_size, operation='Sum', name=block_name+"_sum"))

        Group.__init__(self, seq, name=block_name)

    # Override Group method
    def add_input(self, inputs):
        Group.add_input(self, inputs)
        if self._residual:
            # Connect input directly to ElemWise cell
            self.get_last().add_input(inputs)



class Mobilenet_v2(Group):
    def __init__(self, output_size=1000, alpha=1.0, size=224, l=10, expansion=6):

        self.stem = Group([
            Conv(int(32 * alpha), kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[2, 2], noBias=True, name="conv1"),
            MobileNetBottleneckBlock(int(16 * alpha), int(32 * alpha), 1, l, False, "conv2.1")
        ], name="stem")

        self.body = Group([
            Group([
                MobileNetBottleneckBlock(int(24 * alpha), int(16 * alpha)*expansion, 2, l, False, "conv3.1"),
                MobileNetBottleneckBlock(int(24 * alpha), int(24 * alpha)*expansion, 1, l, True, "conv3.2"),
            ], name=str(size // 4) + "x" + str(size // 4)),
            Group([
                MobileNetBottleneckBlock(int(32 * alpha), int(24 * alpha)*expansion, 2, l, False, "conv4.1"),
                MobileNetBottleneckBlock(int(32 * alpha), int(32 * alpha)*expansion, 1, l, True, "conv4.2"),
                MobileNetBottleneckBlock(int(32 * alpha), int(32 * alpha)*expansion, 1, l, True, "conv4.3"),
            ], name=str(size // 8) + "x" + str(size // 8)),
            Group([
                MobileNetBottleneckBlock(int(64 * alpha), int(32 * alpha)*expansion, 2, l, False, "conv5.1"),
                MobileNetBottleneckBlock(int(64 * alpha), int(64 * alpha)*expansion, 1, l, True, "conv5.2"),
                MobileNetBottleneckBlock(int(64 * alpha), int(64 * alpha)*expansion, 1, l, True, "conv5.3"),
                MobileNetBottleneckBlock(int(64 * alpha), int(64 * alpha)*expansion, 1, l, True, "conv5.4"),
                MobileNetBottleneckBlock(int(96 * alpha), int(64 * alpha)*expansion, 1, l, False, "conv6.1"),
                MobileNetBottleneckBlock(int(96 * alpha), int(96 * alpha)*expansion, 1, l, True, "conv6.2"),
                MobileNetBottleneckBlock(int(96 * alpha), int(96 * alpha)*expansion, 1, l, True, "conv6.3"),
            ], name=str(size // 16) + "x" + str(size // 16)),
            Group([
                MobileNetBottleneckBlock(int(160 * alpha), int(96 * alpha)*expansion, 2, l, False, "conv7.1"),
                MobileNetBottleneckBlock(int(160 * alpha), int(160 * alpha)*expansion, 1, l, True, "conv7.2"),
                MobileNetBottleneckBlock(int(160 * alpha), int(160 * alpha)*expansion, 1, l, True, "conv7.3"),
                MobileNetBottleneckBlock(int(320 * alpha), int(160 * alpha)*expansion, 1, l, False, "conv8.1"),
            ], name=str(size // 32) + "x" + str(size // 32))
        ], name="body")

        self.head =  Group([
            ConvElemWise(max(1280, int(1280 * alpha)), activationFunction=ReLU6(), weightsFiller=He(), name="conv9"),
            Pool2d(max(1280, int(1280 * alpha)), poolDims=[size // 32, size // 32], strideDims=[1, 1], pooling='Average', name="pool"),
        ], name="head")

        self.classifier = Group([
            Fc(output_size, activationFunction=Linear(), weightsFiller=Xavier(scaling=0.0 if l > 0 else 1.0), biasFiller=Constant(value=0.0), name="fc"),
            Softmax(output_size, withLoss=True, name="softmax")
        ], name="classifier")

        Group.__init__(self, [self.stem, self.body, self.head, self.classifier])


    def set_ILSVRC_solvers(self, max_iterations):
        print("Add solvers")
        learning_rate = 0.1

        solver_config = ConfigSection(momentum=0.9, learningRatePolicy='PolyDecay', power=1.0, maxIterations=max_iterations)
        weights_solver = SGD
        weights_solver_config = ConfigSection(learningRate=learning_rate, decay=0.0001, **solver_config.get())
        bias_solver = SGD
        bias_solver_config = ConfigSection(learningRate=2 * learning_rate, decay=0.0, **solver_config.get())

        for name, cell in self.get_cells().items():
            print(name)
            if isinstance(cell, Conv) or isinstance(cell, Fc):
                cell.set_weights_solver(weights_solver(**weights_solver_config.get()))
                cell.set_bias_solver(bias_solver(**bias_solver_config.get()))



def load_from_ONNX(dims=None, batch_size=1, path=None, download=False):
    if dims is None:
        dims = [224, 224, 3]
    #if not dims == [224, 224, 3]:
    #    raise ValueError("This method does not support other dims than [224, 224, 3] yet")
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
    model = n2d2.deepnet.load_from_ONNX(path, dims, batch_size=batch_size)
    return model



def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)



def ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin),
        PadCrop(width=size, height=size),
        RangeAffine(firstOperator='Divides', firstValue=[255.0]),
        ColorSpace(colorSpace='RGB'),
        RangeAffine(firstOperator='Minus', firstValue=[0.485, 0.456, 0.406], secondOperator='Divides', secondValue=[0.229, 0.224, 0.225]),
    ])

    return trans



