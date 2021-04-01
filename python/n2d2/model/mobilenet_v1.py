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
from n2d2.cell import Fc, Conv, ConvDepthWise, ConvPointWise, Softmax, GlobalPool2d, BatchNorm2d
from n2d2.deepnet import Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.global_variables
from n2d2.model.ILSVRC_outils import ILSVRC_preprocessing


def conv_config():
    return ConfigSection(activationFunction=Rectifier(), weightsFiller=He(), noBias=True)
def conv_config_bn():
    return ConfigSection(activationFunction=Linear(), weightsFiller=He(), noBias=True)



class MobileNetv1Extractor(Sequence):
    def __init__(self, alpha):

        Sequence([
            Conv(nbOutputs=int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
                 **conv_config(), name="conv1"),
            ConvDepthWise(kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv1_3x3_dw",
                          **conv_config()),
            ConvPointWise(2 * x.dims()[2], name="conv1_1x1", **conv_config())
        ], "div2")

        x = Conv(deepnet, nbOutputs=int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
                 **conv_config(), name="conv1")
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv1_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, 2 * x.dims()[2], name="conv1_1x1", **conv_config())

        deepnet.begin_group("div4")
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[2, 2], name="conv2_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, 2 * x.dims()[2], name="conv2_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv3_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv3_1x1", **conv_config())
        deepnet.end_group()

        deepnet.begin_group("div8")
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[2, 2], name="conv4_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, 2 * x.dims()[2], name="conv4_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv5_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv5_1x1", **conv_config())
        deepnet.end_group()

        deepnet.begin_group("div16")
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[2, 2], name="conv6_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, 2 * x.dims()[2], name="conv6_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv7_1_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv7_1_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv7_2_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv7_2_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv7_3_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv7_3_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv7_4_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv7_4_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv7_5_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv7_5_1x1", **conv_config())
        deepnet.end_group()

        deepnet.begin_group("div32")
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[2, 2], name="conv8_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, 2 * x.dims()[2], name="conv8_1x1", **conv_config())
        x = ConvDepthWise(x, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[1, 1], name="conv9_3x3_dw",
                          **conv_config())
        x = ConvPointWise(x, x.dims()[2], name="conv9_1x1", **conv_config())
        deepnet.end_group()


        Sequence.__init__(self, [

        ])

    def __call__(self, x):





class MobileNet_v1_FeatureExtractor_BN(Group):
    def __init__(self, inputs, alpha):

        self._deepNet = DeepNet()

        Group.__init__(self, [], name="extractor")

        self.add(Conv(inputs, int(32 * alpha), kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[2, 2],
             **conv_config_bn(), deepNet=self._deepNet, name="conv1"))
        self.add(BatchNorm2d(self, int(32 * alpha), activationFunction=Rectifier(), name="bn1"))
        self.add(ConvDepthWise(self, int(32 * alpha), 1, name="conv1_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(32 * alpha), activationFunction=Rectifier(), name="bn1_3x3_dw"))
        self.add(ConvPointWise(self, int(64 * alpha), name="conv1_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(64 * alpha), activationFunction=Rectifier(), name="bn1_1x1"))
        self.add(ConvDepthWise(self, int(64 * alpha), 2, name="conv2_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(64 * alpha), activationFunction=Rectifier(), name="bn2_3x3_dw"))
        self.add(ConvPointWise(self, int(128 * alpha), name="conv2_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(128 * alpha), activationFunction=Rectifier(), name="bn2_1x1"))
        self.add(ConvDepthWise(self, int(128 * alpha), 1, name="conv3_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(128 * alpha), activationFunction=Rectifier(), name="bn3_3x3_dw"))
        self.add(ConvPointWise(self, int(128 * alpha), name="conv3_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(128 * alpha), activationFunction=Rectifier(), name="bn3_1x1"))
        self.add(ConvDepthWise(self, int(128 * alpha), 2, name="conv4_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(128 * alpha), activationFunction=Rectifier(), name="bn4_3x3_dw"))
        self.add(ConvPointWise(self, int(256 * alpha), name="conv4_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(256 * alpha), activationFunction=Rectifier(), name="bn4_1x1"))
        self.add(ConvDepthWise(self, int(256 * alpha), 1, name="conv5_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(256 * alpha), activationFunction=Rectifier(), name="bn5_3x3_dw"))
        self.add(ConvPointWise(self, int(256 * alpha), name="conv5_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(256 * alpha), activationFunction=Rectifier(), name="bn5_1x1"))
        self.add(ConvDepthWise(self, int(256 * alpha), 2, name="conv6_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(256 * alpha), activationFunction=Rectifier(), name="bn6_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv6_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn6_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_1_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_1_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_2_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_2_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_3_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_3_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_4_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_4_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_5_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_5_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 2, name="conv8_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activationFunction=Rectifier(), name="bn8_3x3_dw"))
        self.add(ConvPointWise(self, int(1024 * alpha), name="conv8_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(1024 * alpha), activationFunction=Rectifier(), name="bn8_1x1"))
        self.add(ConvDepthWise(self, int(1024 * alpha), 1, name="conv9_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(1024 * alpha), activationFunction=Rectifier(), name="bn9_3x3_dw"))
        self.add(ConvPointWise(self, int(1024 * alpha), name="conv9_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(1024 * alpha), activationFunction=Rectifier(), name="bn9_1x1"))
        self.add(GlobalPool2d(self, int(1024 * alpha), pooling='Average', name="pool1"))




def create_mobilenetv1_head(inputs, nb_outputs):

    # We need to identify the deepnet to add groups
    # If a DeepNet is provided as input, add new cell to existing DeepNet
    # If Provider is given, new create new DeepNet object
    if isinstance(inputs, n2d2.provider.Provider):
        deepnet = DeepNet(inputs)
    elif isinstance(inputs, n2d2.deepnet.DeepNet):
        deepnet = inputs
    else:
        raise ValueError("Needs Provider or DeepNet as input")
    x = GlobalPool2d(deepnet, pooling='Average', name="pool1")
    x = Fc(x, nbOutputs=nb_outputs, activationFunction=Linear(), weightsFiller=Xavier(),
            biasFiller=Constant(value=0.0), name="fc")
    Softmax(x, withLoss=True, name="softmax")

    return deepnet



class MobileNet_v1(DeepNet):
    def __init__(self, inputs, nb_outputs=1000, alpha=1.0, with_batchnorm=False):

        DeepNet.__init__(self, inputs, name="MobileNetv1")

        self.begin_group("extractor")
        if with_batchnorm:
            create_mobilenetv1_extractor(self, alpha)
        else:
            create_mobilenetv1_extractor(self, alpha)
        self.end_group()

        self.begin_group("head")
        create_mobilenetv1_head(self, nb_outputs)
        self.end_group()


    """
    def set_ILSVRC_solvers(self, max_iterations):
        print("Add solvers")
        learning_rate = 0.1

        solver_config = ConfigSection(momentum=0.9, learningRatePolicy='PolyDecay', power=1.0, maxIterations=max_iterations)
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




def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)



def load_from_ONNX(dims=None, batch_size=1, path=None, download=False):
    raise RuntimeError("Not implemented")