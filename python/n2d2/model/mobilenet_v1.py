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
from n2d2.cell import Fc, Conv, Conv2D, Softmax, Pool2D, BatchNorm
from n2d2.deepnet import Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant


class ConvConfig(ConfigSection):
    def __init__(self):
        ConfigSection.__init__(self, activationFunction=Rectifier(), weightsFiller=He(), noBias=True)

class ConvConfigBN(ConfigSection):
    def __init__(self):
        ConfigSection.__init__(self, activationFunction=Linear(), weightsFiller=He(), noBias=True)

class ConvDepthWise(Conv2D):
    def __init__(self, nb_outputs, stride, **config_parameters):
        Conv2D.__init__(self, nbOutputs=nb_outputs, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[stride, stride], **config_parameters)

class ConvElemWise(Conv):
    def __init__(self, nb_outputs, **config_parameters):
        Conv.__init__(self, nbOutputs=nb_outputs, kernelDims=[1, 1], strideDims=[1, 1], **config_parameters)

class MobileNet_v1_FeatureExtractor(Sequence):
    def __init__(self, alpha, size):
        config = ConvConfig

        Sequence.__init__(self, [
            Conv(nbOutputs=int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
                 **config().get(), name="conv1"),
            ConvDepthWise(int(32 * alpha), 1, name="conv1_3x3_dw", **config().get()),
            ConvElemWise(int(64 * alpha), name="conv1_1x1", **config().get()),
            ConvDepthWise(int(64 * alpha), 2, name="conv2_3x3_dw", **config().get()),
            ConvElemWise(int(128 * alpha), name="conv2_1x1", **config().get()),
            ConvDepthWise(int(128 * alpha), 1, name="conv3_3x3_dw", **config().get()),
            ConvElemWise(int(128 * alpha), name="conv3_1x1", **config().get()),
            ConvDepthWise(int(128 * alpha), 2, name="conv4_3x3_dw", **config().get()),
            ConvElemWise(int(256 * alpha), name="conv4_1x1", **config().get()),
            ConvDepthWise(int(256 * alpha), 1, name="conv5_3x3_dw", **config().get()),
            ConvElemWise(int(256 * alpha), name="conv5_1x1", **config().get()),
            ConvDepthWise(int(256 * alpha), 2, name="conv6_3x3_dw", **config().get()),
            ConvElemWise(int(512 * alpha), name="conv6_1x1", **config().get()),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_1_3x3_dw", **config().get()),
            ConvElemWise(int(512 * alpha), name="conv7_1_1x1", **config().get()),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_2_3x3_dw", **config().get()),
            ConvElemWise(int(512 * alpha), name="conv7_2_1x1", **config().get()),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_3_3x3_dw", **config().get()),
            ConvElemWise(int(512 * alpha), name="conv7_3_1x1", **config().get()),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_4_3x3_dw", **config().get()),
            ConvElemWise(int(512 * alpha), name="conv7_4_1x1", **config().get()),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_5_3x3_dw", **config().get()),
            ConvElemWise(int(512 * alpha), name="conv7_5_1x1", **config().get()),
            ConvDepthWise(int(512 * alpha), 2, name="conv8_3x3_dw", **config().get()),
            ConvElemWise(int(1024 * alpha), name="conv8_1x1", **config().get()),
            ConvDepthWise(int(1024 * alpha), 1, name="conv9_3x3_dw", **config().get()),
            ConvElemWise(int(1024 * alpha), name="conv9_1x1", **config().get()),
            Pool2D(nbOutputs=int(1024 * alpha), poolDims=[size//32, size//32], strideDims=[1, 1], pooling='Average', name="pool1")
        ], name="extractor")


class MobileNet_v1_FeatureExtractor_BN(Sequence):
    def __init__(self, alpha, size):
        config = ConvConfigBN

        Sequence.__init__(self, [
            Conv(nbOutputs=int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
                 **config().get(), name="conv1"),
            BatchNorm(nbOutputs=int(32 * alpha), activationFunction=Rectifier(), name="bn1"),
            ConvDepthWise(int(32 * alpha), 1, name="conv1_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(32 * alpha), activationFunction=Rectifier(), name="bn1_3x3_dw"),
            ConvElemWise(int(64 * alpha), name="conv1_1x1", **config().get()),
            BatchNorm(nbOutputs=int(64 * alpha), activationFunction=Rectifier(), name="bn1_1x1"),
            ConvDepthWise(int(64 * alpha), 2, name="conv2_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(64 * alpha), activationFunction=Rectifier(), name="bn2_3x3_dw"),
            ConvElemWise(int(128 * alpha), name="conv2_1x1", **config().get()),
            BatchNorm(nbOutputs=int(128 * alpha), activationFunction=Rectifier(), name="bn2_1x1"),
            ConvDepthWise(int(128 * alpha), 1, name="conv3_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(128 * alpha), activationFunction=Rectifier(), name="bn3_3x3_dw"),
            ConvElemWise(int(128 * alpha), name="conv3_1x1", **config().get()),
            BatchNorm(nbOutputs=int(128 * alpha), activationFunction=Rectifier(), name="bn3_1x1"),
            ConvDepthWise(int(128 * alpha), 2, name="conv4_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(128 * alpha), activationFunction=Rectifier(), name="bn4_3x3_dw"),
            ConvElemWise(int(256 * alpha), name="conv4_1x1", **config().get()),
            BatchNorm(nbOutputs=int(256 * alpha), activationFunction=Rectifier(), name="bn4_1x1"),
            ConvDepthWise(int(256 * alpha), 1, name="conv5_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(256 * alpha), activationFunction=Rectifier(), name="bn5_3x3_dw"),
            ConvElemWise(int(256 * alpha), name="conv5_1x1", **config().get()),
            BatchNorm(nbOutputs=int(256 * alpha), activationFunction=Rectifier(), name="bn5_1x1"),
            ConvDepthWise(int(256 * alpha), 2, name="conv6_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(256 * alpha), activationFunction=Rectifier(), name="bn6_3x3_dw"),
            ConvElemWise(int(512 * alpha), name="conv6_1x1", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn6_1x1"),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_1_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_3x3_dw"),
            ConvElemWise(int(512 * alpha), name="conv7_1_1x1", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_1x1"),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_2_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_3x3_dw"),
            ConvElemWise(int(512 * alpha), name="conv7_2_1x1", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_1x1"),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_3_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_3x3_dw"),
            ConvElemWise(int(512 * alpha), name="conv7_3_1x1", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_1x1"),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_4_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_3x3_dw"),
            ConvElemWise(int(512 * alpha), name="conv7_4_1x1", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_1x1"),
            ConvDepthWise(int(512 * alpha), 1, name="conv7_5_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_3x3_dw"),
            ConvElemWise(int(512 * alpha), name="conv7_5_1x1", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_1x1"),
            ConvDepthWise(int(512 * alpha), 2, name="conv8_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(512 * alpha), activationFunction=Rectifier(), name="bn8_3x3_dw"),
            ConvElemWise(int(1024 * alpha), name="conv8_1x1", **config().get()),
            BatchNorm(nbOutputs=int(1024 * alpha), activationFunction=Rectifier(), name="bn8_1x1"),
            ConvDepthWise(int(1024 * alpha), 1, name="conv9_3x3_dw", **config().get()),
            BatchNorm(nbOutputs=int(1024 * alpha), activationFunction=Rectifier(), name="bn9_3x3_dw"),
            ConvElemWise(int(1024 * alpha), name="conv9_1x1", **config().get()),
            BatchNorm(nbOutputs=int(1024 * alpha), activationFunction=Rectifier(), name="bn9_1x1"),
            Pool2D(nbOutputs=int(1024 * alpha), poolDims=[size//32, size//32], strideDims=[1, 1], pooling='Average', name="pool1")
        ], name="extractor")

class Mobilenet_v1(Sequence):
    def __init__(self, output_size=1000, alpha=1.0, size=224, with_batchnorm=False):

        self._with_batchnorm = with_batchnorm
        if self._with_batchnorm:
            extractor = MobileNet_v1_FeatureExtractor_BN(alpha, size)
        else:
            extractor = MobileNet_v1_FeatureExtractor(alpha, size)

        classifier = Sequence([
            Fc(nbOutputs=output_size, activationFunction=Linear(), weightsFiller=Xavier(), biasFiller=Constant(value=0.0), name="fc"),
            Softmax(nbOutputs=output_size, withLoss=True)
        ], name="classifier")

        Sequence.__init__(self, [extractor, classifier])


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

            if self._with_batchnorm and isinstance(cell, BatchNorm):
                cell.set_scale_solver(bn_solver(**bn_solver_config.get()))
                cell.set_bias_solver(bn_solver(**bn_solver_config.get()))





def load_from_ONNX(dims=None, batch_size=1, path=None, download=False):
    raise RuntimeError("Not implemented")