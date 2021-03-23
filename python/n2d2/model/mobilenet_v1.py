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
from n2d2.cell import Fc, Conv, ConvDepthWise, ConvPointWise, Softmax, GlobalPool2D, BatchNorm
from n2d2.deepnet import Group, DeepNet
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.global_variables
from n2d2.model.ILSVRC_outils import ILSVRC_preprocessing


def conv_config():
    return ConfigSection(activationFunction=Rectifier(), weightsFiller=He(), noBias=True)

def conv_config_bn():
    return ConfigSection(activationFunction=Linear(), weightsFiller=He(), noBias=True)


class MobileNet_v1_FeatureExtractor(Group):
    def __init__(self, inputs, alpha):

        self._deepNet = DeepNet()

        seq = Group([], name='div2')
        seq.add(Conv(inputs, nbOutputs=int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
             **conv_config(), deepNet=self._deepNet, name="conv1"))
        seq.add(ConvDepthWise(seq, kernelDims=[3, 3], strideDims=[1, 1], name="conv1_3x3_dw", **conv_config()))
        seq.add(ConvPointWise(seq, 2*seq.dims()[2], name="conv1_1x1", **conv_config()))

        seq1 = Group([], name='div4')
        seq1.add(ConvDepthWise(seq, kernelDims=[3, 3], strideDims=[2, 2], name="conv2_3x3_dw", **conv_config()))
        seq1.add(ConvPointWise(seq1, seq1.dims()[2], name="conv2_1x1", **conv_config()))
        seq1.add(ConvDepthWise(seq1, kernelDims=[3, 3], strideDims=[1, 1], name="conv3_3x3_dw", **conv_config()))
        seq1.add(ConvPointWise(seq1, 2*seq1.dims()[2], name="conv3_1x1", **conv_config()))

        seq2 = Group([], name='div8')
        seq2.add(ConvDepthWise(seq1, kernelDims=[3, 3], strideDims=[2, 2], name="conv4_3x3_dw", **conv_config()))
        seq2.add(ConvPointWise(seq2, seq2.dims()[2], name="conv4_1x1", **conv_config()))
        seq2.add(ConvDepthWise(seq2, kernelDims=[1, 1],strideDims=[1, 1], name="conv5_3x3_dw", **conv_config()))
        seq2.add(ConvPointWise(seq2, 2*seq2.dims()[2], name="conv5_1x1", **conv_config()))

        seq3 = Group([], name='div16')
        seq3.add(ConvDepthWise(seq2, kernelDims=[3, 3], strideDims=[2, 2], name="conv6_3x3_dw", **conv_config()))
        seq3.add(ConvPointWise(seq3, seq3.dims()[2], name="conv6_1x1", **conv_config()))
        seq3.add(ConvDepthWise(seq3, kernelDims=[3, 3], strideDims=[1, 1], name="conv7_1_3x3_dw", **conv_config()))
        seq3.add(ConvPointWise(seq3, seq3.dims()[2], name="conv7_1_1x1", **conv_config()))
        seq3.add(ConvDepthWise(seq3, kernelDims=[3, 3], strideDims=[1, 1], name="conv7_2_3x3_dw", **conv_config()))
        seq3.add(ConvPointWise(seq3, seq3.dims()[2], name="conv7_2_1x1", **conv_config()))
        seq3.add(ConvDepthWise(seq3, kernelDims=[3, 3], strideDims=[1, 1], name="conv7_3_3x3_dw", **conv_config()))
        seq3.add(ConvPointWise(seq3, seq3.dims()[2], name="conv7_3_1x1", **conv_config()))
        seq3.add(ConvDepthWise(seq3, kernelDims=[3, 3], strideDims=[1, 1], name="conv7_4_3x3_dw", **conv_config()))
        seq3.add(ConvPointWise(seq3, seq3.dims()[2], name="conv7_4_1x1", **conv_config()))
        seq3.add(ConvDepthWise(seq3, kernelDims=[3, 3], strideDims=[1, 1], name="conv7_5_3x3_dw", **conv_config()))
        seq3.add(ConvPointWise(seq3, seq3.dims()[2], name="conv7_5_1x1", **conv_config()))

        seq4 = Group([], name='div32')
        seq4.add(ConvDepthWise(seq3, kernelDims=[3, 3], strideDims=[2, 2], name="conv8_3x3_dw", **conv_config()))
        seq4.add(ConvPointWise(seq4, 2*seq4.dims()[2], name="conv8_1x1", **conv_config()))
        seq4.add(ConvDepthWise(seq4, kernelDims=[3, 3], strideDims=[1, 1], name="conv9_3x3_dw", **conv_config()))
        seq4.add(ConvPointWise(seq4, seq4.dims()[2], name="conv9_1x1", **conv_config()))

        pool = GlobalPool2D(seq4, pooling='Average', name="pool1")

        self.scales = {}
        name = str('div2')
        self.scales[name] = Group([seq], name=name)
        name = str('div4')
        self.scales[name] = Group([seq, seq1], name=name)
        name = str('div8')
        self.scales[name] = Group([seq, seq1, seq2], name=name)
        name = str('div16')
        self.scales[name] = Group([seq, seq1, seq2, seq3], name=name)
        name = str('div32')
        self.scales[name] = Group([seq, seq1, seq2, seq3, seq4], name=name)

        Group.__init__(self, [seq, seq1, seq2, seq3, seq4, pool], name="extractor")



class MobileNet_v1_FeatureExtractor_BN(Group):
    def __init__(self, inputs, alpha):

        self._deepNet = DeepNet()

        Group.__init__(self, [], name="extractor")

        self.add(Conv(inputs, int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
             **conv_config_bn(), deepNet=self._deepNet, name="conv1"))
        self.add(BatchNorm(self, int(32 * alpha), activationFunction=Rectifier(), name="bn1"))
        self.add(ConvDepthWise(self, int(32 * alpha), 1, name="conv1_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(32 * alpha), activationFunction=Rectifier(), name="bn1_3x3_dw"))
        self.add(ConvPointWise(self, int(64 * alpha), name="conv1_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(64 * alpha), activationFunction=Rectifier(), name="bn1_1x1"))
        self.add(ConvDepthWise(self, int(64 * alpha), 2, name="conv2_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(64 * alpha), activationFunction=Rectifier(), name="bn2_3x3_dw"))
        self.add(ConvPointWise(self, int(128 * alpha), name="conv2_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(128 * alpha), activationFunction=Rectifier(), name="bn2_1x1"))
        self.add(ConvDepthWise(self, int(128 * alpha), 1, name="conv3_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(128 * alpha), activationFunction=Rectifier(), name="bn3_3x3_dw"))
        self.add(ConvPointWise(self, int(128 * alpha), name="conv3_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(128 * alpha), activationFunction=Rectifier(), name="bn3_1x1"))
        self.add(ConvDepthWise(self, int(128 * alpha), 2, name="conv4_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(128 * alpha), activationFunction=Rectifier(), name="bn4_3x3_dw"))
        self.add(ConvPointWise(self, int(256 * alpha), name="conv4_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(256 * alpha), activationFunction=Rectifier(), name="bn4_1x1"))
        self.add(ConvDepthWise(self, int(256 * alpha), 1, name="conv5_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(256 * alpha), activationFunction=Rectifier(), name="bn5_3x3_dw"))
        self.add(ConvPointWise(self, int(256 * alpha), name="conv5_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(256 * alpha), activationFunction=Rectifier(), name="bn5_1x1"))
        self.add(ConvDepthWise(self, int(256 * alpha), 2, name="conv6_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(256 * alpha), activationFunction=Rectifier(), name="bn6_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv6_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn6_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_1_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_1_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_2_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_2_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_3_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_3_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_4_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_4_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_5_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_5_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 2, name="conv8_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(512 * alpha), activationFunction=Rectifier(), name="bn8_3x3_dw"))
        self.add(ConvPointWise(self, int(1024 * alpha), name="conv8_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(1024 * alpha), activationFunction=Rectifier(), name="bn8_1x1"))
        self.add(ConvDepthWise(self, int(1024 * alpha), 1, name="conv9_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm(self, int(1024 * alpha), activationFunction=Rectifier(), name="bn9_3x3_dw"))
        self.add(ConvPointWise(self, int(1024 * alpha), name="conv9_1x1", **conv_config_bn()))

        self.add(BatchNorm(self, int(1024 * alpha), activationFunction=Rectifier(), name="bn9_1x1"))
        self.add(GlobalPool2D(self, int(1024 * alpha), pooling='Average', name="pool1"))




class MobileNet_v1_head(Group):
    def __init__(self, inputs, nb_outputs, deepnet):

        fc = Fc(inputs, nbOutputs=nb_outputs, activationFunction=Linear(), weightsFiller=Xavier(),
                biasFiller=Constant(value=0.0), name="fc", deepNet=deepnet)
        softmax = Softmax(fc, nbOutputs=nb_outputs, withLoss=True, name="softmax")
        Group.__init__(self, [fc, softmax], name="head")



class MobileNet_v1(Group):
    def __init__(self, inputs, nb_outputs=1000, alpha=1.0, with_batchnorm=False, extractor_as_deepNet=False):

        self._with_batchnorm = with_batchnorm
        if self._with_batchnorm:
            self.extractor = MobileNet_v1_FeatureExtractor_BN(inputs, alpha)
        else:
            self.extractor = MobileNet_v1_FeatureExtractor(inputs, alpha)

        if extractor_as_deepNet:
            head_deepNet = DeepNet()
            head_input = n2d2.provider.DataProvider(n2d2.database.Database(),
                    self.extractor.dims()[0:2],
                    batchSize=self.extractor.dims()[3], streamTensor=True)
            head_input.N2D2().setStreamedTensor(self.extractor.get_outputs())
        else:
            head_deepNet = self.extractor.get_deepnet()
            head_input = self.extractor
        #n2d2.global_variables.default_deepNet = head_deepNet

        #fc = Fc(head_input, nbOutputs=nb_outputs, activationFunction=Linear(), weightsFiller=Xavier(), biasFiller=Constant(value=0.0), name="fc", deepNet=head_deepNet)
        #softmax = Softmax(fc, nbOutputs=nb_outputs, withLoss=True, name="softmax", deepNet=head_deepNet)
        #self.head = Group([fc, softmax], name="head")

        self.head = MobileNet_v1_head(head_input, nb_outputs, head_deepNet)

        Group.__init__(self, [self.extractor, self.head])

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

            if self._with_batchnorm and isinstance(cell, BatchNorm):
                cell.set_scale_solver(bn_solver(**bn_solver_config.get()))
                cell.set_bias_solver(bn_solver(**bn_solver_config.get()))
    """




def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)



def load_from_ONNX(dims=None, batch_size=1, path=None, download=False):
    raise RuntimeError("Not implemented")