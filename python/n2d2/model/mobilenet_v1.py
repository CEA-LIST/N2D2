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
from n2d2.cell import Fc, Conv, ConvDepthWise, Softmax, Pool2D, BatchNorm
from n2d2.deepnet import Sequence, DeepNet
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import He, Xavier, Constant
import n2d2.global_variables
from n2d2.model.ILSVRC_outils import ILSVRC_preprocessing


def conv_config():
    return ConfigSection(activationFunction=Rectifier(), weightsFiller=He(), noBias=True)

def conv_config_bn():
    return ConfigSection(activationFunction=Linear(), weightsFiller=He(), noBias=True)

class ConvDepthWise(ConvDepthWise):
    def __init__(self, inputs, nb_outputs, stride, **config_parameters):
        ConvDepthWise.__init__(self, inputs, nb_outputs, kernelDims=[3, 3], paddingDims=[1, 1], strideDims=[stride, stride], **config_parameters)

class ConvElemWise(Conv):
    def __init__(self, inputs, nb_outputs, **config_parameters):
        Conv.__init__(self, inputs, nb_outputs, kernelDims=[1, 1], strideDims=[1, 1], **config_parameters)


class MobileNet_v1_FeatureExtractor(Sequence):
    def __init__(self, inputs, alpha):

        self._deepNet = DeepNet()

        seq = Sequence([], name='div2')
        seq.add(Conv(inputs, nbOutputs=int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
             **conv_config(), deepNet=self._deepNet, name="conv1"))
        seq.add(ConvDepthWise(seq.get_last(), int(32 * alpha), 1, name="conv1_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq.add(ConvElemWise(seq.get_last(), int(64 * alpha), name="conv1_1x1", **conv_config(), deepNet=self._deepNet))

        seq1 = Sequence([], name='div4')
        seq1.add(ConvDepthWise(seq.get_last(), int(64 * alpha), 2, name="conv2_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq1.add(ConvElemWise(seq1.get_last(), int(128 * alpha), name="conv2_1x1", **conv_config(), deepNet=self._deepNet))
        seq1.add(ConvDepthWise(seq1.get_last(), int(128 * alpha), 1, name="conv3_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq1.add(ConvElemWise(seq1.get_last(), int(128 * alpha), name="conv3_1x1", **conv_config(), deepNet=self._deepNet))

        seq2 = Sequence([], name='div8')
        seq2.add(ConvDepthWise(seq1.get_last(), int(128 * alpha), 2, name="conv4_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq2.add(ConvElemWise(seq2.get_last(), int(256 * alpha), name="conv4_1x1", **conv_config(), deepNet=self._deepNet))
        seq2.add(ConvDepthWise(seq2.get_last(), int(256 * alpha), 1, name="conv5_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq2.add(ConvElemWise(seq2.get_last(), int(256 * alpha), name="conv5_1x1", **conv_config(), deepNet=self._deepNet))

        seq3 = Sequence([], name='div16')
        seq3.add(ConvDepthWise(seq2.get_last(), int(256 * alpha), 2, name="conv6_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvElemWise(seq3.get_last(), int(512 * alpha), name="conv6_1x1", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvDepthWise(seq3.get_last(), int(512 * alpha), 1, name="conv7_1_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvElemWise(seq3.get_last(), int(512 * alpha), name="conv7_1_1x1", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvDepthWise(seq3.get_last(), int(512 * alpha), 1, name="conv7_2_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvElemWise(seq3.get_last(), int(512 * alpha), name="conv7_2_1x1", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvDepthWise(seq3.get_last(), int(512 * alpha), 1, name="conv7_3_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvElemWise(seq3.get_last(), int(512 * alpha), name="conv7_3_1x1", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvDepthWise(seq3.get_last(), int(512 * alpha), 1, name="conv7_4_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvElemWise(seq3.get_last(), int(512 * alpha), name="conv7_4_1x1", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvDepthWise(seq3.get_last(), int(512 * alpha), 1, name="conv7_5_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq3.add(ConvElemWise(seq3.get_last(), int(512 * alpha), name="conv7_5_1x1", **conv_config(), deepNet=self._deepNet))

        seq4 = Sequence([], name='div32')
        seq4.add(ConvDepthWise(seq3.get_last(), int(512 * alpha), 2, name="conv8_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq4.add(ConvElemWise(seq4.get_last(), int(1024 * alpha), name="conv8_1x1", **conv_config(), deepNet=self._deepNet))
        seq4.add(ConvDepthWise(seq4.get_last(), int(1024 * alpha), 1, name="conv9_3x3_dw", **conv_config(), deepNet=self._deepNet))
        seq4.add(ConvElemWise(seq4.get_last(), int(1024 * alpha), name="conv9_1x1", **conv_config(), deepNet=self._deepNet))

        pool = Pool2D(seq4.get_last(), nbOutputs=int(1024 * alpha),
               poolDims=[seq4.get_last().get_outputs().dimX(), seq4.get_last().get_outputs().dimY()],
               strideDims=[1, 1], pooling='Average', name="pool1", deepNet=self._deepNet)

        self.scales = {}
        name = str('div2')
        self.scales[name] = Sequence([seq], name=name)
        name = str('div4')
        self.scales[name] = Sequence([seq, seq1], name=name)
        name = str('div8')
        self.scales[name] = Sequence([seq, seq1, seq2], name=name)
        name = str('div16')
        self.scales[name] = Sequence([seq, seq1, seq2, seq3], name=name)
        name = str('div32')
        self.scales[name] = Sequence([seq, seq1, seq2, seq3, seq4], name=name)

        Sequence.__init__(self, [seq, seq1, seq2, seq3, seq4, pool], name="extractor")



class MobileNet_v1_FeatureExtractor_BN(Sequence):
    def __init__(self, inputs, alpha):

        self._deepNet = DeepNet()

        Sequence.__init__(self, [], name="extractor")

        self.add(Conv(inputs, int(32 * alpha), kernelDims=[3, 3], strideDims=[2, 2], paddingDims=[1, 1],
             **conv_config_bn(), deepNet=self._deepNet, name="conv1"))
        self.add(BatchNorm(self.get_last(), int(32 * alpha), activationFunction=Rectifier(), name="bn1"))
        self.add(ConvDepthWise(self.get_last(), int(32 * alpha), 1, name="conv1_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(32 * alpha), activationFunction=Rectifier(), name="bn1_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(64 * alpha), name="conv1_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(64 * alpha), activationFunction=Rectifier(), name="bn1_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(64 * alpha), 2, name="conv2_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(64 * alpha), activationFunction=Rectifier(), name="bn2_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(128 * alpha), name="conv2_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(128 * alpha), activationFunction=Rectifier(), name="bn2_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(128 * alpha), 1, name="conv3_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(128 * alpha), activationFunction=Rectifier(), name="bn3_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(128 * alpha), name="conv3_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(128 * alpha), activationFunction=Rectifier(), name="bn3_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(128 * alpha), 2, name="conv4_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(128 * alpha), activationFunction=Rectifier(), name="bn4_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(256 * alpha), name="conv4_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(256 * alpha), activationFunction=Rectifier(), name="bn4_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(256 * alpha), 1, name="conv5_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(256 * alpha), activationFunction=Rectifier(), name="bn5_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(256 * alpha), name="conv5_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(256 * alpha), activationFunction=Rectifier(), name="bn5_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(256 * alpha), 2, name="conv6_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(256 * alpha), activationFunction=Rectifier(), name="bn6_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(512 * alpha), name="conv6_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn6_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(512 * alpha), 1, name="conv7_1_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(512 * alpha), name="conv7_1_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_1_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(512 * alpha), 1, name="conv7_2_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(512 * alpha), name="conv7_2_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_2_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(512 * alpha), 1, name="conv7_3_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(512 * alpha), name="conv7_3_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_3_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(512 * alpha), 1, name="conv7_4_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(512 * alpha), name="conv7_4_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_4_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(512 * alpha), 1, name="conv7_5_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(512 * alpha), name="conv7_5_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn7_5_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(512 * alpha), 2, name="conv8_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(512 * alpha), activationFunction=Rectifier(), name="bn8_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(1024 * alpha), name="conv8_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(1024 * alpha), activationFunction=Rectifier(), name="bn8_1x1"))
        self.add(ConvDepthWise(self.get_last(), int(1024 * alpha), 1, name="conv9_3x3_dw", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(1024 * alpha), activationFunction=Rectifier(), name="bn9_3x3_dw"))
        self.add(ConvElemWise(self.get_last(), int(1024 * alpha), name="conv9_1x1", **conv_config_bn(), deepNet=self._deepNet))
        self.add(BatchNorm(self.get_last(), int(1024 * alpha), activationFunction=Rectifier(), name="bn9_1x1"))
        self.add(Pool2D(self.get_last(), int(1024 * alpha),
                        poolDims=[self.get_last().get_outputs().dimX(), self.get_last().get_outputs().dimY()],
                        strideDims=[1, 1], pooling='Average', name="pool1"))




class MobileNet_v1_head(Sequence):
    def __init__(self, inputs, nb_outputs, deepnet):

        fc = Fc(inputs, nbOutputs=nb_outputs, activationFunction=Linear(), weightsFiller=Xavier(),
                biasFiller=Constant(value=0.0), name="fc", deepNet=deepnet)
        softmax = Softmax(fc, nbOutputs=nb_outputs, withLoss=True, name="softmax")
        Sequence.__init__(self, [fc, softmax], name="head")



class MobileNet_v1(Sequence):
    def __init__(self, inputs, nb_outputs=1000, alpha=1.0, with_batchnorm=False, extractor_as_deepNet=False):

        self._with_batchnorm = with_batchnorm
        if self._with_batchnorm:
            self.extractor = MobileNet_v1_FeatureExtractor_BN(inputs, alpha)
        else:
            self.extractor = MobileNet_v1_FeatureExtractor(inputs, alpha)

        if extractor_as_deepNet:
            head_deepNet = DeepNet()
            head_input = n2d2.provider.DataProvider(n2d2.database.Database(),
                    [self.extractor.get_last().get_outputs().dimX(),
                     self.extractor.get_last().get_outputs().dimY(),
                     self.extractor.get_last().get_outputs().dimZ()],
                    batchSize=self.extractor.get_last().get_outputs().dimB(), streamTensor=True)
            head_input.N2D2().setStreamedTensor(self.extractor.get_last().get_outputs())
        else:
            head_deepNet = self.extractor.get_last().get_deepnet()
            head_input = self.extractor
        #n2d2.global_variables.default_deepNet = head_deepNet

        #fc = Fc(head_input, nbOutputs=nb_outputs, activationFunction=Linear(), weightsFiller=Xavier(), biasFiller=Constant(value=0.0), name="fc", deepNet=head_deepNet)
        #softmax = Softmax(fc, nbOutputs=nb_outputs, withLoss=True, name="softmax", deepNet=head_deepNet)
        #self.head = Sequence([fc, softmax], name="head")

        self.head = MobileNet_v1_head(head_input, nb_outputs, head_deepNet)

        Sequence.__init__(self, [self.extractor, self.head])

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