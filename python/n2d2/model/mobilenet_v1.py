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
    return ConfigSection(activation_function=Rectifier(), weights_filler=He(), no_bias=True)

def conv_config_bn():
    return ConfigSection(activation_function=Linear(), weights_filler=He(), no_bias=True)



class MobileNetv1Extractor(Sequence):
    def __init__(self, alpha):

        base_nb_outputs = int(32 * alpha)

        self.div2 = Sequence([
            Conv(3, base_nb_outputs, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1],
                 **conv_config(), name="conv1"),
            ConvDepthWise(base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv1_3x3_dw",
                          **conv_config()),
            ConvPointWise(base_nb_outputs, 2 * base_nb_outputs, name="conv1_1x1", **conv_config())
        ], "div2")

        self.div4 = Sequence([
            ConvDepthWise(2 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv2_3x3_dw",
                          **conv_config()),
            ConvPointWise(2 * base_nb_outputs, 4 * base_nb_outputs, name="conv2_1x1", **conv_config()),
            ConvDepthWise(4 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv3_3x3_dw",
                          **conv_config()),
            ConvPointWise(4 * base_nb_outputs, 4 * base_nb_outputs, name="conv3_1x1", **conv_config()),
        ], "div4")

        self.div8 = Sequence([
            ConvDepthWise(4 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv4_3x3_dw",
                              **conv_config()),
            ConvPointWise(4 * base_nb_outputs, 8 * base_nb_outputs, name="conv4_1x1", **conv_config()),
            ConvDepthWise(8 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv5_3x3_dw",
                          **conv_config()),
            ConvPointWise(8 * base_nb_outputs, 8 * base_nb_outputs, name="conv5_1x1", **conv_config())
        ], "div8")

        self.div16 = Sequence([
            ConvDepthWise(8 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv6_3x3_dw",
                              **conv_config()),
            ConvPointWise(8 * base_nb_outputs, 16 * base_nb_outputs, name="conv6_1x1", **conv_config()),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_1_3x3_dw",
                              **conv_config()),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_1_1x1", **conv_config()),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_2_3x3_dw",
                              **conv_config()),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_2_1x1", **conv_config()),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_3_3x3_dw",
                              **conv_config()),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_3_1x1", **conv_config()),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_4_3x3_dw",
                              **conv_config()),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_4_1x1", **conv_config()),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_5_3x3_dw",
                              **conv_config()),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_5_1x1", **conv_config())
        ], "div16")

        self.div32 = Sequence([
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv8_3x3_dw",
                              **conv_config()),
            ConvPointWise(16 * base_nb_outputs, 32 * base_nb_outputs, name="conv8_1x1", **conv_config()),
            ConvDepthWise(32 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv9_3x3_dw",
                              **conv_config()),
            ConvPointWise(32 * base_nb_outputs, 32 * base_nb_outputs, name="conv9_1x1", **conv_config())
        ], "div32")

        Sequence.__init__(self, [self.div2, self.div4, self.div8, self.div16, self.div32], "extractor")




class MobileNetv1Extractor_BN(Sequence):
     def __init__(self, alpha):

        base_nb_outputs = int(32 * alpha)

        """

        self.add(Conv(inputs, int(32 * alpha), kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2],
             **conv_config_bn(), deepNet=self._deepNet, name="conv1"))
        self.add(BatchNorm2d(self, int(32 * alpha), activation_function=Rectifier(), name="bn1"))
        self.add(ConvDepthWise(self, int(32 * alpha), 1, name="conv1_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(32 * alpha), activation_function=Rectifier(), name="bn1_3x3_dw"))
        self.add(ConvPointWise(self, int(64 * alpha), name="conv1_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(64 * alpha), activation_function=Rectifier(), name="bn1_1x1"))
        self.add(ConvDepthWise(self, int(64 * alpha), 2, name="conv2_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(64 * alpha), activation_function=Rectifier(), name="bn2_3x3_dw"))
        self.add(ConvPointWise(self, int(128 * alpha), name="conv2_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(128 * alpha), activation_function=Rectifier(), name="bn2_1x1"))
        self.add(ConvDepthWise(self, int(128 * alpha), 1, name="conv3_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(128 * alpha), activation_function=Rectifier(), name="bn3_3x3_dw"))
        self.add(ConvPointWise(self, int(128 * alpha), name="conv3_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(128 * alpha), activation_function=Rectifier(), name="bn3_1x1"))
        self.add(ConvDepthWise(self, int(128 * alpha), 2, name="conv4_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(128 * alpha), activation_function=Rectifier(), name="bn4_3x3_dw"))
        self.add(ConvPointWise(self, int(256 * alpha), name="conv4_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(256 * alpha), activation_function=Rectifier(), name="bn4_1x1"))
        self.add(ConvDepthWise(self, int(256 * alpha), 1, name="conv5_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(256 * alpha), activation_function=Rectifier(), name="bn5_3x3_dw"))
        self.add(ConvPointWise(self, int(256 * alpha), name="conv5_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(256 * alpha), activation_function=Rectifier(), name="bn5_1x1"))
        self.add(ConvDepthWise(self, int(256 * alpha), 2, name="conv6_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(256 * alpha), activation_function=Rectifier(), name="bn6_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv6_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn6_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_1_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_1_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_1_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_1_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_2_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_2_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_2_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_2_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_3_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_3_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_3_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_3_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_4_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_4_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_4_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_4_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 1, name="conv7_5_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_5_3x3_dw"))
        self.add(ConvPointWise(self, int(512 * alpha), name="conv7_5_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn7_5_1x1"))
        self.add(ConvDepthWise(self, int(512 * alpha), 2, name="conv8_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(512 * alpha), activation_function=Rectifier(), name="bn8_3x3_dw"))
        self.add(ConvPointWise(self, int(1024 * alpha), name="conv8_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(1024 * alpha), activation_function=Rectifier(), name="bn8_1x1"))
        self.add(ConvDepthWise(self, int(1024 * alpha), 1, name="conv9_3x3_dw", **conv_config_bn()))
        self.add(BatchNorm2d(self, int(1024 * alpha), activation_function=Rectifier(), name="bn9_3x3_dw"))
        self.add(ConvPointWise(self, int(1024 * alpha), name="conv9_1x1", **conv_config_bn()))

        self.add(BatchNorm2d(self, int(1024 * alpha), activation_function=Rectifier(), name="bn9_1x1"))
        self.add(GlobalPool2d(self, int(1024 * alpha), pooling='Average', name="pool1"))
        
        """

        Sequence.__init__(self, [], "extractor")





class MobileNetv1Head(Sequence):

    def __init__(self, nb_outputs, alpha):

        pool = GlobalPool2d(pooling='Average', name="pool1")
        fc = Fc(32 * int(32 * alpha), nb_outputs, activation_function=Linear(), weights_filler=Xavier(),
                bias_filler=Constant(value=0.0), name="fc")

        Sequence.__init__(self, [pool, fc], "head")



class MobileNetv1(Sequence):
    def __init__(self, nb_outputs=1000, alpha=1.0, with_batchnorm=False):

        if with_batchnorm:
            self.extractor = MobileNetv1Extractor_BN(alpha)
        else:
            self.extractor = MobileNetv1Extractor(alpha)

        self.head = MobileNetv1Head(nb_outputs, alpha)

        Sequence.__init__(self, [self.extractor, self.head], "mobilenet_v1")



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