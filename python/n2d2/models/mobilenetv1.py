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
from n2d2.cells.nn import Fc, Conv, ConvDepthWise, ConvPointWise, GlobalPool2d, BatchNorm2d
from n2d2.cells import Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.filler import He, Xavier, Constant, Normal
from n2d2.models.ILSVRC_outils import ILSVRC_preprocessing

# NOTE: This is the filler optimized for quantization. In normal training He might work better
def conv_config(with_bn):
    return ConfigSection(activation=Linear() if with_bn else Rectifier(),
                         weights_filler=Xavier(variance_norm='FanOut', scaling=2.0), no_bias=True)


class MobileNetv1Extractor(Sequence):
    def __init__(self, alpha, with_bn=False):

        base_nb_outputs = int(32 * alpha)

        self.div2 = Sequence([
            Conv(3, base_nb_outputs, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1],
                 **conv_config(with_bn), name="conv1"),
            ConvDepthWise(base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv1_3x3_dw",
                          **conv_config(with_bn)),
            ConvPointWise(base_nb_outputs, 2 * base_nb_outputs, name="conv1_1x1", **conv_config(with_bn))
        ], "div2")

        self.div4 = Sequence([
            ConvDepthWise(2 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv2_3x3_dw",
                          **conv_config(with_bn)),
            ConvPointWise(2 * base_nb_outputs, 4 * base_nb_outputs, name="conv2_1x1", **conv_config(with_bn)),
            ConvDepthWise(4 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv3_3x3_dw",
                          **conv_config(with_bn)),
            ConvPointWise(4 * base_nb_outputs, 4 * base_nb_outputs, name="conv3_1x1", **conv_config(with_bn)),
        ], "div4")

        self.div8 = Sequence([
            ConvDepthWise(4 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv4_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(4 * base_nb_outputs, 8 * base_nb_outputs, name="conv4_1x1", **conv_config(with_bn)),
            ConvDepthWise(8 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv5_3x3_dw",
                          **conv_config(with_bn)),
            ConvPointWise(8 * base_nb_outputs, 8 * base_nb_outputs, name="conv5_1x1", **conv_config(with_bn))
        ], "div8")

        self.div16 = Sequence([
            ConvDepthWise(8 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv6_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(8 * base_nb_outputs, 16 * base_nb_outputs, name="conv6_1x1", **conv_config(with_bn)),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_1_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_1_1x1", **conv_config(with_bn)),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_2_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_2_1x1", **conv_config(with_bn)),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_3_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_3_1x1", **conv_config(with_bn)),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_4_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_4_1x1", **conv_config(with_bn)),
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv7_5_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(16 * base_nb_outputs, 16 * base_nb_outputs, name="conv7_5_1x1", **conv_config(with_bn))
        ], "div16")

        self.div32 = Sequence([
            ConvDepthWise(16 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[2, 2], name="conv8_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(16 * base_nb_outputs, 32 * base_nb_outputs, name="conv8_1x1", **conv_config(with_bn)),
            ConvDepthWise(32 * base_nb_outputs, kernel_dims=[3, 3], padding_dims=[1, 1], stride_dims=[1, 1], name="conv9_3x3_dw",
                              **conv_config(with_bn)),
            ConvPointWise(32 * base_nb_outputs, 32 * base_nb_outputs, name="conv9_1x1", **conv_config(with_bn))
        ], "div32")

        Sequence.__init__(self, [self.div2, self.div4, self.div8, self.div16, self.div32], "extractor")



class MobileNetv1Head(Sequence):

    def __init__(self, nb_outputs, alpha=None):

        pool = GlobalPool2d(pooling='Average', name="pool1")
        fc = Fc(32 * int(32 * alpha), nb_outputs, activation=Linear(), weights_filler=Normal(mean=0.0, std_dev=0.01),
                bias_filler=Constant(value=0.0), name="fc")

        Sequence.__init__(self, [pool, fc], "head")



class MobileNetv1(Sequence):
    def __init__(self, nb_outputs=1000, alpha=1.0, with_bn=False):

        self.extractor = MobileNetv1Extractor(alpha, with_bn)
            
        if with_bn:
            for scale in self.extractor:
                for cell in scale:
                    if isinstance(cell, Conv):
                        bn_name = "bn" + cell.get_name()[4:]
                        scale.insert(scale.index(cell)+1, BatchNorm2d(cell.get_nb_outputs(), activation=Rectifier(), name=bn_name))

        self.head = MobileNetv1Head(nb_outputs, alpha)

        Sequence.__init__(self, [self.extractor, self.head], "mobilenet_v1")



def ILSVRC_preprocessing(size=224):
   return ILSVRC_preprocessing(size)

