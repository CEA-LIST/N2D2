"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
from n2d2.cells.nn import Conv, Deconv, Padding, ElemWise
from n2d2.cells import Block
from n2d2.activation import Linear, Tanh
from n2d2.filler import Constant, Normal
from n2d2.tensor import Interface


class DecoderConv(Conv):
    def __init__(self, nb_inputs, nb_outputs, **config_parameters):
        Conv.__init__(self, nb_inputs, nb_outputs, kernel_dims=[1, 1], stride_dims=[1, 1], activation=Tanh(), back_propagate=False,
                    weights_filler=Normal(std_dev=0.1), bias_filler=Constant(value=0.2),
                    **config_parameters)


class DecoderDeconv(Deconv):
    def __init__(self, nb_inputs, nb_outputs, **config_parameters):
        Deconv.__init__(self, nb_inputs, nb_outputs, activation=Linear(), kernel_dims=[4, 4], stride_dims=[2, 2],
                    weights_filler=Normal(std_dev=0.1), bias_filler=Constant(value=0.2),
                    **config_parameters)


class DecoderPadding(Padding):
    def __init__(self, **config_parameters):
        Padding.__init__(self, top_pad=-1, bot_pad=-1, left_pad=-1, right_pad=-1, **config_parameters)


class DecoderFuse(ElemWise):
    def __init__(self, **config_parameters):
        ElemWise.__init__(self, operation='Sum', **config_parameters)


class SegmentationNetwork(Block):
    def __init__(self, backbone, features, nb_channels):

        self.backbone_features = features
        self.backbone = backbone

        if not len(nb_channels) == 4:
            RuntimeError("'nb_channels' needs exactly 4 elements.")

        self.decoder = Block([
            DecoderConv(nb_channels[0], 5, name="conv_1x1_x4"),
            DecoderConv(nb_channels[1], 5, name="conv_1x1_x8"),
            DecoderConv(nb_channels[2], 5, name="conv_1x1_x16"),
            DecoderConv(nb_channels[3], 5, name="conv_1x1_x32"),

            DecoderDeconv(5, 5, name="deconv1"),
            DecoderPadding(name="deconv1_pad"),
            DecoderFuse(name="fuse1"),

            DecoderDeconv(5, 5, name="deconv_fuse1"),
            DecoderPadding(name="deconv_fuse1_pad"),
            DecoderFuse(name="fuse2"),

            DecoderDeconv(5, 5, name="deconv_fuse2"),
            DecoderPadding(name="deconv_fuse2_pad"),
            DecoderFuse(name="fuse3"),

            Deconv(5, 5, kernel_dims=[8, 8], stride_dims=[4, 4], activation=Linear(),
                  weights_filler=Normal(std_dev=0.1), bias_filler=Constant(value=0.2),
                  name="deconv_fuse3"),
            Padding(top_pad=-2, bot_pad=-2, left_pad=-2, right_pad=-2, name="out_adapt")
        ], name="decoder")

        super().__init__([self.backbone, self.decoder])

    def __call__(self, x):
        super().__call__(x)

        # No need to get output since we use features as different levels
        self.backbone(x)

        # post_backbone_convs
        x_x4 = self.decoder["conv_1x1_x4"](self.backbone_features[0].get_outputs())
        x_x8 = self.decoder["conv_1x1_x8"](self.backbone_features[1].get_outputs())
        x_x16 = self.decoder["conv_1x1_x16"](self.backbone_features[2].get_outputs())
        x_x32 = self.decoder["conv_1x1_x32"](self.backbone_features[3].get_outputs())

        # deconv_sequence1
        x = self.decoder["deconv1"](x_x32)
        x = self.decoder["deconv1_pad"](x)
        x = self.decoder["fuse1"](Interface([x, x_x16]))

        # deconv_sequence2
        x = self.decoder["deconv_fuse1"](x)
        x = self.decoder["deconv_fuse1_pad"](x)
        x = self.decoder["fuse2"](Interface([x, x_x8]))

        # deconv_sequence3
        x = self.decoder["deconv_fuse2"](x)
        x = self.decoder["deconv_fuse2_pad"](x)
        x = self.decoder["fuse3"](Interface([x, x_x4]))

        # deconv_sequence4
        x = self.decoder["deconv_fuse3"](x)
        x = self.decoder["out_adapt"](x)

        return x

