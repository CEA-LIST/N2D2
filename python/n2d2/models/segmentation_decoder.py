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
from n2d2.solver import SGD
from n2d2.filler import Constant, Normal


decoder_solver_config = ConfigSection(learning_rate_policy='CosineDecay', learning_rate=0.01, momentum=0.9,
                              decay=0.00004, warm_up_duration=0, max_iterations=59500, polyak_momentum=False)

class DecoderConv(Conv):
    def __init__(self, nb_inputs, nb_outputs, **config_parameters):
        Conv.__init__(self, nb_inputs, nb_outputs, kernel_dims=[1, 1], stride_dims=[1, 1], activation_function=Tanh(), back_propagate=False,
                    weights_filler=Normal(std_dev=0.1), bias_filler=Constant(value=0.2),
                    weights_solver=SGD(**decoder_solver_config), bias_solver=SGD(**decoder_solver_config),
                    **config_parameters)


class DecoderDeconv(Deconv):
    def __init__(self, nb_inputs, nb_outputs, **config_parameters):
        Deconv.__init__(self, nb_inputs, nb_outputs, activation_function=Linear(), kernel_dims=[4, 4], stride_dims=[2, 2],
                    weights_filler=Normal(std_dev=0.1), bias_filler=Constant(value=0.2),
                    weights_solver=SGD(**decoder_solver_config), bias_solver=SGD(**decoder_solver_config),
                    **config_parameters)


class DecoderPadding(Padding):
    def __init__(self, **config_parameters):
        Padding.__init__(self, top_pad=-1, bot_pad=-1, left_pad=-1, right_pad=-1, **config_parameters)


class DecoderFuse(ElemWise):
    def __init__(self, **config_parameters):
        ElemWise.__init__(self, operation='Sum', **config_parameters)


class SegmentationDecoder(Block):
    def __init__(self, nb_inputs):

        if not len(nb_inputs) == 4:
            RuntimeError("'nb_inputs' needs exactly 4 elements.")

        Block.__init__(self, [
            DecoderConv(nb_inputs[0], 5, name="conv_1x1_x4"),
            DecoderConv(nb_inputs[1], 5, name="conv_1x1_x8"),
            DecoderConv(nb_inputs[2], 5, name="conv_1x1_x16"),
            DecoderConv(nb_inputs[3], 5, name="conv_1x1_x32"),

            DecoderDeconv(5, 5, name="deconv1"),
            DecoderPadding(name="deconv1_pad"),
            DecoderFuse(name="fuse1"),

            DecoderDeconv(5, 5, name="deconv_fuse1"),
            DecoderPadding(name="deconv_fuse1_pad"),
            DecoderFuse(name="fuse2"),

            DecoderDeconv(5, 5, name="deconv_fuse2"),
            DecoderPadding(name="deconv_fuse2_pad"),
            DecoderFuse(name="fuse3"),

            Deconv(5, 5, kernel_dims=[8, 8], stride_dims=[4, 4], activation_function=Linear(),
                  weights_filler=Normal(std_dev=0.1), bias_filler=Constant(value=0.2),
                  weights_solver=SGD(**decoder_solver_config), bias_solver=SGD(**decoder_solver_config),
                  name="deconv_fuse3"),
            Padding(top_pad=-2, bot_pad=-2, left_pad=-2, right_pad=-2, name="out_adapt")
        ])

    def __call__(self, inputs):

        if not len(inputs) == 4:
            RuntimeError("'inputs' needs exactly 4 elements.")

        # post_backbone_convs
        x_x4 = self["conv_1x1_x4"](inputs[0])
        x_x8 = self["conv_1x1_x8"](inputs[1])
        x_x16 = self["conv_1x1_x16"](inputs[2])
        x_x32 = self["conv_1x1_x32"](inputs[3])

        # deconv_sequence1
        x = self["deconv1"](x_x32)
        x = self["deconv1_pad"](x)
        x = self["fuse1"]([x, x_x16])

        # deconv_sequence2
        x = self["deconv_fuse1"](x)
        x = self["deconv_fuse1_pad"](x)
        x = self["fuse2"]([x, x_x8])

        # deconv_sequence3
        x = self["deconv_fuse2"](x)
        x = self["deconv_fuse2_pad"](x)
        x = self["fuse3"]([x, x_x4])

        # deconv_sequence4
        x = self["deconv_fuse3"](x)
        x = self["out_adapt"](x)

        return x


    """
    # Note: Not functional
    def set_Cityscapes_solvers(self, max_iterations):
        print("Add solvers")


        #solver_config = ConfigSection(learning_rate_policy='CosineDecay', learning_rate=0.01, momentum=0.9,
        #                decay=0.00004, warm_up_duration=0, max_iterations=max_iterations, polyak_momentum=False)

        solver_config = ConfigSection(learning_rate=0.03)

        weights_solver = SGD
        weights_solver_config = solver_config
        bias_solver = SGD
        bias_solver_config = solver_config

        for name, cells in self.get_cells().items():
            if isinstance(cells, Conv) or isinstance(cells, Deconv):
                print("Add solver: " + name)
                cells.set_weights_solver(SGD(**weights_solver_config))
                cells.set_bias_solver(SGD(**bias_solver_config))
    """



