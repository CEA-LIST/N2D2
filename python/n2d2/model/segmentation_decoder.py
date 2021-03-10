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
from n2d2.cell import Conv, Deconv, Softmax, Padding, ElemWise
from n2d2.deepnet import Sequence, Layer, DeepNet
from n2d2.activation import Linear, Tanh
from n2d2.solver import SGD
from n2d2.filler import Constant, Normal
from n2d2.provider import TensorPlaceholder
import n2d2.global_variables

decoder_solver_config = ConfigSection(learningRatePolicy='CosineDecay', learningRate=0.01, momentum=0.9,
                              decay=0.00004, warmUpDuration=0, maxIterations=59500, polyakMomentum=False)

class DecoderConv(Conv):
    def __init__(self, inputs, nbOutputs, **config_parameters):
        Conv.__init__(self, inputs, nbOutputs, kernelDims=[1, 1], strideDims=[1, 1], activationFunction=Tanh(), backPropagate=False,
                    weightsFiller=Normal(stdDev=0.1), biasFiller=Constant(value=0.2),
                    weightsSolver=SGD(**decoder_solver_config.get()), biasSolver=SGD(**decoder_solver_config.get()),
                    **config_parameters)


class DecoderDeconv(Deconv):
    def __init__(self, inputs, nbOutputs, **config_parameters):
        Deconv.__init__(self, inputs, nbOutputs, activationFunction=Linear(), kernelDims=[4, 4], strideDims=[2, 2],
                    weightsFiller=Normal(stdDev=0.1), biasFiller=Constant(value=0.2),
                    weightsSolver=SGD(**decoder_solver_config.get()), biasSolver=SGD(**decoder_solver_config.get()),
                    **config_parameters)


class DecoderPadding(Padding):
    def __init__(self, inputs, nbOutputs, **config_parameters):
        Padding.__init__(self, inputs, nbOutputs,
                         topPad=-1, botPad=-1, leftPad=-1, rightPad=-1, **config_parameters)


class DecoderFuse(ElemWise):
    def __init__(self, inputs, nbOutputs, **config_parameters):
        ElemWise.__init__(self, inputs, nbOutputs, operation='Sum', **config_parameters)


class SegmentationDecoder(Sequence):
    def __init__(self, inputs_scales=None, deepNet=None):

        if not len(inputs_scales) == 4:
            RuntimeError("'input_scales' needs exactly 4 elements.")

        inputs = []
        for scale in inputs_scales:
            if isinstance(scale, list):
                size = []
                for elem in scale:
                    size.append(elem)
                inputs.append(size)
            elif isinstance(scale, n2d2.cell.Cell):
                inputs.append(scale.get_outputs())
            else:
                ValueError("'inputs_scales' has to be list of int lists or cells")

        if deepNet is None:
            self._deepNet = DeepNet()
        else:
            self._deepNet = deepNet

        interface_1x1_x4 = TensorPlaceholder(inputs[0], name="interface_1x1_x4")
        interface_1x1_x8 = TensorPlaceholder(inputs[1], name="interface_1x1_x8")
        interface_1x1_x16 = TensorPlaceholder(inputs[2], name="interface_1x1_x16")
        interface_1x1_x32 = TensorPlaceholder(inputs[3], name="interface_1x1_x32")

        # For graph visualization tools
        self._deepNet.add_provider(interface_1x1_x4)

        conv_1x1_x4 = DecoderConv(interface_1x1_x4, nbOutputs=5, name="conv_1x1_x4", deepNet=self._deepNet)
        conv_1x1_x8 = DecoderConv(interface_1x1_x8, nbOutputs=5, name="conv_1x1_x8", deepNet=self._deepNet)
        conv_1x1_x16 = DecoderConv(interface_1x1_x16, nbOutputs=5, name="conv_1x1_x16", deepNet=self._deepNet)
        conv_1x1_x32 = DecoderConv(interface_1x1_x32, nbOutputs=5, name="conv_1x1_x32", deepNet=self._deepNet)

        post_backbone_convs = Layer([conv_1x1_x4, conv_1x1_x8, conv_1x1_x16, conv_1x1_x32], name="post_backbone_convs")

        deconv1 = DecoderDeconv(conv_1x1_x32, conv_1x1_x32.get_outputs().dimZ(), name="deconv1", deepNet=self._deepNet)
        deconv1_pad = DecoderPadding(deconv1, deconv1.get_outputs().dimZ(), name="deconv1_pad", deepNet=self._deepNet)
        fuse1 = DecoderFuse([deconv1_pad, conv_1x1_x16], deconv1_pad.get_outputs().dimZ(), name="fuse1", deepNet=self._deepNet)

        deconv_sequence1 = Sequence([deconv1, deconv1_pad, fuse1], name="deconv_sequence1")

        deconv_fuse1 = DecoderDeconv(fuse1, fuse1.get_outputs().dimZ(), name="deconv_fuse1", deepNet=self._deepNet)
        deconv_fuse1_pad = DecoderPadding(deconv_fuse1, deconv_fuse1.get_outputs().dimZ(), name="deconv_fuse1_pad", deepNet=self._deepNet)
        fuse2 = DecoderFuse([deconv_fuse1_pad, conv_1x1_x8], deconv_fuse1_pad.get_outputs().dimZ(), name="fuse2", deepNet=self._deepNet)

        deconv_sequence2 = Sequence([deconv_fuse1, deconv_fuse1_pad, fuse2], name="deconv_sequence2")

        deconv_fuse2 = DecoderDeconv(fuse2, fuse2.get_outputs().dimZ(), name="deconv_fuse2", deepNet=self._deepNet)
        deconv_fuse2_pad = DecoderPadding(deconv_fuse2, deconv_fuse2.get_outputs().dimZ(), name="deconv_fuse2_pad", deepNet=self._deepNet)
        fuse3 = DecoderFuse([deconv_fuse2_pad, conv_1x1_x4], deconv_fuse2_pad.get_outputs().dimZ(), name="fuse3", deepNet=self._deepNet)

        deconv_sequence3 = Sequence([deconv_fuse2, deconv_fuse2_pad, fuse3], name="deconv_sequence3")

        deconv_fuse3 = Deconv(fuse3, fuse3.get_outputs().dimZ(), kernelDims=[8, 8], strideDims=[4, 4], activationFunction=Linear(),
                    weightsFiller=Normal(stdDev=0.1), biasFiller=Constant(value=0.2),
                    weightsSolver=SGD(**decoder_solver_config.get()), biasSolver=SGD(**decoder_solver_config.get()),
                    name="deconv_fuse3", deepNet=self._deepNet)
        out_adapt = Padding(deconv_fuse3, deconv_fuse3.get_outputs().dimZ(),
                    topPad=-2, botPad=-2, leftPad=-2, rightPad=-2, name="out_adapt", deepNet=self._deepNet)

        deconv_sequence4 = Sequence([deconv_fuse3, out_adapt], name="deconv_sequence4")

        decoder_blocks = Sequence([deconv_sequence1, deconv_sequence2, deconv_sequence3, deconv_sequence4], name="decoder_blocks")

        softmax = Softmax(out_adapt, out_adapt.get_outputs().dimZ(), withLoss=True, name="segmentation_decoder_softmax", deepNet=self._deepNet)

        Sequence.__init__(self, [post_backbone_convs, decoder_blocks, softmax], name="decoder")

    """
    # Note: Not functional
    def set_Cityscapes_solvers(self, max_iterations):
        print("Add solvers")


        #solver_config = ConfigSection(learningRatePolicy='CosineDecay', learningRate=0.01, momentum=0.9,
        #                decay=0.00004, warmUpDuration=0, maxIterations=max_iterations, polyakMomentum=False)

        solver_config = ConfigSection(learningRate=0.03)

        weights_solver = SGD
        weights_solver_config = solver_config
        bias_solver = SGD
        bias_solver_config = solver_config

        for name, cell in self.get_cells().items():
            if isinstance(cell, Conv) or isinstance(cell, Deconv):
                print("Add solver: " + name)
                cell.set_weights_solver(SGD(**weights_solver_config.get()))
                cell.set_bias_solver(SGD(**bias_solver_config.get()))
    """



