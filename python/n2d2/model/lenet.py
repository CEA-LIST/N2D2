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
from n2d2.cell import Fc, Conv, Softmax, Pool2D, BatchNorm, Dropout
from n2d2.deepnet import Sequence, DeepNet
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import Normal, Xavier, Constant
from n2d2.quantizer import SATCell, SATAct
import n2d2.global_variables

solver_config = ConfigSection(learningRate=0.05, momentum=0.0, decay=0.0)

def quant_conv_def():
    weights_quantizer = SATCell(applyScaling=False, applyQuantization=True, range=15)
    weights_filler = Xavier(varianceNorm='FanOut', scaling=1.0)
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activationFunction=Linear(),
                noBias=True, weightsSolver=weights_solver, biasSolver=bias_solver,
                weightsFiller=weights_filler, quantizer=weights_quantizer)

def quant_fc_def():
    weights_quantizer = SATCell(applyScaling=True, applyQuantization=True, range=15)
    sat_solver = SGD(**solver_config)
    act_quantizer = SATAct(alpha=6.0, range=15, solver=sat_solver)
    weights_filler = Normal(mean=0.0, stdDev=0.01)
    bias_filler = Constant(value=0.0) # Usually not used because of disactivated bias
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activationFunction=Linear(quantizer=act_quantizer),
                        noBias=True, weightsSolver=weights_solver, biasSolver=bias_solver,
                        weightsFiller=weights_filler, biasFiller=bias_filler,
                        quantizer=weights_quantizer)

def quant_bn_def():
    sat_solver = SGD(learningRate=0.05, momentum=0.0, decay=0.0)
    act_quantizer = SATAct(alpha=6.0, range=15, solver=sat_solver)
    scale_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activationFunction=Linear(quantizer=act_quantizer), scaleSolver=scale_solver, biasSolver=bias_solver)



class QuantLeNet(Sequence):
    def __init__(self, inputs, nb_outputs=10):

        self.deepnet = DeepNet()

        self.extractor = Sequence([], name='extractor')

        first_layer_config = quant_conv_def()
        first_layer_config['quantizer'].set_range(255)
        self.extractor.add(Conv(inputs, nbOutputs=6, kernelDims=[5, 5], **first_layer_config, name="conv1",
                                deepNet=self.deepnet))
        self.extractor.add(BatchNorm(self.extractor, **quant_bn_def(), name="bn1"))
        self.extractor.add(Pool2D(self.extractor, poolDims=[2, 2], strideDims=[2, 2], pooling='Max', name="pool1"))
        self.extractor.add(Conv(self.extractor, nbOutputs=16, kernelDims=[5, 5], **quant_conv_def(), name="conv2"))
        self.extractor.add(BatchNorm(self.extractor, **quant_bn_def(), name="bn2"))
        self.extractor.add(Pool2D(self.extractor, poolDims=[2, 2], strideDims=[2, 2], pooling='Max', name="pool2"))
        self.extractor.add(Conv(self.extractor, nbOutputs=120, kernelDims=[5, 5], **quant_conv_def(), name="conv3"))
        self.extractor.add(BatchNorm(self.extractor, **quant_bn_def(), name="bn3"))
        self.extractor.add(Fc(self.extractor, nbOutputs=84, **quant_fc_def(), name="fc1"))
        self.extractor.add(Dropout(self.extractor, name="fc1.drop"))

        self.classifier = Sequence([], name="classifier")

        last_layer_config = quant_fc_def()
        last_layer_config['quantizer'].set_range(255)
        last_layer_config['activationFunction'].get_quantizer().set_range(255)
        self.classifier.add(Fc(self.extractor, nbOutputs=nb_outputs, **last_layer_config,  name="fc2"))
        self.classifier.add(Softmax(self.classifier, withLoss=True, name="softmax"))

        Sequence.__init__(self, [self.extractor, self.classifier])




def conv_def():
    weights_filler = Xavier(varianceNorm='FanOut', scaling=1.0)
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activationFunction=Linear(), weightsSolver=weights_solver, biasSolver=bias_solver,
                           noBias=True, weightsFiller=weights_filler)

def fc_def():
    weights_filler = Normal(mean=0.0, stdDev=0.01)
    bias_filler = Constant(value=0.0) # Usually not used because of disactivated bias
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activationFunction=Rectifier(), weightsSolver=weights_solver, biasSolver=bias_solver,
                           noBias=True, weightsFiller=weights_filler, biasFiller=bias_filler)

def bn_def():
    scale_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activationFunction=Rectifier(), scaleSolver=scale_solver, biasSolver=bias_solver)


class LeNet(Sequence):
    def __init__(self, inputs, nb_outputs=10):

        self.deepnet = DeepNet()

        self.extractor = Sequence([], name='extractor')

        self.extractor.add(Conv(inputs, nbOutputs=6, kernelDims=[5, 5], **conv_def(), name="conv1",
                                deepNet=self.deepnet))
        self.extractor.add(BatchNorm(self.extractor, **bn_def(), name="bn1"))
        self.extractor.add(Pool2D(self.extractor, poolDims=[2, 2], strideDims=[2, 2], pooling='Max', name="pool1"))
        self.extractor.add(Conv(self.extractor, nbOutputs=16, kernelDims=[5, 5], **conv_def(), name="conv2"))
        self.extractor.add(BatchNorm(self.extractor, **bn_def(), name="bn2"))
        self.extractor.add(Pool2D(self.extractor, poolDims=[2, 2], strideDims=[2, 2], pooling='Max', name="pool2"))
        self.extractor.add(Conv(self.extractor, nbOutputs=120, kernelDims=[5, 5], **conv_def(), name="conv3"))
        self.extractor.add(BatchNorm(self.extractor, **bn_def(), name="bn3"))
        self.extractor.add(Fc(self.extractor, nbOutputs=84, **fc_def(), name="fc1"))
        self.extractor.add(Dropout(self.extractor, name="fc1.drop"))


        self.classifier = Sequence([], name="classifier")

        self.classifier.add(Fc(self.extractor, nbOutputs=nb_outputs, **fc_def(),  name="fc2"))
        self.classifier.add(Softmax(self.classifier, withLoss=True, name="lenet_softmax"))

        Sequence.__init__(self, [self.extractor, self.classifier])




