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
from n2d2.cells.nn import Fc, Conv, Pool2d, BatchNorm2d
from n2d2.cells import Sequence
from n2d2.activation import Rectifier, Linear
from n2d2.solver import SGD
from n2d2.filler import Normal, Xavier

solver_config = ConfigSection(learning_rate=0.05, momentum=0.0, decay=0.0)

"""
def quant_conv_def():
    weights_quantizer = SATCell(applyScaling=False, applyQuantization=True, range=15)
    weights_filler = Xavier(variance_norm='FanOut', scaling=1.0)
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation_function=Linear(),
                no_bias=True, weights_solver=weights_solver, bias_solver=bias_solver,
                weights_filler=weights_filler, quantizer=weights_quantizer)

def quant_fc_def():
    weights_quantizer = SATCell(applyScaling=True, applyQuantization=True, range=15)
    sat_solver = SGD(**solver_config)
    act_quantizer = SATAct(alpha=6.0, range=15, solver=sat_solver)
    weights_filler = Normal(mean=0.0, std_dev=0.01)
    bias_filler = Constant(value=0.0) # Usually not used because of disactivated bias
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation_function=Linear(quantizer=act_quantizer),
                        no_bias=True, weights_solver=weights_solver, bias_solver=bias_solver,
                        weights_filler=weights_filler, biasFiller=bias_filler,
                        quantizer=weights_quantizer)

def quant_bn_def():
    sat_solver = SGD(learningRate=0.05, momentum=0.0, decay=0.0)
    act_quantizer = SATAct(alpha=6.0, range=15, solver=sat_solver)
    scale_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation_function=Linear(quantizer=act_quantizer), scale_solver=scale_solver, bias_solver=bias_solver)



class QuantLeNet(Group):
    def __init__(self, inputs, nb_outputs=10):

        self.deepnet = DeepNet()

        self.extractor = Group([], name='extractor')

        first_layer_config = quant_conv_def()
        first_layer_config['quantizer'].set_range(255)
        self.extractor.add(Conv(inputs, nbOutputs=6, kernel_dims=[5, 5], **first_layer_config, name="conv1",
                                deepNet=self.deepnet))
        self.extractor.add(BatchNorm2d(self.extractor, **quant_bn_def(), name="bn1"))
        self.extractor.add(Pool2d(self.extractor, pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max', name="pool1"))
        self.extractor.add(Conv(self.extractor, nbOutputs=16, kernel_dims=[5, 5], **quant_conv_def(), name="conv2"))
        self.extractor.add(BatchNorm2d(self.extractor, **quant_bn_def(), name="bn2"))
        self.extractor.add(Pool2d(self.extractor, pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max', name="pool2"))
        self.extractor.add(Conv(self.extractor, nbOutputs=120, kernel_dims=[5, 5], **quant_conv_def(), name="conv3"))
        self.extractor.add(BatchNorm2d(self.extractor, **quant_bn_def(), name="bn3"))
        self.extractor.add(Fc(self.extractor, nbOutputs=84, **quant_fc_def(), name="fc1"))
        self.extractor.add(Dropout(self.extractor, name="fc1.drop"))

        self.classifier = Group([], name="classifier")

        last_layer_config = quant_fc_def()
        last_layer_config['quantizer'].set_range(255)
        last_layer_config['activation_function'].get_quantizer().set_range(255)
        self.classifier.add(Fc(self.extractor, nbOutputs=nb_outputs, **last_layer_config,  name="fc2"))
        self.classifier.add(Softmax(self.classifier, withLoss=True, name="softmax"))

        Group.__init__(self, [self.extractor, self.classifier])
"""



def conv_def():
    weights_filler = Xavier(variance_norm='FanOut', scaling=1.0)
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation_function=Linear(), weights_solver=weights_solver, bias_solver=bias_solver,
                           no_bias=True, weights_filler=weights_filler)

def fc_def():
    weights_filler = Normal(mean=0.0, std_dev=0.01)
    weights_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(weights_solver=weights_solver, bias_solver=bias_solver,
                           no_bias=True, weights_filler=weights_filler)

def bn_def():
    scale_solver = SGD(**solver_config)
    bias_solver = SGD(**solver_config)
    return ConfigSection(activation_function=Rectifier(), scale_solver=scale_solver, bias_solver=bias_solver)


def generate(inputs, nb_outputs=10):
    x = Conv(1, 6, kernel_dims=[5, 5], **conv_def())(inputs)
    #x = BatchNorm2d(**bn_def())(x)
    x = Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max')(x)
    x = Conv(6, 16, kernel_dims=[5, 5], **conv_def())(x)
    #x = BatchNorm2d(x, **bn_def())(x)
    x = Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max')(x)
    x = Conv(16, 120, kernel_dims=[5, 5], **conv_def())(x)
    #x = BatchNorm2d(x, **bn_def())(x)
    x = Fc(120, 84, **fc_def())(x)
    #x = Dropout(name="fc1.drop")(x)
    x = Fc(84, nb_outputs, **fc_def())(x)
    #x = Softmax(withLoss=True)(x)
    return x



class LeNet(Sequence):
    def __init__(self, nb_outputs=10):
        Sequence.__init__(self, [
            Conv(1, 6, kernel_dims=[5, 5], **conv_def()),
            BatchNorm2d(6, **bn_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(6, 16, kernel_dims=[5, 5], **conv_def()),
            BatchNorm2d(16, **bn_def()),
            Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
            Conv(16, 120, kernel_dims=[5, 5], **conv_def()),
            BatchNorm2d(120, **bn_def()),
            Fc(120, 84, activation_function=Rectifier(), **fc_def()),
            #Dropout(name="fc1.drop"),
            Fc(84, nb_outputs, activation_function=Linear(), **fc_def()),
        ])




