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

class QuantConvDef(ConfigSection):
    def __init__(self):
        data_type = "float"
        weights_quantizer = SATCell(applyScaling=False, applyQuantization=True, range=15)
        weights_filler = Xavier(varianceNorm='FanOut', scaling=1.0)
        ConfigSection.__init__(self, dataType=data_type, activationFunction=Linear(),
                               noBias=True, weightsFiller=weights_filler, quantizer=weights_quantizer)

class QuantFcDef(ConfigSection):
    def __init__(self):
        data_type = "float"
        weights_quantizer = SATCell(applyScaling=True, applyQuantization=True, range=15)
        sat_solver = SGD(learningRate=0.05, momentum=0.0, decay=0.0)
        act_quantizer = SATAct(alpha=6.0, range=15, solver=sat_solver)
        weights_filler = Normal(mean=0.0, stdDev=0.01)
        bias_filler = Constant(value=0.0) # Usually not used because of disactivated bias
        ConfigSection.__init__(self, dataType=data_type, activationFunction=Linear(quantizer=act_quantizer),
                               noBias=True, weightsFiller=weights_filler, biasFiller=bias_filler,
                               quantizer=weights_quantizer)

class QuantBnDef(ConfigSection):
    def __init__(self):
        data_type = "float"
        sat_solver = SGD(learningRate=0.05, momentum=0.0, decay=0.0)
        act_quantizer = SATAct(alpha=6.0, range=15, solver=sat_solver)
        ConfigSection.__init__(self, dataType=data_type, activationFunction=Linear(quantizer=act_quantizer))



class QuantLeNet(Sequence):
    def __init__(self, inputs, nb_outputs=10):

        self.deepnet = DeepNet()

        self.extractor = Sequence([], name='extractor')

        first_layer_config = QuantConvDef()
        first_layer_config.get()['quantizer'].set_range(255)
        self.extractor.add(Conv(inputs, nbOutputs=6, kernelDims=[5, 5],
                      **first_layer_config.get(), deepNet=self.deepnet, name="conv1"))
        self.extractor.add(BatchNorm(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),
                           **QuantBnDef().get(), name="bn1"))
        self.extractor.add(Pool2D(self.extractor.get_last(), nbOutputs=6, poolDims=[2, 2],
                        strideDims=[2, 2], pooling='Max', name="pool1"))
        self.extractor.add(Conv(self.extractor.get_last(), nbOutputs=16, kernelDims=[5, 5],
                      **QuantConvDef().get(), name="conv2"))
        self.extractor.add(BatchNorm(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),
                      **QuantBnDef().get(), name="bn2"))
        self.extractor.add(Pool2D(self.extractor.get_last(), nbOutputs=16, poolDims=[2, 2],
                        strideDims=[2, 2], pooling='Max', name="pool2"))
        self.extractor.add(Conv(self.extractor.get_last(), nbOutputs=120, kernelDims=[5, 5],
                      **QuantConvDef().get(), name="conv3"))
        self.extractor.add(BatchNorm(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),
                           **QuantBnDef().get(), name="bn3"))
        self.extractor.add(Fc(self.extractor.get_last(), nbOutputs=84, **QuantFcDef().get(), name="fc1"))
        self.extractor.add(Dropout(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),  name="fc1.drop"))


        self.classifier = Sequence([], name="classifier")

        last_layer_config = QuantFcDef()
        last_layer_config.get()['quantizer'].set_range(255)
        last_layer_config.get()['activationFunction'].get_quantizer().set_range(255)
        self.classifier.add(Fc(self.extractor.get_last(), nbOutputs=nb_outputs, **last_layer_config.get(),  name="fc2"))
        self.classifier.add(Softmax(self.classifier.get_last(), nbOutputs=nb_outputs, withLoss=True, name="softmax"))

        Sequence.__init__(self, [self.extractor, self.classifier])


    def set_MNIST_solvers(self):
        print("Add solvers")
        solver = SGD
        solver_config = ConfigSection(learningRate=0.05, momentum=0.0, decay=0.0)

        for name, cell in self.get_cells().items():
            if isinstance(cell, Fc) or isinstance(cell, Conv):
                cell.set_weights_solver(solver(**solver_config.get()))
                cell.set_bias_solver(solver(**solver_config.get()))

            if isinstance(cell, BatchNorm):
                cell.set_scale_solver(solver(**solver_config.get()))
                cell.set_bias_solver(solver(**solver_config.get()))



class ConvDef(ConfigSection):
    def __init__(self):
        data_type = "float"
        weights_filler = Xavier(varianceNorm='FanOut', scaling=1.0)
        ConfigSection.__init__(self, dataType=data_type, activationFunction=Linear(),
                               noBias=True, weightsFiller=weights_filler)

class FcDef(ConfigSection):
    def __init__(self):
        data_type = "float"
        weights_filler = Normal(mean=0.0, stdDev=0.01)
        bias_filler = Constant(value=0.0) # Usually not used because of disactivated bias
        ConfigSection.__init__(self, dataType=data_type, activationFunction=Rectifier(),
                               noBias=True, weightsFiller=weights_filler, biasFiller=bias_filler)

class BnDef(ConfigSection):
    def __init__(self):
        data_type = "float"
        ConfigSection.__init__(self, dataType=data_type, activationFunction=Rectifier())


class LeNet(Sequence):
    def __init__(self, inputs, nb_outputs=10):

        self.deepnet = DeepNet()

        self.extractor = Sequence([], name='extractor')

        self.extractor.add(Conv(inputs, nbOutputs=6, kernelDims=[5, 5],
                      **ConvDef().get(), deepNet= self.deepnet, name="conv1"))
        self.extractor.add(BatchNorm(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),
                           **BnDef().get(), name="bn1"))
        self.extractor.add(Pool2D(self.extractor.get_last(), nbOutputs=6, poolDims=[2, 2],
                        strideDims=[2, 2], pooling='Max', name="pool1"))
        self.extractor.add(Conv(self.extractor.get_last(), nbOutputs=16, kernelDims=[5, 5],
                      **ConvDef().get(), name="conv2"))
        self.extractor.add(BatchNorm(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),
                      **BnDef().get(), name="bn2"))
        self.extractor.add(Pool2D(self.extractor.get_last(), nbOutputs=16, poolDims=[2, 2],
                        strideDims=[2, 2], pooling='Max', name="pool2"))
        self.extractor.add(Conv(self.extractor.get_last(), nbOutputs=120, kernelDims=[5, 5],
                      **ConvDef().get(), name="conv3"))
        self.extractor.add(BatchNorm(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(),
                           **BnDef().get(), name="bn3"))
        self.extractor.add(Fc(self.extractor.get_last(), nbOutputs=84, **FcDef().get(), name="fc1"))
        self.extractor.add(Dropout(self.extractor.get_last(), nbOutputs=self.extractor.get_last().get_outputs().dimZ(), name="fc1.drop"))


        self.classifier = Sequence([], name="classifier")

        self.classifier.add(Fc(self.extractor.get_last(), nbOutputs=nb_outputs, **FcDef().get(),  name="fc2"))
        self.classifier.add(Softmax(self.classifier.get_last(), nbOutputs=nb_outputs, withLoss=True, name="softmax"))

        Sequence.__init__(self, [self.extractor, self.classifier])


    def set_MNIST_solvers(self):
        print("Add solvers")
        solver = SGD
        solver_config = ConfigSection(learningRate=0.05, momentum=0.0, decay=0.0)

        for name, cell in self.get_cells().items():
            if isinstance(cell, Fc) or isinstance(cell, Conv):
                cell.set_weights_solver(solver(**solver_config.get()))
                cell.set_bias_solver(solver(**solver_config.get()))

            if isinstance(cell, BatchNorm):
                cell.set_scale_solver(solver(**solver_config.get()))
                cell.set_bias_solver(solver(**solver_config.get()))



