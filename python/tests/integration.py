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

"""
This script aims to compare a network defined with an ini file with a network defined with the PythonAPI.
"""

import N2D2
import n2d2


# Fill this list and run the script. If the network described with ini is different from 
# the one described with python an assertion error will be raised.
test_list = [
    ["../../models/LeNet.ini", n2d2.models.lenet.LeNet(10), [128, 1, 32, 32]],
]

weights_value = 1

class IniReader():
    """
    Quick class to create a deepNet from an ini file and to use it.
    """
    def __init__(self, path):
        net = N2D2.Network(1)
        self.deepNet = N2D2.DeepNetGenerator.generate(net, path)
        self.deepNet.initialize() 
        self.cells = self.deepNet.getCells()
        self.first_cell = self.cells[self.deepNet.getLayers()[1][0]]
        self.last_cell = self.cells[self.deepNet.getLayers()[-1][-1]]

    def set_weights(self):
        for name in self.cells:
            cell = self.cells[name]
            if 'Conv' in str(type(cell)) or 'Deconv' in str(type(cell)) or 'Fc' in str(type(cell)):
                for o in range(cell.getNbOutputs()):
                    for c in range(cell.getNbChannels()):
                        weights = N2D2.Tensor_float([])
                        cell.getWeight(o, c,  weights)
                        for i in range(len(weights)):
                            weights[i] = weights_value
                        cell.setWeight(o, c,  weights)

    def forward(self, input_tensor):
        self.first_cell.clearInputs()

        shape = [i for i in reversed(input_tensor.dims())]
        diffOutputs = n2d2.Tensor(shape, value=0)
        self.first_cell.addInputBis(input_tensor.N2D2(), diffOutputs.N2D2())

        N2D2_inputs = self.deepNet.getCell_Frame_Top(self.first_cell.getName()).getInputs(0)
        N2D2_inputs.op_assign(input_tensor.N2D2())
        N2D2_inputs.synchronizeHToD()

        self.deepNet.propagate(N2D2.Database.Learn, False, [])

        outputs = self.deepNet.getCell_Frame_Top(self.last_cell.getName()).getOutputs() 
        outputs.synchronizeDToH()
        return n2d2.Tensor.from_N2D2(outputs)
    
    def backward(self):
        # TODO test if the weight and biases are well initialized
        pass

def set_weights(sequence):
    for cell in sequence._cells:
        if 'Conv' in str(type(cell)) or 'Deconv' in str(type(cell)) or 'Fc' in str(type(cell)):
            for o in range(cell.get_nb_outputs()):
                for c in range(cell.get_nb_channels()):
                    weights = cell.get_weight(o, c)
                    for i in range(len(weights)):
                        weights[i] = weights_value
                    cell.set_weight(o, c, weights)
        

def test_output(ini_path, model, tensor_size):
    inputs = n2d2.Tensor(tensor_size, value=1.0, cuda=True)
    net = IniReader(ini_path)
    net.set_weights()
    ini_output = net.forward(inputs)
    set_weights(model)
    py_output = model(inputs)
    assert(py_output == ini_output)


# BEGINNING TEST 

for descriptor in test_list:
    test_output(descriptor[0], descriptor[1], descriptor[2])

