"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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
Example for main.py file based on LeNet.ini

In the following, "n2d2" refers to the python package, "N2D2" to the C++ core functions
"""

"""
General paradigms:
    * Merges tasks of INI files, generators and n2d2.cpp
    * Pythonic and rather similar to Pytorch (based on lists and dictionaries)
    * Coupled modules with forward and backward methods
    * Hardware agnostic definitions of layers -> High level Python wrappers for all cell types
    * Hardware binding at network initialization time
    * Easy access to weights and activations of any layer at any time
    * Exception and error treatment at the python level
    * Be able to use N2D2 with as little recompilation as possible
    * Basic datatypes: n2d2.Tensor (double, float, half)
"""

import n2d2
import N2D2

batch_size = 128
nb_epochs = 10
epoch_size = 100


print("Create database")
database = N2D2.MNIST_IDX_Database()

# Necessary to initialize random number generator; TODO: Replace
net = N2D2.Network()
deepNet = N2D2.DeepNet(net) # Proposition : overload the constructor to avoid passing a Network object

print("Load database")
database.load("/local/is154584/cm264821/mnist")

print("Create stimuli")
stimuli = n2d2.stimuli_provider.StimuliProvider(database, [24, 24, 1], batch_size, False)


model = n2d2.deepnet.Sequential([
    [
        n2d2.cells.FcCell(name='fc1', nbOutputs=300, activation='Tanh', NoBias=False, Backpropagate=True),
        n2d2.cells.FcCell(name='fc2', nbOutputs=300, activation='Tanh', NoBias=False, Backpropagate=True)
    ],
    [
        [
            n2d2.cells.FcCell(name='fc3', nbOutputs=200, activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(name='fc4', nbOutputs=200, activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(name='fc5', nbOutputs=200, activation='Tanh', NoBias=False, Backpropagate=True)
        ],
        [
            n2d2.cells.FcCell(name='fc6', nbOutputs=100, activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(name='fc7', nbOutputs=100, activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(name='fc8', nbOutputs=100, activation='Tanh', NoBias=False, Backpropagate=True)
        ],
    ],
    n2d2.cells.FcCell(name='fc9', nbOutputs=10, activation='Tanh', NoBias=False, Backpropagate=True),
], model_type='Frame')

print(model)

model.addStimulus(stimuli)


"""

for epoch in range(nb_epochs):
    for i in range(epoch_size):
        # Propagate
        stimuli_provider.readRandomBatch()
        deepNet.propagate()
        
        # Calculate loss
        loss_function.propagate()
        
        # Backpropagate
        loss_function.backpropagate()
        deepNet.backpropagate()
        
        # Update parameters by calling solver on gradients
        deepNet.update()
        
        # This may not be necessary in current implementation
        deepNet.reset()
"""
    
    
