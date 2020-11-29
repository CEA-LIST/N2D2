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
    * Be able to use N2D2 with as little recompilation as possible
    * Use as much as possible pybind methods directly in objects, and avoid parameter reimplementations
      to avoid incoherences between n2d2 and N2D2 objects
    * Keep INI file uppercase name convention for parameters
    * Pythonic and rather similar to Pytorch (based on lists and dictionaries)
    * Hardware agnostic definitions of layers -> High level Python wrappers for all cell types
    * Hardware binding at network initialization time
    * Easy access to weights and activations of any layer at any time
    * Exception and error treatment at the python level
"""

import n2d2

batch_size = 2
nb_epochs = 10
epoch_size = 10


print("Create database")
database = n2d2.database.MNIST_IDX_Database(Validation=0.1)
print("Load database")
database.load("/nvme0/DATABASE/MNIST/raw/")

print("Create model")
model = n2d2.deepnet.Sequential([
    [
        n2d2.cells.FcCell(Name='fc1', NbOutputs=300, Activation='Tanh', NoBias=False, Backpropagate=True),
        n2d2.cells.FcCell(Name='fc2', NbOutputs=300, Activation='Tanh', NoBias=False, Backpropagate=True)
    ],
    [
        [
            n2d2.cells.FcCell(Name='fc3', NbOutputs=200, Activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(Name='fc4', NbOutputs=200, Activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(Name='fc5', NbOutputs=200, Activation='Tanh', NoBias=False, Backpropagate=True)
        ],
        [
            n2d2.cells.FcCell(Name='fc6', NbOutputs=100, Activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(Name='fc7', NbOutputs=100, Activation='Tanh', NoBias=False, Backpropagate=True),
            n2d2.cells.FcCell(Name='fc8', NbOutputs=100, Activation='Tanh', NoBias=False, Backpropagate=True)
        ],
    ],
    n2d2.cells.FcCell(Name='fc9', NbOutputs=10, Activation='Tanh', NoBias=False, Backpropagate=True),
], DefaultModel='Frame')

print(model)


print("Create stimulus")
stimulus = n2d2.stimuli_provider.StimuliProvider(database, [28, 28, 1], batch_size, False)

print("Add stimulus")
model.add_stimulus(stimulus)

print("Initialize model")
model.initialize()

for epoch in range(nb_epochs):

    print("### Epoch: " + str(epoch))

    for i in range(epoch_size):

        print("Batch: " + str(i))

        # Generate target
        stimulus.readRandomBatch(set='Learn')
        #outputTarget = database.getStimulusLabel(id)
        #outputTensor[iteration] = outputTarget

        # Propagate
        model.propagate()

        # Backpropagate
        model.backpropagate()

        # Update parameters by calling solver on gradients
        model.update()




