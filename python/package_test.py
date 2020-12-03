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
import N2D2


batch_size = 128
nb_epochs = 1
epoch_size = int(50000/batch_size)


print("Create database")
database = n2d2.database.MNIST(datapath="/nvme0/DATABASE/MNIST/raw/", Validation=0.1)

print("Create provider")
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size, False)


print("Create model")
model = n2d2.deepnet.Sequential([
    [
        n2d2.cell.Fc(Name='fc1', NbOutputs=300, Activation=n2d2.activation.Rectifier(), Solver=n2d2.solver.SGD(LearningRate='0.1'),
 NoBias=False, Backpropagate=True),
        n2d2.cell.Fc(Name='fc2', NbOutputs=10, Activation=n2d2.activation.Linear(), Solver=n2d2.solver.SGD(LearningRate='0.1'), NoBias=False, Backpropagate=True)
    ],
    n2d2.cell.Softmax(Name='softmax', NbOutputs=10)
], DefaultModel='Frame_CUDA')

print(model)


print("Add provider")
model.add_provider(provider)

print("Create target")
tar = N2D2.TargetScore('target', model.getOutput().N2D2(), provider.N2D2())

print("Initialize model")
model.initialize()

for epoch in range(nb_epochs):

    print("### Epoch: " + str(epoch) + " ###")

    for i in range(epoch_size):

        # Load example
        provider.readRandomBatch(set='Learn')

        # Set target of cell
        tar.provideTargets(N2D2.Database.Learn)

        # Propagate
        model.propagate()

        # Calculate loss and error
        tar.process(N2D2.Database.Learn)

        # Backpropagate
        model.backpropagate()

        # Update parameters
        model.update()

        success = tar.getAverageSuccess(N2D2.Database.Learn, 100)

        print("Batch: " + str(i)  + ", train success: " + "{0:.1f}".format(100*success) + "%", end='\r')

    print("")

print("Final train success: " + "{0:.1f}".format(100*success) + "%")
    #model.getCell('fc1').N2D2().logFreeParameters("fc1.weights")
    #model.getCell('fc2').N2D2().logFreeParameters("fc2.weights")



