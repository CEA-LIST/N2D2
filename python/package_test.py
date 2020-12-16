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

import math

batch_size = 128
nb_epochs = 1
avg_window = int(10000/batch_size)

print(avg_window)

#N2D2.mtSeed(1)
net = N2D2.Network(1)
deepnet = N2D2.DeepNet(net)

N2D2.CudaContext.setDevice(1)

print("Create database")
database = n2d2.database.MNIST(datapath="/nvme0/DATABASE/MNIST/raw/", Validation=0.2)

print("Create provider")
provider = n2d2.provider.DataProvider(Database=database, Size=[28, 28, 1], BatchSize=batch_size)


print("Create model")
model = n2d2.deepnet.Sequential(deepnet,
[
    [
        #n2d2.cell.Fc(Name='fc1', NbOutputs=300, ActivationFunction=n2d2.activation.Rectifier(), WeightsSolver=n2d2.solver.SGD(), WeightsFiller=n2d2.filler.Normal(Mean=0.1, StdDev=0.5)),
        #n2d2.cell.Fc(Name='fc2', NbOutputs=10, ActivationFunction=n2d2.activation.Linear(), WeightsSolver=n2d2.solver.SGD())
        n2d2.cell.Fc(Name='fc1', NbOutputs=300),
        n2d2.cell.Fc(Name='fc2', NbOutputs=10)
    ],
    n2d2.cell.Softmax(Name='softmax', NbOutputs=10)
], Model='Frame_CUDA')

print(model)

#print(model.get_cell('fc1')._cell_parameters['WeightsFiller'])
#print(model.get_cell('fc2')._cell_parameters['WeightsFiller'])


print("Add provider")
model.add_provider(provider)

print("Create target")
tar = n2d2.target.Score('softmax.Target', model.get_output(), provider)

print("Initialize model")
model.initialize()


n2d2.utils.convert_to_INI("model_INI", database, provider, model, tar)


for epoch in range(nb_epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    partition = 'Learn'

    for i in range(math.ceil(database.get_nb_stimuli(partition)/batch_size)):

        # Load example
        provider.read_random_batch(partition=partition)

        # Set target of cell
        tar.provide_targets(partition=partition)

        # Propagate
        model.propagate()

        # Calculate loss and error
        tar.process(partition=partition)

        # Backpropagate
        model.back_propagate()

        # Update parameters
        model.update()

        success = tar.get_average_success(partition=partition, window=avg_window)

        print("Example: " + str(i*batch_size)  + ", train success: " + "{0:.2f}".format(100*success) + "%", end='\r')



    print("\n### Validate Epoch: " + str(epoch) + " ###")

    partition = "Validation"

    for i in range(math.ceil(database.get_nb_stimuli(partition)/batch_size)):

        batch_idx = i*batch_size

        # Load example
        provider.read_batch(partition=partition, idx=batch_idx)

        # Set target of cell
        tar.provide_targets(partition=partition)

        # Propagate
        model.propagate(inference=True)

        # Calculate loss and error
        tar.process(partition=partition)

        success = tar.get_average_score(partition=partition, metric='Sensitivity')

        print("Example: " + str(i*batch_size) + ", val success: " + "{0:.2f}".format(100 * success) + "%", end='\n')


#model.get_cell('fc1').N2D2().exportFreeParameters("")
#model.get_cell('fc1').N2D2().importFreeParameters("")


model.get_cell('fc1').N2D2().importFreeParameters("../build/bin/weights_validation/fc1.syntxt")
model.get_cell('fc2').N2D2().importFreeParameters("../build/bin/weights_validation/fc2.syntxt")

print("\n### Test ###")

partition = "Test"

for i in range(math.ceil(database.get_nb_stimuli(partition)/batch_size)):

    batch_idx = i*batch_size

    # Load example
    provider.read_batch(partition=partition, idx=batch_idx)

    # Set target of cell
    tar.provide_targets(partition=partition)

    # Propagate
    model.propagate(inference=True)

    # Calculate loss and error
    tar.process(partition=partition)

    success = tar.get_average_success(partition=partition)

    print("Example: " + str(i*batch_size) + ", test success: " + "{0:.2f}".format(100 * success) + "%", end='\n')

    #print(tar.N2D2().getEstimatedLabels)


print("")

#print("Final train success: " + "{0:.1f}".format(100*success) + "%")
#model.getCell('fc1').N2D2().logFreeParameters("fc1.weights")
#model.getCell('fc2').N2D2().logFreeParameters("fc2.weights")



