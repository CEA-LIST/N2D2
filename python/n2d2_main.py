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



import n2d2
import N2D2

import math


learn = 200000
log = 50000
seed = 1
dev = 3
avg_window = 10000


net = N2D2.Network(seed)
deepnet = N2D2.DeepNet(net)

N2D2.CudaContext.setDevice(dev)


""" 
Begin INI equivalent
"""

print("Create database")
database = n2d2.database.MNIST(datapath="/nvme0/DATABASE/MNIST/raw/", Validation=0.2)

batch_size = 128

print("Create provider")
provider = n2d2.provider.DataProvider(Database=database, Size=[28, 28, 1], BatchSize=batch_size)


print("Create model")
model = n2d2.deepnet.Sequential(deepnet,
[
    [
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

n2d2.utils.convert_to_INI("model_INI", database, provider, model, tar)

"""
End INI equivalent
"""

print("Initialize model")
model.initialize()

nb_batch = math.ceil(learn/batch_size)

next_log = log

for batch in range(nb_batch):

    partition = 'Learn'

    #print("\n### Train Batch: " + str(batch) + " ###")

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

    success = tar.get_average_success(partition=partition, window=math.floor(avg_window/batch_size))

    print("Example: " + str(batch*batch_size) + ", train success: " + "{0:.2f}".format(100*success) + "%", end='\r')
"""
    if batch*batch_size >= next_log or batch == nb_batch-1:

        print("")

        next_log += log

        tar.N2D2().clear(set=N2D2.Database.Learn)

        partition = "Validation"

        for val_batch in range(math.ceil(database.get_nb_stimuli(partition)/batch_size)):

            #print("\n### Validate Epoch: " + str(val_batch) + " ###")

            batch_idx = val_batch*batch_size

            # Load example
            provider.read_batch(partition=partition, idx=batch_idx)

            # Set target of cell
            tar.provide_targets(partition=partition)

            # Propagate
            model.propagate(inference=True)

            # Calculate loss and error
            tar.process(partition=partition)

            success = tar.get_average_score(partition=partition, metric='Sensitivity')
            #success = tar.get_average_success(partition=partition)

            print("Validate Example: " + str(val_batch*batch_size) + ", val success: " + "{0:.2f}".format(100 * success) + "%", end='\r')

        tar.N2D2().clearScore(set=N2D2.Database.Validation)
        #tar.N2D2().clearSuccess(set=N2D2.Database.Validation)

        print("")
"""

#model.get_cell('fc1').N2D2().importFreeParameters("../build/bin/weights_validation/fc1.syntxt")
#model.get_cell('fc2').N2D2().importFreeParameters("../build/bin/weights_validation/fc2.syntxt")

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

    print("Example: " + str(i*batch_size) + ", test success: " + "{0:.2f}".format(100 * success) + "%", end='\r')

    #print(tar.N2D2().getEstimatedLabels)

