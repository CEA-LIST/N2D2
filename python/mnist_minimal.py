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

import n2d2
import math


nb_epochs = 10
batch_size = 256

database = n2d2.database.MNIST(dataPath="/nvme0/DATABASE/MNIST/raw/")
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batchSize=batch_size)


"""
* First way to define a model
* Create a sequence object on-the-fly and add to sequence by using 'add'. 
* The deepNet can be implicitly derived from a preceding cell or sequence and only has to be provided to the first
* cell of the graph.
"""

"""
model = n2d2.deepnet.Group([])
model.add(n2d2.cell.Conv(provider, 5, kernelDims=[5, 5], activationFunction=n2d2.activation.Rectifier()))
model.add(n2d2.cell.Fc(model, 10, activationFunction=n2d2.activation.Linear()))
model.add(n2d2.cell.Softmax(model, withLoss=True))
"""

"""
* Second way to define a model.
* Create cell objects and add them to a sequence after creation. The created object can also be added with the 'add' method
"""
"""
conv1 = n2d2.cell.Conv(provider, 5, kernelDims=[5, 5], activationFunction=n2d2.activation.Rectifier())
fc = n2d2.cell.Fc(conv1, 10, activationFunction=n2d2.activation.Linear())
softmax = n2d2.cell.Softmax(fc, withLoss=True)
model = n2d2.deepnet.Group([conv1, fc, softmax])"""


"""
* Third way to define a model.
* Pass cells directly to each other
"""
x = n2d2.cell.Conv(provider, 5, kernelDims=[5, 5], activationFunction=n2d2.activation.Rectifier())
x = n2d2.cell.Fc(x, 10, activationFunction=n2d2.activation.Linear())
x = n2d2.cell.Softmax(x, withLoss=True)

classifier = n2d2.application.Classifier(provider, x)
print(x)

for epoch in range(nb_epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        classifier.read_random_batch()

        classifier.process()

        classifier.optimize()

        print("Example: " + str(i*batch_size) + ", loss: "
              + "{0:.3f}".format(classifier.get_current_loss()), end='\r')

print("\n")

classifier.set_mode('Test')

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
    batch_idx = i*batch_size

    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')

print("\n")



