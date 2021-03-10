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



# Change default model
n2d2.global_variables.default_model = "Frame_CUDA"
# Change cuda device (default 0)
n2d2.global_variables.set_cuda_device(1)
# Change seed (default 1)
#n2d2.global_variables.default_seed = 2

nb_epochs = 10
batch_size = 256

print("\n### Create database ###")
database = n2d2.database.MNIST(dataPath="/nvme0/DATABASE/MNIST/raw/", validation=0.1)
print(database)

print("\n### Create Provider ###")
provider = n2d2.provider.DataProvider(database, [32, 32, 1], batchSize=batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
print(provider)

print("\n### Loading Model ###")
model = n2d2.model.LeNet(provider, 10)
print(model)

classifier = n2d2.application.Classifier(provider, model)

print("\n### Training ###")
for epoch in range(nb_epochs):

    print("\n# Train Epoch: " + str(epoch) + " #")

    classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        classifier.read_random_batch()

        classifier.process()

        classifier.optimize()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(classifier.get_current_loss()), end='\r')

    print("\n### Validation ###")

    classifier.set_mode('Validation')
    classifier.clear_success()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        classifier.read_batch(idx=batch_idx)

        classifier.process()

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')


print("\n\n### Testing ###")

classifier.set_mode('Test')

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
    batch_idx = i*batch_size

    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')

print("\n")
# save a confusion matrix
classifier.logConfusionMatrix("lenet_confusion_matrix")
# save a graph of the loss and the validation score as a function of the number of steps
classifier.logSuccess("lenet_success")
