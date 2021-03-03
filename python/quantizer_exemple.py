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


n2d2.global_variables.set_cuda_device(4)
n2d2.global_variables.default_model = "Frame_CUDA"

nb_epochs = 10
batch_size = 256
avg_window = 1

print("Load database")
database = n2d2.database.MNIST(dataPath="/nvme0/DATABASE/MNIST/raw/", randomPartitioning=True)
print(database)

print("Create Provider")
provider = n2d2.provider.DataProvider(database, [32, 32, 1], batchSize=batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
print(provider)

print("\n### Loading Model ###")
model = n2d2.model.QuantLeNet(provider, 10)
#model = n2d2.model.LeNet(provider, 10)
print(model)

classifier = n2d2.application.Classifier(provider, model)


print("\n### Train ###")

for epoch in range(nb_epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Learn')

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        # Load example
        classifier.read_random_batch()

        classifier.process()

        classifier.optimize()

        print("Example: " + str(i*batch_size) + ", train success: "
              + "{0:.2f}".format(100*classifier.get_average_success(window=avg_window)) + "%", end='\r')


#model.deepnet.N2D2().exportNetworkFreeParameters("posttrain_free_parameters")


#model.deepnet.N2D2().importNetworkFreeParameters("/home/jt251134/N2D2-IP/build/bin/weights")
#model.deepnet.N2D2().importNetworkFreeParameters("/home/jt251134/N2D2-IP/N2D2/tests/tests_data/mnist_model/weights_test_SAT")
classifier.set_mode('Test')

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\n')



