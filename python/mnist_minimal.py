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

n2d2.global_variables.default_model = "Frame_CUDA"
n2d2.global_variables.set_cuda_device(5)

nb_epochs = 10
batch_size = 256

database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw/")
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)

database2 = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw/")
provider2 = n2d2.provider.DataProvider(database2, [28, 28, 1], batch_size=batch_size)


print("Create Conv")
cell1 = n2d2.cells.cell.Conv(1, 10, kernel_dims=[5, 5])
#cell1 = n2d2.cells.Fc(28*28, 50)
#fc1 = n2d2.cells.Fc(50, 50)
#TODO: Fc input dimension check before call
print("Create Fc")
fc2 = n2d2.cells.cell.Fc(10 * 24 * 24, 10)
#fc2 = n2d2.cells.Fc(50, 10)

#cell1.N2D2().exportFreeParameters("exported_parameters")
#fc2.N2D2().exportFreeParameters("exported_parameters")

loss_function = n2d2.application.CrossEntropyClassifier(provider)

provider.set_partition("Learn")

#input_provider = n2d2.provider.Input([batch_size, 1, 28, 28])



for epoch in range(nb_epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        #x = n2d2.Tensor([batch_size, 1, 28, 28], value=1.0)
        #x = input_provider(x)

        #if i%2 == 0:
        #    x = provider.read_random_batch()
        #else:
        #    x = provider2.read_random_batch()
        x = provider.read_random_batch()
        x = cell1(x)
        x = fc2(x)
        x = loss_function(x)

        #print(x.get_deepnet())

        #print(id(x.get_deepnet().N2D2()))

        x.back_propagate()
        x.update()

        print("Example: " + str(i*batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')

print("\n")

provider.set_partition('Test')

cell1.test()
fc2.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
    batch_idx = i*batch_size

    x = provider.read_batch(idx=batch_idx)
    #x = tensor_placeholder(x)
    x = cell1(x)
    x = fc2(x)
    x = loss_function(x)

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')

print("\n")



