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

def one_hot_encode(target, inputs):
    """
    Small function to one encode targets.
    If we have 5 channels
    [1, 4] -> [[01000][00001]]
    """
    encoded_target = n2d2.Tensor(inputs.shape(), datatype="int", value=0)
    cpt = 0
    for i in target:
        encoded_target[i + (cpt * inputs.dimZ())] = 1
        cpt += 1
    return encoded_target


# n2d2.global_variables.default_model = "Frame_CUDA"
# n2d2.global_variables.set_cuda_device(5)

nb_epochs = 1
batch_size = 256

print("\n### Create database ###")
database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw", validation=0.1)
print(database)

print("\n### Create Provider ###")
provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
print(provider)

print("\n### Loading Model ###")
model = n2d2.models.lenet.LeNet(10)
print(model)


class test_loss(n2d2.loss_function.LossFunction):
    def __init__(self) -> None:
        super().__init__()

    def compute_loss(self, inputs: n2d2.Tensor, target: n2d2.Tensor, **kwargs) -> n2d2.Tensor:
        loss_t = n2d2.Tensor(inputs.shape())
        real_target = one_hot_encode(target, inputs)
        for i in range(len(loss_t)):
            loss_t[i] = - real_target[i] * math.log(inputs[i])
        return loss_t


loss_function = test_loss()
# loss_function = n2d2.application.CrossEntropyClassifier(provider)

print("\n### Training ###")
for epoch in range(nb_epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        x = provider.read_random_batch()
        x = model(x)
        x = loss_function(x, provider.get_labels())
        x.back_propagate()
        x.update()
        print("Example: " + str(i * batch_size) + ", loss: "
                    + "{0:.5f}".format(x[0]), end='\r')


    print("\n### Validation ###")
    
    provider.set_partition('Validation')
    model.test()
    success = 0
    cpt = 0
    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size
        labels =  provider.get_labels()
        
        x = provider.read_batch(batch_idx)
        x = model(x)
        
        encoded_labels = one_hot_encode(labels, x)

        for index in range(x.dimB()):
            batch = [x[temp_idx] for temp_idx in range(index * x.dimZ(), (index+1) * x.dimZ())]
            max = batch[0]
            i_max = 0
            for i_val in range(len(batch)):
                if batch[i_val] > max:
                    i_max = i_val
                    max = batch[i_val]
            predicted_index = i_max

            success += encoded_labels[predicted_index + index * x.dimZ()]
            cpt += 1
        x = loss_function(x, labels)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * (success / cpt)) + "%", end='\r')
print("\n")