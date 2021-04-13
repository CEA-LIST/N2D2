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
This script reproduce the "LAB : Optimized DNN implementation with N2D2" practical work.
"""
import N2D2
import n2d2
from n2d2.cells.nn import Conv, Pool, Fc, Dropout
from n2d2.activation import Linear, Rectifier
from n2d2.filler import He
from n2d2.solver import SGD
from n2d2 import ConfigSection

n2d2.global_variables.default_model = "Frame_CUDA"

db = n2d2.database.GTSRB(0.2)
db.load("/nvme0/DATABASE/GTSRB")

print("\n### Data imported ###\n")
db.get_partition_summary()

provider = n2d2.provider.DataProvider(db, [29, 29, 3], batch_size=24)
provider.add_transformation(n2d2.transform.Rescale(width=29, height=29))

solver_config = ConfigSection(
    learning_rate=0.01, 
    momentum=0.9, 
    decay=0.0005, 
    learning_rate_decay=0.993)

fc_config   = ConfigSection(no_bias=True)
conv_config = ConfigSection(activation_function=Rectifier(), 
                            weights_filler=He(), 
                            weights_solver=SGD(**solver_config), 
                            no_bias=True)

# TODO :  look to support Pool mapping with tensor 

model = n2d2.cells.Sequence([
    Conv(3, 32, [4, 4], **conv_config),
    Pool([2, 2], stride_dims=[2, 2], pooling='Max'),
    Conv(32, 48, [5, 5], **conv_config),
    Pool([3, 3], stride_dims=[2, 2], pooling='Max'),
    Fc(48, 200, activation_function=Rectifier(), weights_filler=He(), weights_solver=SGD(**solver_config), **fc_config),
    Dropout(name="fc1.drop"),
    Fc(200, 43, activation_function=Linear(), **fc_config)
    # We don't add a Softmax layer because it's already in the CrossEntropyClassifier.
])
print("\n### Model ###\n")
print(model)
loss_function = n2d2.application.CrossEntropyClassifier(provider)
for epoch in range(50):
    print("\n### Learning ###")

    provider.set_partition("Learn")
    model.learn()
    provider.set_reading_randomly(True)
    for stimuli in provider:
        output = model(stimuli)
        loss = loss_function(output)
        loss.back_propagate()
        loss.update()
        print("Batch number : " + str(provider.batch_number()) + ", loss: " + "{0:.3f}".format(loss[0]), end='\r')

    print("\n### Validation ###")

    loss_function.clear_success()

    provider.set_partition('Validation')
    model.test()

    for stimuli in provider:

        x = model(stimuli)
        x = loss_function(x)
        print("Batch number : " + str(provider.batch_number()) + ", val success: "
                + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')


print("\n### Testing ###")

provider.set_partition('Test')
model.test()

for stimuli in provider:
    x = model(stimuli)
    x = loss_function(x)
    print("Batch number : " + str(provider.batch_number()) + ", test success: "
          + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')

# print("\n")
# # save a confusion matrix
# loss_function.log_confusion_matrix("TP_confusion_matrix")
# # save a graph of the loss and the validation score as a function of the number of steps
# loss_function.log_success("TP_success")
