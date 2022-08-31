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
### Import + global var ###
from os import getenv
import n2d2
import math
from n2d2.cells.nn import Dropout, Fc, Conv, Pool2d, BatchNorm2d
from n2d2.quantizer import SATCell

nb_epochs = 100
batch_size = 256
n2d2.global_variables.set_cuda_device(0)
n2d2.global_variables.default_model = "Frame_CUDA"

DATA_PATH = getenv("N2D2_DATA")
if DATA_PATH is None:
    DATA_PATH="/local/DATABASE/"


print("\n### Create database ###")
database = n2d2.database.MNIST(data_path="{DATA_PATH}/mnist", validation=0.1)
print(database)

print("\n### Create Provider ###")
provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
print(provider)

### Configuration ###

solver_conf = n2d2.ConfigSection(
    learning_rate=0.05,
    learning_rate_policy="None",
    momentum=0.0,
    decay=0.0, 
)
def conv_conf():
    return n2d2.ConfigSection(
        activation=n2d2.activation.Linear(),
        no_bias=True,
        weights_solver=n2d2.solver.SGD(**solver_conf),
        bias_solver=n2d2.solver.SGD(**solver_conf),
        quantizer=SATCell(
            apply_scaling=False, # No scaling needed because each conv is followed by batch-normalization layers
            apply_quantization=False, # Only clamp mode for the 1st step
    ),)
def fc_conf():
    return n2d2.ConfigSection(
        activation=n2d2.activation.Linear(),
        no_bias=True,
        weights_solver=n2d2.solver.SGD(**solver_conf),
        bias_solver=n2d2.solver.SGD(**solver_conf),
        quantizer=SATCell(
            apply_scaling=True, # Scaling needed for Full-Connected
            apply_quantization=False, # Only clamp mode for the 1st step
        ),
    )
def bn_conf(): 
    return n2d2.ConfigSection(
        activation=n2d2.activation.Rectifier(),
        scale_solver=n2d2.solver.SGD(**solver_conf),
        bias_solver=n2d2.solver.SGD(**solver_conf),
    )

### Creating model ###
print("\n### Loading Model ###")
model = n2d2.cells.Sequence([
    Conv(1, 6, kernel_dims=[5, 5], **conv_conf()),
    BatchNorm2d(6, **bn_conf()),
    Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling="Max"),
    Conv(6, 16, [5, 5], **conv_conf()),
    BatchNorm2d(16, **bn_conf()),
    Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling="Max"),
    Conv(16, 120, [5, 5], **conv_conf()),
    Dropout(name="Conv3.Dropout"),
    BatchNorm2d(120, **bn_conf()),
    Fc(120, 84, **fc_conf()),
    Dropout(name="Fc1.Dropout"),
    Fc(84, 10, **fc_conf()),
])
print(model)

loss_function = n2d2.application.CrossEntropyClassifier(provider)

print("\n### Training ###")
for epoch in range(nb_epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        x = provider.read_random_batch()
        x = model(x)
        x = loss_function(x)
        x.back_propagate()
        x.update()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')


    print("\n### Validation ###")

    loss_function.clear_success()
    
    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')


print("\n\n### Testing ###")

provider.set_partition('Test')
model.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
    batch_idx = i*batch_size

    x = provider.read_batch(batch_idx)
    x = model(x)
    x = loss_function(x)

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')

print("\n")

### Exporting weights ###
x.get_deepnet().export_network_free_parameters("./weights_clamped")
