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
import argparse

parser = argparse.ArgumentParser(description="Train LeNet on MNIST dataset")
parser.add_argument('--epochs', type=int, default=10, metavar='S',
                    help='Nb Epochs. 0 is testing only (default: 120)')
parser.add_argument('--batch_size', type=int, default=64, metavar='S',
                    help='Batch size')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
parser.add_argument("--data_path", type=str, help="Path to MNIST dataset")

args = parser.parse_args()

n2d2.global_variables.cuda_device = args.dev
n2d2.global_variables.default_model = "Frame_CUDA"
n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.detailed


print("\n### Create database ###")
database = n2d2.database.MNIST(data_path=args.data_path, validation=0.1)
print(database)

print("\n### Create Provider ###")
provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=args.batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
print(provider)

print("\n### Loading Model ###")
model = n2d2.models.lenet.LeNet(10)

softmax = n2d2.cells.nn.Softmax(with_loss=True)
target = n2d2.target.Score(provider)

model.set_solver(n2d2.solver.SGD(learning_rate=0.01, momentum=0.9))

print(model)

print("\n### Training ###")
for epoch in range(args.epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/args.batch_size)):

        x = provider.read_random_batch()

        x = model(x)
        x = softmax(x)
        x = target(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * args.batch_size) + ", loss: "
              + "{0:.3f}".format(target.loss()), end='\r')

    print("\n### Validation ###")

    target.clear_success()
    
    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation')/args.batch_size)):
        batch_idx = i * args.batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = target(x)

        print("Validate example: " + str(i * args.batch_size) + ", val success: "
              + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')


print("\n\n### Testing ###")

provider.set_partition('Test')
model.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/args.batch_size)):
    batch_idx = i*args.batch_size

    x = provider.read_batch(batch_idx)
    x = model(x)
    x = softmax(x)
    x = target(x)

    print("Example: " + str(i * args.batch_size) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')

print("\n")
# save a confusion matrix
target.log_confusion_matrix("lenet_confusion_matrix")
# save a graph of the loss and the validation score as a function of the number of steps
target.log_success("lenet_success")
