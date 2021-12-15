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

parser = argparse.ArgumentParser(description="Train MLP on MNIST dataset")
parser.add_argument('--epochs', type=int, default=5, metavar='S',
                    help='Nb Epochs. 0 is testing only (default: 120)')
parser.add_argument('--batch_size', type=int, default=64, metavar='S',
                    help='Batch size')
parser.add_argument("--data_path", type=str, help="Path to MNIST dataset")

args = parser.parse_args()

if args.data_path is None:
    raise RuntimeError("Please give a path for the MNIST dataset")

n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.detailed

database = n2d2.database.MNIST(data_path=args.data_path)
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=args.batch_size)

fc1 = n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier())
fc2 = n2d2.cells.Fc(50, 10)
softmax = n2d2.cells.nn.Softmax(with_loss=True)
target = n2d2.target.Score(provider)

fc1.solver = n2d2.solver.SGD(learning_rate=0.01)
fc2.solver = n2d2.solver.SGD(learning_rate=0.01)

provider.set_partition("Learn")

for epoch in range(args.epochs):

    print("\n### Train Epoch: " + str(epoch) + " ###")

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/args.batch_size)):

        x = provider.read_random_batch()
        x = fc1(x)
        x = fc2(x)
        x = softmax(x)
        x = target(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i*args.batch_size) + ", loss: "
              + "{0:.3f}".format(target.loss()), end='\r')

print("\n")

provider.set_partition('Test')

fc1.test()
fc2.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/args.batch_size)):
    batch_idx = i*args.batch_size

    x = provider.read_batch(idx=batch_idx)
    x = fc1(x)
    x = fc2(x)
    x = softmax(x)
    x = target(x)

    print("Example: " + str(i * args.batch_size) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')

print("\n")



