"""
    (C) Copyright 2023 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
This script showcase how to use data loaded with numpy to train your Network.
In this example we use the Keras dataloader fashion mnist : https://keras.io/api/datasets/fashion_mnist/
To learn a minimal LeNet Network
"""

from tensorflow.keras.datasets.fashion_mnist import load_data
import n2d2
from n2d2.cells.nn import Fc, Softmax
from n2d2.cells import Sequence
from n2d2.solver import SGD
from n2d2.activation import Rectifier, Linear
from math import ceil
import argparse

# ARGUMENTS PARSING
parser = argparse.ArgumentParser()

parser.add_argument('--dev', '-d', type=int, default=0, help='GPU device, only if CUDA is available. (default=0)')
parser.add_argument('--epochs', "-e", type=int, default=10, help='Number of epochs (default=10)')
parser.add_argument('--batch_size', "-b", type=int, default=32, help='Batchsize (default=32)')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCH = args.epochs

if n2d2.global_variables.cuda_available:
    n2d2.global_variables.default_model = "Frame_CUDA"
    n2d2.global_variables.cuda_device = args.dev
else:
    print("CUDA is not available")
(x_train, y_train), (x_test, y_test) = load_data()

db = n2d2.database.Numpy()

# x_train is a numpy array of shape [nb train, 28, 28].
# `n2d2.database.numpy.load` only take a list of stimuli.
# So we create a list of numpy array of shape [28, 28] using list comprehension.
db.load([a for a in x_train], [(int)(i.item()) for i in y_train])
db.partition_stimuli(1., 0., 0.) # Learn Validation Test

# Using test set for validation
db.load([a for a in x_test], [(int)(i.item()) for i in y_test], partition="Validation")

db.get_partition_summary()

model = Sequence([
        Fc(28*28, 128, activation=Rectifier()),
        Fc(128, 10, activation=Linear()),
    ])
softmax = Softmax(with_loss=True)
model.set_solver(SGD(learning_rate=0.001))

print("Model :")
print(model)


provider = n2d2.provider.DataProvider(db, [28, 28, 1], batch_size=BATCH_SIZE)

provider.set_partition("Learn")

target = n2d2.target.Score(provider)

print("\n### Training ###")
for epoch in range(EPOCH):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(ceil(db.get_nb_stimuli('Learn')/BATCH_SIZE)):

        x = provider.read_random_batch()
        x = model(x)
        x = softmax(x)
        x = target(x)
        x.back_propagate()
        x.update()

        print("Example: " + str(i * BATCH_SIZE) + ", loss: "
              + "{0:.3f}".format(target.loss()), end='\r')

    print("\n### Validation ###")

    target.clear_success()
    
    provider.set_partition('Validation')
    model.test()

    for i in range(ceil(db.get_nb_stimuli('Validation')/BATCH_SIZE)):
        batch_idx = i * BATCH_SIZE

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = target(x)

        print("Validate example: " + str(i * BATCH_SIZE) + ", val success: "
              + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')
print("\nEND")
