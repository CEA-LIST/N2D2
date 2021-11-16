"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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

import n2d2
import math
import argparse
import time

t = time.time()

# ARGUMENTS PARSING
parser = argparse.ArgumentParser()

parser.add_argument('--device', '-d', type=int, default=0, help='GPU device (default=0)')
parser.add_argument('--fit', action='store_true', help='Train with the fit method')
parser.add_argument('--epochs', "-e", type=int, default=10, help='Number of epochs (default=10)')
parser.add_argument("--onnx", type=str, default="$ONNX_PATH/LeNet.onnx", help='Path to the onnx file')
args = parser.parse_args()



# Change default model
n2d2.global_variables.default_model = "Frame_CUDA"
# Change cuda device (default 0)
n2d2.global_variables.cuda_device = args.device
nb_epochs = args.epochs
batch_size = 256

print("\n### Create database ###")
database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw/", validation=0.1)

print("\n### Create Provider ###")
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=28, height=28))
print(provider)

print("\n### Loading Model ###")
model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, args.onnx)
print(model)

if args.fit:
    model.fit(learn_epoch=nb_epochs, valid_metric='Accuracy')

    model.run_test()
else:
    target = n2d2.target.Score(provider)
    print("\n### Training ###")
    for epoch in range(nb_epochs):

        provider.set_partition("Learn")
        model.learn()

        print("\n# Train Epoch: " + str(epoch) + " #")

        for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

            x = provider.read_random_batch()
            x = model(x)
            x = target(x)

            x.back_propagate()
            x.update()

            print("Example: " + str(i * batch_size) + ", loss: "
                + "{0:.3f}".format(x[0]), end='\r')


        print("\n### Validation ###")

        target.clear_success()
        
        provider.set_partition('Validation')
        model.test()

        for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
            batch_idx = i * batch_size

            x = provider.read_batch(batch_idx)
            x = model(x)
            x = target(x)

            print("Validate example: " + str(i * batch_size) + ", val success: "
                + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')


    print("\n\n### Testing ###")

    provider.set_partition('Test')
    model.test()
    target.clear_success()
    for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
        batch_idx = i*batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = target(x)

        print("Example: " + str(i * batch_size) + ", test success: "
            + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')

    print("\n")

    # save a confusion matrix
    target.log_confusion_matrix("lenet_confusion_matrix")
    # save a graph of the loss and the validation score as a function of the number of steps
    target.log_success("lenet_success")


print(f"\n{time.time()-t}")