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

parser = argparse.ArgumentParser(description="Train mobilenetv1 on the ILSVRC2012 dataset")
parser.add_argument('--with_batchnorm', action='store_true',
                    help='use mobilenet with batch norm layers or without')
parser.add_argument('--epochs', type=int, default=120, metavar='S',
                    help='Nb Epochs. 0 is testing only (default: 120)')
parser.add_argument('--batch_size', type=int, default=64, metavar='S',
                    help='Batch size')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
parser.add_argument("--data_path", type=str, help="Path to ILSVRC2012 dataset")
parser.add_argument("--label_path", type=str, help="Path to ILSVRC2012 labels")
parser.add_argument("--save_path", type=str, help="Path to save trained parameters")

args = parser.parse_args()

n2d2.global_variables.cuda_device = args.dev
n2d2.global_variables.default_model = "Frame_CUDA"
n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.detailed

print("Create database")
database = n2d2.database.ILSVRC2012(learn=0.95, random_partitioning=False)
database.load(args.data_path, label_path=args.label_path)
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[224, 224, 3], batch_size=args.batch_size)

model = n2d2.models.MobileNetv1(alpha=1.0, with_bn=args.with_batchnorm)
trans, otf_trans = n2d2.models.ILSVRC_preprocessing(size=224)
provider.add_transformation(trans)
provider.add_on_the_fly_transformation(otf_trans)

print("Create classifier")
softmax = n2d2.cells.nn.Softmax(with_loss=True)
target = n2d2.target.Score(provider, top_n=1)

model.set_solver(n2d2.solver.SGD(learning_rate=0.01, momentum=0.9, decay=0.0001, learning_rate_policy='PolyDecay',
                                 power=1.0,
                                 max_iterations=int(args.epochs*database.get_nb_stimuli('Learn')/args.batch_size)))

print(model)

print("\n### Training ###")
for epoch in range(args.epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / args.batch_size)):
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

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / args.batch_size)):
        batch_idx = i * args.batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = target(x)

        print("Validate example: " + str(i * args.batch_size) + ", val success: "
              + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')

if args.save_path is not None:
    model.export_free_parameters(args.save_path)

print("\n\n### Testing ###")

provider.set_partition('Test')
model.test()

for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test') / args.batch_size)):
    batch_idx = i * args.batch_size

    x = provider.read_batch(batch_idx)
    x = model(x)
    x = softmax(x)
    x = target(x)

    print("Example: " + str(i * args.batch_size) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')


