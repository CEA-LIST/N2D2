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




import n2d2

import math
import argparse


# ARGUMENTS PARSING
parser = argparse.ArgumentParser(description="Testbench for several standard architectures on the ILSVRC2012 dataset")
parser.add_argument('--arch', type=str, default='MobileNet_v1', metavar='N',
                    help='MobileNet_v1|MobileNet_v1_batchnorm|')
parser.add_argument('--weights', type=str, default='', metavar='N',
                    help='weights directory')
parser.add_argument('--epochs', type=int, default=-1, metavar='S',
                    help='Nb Epochs. 0 is testing only (default: architecture dependent)')
parser.add_argument('--dev', type=int, default=0, metavar='S',
                    help='cuda device (default: 0)')
args = parser.parse_args()


n2d2.global_variables.set_cuda_device(args.dev)
n2d2.global_variables.default_model = "Frame_CUDA"


if args.arch == 'MobileNet_v1':
    batch_size = 256
elif args.arch == 'MobileNet_v1_batchnorm':
    batch_size = 64
elif args.arch == 'MobileNet_v2':
    batch_size = 256
elif args.arch == 'ResNet-50-BN':
    batch_size = 64
elif args.arch == 'ResNet-onnx':
    batch_size = 16
else:
    raise ValueError("Invalid architecture: " + args.arch)

avg_window = int(10000 / batch_size)

size = 224

print("Create database")
database = n2d2.database.ILSVRC2012(learn=1.0, randomPartitioning=False)
database.load("/nvme0/DATABASE/ILSVRC2012", labelPath="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
print(database)

print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[size, size, 3], batchSize=batch_size)


print("Add transformation")
trans, otf_trans = n2d2.model.ILSVRC_preprocessing(size=size)

print(trans)
print(otf_trans)
provider.add_transformation(trans)
provider.add_on_the_fly_transformation(otf_trans)



if args.arch == 'MobileNet_v1':
    """Equivalent to N2D2/models/MobileNet_v1.ini. Typically around 56% test Top1"""
    nb_epochs = 120 if args.epochs == -1 else args.epochs
    size = 160
    model = n2d2.model.Mobilenet_v1(provider, alpha=0.5, extractor_as_deepnet=True)
    # 55.87% test Top1
    # "/local/is154584/jt251134/ILSVRC_testbench/MobileNet_v1/weights_validation/"
elif args.arch == 'MobileNet_v1_batchnorm':
    """Equivalent to N2D2/models/MobileNet_v1_batchnorm.ini. Typically around 59.1% test Top1"""
    nb_epochs = 120 if args.epochs == -1 else args.epochs
    size = 224
    model = n2d2.model.Mobilenet_v1(provider, alpha=0.5, with_batchnorm=True)
elif args.arch == 'MobileNet_v2':
    """Equivalent to N2D2/models/MobileNet_v2.ini"""
    nb_epochs = 120 if args.epochs == -1 else args.epochs
    size = 160
    model = n2d2.model.Mobilenet_v2(alpha=0.5, size=size, l=10, expansion=6)
elif args.arch == 'ResNet-50-BN':
    nb_epochs = 90 if args.epochs == -1 else args.epochs
    size = 224
    model = n2d2.model.ResNet50BN(provider, output_size=1000, alpha=1.0, l=0)
elif args.arch == 'ResNet-onnx':
    nb_epochs = 0 if args.epochs == -1 else args.epochs
    model = n2d2.model.resnet.load_from_ONNX('18', 'post_act', download=True, batch_size=batch_size)
else:
    raise ValueError("Invalid architecture: " + args.arch)

print(model)

#model.set_ILSVRC_solvers(int((database.get_nb_stimuli('Learn')*nb_epochs)/batch_size))

print("Create classifier")
classifier = n2d2.application.Classifier(provider, model, topN=1)


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


    print("\n### Validate Epoch: " + str(epoch) + " ###")

    classifier.set_mode('Validation')

    for i in range(math.ceil(database.get_nb_stimuli('Validation')/batch_size)):

        batch_idx = i*batch_size

        # Load example
        classifier.read_batch(idx=batch_idx)

        classifier.process()

        print("Example: " + str(i*batch_size) + ", val success: "
              + "{0:.2f}".format(100 * classifier.get_average_score(metric='Sensitivity')) + "%", end='\r')




print("\n### Test ###")

classifier.set_mode('Test')

for i in range(math.ceil(database.get_nb_stimuli('Test')/batch_size)):

    batch_idx = i*batch_size

    # Load example
    classifier.read_batch(idx=batch_idx)

    classifier.process()

    print("Example: " + str(i*batch_size) + ", test success: "
          + "{0:.2f}".format(100 * classifier.get_average_success()) + "%", end='\r')


